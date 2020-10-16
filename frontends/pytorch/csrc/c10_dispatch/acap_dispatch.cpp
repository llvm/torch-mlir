//===- acap_dispatch.cpp --------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "acap_dispatch.h"

#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Types.h"
#include "npcomp/Python/PybindUtils.h"

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKey.h>
#include <torch/library.h>

using namespace torch_mlir;

namespace py = pybind11;

using c10::FunctionSchema;
using c10::OperatorHandle;
using c10::Stack;

// TODO: Private use dispatch keys are not made for real uses. Allocate a proper
// dispatch key in upstream PyTorch (DispatchKey.h) prior to maturity. Note
// that the TORCH_LIBRARY_* macros expand this by name and other APIs use its
// enum value, so we define both. We can get rid of both once we have our
// own key.
#define ACAP_DISPATCH_KEY PrivateUse1
static c10::DispatchKey kAcapDispatchKey = c10::DispatchKey::ACAP_DISPATCH_KEY;

std::list<AcapController::Activation> &
AcapController::getThreadLocalActiveStack() {
  static thread_local std::list<Activation> threadLocalActiveStack;
  return threadLocalActiveStack;
}

py::object AcapController::contextEnter() {
  auto &stack = getThreadLocalActiveStack();
  stack.emplace_front(shared_from_this());
  Activation &current = stack.front();
  current.dispatchGuard =
      std::make_unique<c10::impl::IncludeDispatchKeyGuard>(kAcapDispatchKey);
  return py::cast(this);
}

void AcapController::contextExit(py::object exc_type, py::object exc_val,
                                 py::object exc_tb) {
  auto &stack = getThreadLocalActiveStack();
  if (stack.empty() || stack.front().controller.get() != this) {
    throw py::raisePyError(PyExc_RuntimeError,
                           "Mismatched context manager __exit__");
  }
  stack.pop_front();

  if (!hasReturned) {
    returns({});
  }
}

void AcapController::returns(std::vector<at::Tensor> tensors) {
  verifyHasNotReturned();

  llvm::SmallVector<MlirType, 4> returnsTypes;
  llvm::SmallVector<MlirValue, 4> returnsValues;
  for (auto &tensor : tensors) {
    MlirValue v = funcBuilder->lookupTensor(tensor);
    if (mlirValueIsNull(v)) {
      // Exclude recursive dispatch in order to print tensor.
      c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);
      std::stringstream msg;
      msg << "Cannot return a tensor that is not from the capture context: ";
      msg << tensor;
      throw std::invalid_argument(msg.str());
    }

    returnsTypes.push_back(mlirValueGetType(v));
    returnsValues.push_back(v);
  }

  MlirLocation loc = getCurrentLocation();
  OperationStateHolder s("std.return", loc);
  mlirOperationStateAddOperands(&s.state, returnsValues.size(),
                                returnsValues.data());
  funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(
      s.createOperation());
  funcBuilder->rewriteFuncReturnTypes(returnsTypes);
  hasReturned = true;
}

std::vector<std::string> AcapController::getDebugLog() {
  std::vector<std::string> copy;
  captureLog.swap(copy);
  return copy;
}

std::shared_ptr<AcapController> AcapController::getCurrent() {
  auto &stack = getThreadLocalActiveStack();
  if (stack.empty())
    return nullptr;
  return stack.front().controller;
}

void AcapController::verifyHasNotReturned() {
  if (hasReturned) {
    throw std::runtime_error(
        "Function has already returned. Cannot trace more operations.");
  }
}

/* static */
void AcapController::fallbackKernel(const OperatorHandle &opHandle,
                                    Stack *stack) {
  auto current = getCurrent();
  if (!current) {
    current->redispatch(opHandle, stack);
    return;
  }
  current->fallbackKernelImpl(opHandle, stack);
}

MlirLocation AcapController::getCurrentLocation() {
  return mlirLocationUnknownGet(funcBuilder->getContext());
}

void AcapController::redispatch(const c10::OperatorHandle &opHandle,
                                c10::Stack *stack) {
  // Exclude recursive dispatch to this kernel.
  c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);
  // Passthrough.
  auto &dispatcher = c10::Dispatcher::singleton();
  dispatcher.callBoxed(opHandle, stack);
}

void AcapController::fallbackKernelImpl(const OperatorHandle &opHandle,
                                        Stack *stack) {
  verifyHasNotReturned();
  // Exclude recursive dispatch to this kernel.
  c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);

  const FunctionSchema &schema = opHandle.schema();

  // Check for unsupported.
  if (schema.is_vararg() || schema.is_varret()) {
    throw std::invalid_argument(
        "Cannot capture ops with variable arguments or returns");
  }

  // TODO: Extract actual location from stack.
  MlirContext context = funcBuilder->getContext();
  MlirLocation loc = mlirLocationUnknownGet(context);
  OperationStateHolder stateHolder("torch.kernel_call", loc);

  // Add the kernel_name attribute.
  auto kernelName = schema.name();
  MlirNamedAttribute kernelNameAttr = mlirNamedAttributeGet(
      "kernel_name",
      mlirStringAttrGet(context, kernelName.size(), kernelName.data()));
  mlirOperationStateAddAttributes(&stateHolder.state, 1, &kernelNameAttr);

  // Map arguments to operands.
  // This must be accumulated into the OperationState prior to re-dispatch
  // since the stack is modified at that point.
  size_t argCount = schema.arguments().size();
  assert(stack->size() >= argCount && "stack too short");
  llvm::SmallVector<MlirValue, 4> operands;
  for (auto argIt = stack->end() - argCount; argIt != stack->end(); ++argIt) {
    MlirValue mlirValue = mapIValueToMlirValue(loc, *argIt);
    if (mlirValueIsNull(mlirValue)) {
      std::stringstream out;
      out << "Unsupported capture value returned from kernel '" << kernelName
          << "' (" << argIt->tagKind() << "): " << *argIt;
      throw std::invalid_argument(out.str());
    }
    operands.push_back(mlirValue);
  }
  mlirOperationStateAddOperands(&stateHolder.state, operands.size(),
                                operands.data());

  // Invoke the original kernel.
  redispatch(opHandle, stack);

  // Map returns to results.
  size_t returnCount = schema.returns().size();
  assert(stack->size() >= returnCount && "stack too short");
  llvm::SmallVector<MlirType, 4> resultTypes;
  llvm::SmallVector<std::pair<size_t, at::Tensor>, 4> resultIndexToTensorMap;
  for (auto returnIt = stack->end() - returnCount; returnIt != stack->end();
       ++returnIt) {
    size_t resultIndex = resultTypes.size();
    MlirType resultType = mapIValueToMlirType(loc, *returnIt);
    if (mlirTypeIsNull(resultType)) {
      std::stringstream out;
      out << "Unsupported capture value returned from kernel '" << kernelName
          << "' (" << returnIt->tagKind() << "): " << *returnIt;
      throw std::invalid_argument(out.str());
    }
    resultTypes.push_back(resultType);
    if (returnIt->isTensor()) {
      resultIndexToTensorMap.emplace_back(resultIndex, returnIt->toTensor());
    }
  }
  mlirOperationStateAddResults(&stateHolder.state, resultTypes.size(),
                               resultTypes.data());

  // Create operation.
  MlirOperation op = stateHolder.createOperation();
  funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(op);

  // Map result tensors.
  for (auto &it : resultIndexToTensorMap) {
    MlirValue result = mlirOperationGetResult(op, it.first);
    funcBuilder->mapTensor(it.second, result);
  }

  // Add to debug log.
  std::stringstream sout;
  sout << "CAPTURE: " << opHandle.schema() << "\n";
  captureLog.push_back(sout.str());
}

MlirValue AcapController::mapIValueToMlirValue(MlirLocation loc,
                                               c10::IValue &ival) {
  if (ival.isScalar()) {
    return funcBuilder->getScalarConstant(loc, ival.toScalar());
  }
  if (ival.isTensor()) {
    // Is it an already mapped tensor?
    MlirValue mappedValue = funcBuilder->lookupTensor(ival.toTensor());
    if (!mlirValueIsNull(mappedValue)) {
      return mappedValue;
    }

    mappedValue = importTensorByValue(ival.toTensor());
    assert(mappedValue.ptr);
    return mappedValue;
  }
  if (ival.isBool()) {
    // TODO: Switch to the numpy.bool type as that is a closer domain match.
    return funcBuilder->getBoolConstant(loc, ival.toBool());
  }
  return {nullptr};
  // TODO: Implement mappings for the whole set (relevant to this use case):
  // _(None)
  // _(Tensor)
  // _(Double)
  // _(Int)
  // _(Tuple)
  // _(String)
  // _(Blob)
  // _(GenericList)
  // _(GenericDict)
  // _(Future)
  // _(Device)
  // _(Object)
  // _(PyObject)
  // _(Uninitialized)
  // _(Capsule)
  // _(RRef)
  // _(Generator)
}

MlirType AcapController::mapIValueToMlirType(MlirLocation loc,
                                             c10::IValue &ival) {
  if (ival.isScalar()) {
    return typeMapper.mapScalarType(ival.toScalar().type());
  }
  if (ival.isTensor()) {
    return typeMapper.forwardTensorToType(ival.toTensor());
  }
  if (ival.isBool()) {
    // TODO: Switch to the numpy.bool type as that is a closer domain match.
    return mlirIntegerTypeGet(funcBuilder->getContext(), 1);
  }
  return {nullptr};
}

MlirValue AcapController::importTensorByValue(at::Tensor tensor) {
  using at::ScalarType;

  auto throwUnsupportedTensorError = [&]() {
    std::stringstream msg;
    msg << "Unsupported import tensor type: " << tensor;
    throw std::invalid_argument(msg.str());
  };

  // Get a C-contiguous form as we can bulk-load that into a DenseElementsAttr.
  if (!tensor.is_contiguous())
    tensor = tensor.contiguous();

  // The flat number of bytes throws an exception for tensors that are not
  // dense and accessible as such.
  at::checkLayout(at::CheckedFrom("accessing contiguous"), tensor,
                  c10::Layout::Strided);

  // Construct the ShapedType.
  auto loc = getCurrentLocation();
  MlirType elementType = typeMapper.mapScalarType(tensor.scalar_type());
  llvm::SmallVector<int64_t, 4> shape(tensor.sizes().begin(),
                                      tensor.sizes().end());
  MlirType shapedType = mlirRankedTensorTypeGetChecked(
      shape.size(), shape.data(), elementType, loc);
  if (mlirTypeIsNull(shapedType)) {
    throwUnsupportedTensorError();
  }

  // Import DenseElementsAttr data.
  // TODO: Support bool tensors.
  // TODO: More import formats in C-API.
  MlirAttribute valueAttribute;
  auto numElements = tensor.numel();
  auto tensorData = tensor.data_ptr();
  switch (tensor.scalar_type()) {
  case ScalarType::Int:
    valueAttribute = mlirDenseElementsAttrInt32Get(
        shapedType, numElements, static_cast<const int32_t *>(tensorData));
    break;
  case ScalarType::Long:
    valueAttribute = mlirDenseElementsAttrInt64Get(
        shapedType, numElements, static_cast<const int64_t *>(tensorData));
    break;
  case ScalarType::Float:
    valueAttribute = mlirDenseElementsAttrFloatGet(
        shapedType, numElements, static_cast<const float *>(tensorData));
    break;
  case ScalarType::Double:
    valueAttribute = mlirDenseElementsAttrDoubleGet(
        shapedType, numElements, static_cast<const double *>(tensorData));
    break;
  default:
    throwUnsupportedTensorError();
  }
  MlirValue constTensorValue =
      funcBuilder->getGeneralConstant(loc, valueAttribute);

  // Create an array from the tensor constant via the
  // numpy.create_array_from_tensor op.
  MlirType constArrayType = npcompNdArrayTypeGetFromShaped(shapedType);
  MlirOperationState state =
      mlirOperationStateGet("numpy.create_array_from_tensor", loc);
  mlirOperationStateAddOperands(&state, 1, &constTensorValue);
  mlirOperationStateAddResults(&state, 1, &constArrayType);
  MlirOperation constArrayOp = mlirOperationCreate(&state);

  funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(constArrayOp);
  MlirValue constArrayValue = mlirOperationGetResult(constArrayOp, 0);
  funcBuilder->mapTensor(tensor, constArrayValue);
  return constArrayValue;
}

TORCH_LIBRARY_IMPL(_, ACAP_DISPATCH_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &AcapController::fallbackKernel>());
}
