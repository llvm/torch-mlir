//===- acap_dispatch.cpp --------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "acap_dispatch.h"

#include "mlir-c/StandardAttributes.h"
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
    // TODO: Add mlirValueIsNull()
    if (!v.ptr) {
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

  // TODO: Get location from traceback.
  MlirLocation loc = mlirLocationUnknownGet(funcBuilder->getContext());
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
    // TODO: Add upstream mlirValueIsNull
    if (!mlirValue.ptr) {
      std::stringstream out;
      out << "Unsupported capture value passed to kernel (" << argIt->tagKind()
          << "): " << *argIt;
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
      out << "Unsupported capture value returned from kernel ("
          << returnIt->tagKind() << "): " << *returnIt;
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
    // TODO: Add mlirValueIsNull()
    if (mappedValue.ptr) {
      return mappedValue;
    }

    throw std::invalid_argument(
        "TODO: implement tensor import for non-arg tensors");
  }
  return {nullptr};
  // TODO: Implement mappings for the whole set (relevant to this use case):
  // _(None)
  // _(Tensor)
  // _(Double)
  // _(Int)
  // _(Bool)
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
  return {nullptr};
}

TORCH_LIBRARY_IMPL(_, ACAP_DISPATCH_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &AcapController::fallbackKernel>());
}
