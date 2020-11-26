//===- acap_dispatch.cpp --------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "acap_dispatch.h"
#include "debug.h"
#include "mlir_utils.h"

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
using c10::IValue;
using c10::OperatorHandle;
using c10::Stack;

// TODO: Private use dispatch keys are not made for real uses. Allocate a proper
// dispatch key in upstream PyTorch (DispatchKey.h) prior to maturity. Note
// that the TORCH_LIBRARY_* macros expand this by name and other APIs use its
// enum value, so we define both. We can get rid of both once we have our
// own key.
// TODO: Ask the PT devs why conv is special and only shows up if dispatching
// through the autograd keys.
// https://github.com/llvm/mlir-npcomp/issues/86
#define ACAP_DISPATCH_KEY PrivateUse2
#define ACAP_GRAD_DISPATCH_KEY AutogradPrivateUse2
static c10::DispatchKey kAcapDispatchKey = c10::DispatchKey::ACAP_DISPATCH_KEY;
static c10::DispatchKey kAcapGradDispatchKey =
    c10::DispatchKey::ACAP_GRAD_DISPATCH_KEY;

AcapController::TracedKernelCallBuilder::TracedKernelCallBuilder(
    AcapController &parent, MlirContext context, MlirLocation loc,
    const c10::OperatorHandle &opHandle,
    llvm::Optional<std::string> overrideKernelName)
    : KernelCallBuilder(context, loc,
                        overrideKernelName ? *overrideKernelName
                                           : opHandle.operator_name().name,
                        opHandle.schema()),
      parent(parent), opHandle(opHandle) {}

void AcapController::TracedKernelCallBuilder::addOperand(const IValue &value) {
  MlirValue mlirValue = parent.mapIValueToMlirValue(loc, value);
  if (mlirValueIsNull(mlirValue)) {
    std::stringstream out;
    const std::string &kernelName = opHandle.operator_name().name;
    out << "Unsupported capture value passed to kernel '" << kernelName << "' ("
        << value.tagKind() << "): " << value;
    throw std::invalid_argument(out.str());
  }
  KernelCallBuilder::addOperand(mlirValue);
}

void AcapController::TracedKernelCallBuilder::addResult(const IValue &value) {
  MlirType resultType = parent.mapIValueToMlirType(loc, value);
  if (mlirTypeIsNull(resultType)) {
    std::stringstream out;
    const std::string &kernelName = opHandle.operator_name().name;
    out << "Unsupported capture value returned from kernel '" << kernelName
        << "' (" << value.tagKind() << "): " << value;
    throw std::invalid_argument(out.str());
  }
  if (value.isTensor()) {
    resultIndexToTensorMap.emplace_back(resultCount++, value.toTensor());
  }
  KernelCallBuilder::addResultType(resultType);
}

MlirOperation AcapController::TracedKernelCallBuilder::create() {
  MlirOperation op = KernelCallBuilder::create();
  parent.funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(op);

  // Map result tensors.
  for (auto &it : resultIndexToTensorMap) {
    MlirValue result = mlirOperationGetResult(op, it.first);
    parent.funcBuilder->mapTensor(it.second, result);
  }
  return op;
}

std::list<AcapController::Activation> &
AcapController::getThreadLocalActiveStack() {
  static thread_local std::list<Activation> threadLocalActiveStack;
  return threadLocalActiveStack;
}

py::object AcapController::contextEnter() {
  auto &stack = getThreadLocalActiveStack();
  stack.emplace_front(shared_from_this());
  Activation &current = stack.front();
  c10::DispatchKeySet keySet{kAcapDispatchKey, kAcapGradDispatchKey};
  current.includeGuard =
      std::make_unique<c10::impl::IncludeDispatchKeyGuard>(keySet);
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
      debugTrace(
          "Return of imported-constant tensor (intentional memorization?)");
      v = importTensorByValue(tensor);
    }

    returnsTypes.push_back(mlirValueGetType(v));
    returnsValues.push_back(v);
  }

  MlirLocation loc = getCurrentLocation();
  OperationStateHolder s("std.return", loc);
  mlirOperationStateAddOperands(s, returnsValues.size(), returnsValues.data());
  funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(
      s.createOperation());
  funcBuilder->rewriteFuncReturnTypes(returnsTypes);
  hasReturned = true;
  if (onReturn)
    onReturn(returnsTypes);
}

std::shared_ptr<AcapController>
AcapController::getCurrentThreadAcapController() {
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
  auto redispatchCallback = [&]() {
    // Exclude recursive dispatch to this kernel.
    c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);
    // Passthrough.
    auto &dispatcher = c10::Dispatcher::singleton();
    dispatcher.callBoxed(opHandle, stack);
  };

  auto current = getCurrentThreadAcapController();
  if (!current) {
    redispatchCallback();
    return;
  }
  current->fallbackKernelImpl(opHandle, stack, redispatchCallback);
}

at::Tensor AcapController::convolutionKernel(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const at::IntArrayRef stride,
    const at::IntArrayRef padding, const at::IntArrayRef dilation,
    const bool transposed, const at::IntArrayRef output_padding,
    const int64_t groups) {
  static c10::OperatorName opName{"aten::convolution", ""};
  auto &dispatcher = c10::Dispatcher::singleton();
  auto opHandle = dispatcher.findOp(opName);
  assert(opHandle && "could not find convolution op");
  if (isDebugTraceEnabled()) {
    std::stringstream s;
    s << "Convolution (unboxed) dispatch: " << opHandle->schema();
    debugTrace(s.str());
  }

  auto opTyped = opHandle->typed<at::Tensor(
      const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &,
      const at::IntArrayRef, const at::IntArrayRef, const at::IntArrayRef,
      const bool, const at::IntArrayRef, const int64_t)>();

  // Exclude recursive calls: convolution is completely emitted by this
  // kernel.
  c10::DispatchKeySet keySet{kAcapDispatchKey, kAcapGradDispatchKey};
  c10::impl::ExcludeDispatchKeyGuard exclusion(keySet);

  auto current = getCurrentThreadAcapController();
  if (!current) {
    return opTyped.callWithDispatchKey(c10::DispatchKey::AutogradOther, input,
                                       weight, bias, stride, padding, dilation,
                                       transposed, output_padding, groups);
  }

  MlirContext context = current->funcBuilder->getContext();
  MlirLocation loc = current->getCurrentLocation();
  std::string kernelName{"aten::convolution"};
  TracedKernelCallBuilder callBuilder{*current, context, loc, *opHandle};

  callBuilder.addOperand(IValue(input));
  callBuilder.addOperand(IValue(weight));
  // This is really sad: instead of storing a none in the optional, it stores
  // an undefined tensor, which cannot convert to an IValue :(
  // TODO: File PyTorch bug. Perhaps this is why they don't support boxing
  // for it.
  IValue biasIValue;
  if (bias && bias->defined()) {
    biasIValue = IValue(bias);
  } else {
    biasIValue = IValue(c10::optional<at::Tensor>());
  }
  callBuilder.addOperand(biasIValue);
  callBuilder.addOperand(IValue(stride));
  callBuilder.addOperand(IValue(padding));
  callBuilder.addOperand(IValue(dilation));
  callBuilder.addOperand(IValue(transposed));
  callBuilder.addOperand(IValue(output_padding));
  callBuilder.addOperand(IValue(groups));

  auto result = opTyped.callWithDispatchKey(
      c10::DispatchKey::AutogradOther, input, weight, bias, stride, padding,
      dilation, transposed, output_padding, groups);
  callBuilder.addResult(result);
  callBuilder.create();
  return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AcapController::mklConvolutionBackward(
    const at::Tensor &input, const at::Tensor &grad_output,
    const at::Tensor &weight, const at::IntArrayRef padding,
    const at::IntArrayRef stride, const at::IntArrayRef dilation,
    const int64_t groups, std::array<bool, 3> output_mask) {
  static c10::OperatorName opName{"aten::mkldnn_convolution_backward", ""};
  auto &dispatcher = c10::Dispatcher::singleton();
  auto opHandle = dispatcher.findOp(opName);
  assert(opHandle && "could not find mkldnn_convolution_backward op");
  if (isDebugTraceEnabled()) {
    std::stringstream s;
    s << "mkldnn_convolution_backward dispatch: " << opHandle->schema();
    debugTrace(s.str());
  }

  auto opTyped = opHandle->typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      const at::Tensor &input, const at::Tensor &grad_output,
      const at::Tensor &weight, const at::IntArrayRef padding,
      const at::IntArrayRef stride, const at::IntArrayRef dilation,
      const int64_t groups, std::array<bool, 3> output_mask)>();

  // Exclude recursive calls: convolution is completely emitted by this
  // kernel.
  c10::DispatchKeySet keySet{kAcapDispatchKey, kAcapGradDispatchKey};
  c10::impl::ExcludeDispatchKeyGuard exclusion(keySet);

  auto current = getCurrentThreadAcapController();
  if (!current) {
    return opTyped.callWithDispatchKey(c10::DispatchKey::AutogradOther, input,
                                       grad_output, weight, padding, stride,
                                       dilation, groups, output_mask);
  }

  // Emit the call as if to aten::convolution_overridable, the generic, full
  // parameterized versions that backends are supposed to implement.
  // Requires some parameter swizzling.
  // It has the signature:
  // convolution_backward_overrideable(Tensor grad_output, Tensor input,
  //   Tensor weight, int[] stride, int[] padding, int[] dilation,
  //   bool transposed, int[] output_padding, int groups,
  //   bool[3] output_mask) ->
  //     (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
  MlirContext context = current->funcBuilder->getContext();
  MlirLocation loc = current->getCurrentLocation();
  std::string kernelName{"aten::convolution_backward"};
  static c10::OperatorName emitOpName{"aten::convolution_backward_overrideable",
                                      ""};
  auto emitOpHandle = dispatcher.findOp(emitOpName);
  assert(emitOpHandle && "could not find convolution_backward_overrideable op");
  TracedKernelCallBuilder callBuilder{*current, context, loc, *emitOpHandle,
                                      kernelName};

  callBuilder.addOperand(IValue(grad_output));
  callBuilder.addOperand(IValue(input));
  callBuilder.addOperand(IValue(weight));
  callBuilder.addOperand(IValue(stride));
  callBuilder.addOperand(IValue(padding));
  callBuilder.addOperand(IValue(dilation));
  callBuilder.addOperand(IValue(false));
  std::vector<int64_t> output_padding(padding.size()); // Not provided.
  callBuilder.addOperand(IValue(at::IntArrayRef(output_padding)));
  callBuilder.addOperand(IValue(groups));
  callBuilder.addOperand(IValue(output_mask));

  auto results = opTyped.callWithDispatchKey(
      c10::DispatchKey::AutogradCPU, input, grad_output, weight, padding,
      stride, dilation, groups, output_mask);

  callBuilder.addResult(std::get<0>(results));
  callBuilder.addResult(std::get<1>(results));
  callBuilder.addResult(std::get<2>(results));
  callBuilder.create();
  return results;
}

at::Tensor &AcapController::copyUnderKernel(at::Tensor &self,
                                            const at::Tensor &src,
                                            bool non_blocking) {
  static c10::OperatorName opName{"aten::copy_", ""};
  auto &dispatcher = c10::Dispatcher::singleton();
  auto opHandle = dispatcher.findOp(opName);
  assert(opHandle && "could not find copy_ op");
  if (isDebugTraceEnabled()) {
    std::stringstream s;
    s << "copy_ dispatch: " << opHandle->schema();
    debugTrace(s.str());
  }

  auto opTyped = opHandle->typed<at::Tensor &(
      at::Tensor & self, const at::Tensor &src, bool non_blocking)>();

  // Exclude recursive calls.
  c10::DispatchKeySet keySet{kAcapDispatchKey, kAcapGradDispatchKey};
  c10::impl::ExcludeDispatchKeyGuard exclusion(keySet);

  auto current = getCurrentThreadAcapController();
  if (!current) {
    return opTyped.callWithDispatchKey(c10::DispatchKey::AutogradOther, self,
                                       src, non_blocking);
  }

  MlirContext context = current->funcBuilder->getContext();
  MlirLocation loc = current->getCurrentLocation();
  TracedKernelCallBuilder callBuilder{*current, context, loc, *opHandle};

  callBuilder.addOperand(IValue(self));
  callBuilder.addOperand(IValue(src));
  auto &result = opTyped.callWithDispatchKey(c10::DispatchKey::CPU, self, src,
                                             non_blocking);
  callBuilder.addResult(result);
  callBuilder.create();
  return result;
}

at::Tensor AcapController::arangeBackendSelectKernel(
    at::Scalar end, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  static c10::OperatorName opName{"aten::arange", ""};
  auto &dispatcher = c10::Dispatcher::singleton();
  auto opHandle = dispatcher.findOp(opName);
  assert(opHandle && "could not find arange op");

  // Exclude recursive calls.
  c10::DispatchKeySet keySet{kAcapDispatchKey, kAcapGradDispatchKey};
  c10::impl::ExcludeDispatchKeyGuard exclusion(keySet);

  // Dispatching in this fashion replicates the exact way that PyTorch
  // built-in handlers dispatch to BackendSelect kernels.
  auto targetDk = c10::computeDispatchKey(dtype, layout, device);
  auto opTyped = opHandle->typed<at::Tensor(
      at::Scalar end, c10::optional<at::ScalarType> dtype,
      c10::optional<at::Layout> layout, c10::optional<at::Device> device,
      c10::optional<bool> pin_memory)>();
  return opTyped.callWithDispatchKey(targetDk, end, dtype, layout, device,
                                     pin_memory);
}

MlirLocation AcapController::getCurrentLocation() {
  return mlirLocationUnknownGet(funcBuilder->getContext());
}

void AcapController::fallbackKernelImpl(
    const OperatorHandle &opHandle, Stack *stack,
    std::function<void()> redispatchCallback) {
  verifyHasNotReturned();
  if (isDebugTraceEnabled()) {
    std::stringstream s;
    s << "Fallback (boxed) dispatch: " << opHandle.schema()
      << " (stack size=" << stack->size() << ")";
    debugTrace(s.str());
  }

  // Exclude recursive dispatch to this kernel.
  c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);

  const FunctionSchema &schema = opHandle.schema();

  // Check for unsupported.
  if (schema.is_vararg() || schema.is_varret()) {
    throw std::invalid_argument(
        "Cannot capture ops with variable arguments or returns");
  }

  MlirContext context = funcBuilder->getContext();
  MlirLocation loc = getCurrentLocation();
  auto kernelName = schema.name();
  TracedKernelCallBuilder callBuilder{*this, context, loc, opHandle};

  // Map arguments to operands.
  // This must be accumulated into the OperationState prior to re-dispatch
  // since the stack is modified at that point.
  size_t argCount = schema.arguments().size();
  assert(stack->size() >= argCount && "stack too short");
  for (auto argIt = stack->end() - argCount; argIt != stack->end(); ++argIt) {
    callBuilder.addOperand(*argIt);
  }

  // Invoke the original kernel.
  redispatchCallback();

  // Map returns to results.
  size_t returnCount = schema.returns().size();
  assert(stack->size() >= returnCount && "stack too short");
  for (auto returnIt = stack->end() - returnCount; returnIt != stack->end();
       ++returnIt) {
    callBuilder.addResult(*returnIt);
  }

  callBuilder.create();
}

MlirValue AcapController::mapIValueToMlirValue(MlirLocation loc,
                                               const IValue &ival) {
  if (ival.isScalar()) {
    return funcBuilder->getScalarConstant(loc, ival.toScalar());
  }
  if (ival.isTensor()) {
    auto tensor = ival.toTensor();
    if (!tensor.defined()) {
      // Optional tensors ("Tensor?" type) are represented as Tensor ivals
      // that are undefined.
      return funcBuilder->getNoneConstant(loc);
    }

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
  if (ival.isList()) {
    auto list = ival.toList();
    llvm::SmallVector<MlirValue, 4> elements;
    for (IValue element : list) {
      elements.push_back(mapIValueToMlirValue(loc, element));
    }
    return funcBuilder->buildList(loc, elements);
  }
  if (ival.isNone()) {
    return funcBuilder->getNoneConstant(loc);
  }
  if (ival.isDevice()) {
    // TODO: Do we need to model/preserve device? Currently, just None'ing
    // it out.
    return funcBuilder->getNoneConstant(loc);
  }
  return {nullptr};
  // TODO: Implement mappings for the whole set (relevant to this use case):
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
                                             const IValue &ival) {
  if (ival.isScalar()) {
    return typeMapper.mapFromTorchScalarType(ival.toScalar().type());
  }
  if (ival.isTensor()) {
    return typeMapper.forwardTensorToType(ival.toTensor());
  }
  if (ival.isBool()) {
    // TODO: Switch to the numpy.bool type as that is a closer domain match.
    return mlirIntegerTypeGet(funcBuilder->getContext(), 1);
  }
  if (ival.isList()) {
    return npcompListTypeGet(funcBuilder->getContext());
  }
  if (ival.isNone()) {
    return npcompNoneTypeGet(funcBuilder->getContext());
  }
  if (ival.isDevice()) {
    return npcompNoneTypeGet(funcBuilder->getContext());
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
  MlirType elementType;
  if (tensor.scalar_type() == ScalarType::Bool) {
    // Bool is a special case. When used as an element type, it must be i1.
    // The generalized (non-Tensor) conversion, assumes that Bool is the
    // Basicpy bool type.
    elementType = mlirIntegerTypeGet(funcBuilder->getContext(), 1);
  } else {
    elementType = typeMapper.mapFromTorchScalarType(tensor.scalar_type());
  }
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
  case ScalarType::Bool:
    // TODO: Add a test specifically for bool and ensure consistency between
    // storage format and load format
    // (https://github.com/llvm/mlir-npcomp/issues/100).
    valueAttribute = mlirDenseElementsAttrBoolGet(
        shapedType, numElements, static_cast<const int *>(tensorData));
    break;
  default:
    throwUnsupportedTensorError();
  }
  MlirValue constTensorValue =
      funcBuilder->getGeneralConstant(loc, valueAttribute);

  // Create an array from the tensor constant via the
  // numpy.create_array_from_tensor op.
  MlirType constArrayType = npcompNdArrayTypeGetFromShaped(shapedType);
  MlirOperationState state = mlirOperationStateGet(
      toMlirStringRef("numpy.create_array_from_tensor"), loc);
  mlirOperationStateAddOperands(&state, 1, &constTensorValue);
  mlirOperationStateAddResults(&state, 1, &constArrayType);
  MlirOperation constArrayOp = mlirOperationCreate(&state);

  funcBuilder->getEntryBlockBuilder().insertBeforeTerminator(constArrayOp);
  MlirValue constArrayValue = mlirOperationGetResult(constArrayOp, 0);
  funcBuilder->mapTensor(tensor, constArrayValue);
  return constArrayValue;
}

TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  // PyTorch logs a warning when kernels are overriden, which is unavoidable
  // for factory-function BackendSelect kernels (there is not yet a "safe"
  // override mechanism). So, just silence it. Any of them here are coded
  // to be a superset of the default functionality, and there are only a few.
  auto orig_log_level = FLAGS_caffe2_log_level;
  FLAGS_caffe2_log_level = c10::GLOG_ERROR;

  // Disable capture of arange: causes it to memorize the resulting tensor.
  m.impl("arange", &AcapController::arangeBackendSelectKernel);

  // Restore log level.
  FLAGS_caffe2_log_level = orig_log_level;
}

TORCH_LIBRARY_IMPL(_, ACAP_DISPATCH_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &AcapController::fallbackKernel>());
}

TORCH_LIBRARY_IMPL(aten, ACAP_DISPATCH_KEY, m) {
  m.impl("copy_", &AcapController::copyUnderKernel);
}

TORCH_LIBRARY_IMPL(aten, ACAP_GRAD_DISPATCH_KEY, m) {
  // The at::convolution op is special in several ways. First, it presently
  // does not support boxing, so all of the usual fanciness does not apply
  // and it cannot be intercepted by generic fallthroughs, which is what
  // would usually allow us to avoid intercepting it at the gradient phase.
  // Second, the default implementation (see
  // aten/src/ATen/native/Convolution.cpp) is very switchy based on hard-coded
  // assumptions about device type. If we do nothing here, we will at best
  // intercept an mkldnn_convolution, cudnn_convolution, etc on the backend
  // dispatch keys. Non standard backends that don't have these switches
  // just route to aten::convolution_overrideable (see the else in
  // aten::convolution) as a convenience, but that is mostly a pass-through
  // (except for 3d convolutions which contain a trailing squeeze that needs
  // special casing). Therefore, we just intercept the aten::convolution op,
  // record it specially, and then mask ourselves off and ask the CPU backend
  // to invoke it. Not awesome.
  // Presumably this is on someone's list to adapt to the dispatch machinery
  // in a more appropriate way, but as the core of what the framework is,
  // perhaps people are reticent to touch it. Maybe someday, this can go away.
  m.impl_UNBOXED("convolution", &AcapController::convolutionKernel);

  // Sadly, there is no easy intercept point for the backwards convolution
  // kernel which allows for chaining to an existing backend. And convolution
  // is exceptionally special cased in this way, moreso than other ops.
  // The "solution" is to intercept the backend specific backward convolution
  // ops, emit it with the signature of the more generic
  // "convolution_backward_overrideable" op, which is available for generic
  // backends, and twiddle the parameters needed to get back to that form.
  // For MKL, which is effectively the CPU implementation, this just means that
  // some parameters are swapped and the full generality is not supported.
  // The "right" answer at some point is probably just to implement a
  // convolution kernel that fully does what is needed and delegates to an
  // appropriate implementation behind the scenes.
  m.impl("mkldnn_convolution_backward", AcapController::mklConvolutionBackward);
}
