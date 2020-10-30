//===- acap_dispatch.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//
// "ATen Capture" dispatcher: Defines facility for capturing programs by
// registering dispatch keys to intercept op execution.
// References:
//   http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_ACAP_DISPATCH_H
#define NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_ACAP_DISPATCH_H

#include <list>
#include <memory>

#include "../pybind.h"

#include "func_builder.h"

#include "mlir-c/IR.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace torch_mlir {

/// Main entry point for managing device capture.
class AcapController : public std::enable_shared_from_this<AcapController> {
public:
  AcapController(TypeMapper &typeMapper,
                 std::unique_ptr<FuncBuilder> funcBuilder)
      : typeMapper(typeMapper), funcBuilder(std::move(funcBuilder)) {}

  // Enter and exit the context manager.
  pybind11::object contextEnter();
  void contextExit(pybind11::object exc_type, pybind11::object exc_val,
                   pybind11::object exc_tb);

  // Terminates capture and returns tensors from the function.
  void returns(std::vector<at::Tensor> tensors);

  // Returns the current AcapController (if it has been activated on this
  // thread. Returns nullptr if none (not active on the current thread).
  static std::shared_ptr<AcapController> getCurrentThreadAcapController();

  // The fallback boxed kernel that we route captured dispatches through.
  static void fallbackKernel(const c10::OperatorHandle &opHandle,
                             c10::Stack *stack);

  // Kernel implementation for the boxing-incompatible convolution kernel.
  static at::Tensor
  convolutionKernel(const at::Tensor &input, const at::Tensor &weight,
                    const c10::optional<at::Tensor> &bias,
                    const at::IntArrayRef stride, const at::IntArrayRef padding,
                    const at::IntArrayRef dilation, const bool transposed,
                    const at::IntArrayRef output_padding, const int64_t groups);

private:
  /// Builds a kernel call step by step.
  class KernelCallBuilder {
  public:
    KernelCallBuilder(AcapController &parent, MlirContext context,
                      MlirLocation loc, const c10::OperatorHandle &opHandle);
    void addOperand(const c10::IValue &value);
    void addResult(const c10::IValue &result);
    MlirOperation create();

  private:
    void addSchemaAttrs();
    AcapController &parent;
    MlirContext context;
    MlirLocation loc;
    const c10::OperatorHandle &opHandle;
    OperationStateHolder state;
    int resultCount = 0;
    llvm::SmallVector<std::pair<size_t, at::Tensor>, 4> resultIndexToTensorMap;
  };

  MlirLocation getCurrentLocation();
  void redispatch(const c10::OperatorHandle &opHandle, c10::Stack *stack);
  void fallbackKernelImpl(const c10::OperatorHandle &opHandle,
                          c10::Stack *stack);
  MlirValue mapIValueToMlirValue(MlirLocation loc, const c10::IValue &ival);
  MlirType mapIValueToMlirType(MlirLocation loc, const c10::IValue &ival);
  /// Imports a tensor by value (as a constant), remembering the association.
  MlirValue importTensorByValue(at::Tensor tensor);
  void verifyHasNotReturned();
  struct Activation {
    Activation(std::shared_ptr<AcapController> controller)
        : controller(std::move(controller)) {}
    std::shared_ptr<AcapController> controller;
    // The RAII dispatch key guard is not movable, so heap allocate it. This is
    // a bit outside of its intended design, but since this is thread local as
    // well, it should be fine.
    std::unique_ptr<c10::impl::IncludeDispatchKeyGuard> includeGuard;
    std::unique_ptr<c10::impl::ExcludeDispatchKeyGuard> excludeGuard;
  };
  // Gets the thread local stack of active acap controllers.
  static std::list<Activation> &getThreadLocalActiveStack();

  TypeMapper &typeMapper;
  std::unique_ptr<FuncBuilder> funcBuilder;
  bool hasReturned = false;
};

} // namespace torch_mlir

#endif // NPCOMP_FRONTENDS_PYTORCH_CSRC_C10_DISPATCH_ACAP_DISPATCH_H
