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

#include <list>
#include <memory>

#include <pybind11/pybind11.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

namespace torch_mlir {

/// Main entry point for managing device capture.
class AcapController : public std::enable_shared_from_this<AcapController> {
public:
  AcapController() = default;

  // Enter and exit the context manager.
  pybind11::object contextEnter();
  void contextExit(pybind11::object exc_type, pybind11::object exc_val,
                   pybind11::object exc_tb);

  // Gets and clears the current debug log.
  std::vector<std::string> getDebugLog();

  // Returns the current AcapController (if it has been activated on this
  // thread. Returns nullptr if none.
  static std::shared_ptr<AcapController> getCurrent();

  // The fallback boxed kernel that we route captured dispatches through.
  static void fallbackKernel(const c10::OperatorHandle &opHandle,
                             c10::Stack *stack);

private:
  struct Activation {
    Activation(std::shared_ptr<AcapController> controller)
        : controller(std::move(controller)) {}
    std::shared_ptr<AcapController> controller;
    // The RAII dispatch key guard is not movable, so heap allocate it. This is
    // a bit outside of its intended design, but since this is thread local as
    // well, it should be fine.
    std::unique_ptr<c10::impl::IncludeDispatchKeyGuard> dispatchGuard;
  };
  // Gets the thread local stack of active acap controllers.
  static std::list<Activation> &getThreadLocalActiveStack();
  std::vector<std::string> captureLog;
};

} // namespace torch_mlir
