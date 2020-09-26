//===- acap_dispatch.cpp --------------------------------------------------===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "acap_dispatch.h"

#include "npcomp/Python/PybindUtils.h"

#include <c10/core/DispatchKey.h>
#include <torch/library.h>

using namespace torch_mlir;

namespace py = pybind11;

// TODO: Private use dispatch keys are not made for real uses. Allocate a proper
// dispatch key in upstream PyTorch (DispatchKey.h) prior to maturity.
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

void AcapController::fallbackKernel(const c10::OperatorHandle &opHandle,
                                    c10::Stack *stack) {
  // Exclude recursive dispatch to this kernel.
  c10::impl::ExcludeDispatchKeyGuard exclusion(kAcapDispatchKey);

  auto current = getCurrent();
  if (current) {
    // Capture the dispatch.
    std::stringstream sout;
    sout << "CAPTURE: " << opHandle.schema() << "\n";
    current->captureLog.push_back(sout.str());
  }

  auto &dispatcher = c10::Dispatcher::singleton();
  dispatcher.callBoxed(opHandle, stack);
}

TORCH_LIBRARY_IMPL(_, ACAP_DISPATCH_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &AcapController::fallbackKernel>());
}
