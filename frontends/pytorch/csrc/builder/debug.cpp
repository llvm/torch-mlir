//===- debug.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "debug.h"

#include <iostream>

namespace torch_mlir {

static bool debugTraceToStderrEnabled = false;

/// Whether debug tracing is enabled and calls to debugTrace() are more than
/// a no-op.
bool isDebugTraceEnabled() { return debugTraceToStderrEnabled; }

/// Writes a message to the debug trace log.
void debugTrace(const std::string &message) {
  if (debugTraceToStderrEnabled)
    std::cerr << "TORCH_MLIR TRACE: " << message << "\n" << std::flush;
}

/// Enables writing debug traces to stderr.
void enableDebugTraceToStderr() { debugTraceToStderrEnabled = true; }

} // namespace torch_mlir
