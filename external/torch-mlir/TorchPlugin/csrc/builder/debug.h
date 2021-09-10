//===- debug.h --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRPLUGIN_CSRC_BUILDER_DEBUG_H
#define TORCHMLIRPLUGIN_CSRC_BUILDER_DEBUG_H

#include <string>

namespace torch_mlir {

/// Whether debug tracing is enabled and calls to debugTrace() are more than
/// a no-op.
bool isDebugTraceEnabled();

/// Writes a message to the debug trace log.
void debugTrace(const std::string &message);

/// Enables writing debug traces to stderr.
void enableDebugTraceToStderr();

} // namespace torch_mlir

#endif // TORCHMLIRPLUGIN_CSRC_BUILDER_DEBUG_H
