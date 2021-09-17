//===- python_bindings.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the MLIR <-> PyTorch bridge.

#include <ATen/core/dispatch/Dispatcher.h>

#include "class_annotator.h"
#include "get_registered_ops.h"
#include "module_builder.h"

using namespace torch_mlir;

PYBIND11_MODULE(_torch_mlir, m) {
  ModuleBuilder::bind(m);
  initClassAnnotatorBindings(m);
  initGetRegisteredOpsBindings(m);
}
