//===- python_bindings.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE.pytorch for license information.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the JIT IR -> MLIR importer.

#include <ATen/core/dispatch/Dispatcher.h>

#include "class_annotator.h"
#include "get_registered_ops.h"
#include "module_builder.h"

using namespace torch_mlir;

PYBIND11_MODULE(_jit_ir_importer, m) {
  ModuleBuilder::bind(m);
  initClassAnnotatorBindings(m);
  initGetRegisteredOpsBindings(m);
}
