//===- python_bindings.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the JIT IR -> MLIR importer.

#include <ATen/core/dispatch/Dispatcher.h>

#include "class_annotator_pybind.h"
#include "get_registered_ops.h"
#include "import_options_pybind.h"
#include "module_builder.h"

using namespace torch_mlir;

PYBIND11_MODULE(_jit_ir_importer, m) {
  ModuleBuilder::bind(m);
  initClassAnnotatorBindings(m);
  initGetRegisteredOpsBindings(m);
  initImportOptionsBindings(m);
}
