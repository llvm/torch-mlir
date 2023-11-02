//===- module_builder.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_BUILDER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_BUILDER_H

#include "class_annotator.h"

#include "mlir-c/IR.h"

#include <ATen/Tensor.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/pybind.h>

namespace torch_mlir {

/// Main entry-point for constructing an MLIR module from some combination
/// of PyTorch programs/execution.
class ModuleBuilder {
public:
  ModuleBuilder(pybind11::object contextObj);

  /// Creates Python bindings for the class.
  static void bind(pybind11::module &m);

  pybind11::object getContextObj() { return contextObj; }
  pybind11::object getModuleObj() { return moduleObj; }

  // Imports a traced function. Note that the python type
  // torch.jit.ScriptFunction is the C++ type torch::jit::StrongFunctionPtr.
  // Just a bit of naming cruft.
  // Returns the same function, making it suitable as a nested decorator.
  torch::jit::StrongFunctionPtr
  importFunction(torch::jit::StrongFunctionPtr function,
                 py::object maybeImportOptions);

  // Imports a torch::jit::Module into the current module, using the
  // annotations, if not none, provided in `maybeClassAnnotator` which should be
  // a ClassAnnotator.
  void importModule(torch::jit::Module jitModule,
                    py::object maybeClassAnnotator,
                    py::object maybeImportOptions);

private:
  MlirBlock getBodyBlock();

  // Capture references to the python-owned context and module. Ownership
  // is delegated to python for these, and the C-API types are extracted via
  // the capsule API.
  pybind11::object contextObj;
  MlirContext context;
  MlirModule module;
  pybind11::object moduleObj;
  MlirOperation terminator;
  MlirLocation unknownLoc;
};

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_C10_DISPATCH_MODULE_BUILDER_H
