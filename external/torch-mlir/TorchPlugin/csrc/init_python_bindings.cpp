//===- init_python_bindings.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the MLIR <-> PyTorch bridge.

#include "init_python_bindings.h"

#include <string>

namespace py = pybind11;
namespace torch_mlir {
namespace {

void InitModuleBindings(py::module &m) {}

} // namespace

void InitBindings(py::module &m) {
  InitModuleBindings(m);
  InitBuilderBindings(m);
}

} // namespace torch_mlir

PYBIND11_MODULE(_torch_mlir, m) { torch_mlir::InitBindings(m); }
