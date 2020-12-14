//===- init_python_bindings.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// This is the top-level entry point for the MLIR/NPCOMP <-> PyTorch bridge.
// It provides several mechanisms for extracting programs from PyTorch via:
//   a) A pseudo-device which captures the operations to an MLIR module
//      (implemented via the legacy type_dispatch mechanism for PyTorch 1.3).
//   b) Direct IR translation from PyTorch Graphs (not implemented).
//   c) Using the PyTorch JIT facility (not implemented).

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
