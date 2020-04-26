//===- native.cpp - MLIR Python bindings ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace npcomp {
namespace python {

// Externs
void npcompMlirInitialize();
void defineMlirEdscModule(py::module m);

PYBIND11_MODULE(native, m) {
  npcompMlirInitialize();
  m.doc() = "Npcomp native python bindings";

  auto mlir_m = m.def_submodule("mlir", "MLIR interop");
  auto mlir_edsc_m = mlir_m.def_submodule("edsc");
  defineMlirEdscModule(mlir_edsc_m);
}

}  // namespace python
}  // namespace npcomp
