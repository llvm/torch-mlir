//===- mlir_if.cpp - MLIR IR Bindings -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mlir {

void defineMlirIrModule(py::module m) {
  m.doc() = "Python bindings for constructs in the mlir/IR library";
}

} // namespace mlir
