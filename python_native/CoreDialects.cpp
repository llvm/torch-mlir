//===- NpcompDialect.cpp - Custom dialect classes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirIr.h"
#include "NpcompModule.h"

#include "npcomp/Dialect/Basicpy/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/BasicpyOps.h"

namespace mlir {
namespace {

class ScfDialectHelper : public PyDialectHelper {
public:
  using PyDialectHelper::PyDialectHelper;

  static void bind(py::module m) {
    py::class_<ScfDialectHelper, PyDialectHelper>(m, "ScfDialectHelper")
      .def(py::init<PyContext &, PyOpBuilder &>(), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>());
  }
};

} // namespace
} // namespace mlir

void mlir::defineMlirCoreDialects(py::module m) { ScfDialectHelper::bind(m); }
