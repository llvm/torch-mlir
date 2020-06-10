//===- NpcompDialect.cpp - Custom dialect classes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MlirIr.h"
#include "NpcompModule.h"

#include "mlir/Dialect/SCF/SCF.h"

namespace mlir {
namespace {

class ScfDialectHelper : public PyDialectHelper {
public:
  using PyDialectHelper::PyDialectHelper;

  static void bind(py::module m) {
    py::class_<ScfDialectHelper, PyDialectHelper>(m, "ScfDialectHelper")
        .def(py::init<PyContext &, PyOpBuilder &>(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def("scf_yield_op",
             [](ScfDialectHelper &self,
                std::vector<PyValue> pyYields) -> PyOperationRef {
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               llvm::SmallVector<Value, 4> yields(pyYields.begin(),
                                                  pyYields.end());
               auto op = opBuilder.create<scf::YieldOp>(loc, yields);
               return op.getOperation();
             })
        .def("scf_if_op",
             [](ScfDialectHelper &self, std::vector<PyType> pyResultTypes,
                PyValue cond, bool withElseRegion) {
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               llvm::SmallVector<Type, 4> resultTypes(pyResultTypes.begin(),
                                                      pyResultTypes.end());
               auto op = opBuilder.create<scf::IfOp>(loc, resultTypes, cond,
                                                     withElseRegion);
               if (withElseRegion) {
                 return py::make_tuple(
                     PyOperationRef(op),
                     op.getThenBodyBuilder().saveInsertionPoint(),
                     op.getElseBodyBuilder().saveInsertionPoint());
               } else {
                 return py::make_tuple(
                     PyOperationRef(op),
                     op.getThenBodyBuilder().saveInsertionPoint());
               }
             },
             py::arg("result_types"), py::arg("cond"),
             py::arg("with_else_region") = false);
  }
};

} // namespace
} // namespace mlir

void mlir::defineMlirCoreDialects(py::module m) { ScfDialectHelper::bind(m); }
