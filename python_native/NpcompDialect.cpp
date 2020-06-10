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
namespace NPCOMP {

class BasicpyDialectHelper : public PyDialectHelper {
public:
  using PyDialectHelper::PyDialectHelper;

  static void bind(py::module m) {
    py::class_<BasicpyDialectHelper, PyDialectHelper>(m, "BasicpyDialectHelper")
        .def(py::init<PyContext &, PyOpBuilder &>(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        .def_property_readonly("basicpy_BoolType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::BoolType::get(
                                     self.getContext());
                               })
        .def_property_readonly("basicpy_BytesType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::BytesType::get(
                                     self.getContext());
                               })
        .def_property_readonly("basicpy_EllipsisType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::EllipsisType::get(
                                     self.getContext());
                               })
        .def_property_readonly("basicpy_NoneType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::NoneType::get(
                                     self.getContext());
                               })
        .def("basicpy_SlotObject_type",
             [](BasicpyDialectHelper &self, std::string className,
                py::args pySlotTypes) -> PyType {
               SmallVector<Type, 4> slotTypes;
               for (auto pySlotType : pySlotTypes) {
                 slotTypes.push_back(pySlotType.cast<PyType>());
               }
               auto classNameAttr =
                   StringAttr::get(className, self.getContext());
               return Basicpy::SlotObjectType::get(classNameAttr, slotTypes);
             },
             py::arg("className"))
        .def_property_readonly("basicpy_StrType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::StrType::get(
                                     self.getContext());
                               })
        .def_property_readonly("basicpy_UnknownType",
                               [](BasicpyDialectHelper &self) -> PyType {
                                 return Basicpy::UnknownType::get(
                                     self.getContext());
                               })
        .def("basicpy_exec_op",
             [](BasicpyDialectHelper &self) {
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               auto op = opBuilder.create<Basicpy::ExecOp>(loc);
               return py::make_tuple(PyOperationRef(op),
                                     op.getBodyBuilder().saveInsertionPoint());
             })
        .def("basicpy_exec_discard_op",
             [](BasicpyDialectHelper &self, std::vector<PyValue> pyOperands) {
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               llvm::SmallVector<Value, 4> operands(pyOperands.begin(),
                                                    pyOperands.end());
               auto op =
                   opBuilder.create<Basicpy::ExecDiscardOp>(loc, operands);
               return PyOperationRef(op);
             })
        .def("basicpy_slot_object_get_op",
             [](BasicpyDialectHelper &self, PyValue slotObject,
                unsigned index) -> PyOperationRef {
               auto slotObjectType = slotObject.value.getType()
                                         .dyn_cast<Basicpy::SlotObjectType>();
               if (!slotObjectType) {
                 throw py::raiseValueError("Operand must be a SlotObject");
               }
               if (index >= slotObjectType.getSlotCount()) {
                 throw py::raiseValueError("Out of range slot index");
               }
               auto resultType = slotObjectType.getSlotTypes()[index];
               auto indexAttr =
                   IntegerAttr::get(IndexType::get(self.getContext()), index);
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               auto op = opBuilder.create<Basicpy::SlotObjectGetOp>(
                   loc, resultType, slotObject, indexAttr);
               return op.getOperation();
             });
  }
};

} // namespace NPCOMP
} // namespace mlir

using namespace ::mlir::NPCOMP;

void mlir::npcomp::python::defineNpcompDialect(py::module m) {
  BasicpyDialectHelper::bind(m);
}
