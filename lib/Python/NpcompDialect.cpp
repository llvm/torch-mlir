//===- NpcompDialect.cpp - Custom dialect classes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Python/MlirIr.h"
#include "npcomp/Python/NpcompModule.h"

#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"

namespace mlir {
namespace NPCOMP {

class BasicpyDialectHelper : public PyDialectHelper {
public:
  using PyDialectHelper::PyDialectHelper;

  static void bind(py::module m) {
    py::class_<BasicpyDialectHelper, PyDialectHelper>(m, "BasicpyDialectHelper")
        .def(py::init<PyContext &, PyOpBuilder &>(), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())
        // ---------------------------------------------------------------------
        // Basicpy dialect
        // ---------------------------------------------------------------------
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
             })
        // ---------------------------------------------------------------------
        // Numpy dialect
        // ---------------------------------------------------------------------
        .def("numpy_copy_to_tensor_op",
             [](BasicpyDialectHelper &self, PyValue source) -> PyOperationRef {
               auto sourceType =
                   source.value.getType().dyn_cast<Numpy::NdArrayType>();
               if (!sourceType) {
                 source.value.dump();
                 throw py::raiseValueError("expected ndarray type for "
                                           "numpy_copy_to_tensor_op");
               }
               auto dtype = sourceType.getDtype();
               auto tensorType = UnrankedTensorType::get(dtype);
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               auto op = opBuilder.create<Numpy::CopyToTensorOp>(
                   loc, tensorType, source.value);
               return op.getOperation();
             })
        .def("numpy_create_array_from_tensor_op",
             [](BasicpyDialectHelper &self, PyValue source) -> PyOperationRef {
               auto sourceType = source.value.getType().dyn_cast<TensorType>();
               if (!sourceType) {
                 throw py::raiseValueError("expected tensor type for "
                                           "numpy_create_array_from_tensor_op");
               }
               auto dtype = sourceType.getElementType();
               auto ndarrayType = Numpy::NdArrayType::get(dtype);
               OpBuilder &opBuilder = self.pyOpBuilder.getBuilder(true);
               Location loc = self.pyOpBuilder.getCurrentLoc();
               auto op = opBuilder.create<Numpy::CreateArrayFromTensorOp>(
                   loc, ndarrayType, source.value);
               return op.getOperation();
             })
        .def("numpy_NdArrayType",
             [](BasicpyDialectHelper &self, PyType dtype) -> PyType {
               return Numpy::NdArrayType::get(dtype.type);
             });
  }
};

} // namespace NPCOMP
} // namespace mlir

using namespace ::mlir::NPCOMP;

void mlir::npcomp::python::defineNpcompDialect(py::module m) {
  BasicpyDialectHelper::bind(m);
}
