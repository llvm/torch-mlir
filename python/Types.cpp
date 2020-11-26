//===- NpcompModule.cpp - Top-level _npcomp extension module --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Modules.h"

#include "mlir-c/StandardTypes.h"
#include "npcomp-c/Types.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir::npcomp::python;

static py::object mapMlirDTypeToPyType(MlirType dtype) {
  auto np = py::module::import("numpy");

  if (mlirTypeIsAF32(dtype))
    return np.attr("float32");
  if (mlirTypeIsAF64(dtype))
    return np.attr("float64");
  if (mlirTypeIsAF16(dtype))
    return np.attr("float16");

  if (mlirTypeIsAInteger(dtype)) {
    unsigned width = mlirIntegerTypeGetWidth(dtype);
    bool isSigned =
        mlirIntegerTypeIsSigned(dtype) || mlirIntegerTypeIsSignless(dtype);
    if (isSigned && width == 8)
      return np.attr("int8");
    if (!isSigned && width == 8)
      return np.attr("uint8");
    if (isSigned && width == 16)
      return np.attr("int16");
    if (!isSigned && width == 16)
      return np.attr("uint16");
    if (isSigned && width == 32)
      return np.attr("int32");
    if (!isSigned && width == 32)
      return np.attr("uint32");
    if (isSigned && width == 64)
      return np.attr("int64");
    if (!isSigned && width == 64)
      return np.attr("uint64");
  }

  return {};
}

static py::object mapMlirTypeToMetaType(MlirType mlirType) {
  auto typesModule = py::module::import("npcomp.meta.types");
  py::object valueType = typesModule.attr("ValueType")();

  if (npcompTypeIsANdArray(mlirType)) {
    py::list constraints;

    // DType constraints.
    MlirType dtype = npcompNdArrayTypeGetDType(mlirType);
    if (!npcompTypeIsAAnyDtype(dtype)) {
      // Explicit dtype. Map.
      auto pyDType = mapMlirDTypeToPyType(dtype);
      if (pyDType)
        constraints.append(typesModule.attr("DType")(std::move(pyDType)));
    }

    // Shape constraints.
    if (npcompNdArrayTypeHasRank(mlirType)) {
      // Add rank and dim constraints.
      int64_t rank = npcompNdArrayTypeGetRank(mlirType);

      // Get the dims.
      bool isStatic = true;
      py::list dims;
      for (int64_t i = 0; i < rank; ++i) {
        auto dim = npcompNdArrayTypeGetDimSize(mlirType, i);
        if (dim < 0)
          isStatic = false;
        dims.append(dim);
      }

      if (isStatic) {
        // Just emit a static shape constraint.
        constraints.append(typesModule.attr("Shape")(*dims));
      } else {
        // Some dynamic dims. Emit Rank and DimSize constraints.
        constraints.append(typesModule.attr("Rank")(rank));
        // TODO: Implement DimSize.
      }
    }
    return typesModule.attr("ValueType")("NdArray", *constraints);
  }

  return py::none();
}

/// Populates bindings for the _npcomp.type_mapping module.
void mlir::npcomp::python::populateTypesModule(py::module &m) {
  m.def("map_mlir_type_to_meta_type", &mapMlirTypeToMetaType);
}
