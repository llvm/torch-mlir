//===- NumpyTypes.cpp - C Interface for numpy types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/NumpyTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// !numpy.any_dtype type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsANumpyAnyDtype(MlirType t) {
  return unwrap(t).isa<Numpy::AnyDtypeType>();
}

MlirType npcompAnyDtypeTypeGet(MlirContext context) {
  return wrap(Numpy::AnyDtypeType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// NDArray type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsANumpyNdArray(MlirType t) {
  return unwrap(t).isa<Numpy::NdArrayType>();
}

MlirType npcompNumpyNdArrayTypeGetUnranked(MlirType elementType) {
  return wrap(Numpy::NdArrayType::get(unwrap(elementType)));
}

MlirType npcompNumpyNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                         MlirType elementType) {
  llvm::ArrayRef<int64_t> shapeArray(shape, rank);
  return wrap(Numpy::NdArrayType::get(unwrap(elementType), shapeArray));
}

MlirType npcompNumpyNdArrayTypeGetFromShaped(MlirType shapedType) {
  return wrap(Numpy::NdArrayType::getFromShapedType(
      unwrap(shapedType).cast<ShapedType>()));
}

MlirType npcompNumpyNdArrayTypeToTensor(MlirType ndarrayType) {
  return wrap(unwrap(ndarrayType).cast<Numpy::NdArrayType>().toTensorType());
}
