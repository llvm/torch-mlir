//===- Types.cpp - C Interface for NPComp types ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/StandardTypes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP::Basicpy;
using namespace mlir::NPCOMP::Numpy;

/*============================================================================*/
/* Bool type.                                                                 */
/*============================================================================*/

int npcompTypeIsABool(MlirType t) { return unwrap(t).isa<BoolType>(); }

MlirType npcompBoolTypeGet(MlirContext context) {
  return wrap(BoolType::get(unwrap(context)));
}

/*============================================================================*/
/* Any dtype type.                                                            */
/*============================================================================*/

int npcompTypeIsAAnyDtype(MlirType t) { return unwrap(t).isa<AnyDtypeType>(); }

MlirType npcompAnyDtypeTypeGet(MlirContext context) {
  return wrap(AnyDtypeType::get(unwrap(context)));
}

/*============================================================================*/
/* NDArray type.                                                              */
/*============================================================================*/

int npcompTypeIsANdArray(MlirType t) { return unwrap(t).isa<NdArrayType>(); }

MlirType npcompNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                    MlirType elementType) {
  llvm::ArrayRef<int64_t> shapeArray(shape, rank);
  return wrap(NdArrayType::get(unwrap(elementType), shapeArray));
}

MlirType npcompNdArrayTypeGetFromShaped(MlirType shapedType) {
  return wrap(
      NdArrayType::getFromShapedType(unwrap(shapedType).cast<ShapedType>()));
}
