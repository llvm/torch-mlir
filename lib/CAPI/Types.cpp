//===- Types.cpp - C Interface for NPComp types ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

/*============================================================================*/
/* Any dtype type.                                                            */
/*============================================================================*/

int npcompTypeIsAAnyDtype(MlirType t) {
  return unwrap(t).isa<Numpy::AnyDtypeType>();
}

MlirType npcompAnyDtypeTypeGet(MlirContext context) {
  return wrap(Numpy::AnyDtypeType::get(unwrap(context)));
}

/*============================================================================*/
/* Bool type.                                                                 */
/*============================================================================*/

int npcompTypeIsABool(MlirType t) { return unwrap(t).isa<Basicpy::BoolType>(); }

MlirType npcompBoolTypeGet(MlirContext context) {
  return wrap(Basicpy::BoolType::get(unwrap(context)));
}

/*============================================================================*/
/* Dict type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "dict" type. */
int npcompTypeIsADict(MlirType t) { return unwrap(t).isa<Basicpy::DictType>(); }

/** Gets the generic Python "dict" type. */
MlirType npcompDictTypeGet(MlirContext context) {
  return wrap(Basicpy::DictType::get(unwrap(context)));
}

/*============================================================================*/
/* List type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the Python "list" type. */
int npcompTypeIsAList(MlirType t) { return unwrap(t).isa<Basicpy::ListType>(); }

/** Gets the generic Python "dict" type. */
MlirType npcompListTypeGet(MlirContext context) {
  return wrap(Basicpy::ListType::get(unwrap(context)));
}

/*============================================================================*/
/* NDArray type.                                                              */
/*============================================================================*/

int npcompTypeIsANdArray(MlirType t) {
  return unwrap(t).isa<Numpy::NdArrayType>();
}

MlirType npcompNdArrayTypeGetUnranked(MlirType elementType) {
  return wrap(Numpy::NdArrayType::get(unwrap(elementType)));
}

MlirType npcompNdArrayTypeGetRanked(intptr_t rank, const int64_t *shape,
                                    MlirType elementType) {
  llvm::ArrayRef<int64_t> shapeArray(shape, rank);
  return wrap(Numpy::NdArrayType::get(unwrap(elementType), shapeArray));
}

MlirType npcompNdArrayTypeGetFromShaped(MlirType shapedType) {
  return wrap(Numpy::NdArrayType::getFromShapedType(
      unwrap(shapedType).cast<ShapedType>()));
}

MlirType npcompNdArrayTypeToTensor(MlirType ndarrayType) {
  return wrap(unwrap(ndarrayType).cast<Numpy::NdArrayType>().toTensorType());
}

/*============================================================================*/
/* None type.                                                                 */
/*============================================================================*/

/** Checks whether the given type is the type of the singleton 'None' value. */
int npcompTypeIsANone(MlirType t) { return unwrap(t).isa<Basicpy::NoneType>(); }

/** Gets the type of the singleton 'None'. */
MlirType npcompNoneTypeGet(MlirContext context) {
  return wrap(Basicpy::NoneType::get(unwrap(context)));
}

/*============================================================================*/
/* SlotObject type.                                                           */
/*============================================================================*/

MlirType npcompSlotObjectTypeGet(MlirContext context, MlirStringRef className,
                                 intptr_t slotTypeCount,
                                 const MlirType *slotTypes) {
  MLIRContext *cppContext = unwrap(context);
  auto classNameAttr = StringAttr::get(unwrap(className), cppContext);
  SmallVector<Type> slotTypesCpp;
  slotTypesCpp.resize(slotTypeCount);
  for (intptr_t i = 0; i < slotTypeCount; ++i) {
    slotTypesCpp[i] = unwrap(slotTypes[i]);
  }
  return wrap(Basicpy::SlotObjectType::get(classNameAttr, slotTypesCpp));
}

/*============================================================================*/
/* Tuple type.                                                                */
/*============================================================================*/

/** Checks whether the given type is the special "any dtype" type that is used
 * to signal an NDArray or tensor of unknown type. */
int npcompTypeIsATuple(MlirType t) {
  return unwrap(t).isa<Basicpy::TupleType>();
}

/** Gets the "any dtype" type. */
MlirType npcompTupleTypeGet(MlirContext context) {
  return wrap(Basicpy::TupleType::get(unwrap(context)));
}
