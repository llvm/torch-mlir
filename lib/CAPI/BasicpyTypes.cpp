//===- BasicpyTypes.cpp - C Interface for basicpy types -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/BasicpyTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// Bool type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsABasicpyBool(MlirType t) {
  return unwrap(t).isa<Basicpy::BoolType>();
}

MlirType npcompBasicpyBoolTypeGet(MlirContext context) {
  return wrap(Basicpy::BoolType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// Bytes type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsABasicpyBytes(MlirType t) {
  return unwrap(t).isa<Basicpy::BytesType>();
}

MlirType npcompBasicpyBytesTypeGet(MlirContext context) {
  return wrap(Basicpy::BytesType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// Dict type.
//===----------------------------------------------------------------------===//

/** Checks whether the given type is the Python "dict" type. */
bool npcompTypeIsABasicpyDict(MlirType t) {
  return unwrap(t).isa<Basicpy::DictType>();
}

/** Gets the generic Python "dict" type. */
MlirType npcompBasicpyDictTypeGet(MlirContext context) {
  return wrap(Basicpy::DictType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// List type.
//===----------------------------------------------------------------------===//

/** Checks whether the given type is the Python "list" type. */
bool npcompTypeIsABasicpyList(MlirType t) {
  return unwrap(t).isa<Basicpy::ListType>();
}

/** Gets the generic Python "dict" type. */
MlirType npcompBasicpyListTypeGet(MlirContext context) {
  return wrap(Basicpy::ListType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// !basicpy.NoneType type.
//===----------------------------------------------------------------------===//

/** Checks whether the given type is a `!basicpy.NoneType`. */
bool npcompTypeIsANone(MlirType t) {
  return unwrap(t).isa<Basicpy::NoneType>();
}

/** Gets the `!basicpy.NoneType` type. */
MlirType npcompBasicpyNoneTypeGet(MlirContext context) {
  return wrap(Basicpy::NoneType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// SlotObject type.
//===----------------------------------------------------------------------===//

MlirType npcompBasicPySlotObjectTypeGet(MlirContext context,
                                        MlirStringRef className,
                                        intptr_t slotTypeCount,
                                        const MlirType *slotTypes) {
  MLIRContext *cppContext = unwrap(context);
  auto classNameAttr = StringAttr::get(cppContext, unwrap(className));
  SmallVector<Type> slotTypesCpp;
  slotTypesCpp.resize(slotTypeCount);
  for (intptr_t i = 0; i < slotTypeCount; ++i) {
    slotTypesCpp[i] = unwrap(slotTypes[i]);
  }
  return wrap(Basicpy::SlotObjectType::get(classNameAttr, slotTypesCpp));
}

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//

/** Checks whether the given type is the special "any dtype" type that is used
 * to signal an NDArray or tensor of unknown type. */
bool npcompTypeIsABasicpyTuple(MlirType t) {
  return unwrap(t).isa<Basicpy::TupleType>();
}

/** Gets the "any dtype" type. */
MlirType npcompBasicpyTupleTypeGet(MlirContext context) {
  return wrap(Basicpy::TupleType::get(unwrap(context)));
}
