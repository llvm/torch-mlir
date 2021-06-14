//===-- npcomp-c/BasicpyTypes.h - C API for basicpy types ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_C_BASICPYTYPES_H
#define NPCOMP_C_BASICPYTYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// !basicpy.BoolType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "bool" type.
bool npcompTypeIsABasicpyBool(MlirType t);

/// Gets the Python "bool" type.
MlirType npcompBasicpyBoolTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.BytesType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "bytes" type.
bool npcompTypeIsABasicpyBytes(MlirType t);

/// Gets the Python "bytes" type.
MlirType npcompBasicpyBytesTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.DictType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "dict" type.
bool npcompTypeIsABasicpyDict(MlirType t);

/// Gets the generic Python "dict" type.
MlirType npcompBasicpyDictTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// List type
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "list" type.
bool npcompTypeIsABasicpyList(MlirType t);

/// Gets the generic Python "list" type.
MlirType npcompBasicpyListTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.NoneType type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a `!basicpy.NoneType`.
bool npcompTypeIsABasicpyNone(MlirType t);

/// Gets the `!basicpy.NoneType` type.
MlirType npcompBasicpyNoneTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// SlotObject type.
//===----------------------------------------------------------------------===//

MlirType npcompBasicPySlotObjectTypeGet(MlirContext context,
                                        MlirStringRef className,
                                        intptr_t slotTypeCount,
                                        const MlirType *slotTypes);

//===----------------------------------------------------------------------===//
// !basicpy.TupleType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a `!basicpy.TupleType`.
bool npcompTypeIsABasicpyTuple(MlirType t);

/// Gets the generic Python "tuple" type.
MlirType npcompBasicpyTupleTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_BASICPYTYPES_H
