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
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// !basicpy.BoolType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "bool" type.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyBool(MlirType t);

/// Gets the Python "bool" type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyBoolTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.BytesType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "bytes" type.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyBytes(MlirType t);

/// Gets the Python "bytes" type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyBytesTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.DictType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "dict" type.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyDict(MlirType t);

/// Gets the generic Python "dict" type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyDictTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// List type
//===----------------------------------------------------------------------===//

/// Checks whether the given type is the Python "list" type.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyList(MlirType t);

/// Gets the generic Python "list" type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyListTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !basicpy.NoneType type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a `!basicpy.NoneType`.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyNone(MlirType t);

/// Gets the `!basicpy.NoneType` type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyNoneTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// SlotObject type.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirType npcompBasicPySlotObjectTypeGet(
    MlirContext context, MlirStringRef className, intptr_t slotTypeCount,
    const MlirType *slotTypes);

//===----------------------------------------------------------------------===//
// !basicpy.TupleType
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a `!basicpy.TupleType`.
MLIR_CAPI_EXPORTED bool npcompTypeIsABasicpyTuple(MlirType t);

/// Gets the generic Python "tuple" type.
MLIR_CAPI_EXPORTED MlirType npcompBasicpyTupleTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_BASICPYTYPES_H
