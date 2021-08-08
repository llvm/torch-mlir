//===-- npcomp-c/TorchTypes.h - C API for torch types -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_C_TORCHTYPES_H
#define NPCOMP_C_TORCHTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a torch.nn.Module type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchNnModule(MlirType t);

/// Gets the !torch.nn.Module type of the specified class.
MLIR_CAPI_EXPORTED MlirType npcompTorchNnModuleTypeGet(MlirContext context,
                                                       MlirStringRef className);

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.optional<T> type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchOptional(MlirType t);

/// Gets the !torch.optional<T> type with subtype T.
MLIR_CAPI_EXPORTED MlirType npcompTorchOptionalTypeGet(MlirContext context,
                                                       MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tuple type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchTuple(MlirType t);

/// Gets the !torch.tuple type with contained types `containedTypes`.
MLIR_CAPI_EXPORTED MlirType
npcompTorchTupleTypeGet(MlirContext context, intptr_t numContainedTypes,
                        MlirType const *containedTypes);

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.list<T> type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchList(MlirType t);

/// Gets the !torch.list<T> type with contained T.
MLIR_CAPI_EXPORTED MlirType npcompTorchListTypeGet(MlirContext context,
                                                   MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Device type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchDevice(MlirType t);

/// Gets the !torch.Device type.
MLIR_CAPI_EXPORTED MlirType npcompTorchDeviceTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.bool type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchBool(MlirType t);

/// Gets the !torch.bool type.
MLIR_CAPI_EXPORTED MlirType npcompTorchBoolTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.int type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchInt(MlirType t);

/// Gets the !torch.int type.
MLIR_CAPI_EXPORTED MlirType npcompTorchIntTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.float type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchFloat(MlirType t);

/// Gets the !torch.float type.
MLIR_CAPI_EXPORTED MlirType npcompTorchFloatTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.LinearParams type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchLinearParams(MlirType t);

/// Gets the !torch.LinearParams type.
MLIR_CAPI_EXPORTED MlirType npcompTorchLinearParamsTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.qint8 type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchQInt8(MlirType t);

/// Gets the !torch.qint8 type.
MLIR_CAPI_EXPORTED MlirType npcompTorchQInt8TypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.tensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tensor type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchNonValueTensor(MlirType t);

/// Gets a !torch.tensor type.
///
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).  -
/// `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MLIR_CAPI_EXPORTED MlirType npcompTorchNonValueTensorTypeGet(
    MlirContext context, intptr_t numSizes, const int64_t *optionalSizes,
    MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MLIR_CAPI_EXPORTED MlirType
npcompTorchNonValueTensorTypeGetWithLeastStaticInformation(MlirContext context);

/// Gets a !torch.tensor type, taking shape/dtype from a ShapedType `type`.
MLIR_CAPI_EXPORTED MlirType
npcompTorchNonValueTensorTypeGetFromShaped(MlirType type);

//===----------------------------------------------------------------------===//
// torch.vtensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.vtensor type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchValueTensor(MlirType t);

/// Gets a !torch.vtensor type.
///
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).
/// - `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MLIR_CAPI_EXPORTED MlirType npcompTorchValueTensorTypeGet(
    MlirContext context, intptr_t numSizes, const int64_t *optionalSizes,
    MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MLIR_CAPI_EXPORTED MlirType
npcompTorchValueTensorTypeGetWithLeastStaticInformation(MlirContext context);

/// Gets a !torch.tensor type, taking shape/dtype from a ShapedType `type`.
MLIR_CAPI_EXPORTED MlirType
npcompTorchValueTensorTypeGetFromShaped(MlirType type);

//===----------------------------------------------------------------------===//
// !torch.none type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.none type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchNone(MlirType t);

/// Gets the !torch.none type.
MLIR_CAPI_EXPORTED MlirType npcompTorchNoneTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.str type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.str type
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchString(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType npcompTorchStringTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.any type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.any type.
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchAny(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType npcompTorchAnyTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.number type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.number type.
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchNumber(MlirType t);

/// Gets the !torch.number type.
MLIR_CAPI_EXPORTED MlirType npcompTorchNumberTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.dict type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.dict type.
MLIR_CAPI_EXPORTED bool npcompTypeIsATorchDict(MlirType t);

/// Gets the !torch.dict type.
MLIR_CAPI_EXPORTED MlirType npcompTorchDictTypeGet(MlirContext context,
                                                   MlirType keyType,
                                                   MlirType valueType);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_TORCHTYPES_H
