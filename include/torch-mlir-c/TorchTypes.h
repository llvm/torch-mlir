//===-- torch-mlir-c/TorchTypes.h - C API for torch types ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_C_TORCHTYPES_H
#define TORCHMLIR_C_TORCHTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a torch.nn.Module type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNnModule(MlirType t);

/// Gets the !torch.nn.Module type of the specified class.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchNnModuleTypeGet(MlirContext context, MlirStringRef className);

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.optional<T> type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchOptional(MlirType t);

/// Gets the !torch.optional<T> type with subtype T.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchOptionalTypeGet(MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tuple type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchTuple(MlirType t);

/// Gets the !torch.tuple type with contained types `containedTypes`.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchTupleTypeGet(MlirContext context, intptr_t numContainedTypes,
                           MlirType const *containedTypes);

//===----------------------------------------------------------------------===//
// torch.union<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.union type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchUnion(MlirType t);

/// Gets the !torch.union type with contained types `containedTypes`.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchUnionTypeGet(MlirContext context, intptr_t numContainedTypes,
                           MlirType const *containedTypes);

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.list<T> type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchList(MlirType t);

/// Gets the !torch.list<T> type with contained T.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchListTypeGet(MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Device type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchDevice(MlirType t);

/// Gets the !torch.Device type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDeviceTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.Generator type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Generator type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchGenerator(MlirType t);

/// Gets the !torch.Generator type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchGeneratorTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.bool type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchBool(MlirType t);

/// Gets the !torch.bool type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchBoolTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.int type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchInt(MlirType t);

/// Gets the !torch.int type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchIntTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.float type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchFloat(MlirType t);

/// Gets the !torch.float type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchFloatTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.LinearParams type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchLinearParams(MlirType t);

/// Gets the !torch.LinearParams type.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchLinearParamsTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.qint8 type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchQInt8(MlirType t);

/// Gets the !torch.qint8 type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchQInt8TypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.tensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tensor type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNonValueTensor(MlirType t);

/// Gets a !torch.tensor type.
///
/// - `numSizes` having a value of -1 denotes an unranked tensor.
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).  -
/// `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchNonValueTensorTypeGet(
    MlirContext context, intptr_t numSizes, const int64_t *optionalSizes,
    MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context);

/// Gets the !torch.tensor type with the tensor attribute.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchNonValueTensorTypeGetFromAttribute(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// torch.vtensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.vtensor type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchValueTensor(MlirType t);

/// Gets a !torch.vtensor type.
///
/// - `numSizes` having a value of -1 denotes an unranked tensor.
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).
/// - `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchValueTensorTypeGet(
    MlirContext context, intptr_t numSizes, const int64_t *optionalSizes,
    MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.none type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.none type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNone(MlirType t);

/// Gets the !torch.none type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchNoneTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.str type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.str type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchString(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchStringTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.any type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.any type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchAny(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchAnyTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.number type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.number type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNumber(MlirType t);

/// Gets the !torch.number type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchNumberTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.dict type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.dict type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchDict(MlirType t);

/// Gets the !torch.dict type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDictTypeGet(MlirType keyType,
                                                      MlirType valueType);

#ifdef __cplusplus
}
#endif

#endif // TORCHMLIR_C_TORCHTYPES_H
