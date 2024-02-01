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

/// A restricted subset of PyTorch subtyping that only handles a few special
/// cases that we need to model.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsValidSubtype(MlirType subtype,
                                                    MlirType type);

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a torch.nn.Module type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNnModule(MlirType t);

/// Gets the !torch.nn.Module type of the specified class.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchNnModuleTypeGet(MlirContext context, MlirStringRef className);

/// Gets the !torch.nn.Module typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchNnModuleTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.optional<T> type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchOptional(MlirType t);

/// Gets the !torch.optional<T> type with subtype T.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchOptionalTypeGet(MlirType containedType);

/// Gets the subtype T of !torch.optional<T> type.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchOptionalTypeGetContained(MlirType containedType);

/// Gets the !torch.optional typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchOptionalTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tuple type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchTuple(MlirType t);

/// Gets the !torch.tuple type with contained types `containedTypes`.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchTupleTypeGet(MlirContext context, intptr_t numContainedTypes,
                           MlirType const *containedTypes);

/// Returns the number of types contained in a !torch.tuple type.
MLIR_CAPI_EXPORTED size_t torchMlirTorchTupleTypeGetNumTypes(MlirType t);

/// Returns the pos-th type in the !torch.tuple type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchTupleTypeGetType(MlirType t,
                                                           intptr_t pos);

/// Gets the !torch.tuple typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchTupleTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.union<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.union type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchUnion(MlirType t);

/// Gets the !torch.union type with contained types `containedTypes`.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchUnionTypeGet(MlirContext context, intptr_t numContainedTypes,
                           MlirType const *containedTypes);

/// Returns the number of types contained in a !torch.union type.
MLIR_CAPI_EXPORTED size_t torchMlirTorchUnionTypeGetNumTypes(MlirType t);

/// Returns the pos-th type in the !torch.union type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchUnionTypeGetType(MlirType t,
                                                           intptr_t pos);

/// Gets the !torch.union typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchUnionTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.list<T> type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchList(MlirType t);

/// Gets the !torch.list<T> type with contained T.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchListTypeGet(MlirType containedType);

/// Gets contained T in a !torch.list<T> type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchListTypeGetContainedType(MlirType t);

/// Gets the !torch.list typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchListTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Device type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchDevice(MlirType t);

/// Gets the !torch.Device type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDeviceTypeGet(MlirContext context);

/// Gets the !torch.device typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchDeviceTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.Generator type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Generator type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchGenerator(MlirType t);

/// Gets the !torch.Generator type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchGeneratorTypeGet(MlirContext context);

/// Gets the !torch.generator typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchGeneratorTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.bool type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchBool(MlirType t);

/// Gets the !torch.bool type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchBoolTypeGet(MlirContext context);

/// Gets the !torch.bool typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchBoolTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.int type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchInt(MlirType t);

/// Gets the !torch.int type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchIntTypeGet(MlirContext context);

/// Gets the !torch.int typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchIntTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.float type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchFloat(MlirType t);

/// Gets the !torch.float type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchFloatTypeGet(MlirContext context);

/// Gets the !torch.float typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchFloatTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.LinearParams type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchLinearParams(MlirType t);

/// Gets the !torch.LinearParams type.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchLinearParamsTypeGet(MlirContext context);

/// Gets the !torch.linearparams typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchLinearParamsTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.qint8 type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchQInt8(MlirType t);

/// Gets the !torch.qint8 type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchQInt8TypeGet(MlirContext context);

/// Gets the !torch.qint8 typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchQInt8TypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// torch.quint8 type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.quint8 type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchQUInt8(MlirType t);

/// Gets the !torch.quint8 type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchQUInt8TypeGet(MlirContext context);

/// Gets the !torch.quint8 typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchQUInt8TypeGetTypeID(void);

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

/// Gets the the rank (number of dimensions) of a !torch.tensor
MLIR_CAPI_EXPORTED int64_t torchMlirTorchNonValueTensorTypeGetRank(MlirType t);

/// Return true if this type has a list of sizes.
MLIR_CAPI_EXPORTED bool torchMlirTorchNonValueTensorTypeHasSizes(MlirType t);

/// Return true if this type has a dtype.
MLIR_CAPI_EXPORTED bool torchMlirTorchNonValueTensorTypeHasDtype(MlirType t);

/// Gets the the sizes of the dimensions of a !torch.tensor; note -1 size
/// indicates an unrefined/unknown size dimension.
MLIR_CAPI_EXPORTED int64_t
torchMlirTorchNonValueTensorTypeGetSizes(MlirType t, int64_t *sizes);

/// Gets the the dtype (data type) of a !torch.tensor.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchNonValueTensorTypeGetDtype(MlirType t);

/// Gets the !torch.tensor typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchNonValueTensorTypeGetTypeID(void);

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

/// Gets the !torch.vtensor type with the tensor attribute.
MLIR_CAPI_EXPORTED MlirType
torchMlirTorchValueTensorTypeGetFromAttribute(MlirAttribute attr);

/// Gets the the rank (number of dimensions) of a !torch.vtensor
MLIR_CAPI_EXPORTED int64_t torchMlirTorchValueTensorTypeGetRank(MlirType t);

/// Return true if this type has a list of sizes.
MLIR_CAPI_EXPORTED bool torchMlirTorchValueTensorTypeHasSizes(MlirType t);

/// Return true if this type has a dtype.
MLIR_CAPI_EXPORTED bool torchMlirTorchValueTensorTypeHasDtype(MlirType t);

/// Gets the the sizes of the dimensions of a !torch.vtensor; note -1 size
/// indicates an unrefined/unknown size dimension.
MLIR_CAPI_EXPORTED int64_t
torchMlirTorchValueTensorTypeGetSizes(MlirType t, int64_t *sizes);

/// Gets the the dtype (data type) of a !torch.vtensor.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchValueTensorTypeGetDtype(MlirType t);

/// Gets the !torch.vtensor typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchValueTensorTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// !torch.none type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.none type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNone(MlirType t);

/// Gets the !torch.none type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchNoneTypeGet(MlirContext context);

/// Gets the !torch.none typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchNoneTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// !torch.str type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.str type
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchString(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchStringTypeGet(MlirContext context);

/// Gets the !torch.str typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchStringTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// !torch.any type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.any type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchAny(MlirType t);

/// Gets the !torch.str type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchAnyTypeGet(MlirContext context);

/// Gets the !torch.any typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchAnyTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// !torch.number type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.number type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchNumber(MlirType t);

/// Gets the !torch.number type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchNumberTypeGet(MlirContext context);

/// Gets the !torch.number typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchNumberTypeGetTypeID(void);

//===----------------------------------------------------------------------===//
// !torch.dict type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.dict type.
MLIR_CAPI_EXPORTED bool torchMlirTypeIsATorchDict(MlirType t);

/// Gets the !torch.dict type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDictTypeGet(MlirType keyType,
                                                      MlirType valueType);

MLIR_CAPI_EXPORTED MlirType torchMlirTorchDictTypeGetChecked(
    MlirContext context, MlirType keyType, MlirType valueType);

/// Gets the key type of a !torch.dict<key, value> type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDictTypeGetKeyType(MlirType t);

/// Gets the value type of a !torch.dict<key, value> type.
MLIR_CAPI_EXPORTED MlirType torchMlirTorchDictTypeGetValueType(MlirType t);

/// Gets the !torch.dict typeid.
MLIR_CAPI_EXPORTED MlirTypeID torchMlirTorchDictTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // TORCHMLIR_C_TORCHTYPES_H
