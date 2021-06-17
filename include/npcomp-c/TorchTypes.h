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

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a torch.nn.Module type
bool npcompTypeIsATorchNnModule(MlirType t);

/// Gets the !torch.nn.Module type of the specified class.
MlirType npcompTorchNnModuleTypeGet(MlirContext context,
                                    MlirStringRef className);

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.optional<T> type
bool npcompTypeIsATorchOptional(MlirType t);

/// Gets the !torch.optional<T> type with subtype T.
MlirType npcompTorchOptionalTypeGet(MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tuple type
bool npcompTypeIsATorchTuple(MlirType t);

/// Gets the !torch.tuple type with contained types `containedTypes`.
MlirType npcompTorchTupleTypeGet(MlirContext context,
                                 intptr_t numContainedTypes,
                                 MlirType const *containedTypes);

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.list<T> type
bool npcompTypeIsATorchList(MlirType t);

/// Gets the !torch.list<T> type with contained T.
MlirType npcompTorchListTypeGet(MlirType containedType);

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.Device type
bool npcompTypeIsATorchDevice(MlirType t);

/// Gets the !torch.Device type.
MlirType npcompTorchDeviceTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.bool type
bool npcompTypeIsATorchBool(MlirType t);

/// Gets the !torch.bool type.
MlirType npcompTorchBoolTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.int type
bool npcompTypeIsATorchInt(MlirType t);

/// Gets the !torch.int type.
MlirType npcompTorchIntTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.float type
bool npcompTypeIsATorchFloat(MlirType t);

/// Gets the !torch.float type.
MlirType npcompTorchFloatTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.LinearParams type
bool npcompTypeIsATorchLinearParams(MlirType t);

/// Gets the !torch.LinearParams type.
MlirType npcompTorchLinearParamsTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.qint8 type
bool npcompTypeIsATorchQInt8(MlirType t);

/// Gets the !torch.qint8 type.
MlirType npcompTorchQInt8TypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// torch.tensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.tensor type
bool npcompTypeIsATorchNonValueTensor(MlirType t);

/// Gets a !torch.tensor type.
///
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).  -
/// `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MlirType npcompTorchNonValueTensorTypeGet(MlirContext context,
                                          intptr_t numSizes,
                                          const int64_t *optionalSizes,
                                          MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MlirType
npcompTorchNonValueTensorTypeGetWithLeastStaticInformation(MlirContext context);

/// Gets a !torch.tensor type, taking shape/dtype from a ShapedType `type`.
MlirType npcompTorchNonValueTensorTypeGetFromShaped(MlirType type);

//===----------------------------------------------------------------------===//
// torch.vtensor type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.vtensor type
bool npcompTypeIsATorchValueTensor(MlirType t);

/// Gets a !torch.vtensor type.
///
/// - `optionalSizes` is allowed to be null, meaning that no size
/// information is present (and `numSizes` is ignored in that case).
/// - `optionalDtype` is allowed to be null, meaning that no dtype
/// information is present.
MlirType npcompTorchValueTensorTypeGet(MlirContext context, intptr_t numSizes,
                                       const int64_t *optionalSizes,
                                       MlirType optionalDtype);

/// Gets the !torch.tensor type with the least static information.
MlirType
npcompTorchValueTensorTypeGetWithLeastStaticInformation(MlirContext context);

/// Gets a !torch.tensor type, taking shape/dtype from a ShapedType `type`.
MlirType npcompTorchValueTensorTypeGetFromShaped(MlirType type);

//===----------------------------------------------------------------------===//
// !torch.none type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.none type
bool npcompTypeIsATorchNone(MlirType t);

/// Gets the !torch.none type.
MlirType npcompTorchNoneTypeGet(MlirContext context);

//===----------------------------------------------------------------------===//
// !torch.str type.
//===----------------------------------------------------------------------===//

/// Checks whether the given type is a !torch.str type
bool npcompTypeIsATorchString(MlirType t);

/// Gets the !torch.str type.
MlirType npcompTorchStringTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // NPCOMP_C_TORCHTYPES_H
