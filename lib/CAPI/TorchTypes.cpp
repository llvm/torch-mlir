//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-c/TorchTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchNnModule(MlirType t) {
  return unwrap(t).isa<Torch::NnModuleType>();
}

MlirType torchMlirTorchNnModuleTypeGet(MlirContext context,
                                       MlirStringRef className) {
  return wrap(Torch::NnModuleType::get(unwrap(context), unwrap(className)));
}

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchOptional(MlirType t) {
  return unwrap(t).isa<Torch::OptionalType>();
}

MlirType torchMlirTorchOptionalTypeGet(MlirType containedType) {
  return wrap(Torch::OptionalType::get(unwrap(containedType)));
}

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchTuple(MlirType t) {
  return unwrap(t).isa<Torch::TupleType>();
}

MlirType torchMlirTorchTupleTypeGet(MlirContext context,
                                    intptr_t numContainedTypes,
                                    MlirType const *containedTypes) {
  return wrap(Torch::TupleType::get(
      unwrap(context),
      llvm::to_vector<6>(
          llvm::map_range(llvm::makeArrayRef(containedTypes, numContainedTypes),
                          [](MlirType t) { return unwrap(t); }))));
}

//===----------------------------------------------------------------------===//
// torch.union<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchUnion(MlirType t) {
  return unwrap(t).isa<Torch::UnionType>();
}

MlirType torchMlirTorchUnionTypeGet(MlirContext context,
                                    intptr_t numContainedTypes,
                                    MlirType const *containedTypes) {
  return wrap(Torch::UnionType::get(
      unwrap(context),
      llvm::to_vector<6>(
          llvm::map_range(llvm::makeArrayRef(containedTypes, numContainedTypes),
                          [](MlirType t) { return unwrap(t); }))));
}

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchList(MlirType t) {
  return unwrap(t).isa<Torch::ListType>();
}

MlirType torchMlirTorchListTypeGet(MlirType containedType) {
  return wrap(Torch::ListType::get(unwrap(containedType)));
}

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchDevice(MlirType t) {
  return unwrap(t).isa<Torch::DeviceType>();
}

MlirType torchMlirTorchDeviceTypeGet(MlirContext context) {
  return wrap(Torch::DeviceType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.Generator type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchGenerator(MlirType t) {
  return unwrap(t).isa<Torch::GeneratorType>();
}

MlirType torchMlirTorchGeneratorTypeGet(MlirContext context) {
  return wrap(Torch::GeneratorType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchBool(MlirType t) {
  return unwrap(t).isa<Torch::BoolType>();
}

MlirType torchMlirTorchBoolTypeGet(MlirContext context) {
  return wrap(Torch::BoolType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchInt(MlirType t) {
  return unwrap(t).isa<Torch::IntType>();
}

MlirType torchMlirTorchIntTypeGet(MlirContext context) {
  return wrap(Torch::IntType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchFloat(MlirType t) {
  return unwrap(t).isa<Torch::FloatType>();
}

MlirType torchMlirTorchFloatTypeGet(MlirContext context) {
  return wrap(Torch::FloatType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchLinearParams(MlirType t) {
  return unwrap(t).isa<Torch::LinearParamsType>();
}

MlirType torchMlirTorchLinearParamsTypeGet(MlirContext context) {
  return wrap(Torch::LinearParamsType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchQInt8(MlirType t) {
  return unwrap(t).isa<Torch::QInt8Type>();
}

MlirType torchMlirTorchQInt8TypeGet(MlirContext context) {
  return wrap(Torch::QInt8Type::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.tensor type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchNonValueTensor(MlirType t) {
  return unwrap(t).isa<Torch::NonValueTensorType>();
}

MlirType torchMlirTorchNonValueTensorTypeGet(MlirContext context,
                                             intptr_t numSizes,
                                             const int64_t *optionalSizes,
                                             MlirType optionalDtype) {
  Optional<ArrayRef<int64_t>> optionalSizesArrayRef = None;
  // if numSizes == -1, then it is unranked.
  if (numSizes > -1)
    optionalSizesArrayRef = llvm::makeArrayRef(optionalSizes, numSizes);
  return wrap(Torch::NonValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context) {
  return wrap(Torch::NonValueTensorType::getWithLeastStaticInformation(
      unwrap(context)));
}

MlirType torchMlirTorchNonValueTensorTypeGetFromAttribute(MlirAttribute attr) {
  auto attrTensorType = unwrap(attr).getType().cast<RankedTensorType>();
  return wrap(Torch::NonValueTensorType::get(attrTensorType.getContext(),
                                             attrTensorType.getShape(),
                                             attrTensorType.getElementType()));
}

//===----------------------------------------------------------------------===//
// torch.vtensor type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchValueTensor(MlirType t) {
  return unwrap(t).isa<Torch::ValueTensorType>();
}

MlirType torchMlirTorchValueTensorTypeGet(MlirContext context,
                                          intptr_t numSizes,
                                          const int64_t *optionalSizes,
                                          MlirType optionalDtype) {
  Optional<ArrayRef<int64_t>> optionalSizesArrayRef = None;
  // if numSizes == -1, then it is unranked.
  if (numSizes > -1)
    optionalSizesArrayRef = llvm::makeArrayRef(optionalSizes, numSizes);
  return wrap(Torch::ValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context) {
  return wrap(
      Torch::ValueTensorType::getWithLeastStaticInformation(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.none type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchNone(MlirType t) {
  return unwrap(t).isa<Torch::NoneType>();
}

MlirType torchMlirTorchNoneTypeGet(MlirContext context) {
  return wrap(Torch::NoneType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.str type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchString(MlirType t) {
  return unwrap(t).isa<Torch::StringType>();
}

MlirType torchMlirTorchStringTypeGet(MlirContext context) {
  return wrap(Torch::StringType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.any type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchAny(MlirType t) {
  return unwrap(t).isa<Torch::AnyType>();
}

MlirType torchMlirTorchAnyTypeGet(MlirContext context) {
  return wrap(Torch::AnyType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.number type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchNumber(MlirType t) {
  return unwrap(t).isa<Torch::NumberType>();
}

MlirType torchMlirTorchNumberTypeGet(MlirContext context) {
  return wrap(Torch::NumberType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.Dict type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchDict(MlirType t) {
  return unwrap(t).isa<Torch::DictType>();
}

MlirType torchMlirTorchDictTypeGet(MlirType keyType, MlirType valueType) {
  return wrap(Torch::DictType::get(unwrap(keyType), unwrap(valueType)));
}
