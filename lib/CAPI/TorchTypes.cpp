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

bool torchMlirTypeIsValidSubtype(MlirType subtype, MlirType type) {
  return Torch::isValidSubtype(unwrap(subtype), unwrap(type));
}

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

MlirTypeID torchMlirTorchNnModuleTypeGetTypeID() {
  return wrap(Torch::NnModuleType::getTypeID());
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

MlirType torchMlirTorchOptionalTypeGetContained(MlirType t) {
  auto type = unwrap(t).cast<Torch::OptionalType>();
  return wrap(type.getContainedType());
}

MlirTypeID torchMlirTorchOptionalTypeGetTypeID() {
  return wrap(Torch::OptionalType::getTypeID());
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
      unwrap(context), llvm::to_vector<6>(llvm::map_range(
                           llvm::ArrayRef(containedTypes, numContainedTypes),
                           [](MlirType t) { return unwrap(t); }))));
}

size_t torchMlirTorchTupleTypeGetNumTypes(MlirType t) {
  auto type = unwrap(t).cast<Torch::TupleType>();
  return type.getContainedTypes().size();
}

MlirType torchMlirTorchTupleTypeGetType(MlirType t, intptr_t pos) {
  auto type = unwrap(t).cast<Torch::TupleType>();
  return wrap(type.getContainedTypes()[pos]);
}

MlirTypeID torchMlirTorchTupleTypeGetTypeID() {
  return wrap(Torch::TupleType::getTypeID());
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
      unwrap(context), llvm::to_vector<6>(llvm::map_range(
                           llvm::ArrayRef(containedTypes, numContainedTypes),
                           [](MlirType t) { return unwrap(t); }))));
}

size_t torchMlirTorchUnionTypeGetNumTypes(MlirType t) {
  auto type = unwrap(t).cast<Torch::UnionType>();
  return type.getContainedTypes().size();
}

MlirType torchMlirTorchUnionTypeGetType(MlirType t, intptr_t pos) {
  auto type = unwrap(t).cast<Torch::UnionType>();
  return wrap(type.getContainedTypes()[pos]);
}

MlirTypeID torchMlirTorchUnionTypeGetTypeID() {
  return wrap(Torch::UnionType::getTypeID());
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

MlirType torchMlirTorchListTypeGetContainedType(MlirType t) {
  return wrap(unwrap(t).cast<Torch::ListType>().getContainedType());
}

MlirTypeID torchMlirTorchListTypeGetTypeID() {
  return wrap(Torch::ListType::getTypeID());
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

MlirTypeID torchMlirTorchDeviceTypeGetTypeID() {
  return wrap(Torch::DeviceType::getTypeID());
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

MlirTypeID torchMlirTorchGeneratorTypeGetTypeID() {
  return wrap(Torch::GeneratorType::getTypeID());
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

MlirTypeID torchMlirTorchBoolTypeGetTypeID() {
  return wrap(Torch::BoolType::getTypeID());
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

MlirTypeID torchMlirTorchIntTypeGetTypeID() {
  return wrap(Torch::IntType::getTypeID());
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

MlirTypeID torchMlirTorchFloatTypeGetTypeID() {
  return wrap(Torch::FloatType::getTypeID());
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

MlirTypeID torchMlirTorchLinearParamsTypeGetTypeID() {
  return wrap(Torch::LinearParamsType::getTypeID());
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

MlirTypeID torchMlirTorchQInt8TypeGetTypeID() {
  return wrap(Torch::QInt8Type::getTypeID());
}

//===----------------------------------------------------------------------===//
// torch.quint8 type.
//===----------------------------------------------------------------------===//

bool torchMlirTypeIsATorchQUInt8(MlirType t) {
  return unwrap(t).isa<Torch::QUInt8Type>();
}

MlirType torchMlirTorchQUInt8TypeGet(MlirContext context) {
  return wrap(Torch::QUInt8Type::get(unwrap(context)));
}

MlirTypeID torchMlirTorchQUInt8TypeGetTypeID() {
  return wrap(Torch::QUInt8Type::getTypeID());
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
  std::optional<ArrayRef<int64_t>> optionalSizesArrayRef = std::nullopt;
  // if numSizes == -1, then it is unranked.
  if (numSizes > -1)
    optionalSizesArrayRef = llvm::ArrayRef(optionalSizes, numSizes);
  return wrap(Torch::NonValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context) {
  return wrap(Torch::NonValueTensorType::getWithLeastStaticInformation(
      unwrap(context)));
}

MlirType torchMlirTorchNonValueTensorTypeGetFromAttribute(MlirAttribute attr) {
  auto attrTensorType =
      unwrap(attr).cast<TypedAttr>().getType().cast<RankedTensorType>();
  return wrap(Torch::NonValueTensorType::get(attrTensorType.getContext(),
                                             attrTensorType.getShape(),
                                             attrTensorType.getElementType()));
}

int64_t torchMlirTorchNonValueTensorTypeGetRank(MlirType t) {
  return unwrap(t).cast<Torch::NonValueTensorType>().getSizes().size();
}

bool torchMlirTorchNonValueTensorTypeHasSizes(MlirType t) {
  return unwrap(t).cast<Torch::NonValueTensorType>().hasSizes();
}

bool torchMlirTorchNonValueTensorTypeHasDtype(MlirType t) {
  return unwrap(t).cast<Torch::NonValueTensorType>().hasDtype();
}

int64_t torchMlirTorchNonValueTensorTypeGetSizes(MlirType t, int64_t *sizes) {
  auto tensorType = unwrap(t).cast<Torch::NonValueTensorType>();
  bool hasSizes = tensorType.hasSizes();
  if (!hasSizes)
    return -1;

  auto sizes_ = tensorType.getSizes();
  for (const auto &s : llvm::enumerate(sizes_)) {
    sizes[s.index()] = s.value();
  }
  return 0;
}

MlirType torchMlirTorchNonValueTensorTypeGetDtype(MlirType t) {
  return wrap(unwrap(t).cast<Torch::NonValueTensorType>().getDtype());
}

MlirTypeID torchMlirTorchNonValueTensorTypeGetTypeID() {
  return wrap(Torch::NonValueTensorType::getTypeID());
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
  std::optional<ArrayRef<int64_t>> optionalSizesArrayRef = std::nullopt;
  // if numSizes == -1, then it is unranked.
  if (numSizes > -1)
    optionalSizesArrayRef = llvm::ArrayRef(optionalSizes, numSizes);
  return wrap(Torch::ValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context) {
  return wrap(
      Torch::ValueTensorType::getWithLeastStaticInformation(unwrap(context)));
}

MlirType torchMlirTorchValueTensorTypeGetFromAttribute(MlirAttribute attr) {
  auto attrTensorType =
      unwrap(attr).cast<TypedAttr>().getType().cast<RankedTensorType>();
  return wrap(Torch::ValueTensorType::get(attrTensorType.getContext(),
                                          attrTensorType.getShape(),
                                          attrTensorType.getElementType()));
}

int64_t torchMlirTorchValueTensorTypeGetRank(MlirType t) {
  return unwrap(t).cast<Torch::ValueTensorType>().getSizes().size();
}

bool torchMlirTorchValueTensorTypeHasSizes(MlirType t) {
  return unwrap(t).cast<Torch::ValueTensorType>().hasSizes();
}

bool torchMlirTorchValueTensorTypeHasDtype(MlirType t) {
  return unwrap(t).cast<Torch::ValueTensorType>().hasDtype();
}

int64_t torchMlirTorchValueTensorTypeGetSizes(MlirType t, int64_t *sizes) {
  auto tensorType = unwrap(t).cast<Torch::ValueTensorType>();
  bool hasSizes = tensorType.hasSizes();
  if (!hasSizes)
    return -1;

  auto sizes_ = tensorType.getSizes();
  for (const auto &s : llvm::enumerate(sizes_)) {
    sizes[s.index()] = s.value();
  }
  return 0;
}

MlirType torchMlirTorchValueTensorTypeGetDtype(MlirType t) {
  return wrap(unwrap(t).cast<Torch::ValueTensorType>().getDtype());
}

MlirTypeID torchMlirTorchValueTensorTypeGetTypeID() {
  return wrap(Torch::ValueTensorType::getTypeID());
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

MlirTypeID torchMlirTorchNoneTypeGetTypeID() {
  return wrap(Torch::NoneType::getTypeID());
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

MlirTypeID torchMlirTorchStringTypeGetTypeID() {
  return wrap(Torch::StringType::getTypeID());
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

MlirTypeID torchMlirTorchAnyTypeGetTypeID() {
  return wrap(Torch::AnyType::getTypeID());
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

MlirTypeID torchMlirTorchNumberTypeGetTypeID() {
  return wrap(Torch::NumberType::getTypeID());
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

MlirType torchMlirTorchDictTypeGetChecked(MlirContext context, MlirType keyType,
                                          MlirType valueType) {
  auto unknownLoc = unwrap(mlirLocationUnknownGet(context));
  return wrap(Torch::DictType::getChecked(unknownLoc, unwrap(context),
                                          unwrap(keyType), unwrap(valueType)));
}

MlirType torchMlirTorchDictTypeGetKeyType(MlirType t) {
  auto type = unwrap(t).cast<Torch::DictType>();
  return wrap(type.getKeyType());
}

MlirType torchMlirTorchDictTypeGetValueType(MlirType t) {
  auto type = unwrap(t).cast<Torch::DictType>();
  return wrap(type.getValueType());
}

MlirTypeID torchMlirTorchDictTypeGetTypeID() {
  return wrap(Torch::DictType::getTypeID());
}
