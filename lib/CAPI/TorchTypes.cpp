//===- TorchTypes.cpp - C Interface for torch types -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/TorchTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// torch.nn.Module type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchNnModule(MlirType t) {
  return unwrap(t).isa<Torch::NnModuleType>();
}

MlirType npcompTorchNnModuleTypeGet(MlirContext context,
                                    MlirStringRef className) {
  return wrap(Torch::NnModuleType::get(unwrap(context), unwrap(className)));
}

//===----------------------------------------------------------------------===//
// torch.optional type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchOptional(MlirType t) {
  return unwrap(t).isa<Torch::OptionalType>();
}

MlirType npcompTorchOptionalTypeGet(MlirType containedType) {
  return wrap(Torch::OptionalType::get(unwrap(containedType)));
}

//===----------------------------------------------------------------------===//
// torch.tuple<T1, T2, T3> type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchTuple(MlirType t) {
  return unwrap(t).isa<Torch::TupleType>();
}

MlirType npcompTorchTupleTypeGet(MlirContext context,
                                 intptr_t numContainedTypes,
                                 MlirType const *containedTypes) {
  return wrap(Torch::TupleType::get(
      unwrap(context),
      llvm::to_vector<6>(
          llvm::map_range(llvm::makeArrayRef(containedTypes, numContainedTypes),
                          [](MlirType t) { return unwrap(t); }))));
}

//===----------------------------------------------------------------------===//
// torch.list<T> type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchList(MlirType t) {
  return unwrap(t).isa<Torch::ListType>();
}

MlirType npcompTorchListTypeGet(MlirType containedType) {
  return wrap(Torch::ListType::get(unwrap(containedType)));
}

//===----------------------------------------------------------------------===//
// torch.Device type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchDevice(MlirType t) {
  return unwrap(t).isa<Torch::DeviceType>();
}

MlirType npcompTorchDeviceTypeGet(MlirContext context) {
  return wrap(Torch::DeviceType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.bool type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchBool(MlirType t) {
  return unwrap(t).isa<Torch::BoolType>();
}

MlirType npcompTorchBoolTypeGet(MlirContext context) {
  return wrap(Torch::BoolType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.int type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchInt(MlirType t) {
  return unwrap(t).isa<Torch::IntType>();
}

MlirType npcompTorchIntTypeGet(MlirContext context) {
  return wrap(Torch::IntType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.float type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchFloat(MlirType t) {
  return unwrap(t).isa<Torch::FloatType>();
}

MlirType npcompTorchFloatTypeGet(MlirContext context) {
  return wrap(Torch::FloatType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.LinearParams type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchLinearParams(MlirType t) {
  return unwrap(t).isa<Torch::LinearParamsType>();
}

MlirType npcompTorchLinearParamsTypeGet(MlirContext context) {
  return wrap(Torch::LinearParamsType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.qint8 type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchQInt8(MlirType t) {
  return unwrap(t).isa<Torch::QInt8Type>();
}

MlirType npcompTorchQInt8TypeGet(MlirContext context) {
  return wrap(Torch::QInt8Type::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.tensor type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchNonValueTensor(MlirType t) {
  return unwrap(t).isa<Torch::NonValueTensorType>();
}

MlirType npcompTorchNonValueTensorTypeGet(MlirContext context,
                                          intptr_t numSizes,
                                          const int64_t *optionalSizes,
                                          MlirType optionalDtype) {
  Optional<ArrayRef<int64_t>> optionalSizesArrayRef = None;
  if (optionalSizes)
    optionalSizesArrayRef = llvm::makeArrayRef(optionalSizes, numSizes);
  return wrap(Torch::NonValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType npcompTorchNonValueTensorTypeGetWithLeastStaticInformation(
    MlirContext context) {
  return wrap(Torch::NonValueTensorType::getWithLeastStaticInformation(
      unwrap(context)));
}

MlirType npcompTorchNonValueTensorTypeGetFromShaped(MlirType type) {
  return wrap(Torch::NonValueTensorType::getFromShaped(
      unwrap(type).cast<ShapedType>()));
}

//===----------------------------------------------------------------------===//
// torch.vtensor type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchValueTensor(MlirType t) {
  return unwrap(t).isa<Torch::ValueTensorType>();
}

MlirType npcompTorchValueTensorTypeGet(MlirContext context, intptr_t numSizes,
                                       const int64_t *optionalSizes,
                                       MlirType optionalDtype) {
  Optional<ArrayRef<int64_t>> optionalSizesArrayRef = None;
  if (optionalSizes)
    optionalSizesArrayRef = llvm::makeArrayRef(optionalSizes, numSizes);
  return wrap(Torch::ValueTensorType::get(
      unwrap(context), optionalSizesArrayRef, unwrap(optionalDtype)));
}

MlirType
npcompTorchValueTensorTypeGetWithLeastStaticInformation(MlirContext context) {
  return wrap(
      Torch::ValueTensorType::getWithLeastStaticInformation(unwrap(context)));
}

MlirType npcompTorchValueTensorTypeGetFromShaped(MlirType type) {
  return wrap(
      Torch::ValueTensorType::getFromShaped(unwrap(type).cast<ShapedType>()));
}

//===----------------------------------------------------------------------===//
// torch.none type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchNone(MlirType t) {
  return unwrap(t).isa<Torch::NoneType>();
}

MlirType npcompTorchNoneTypeGet(MlirContext context) {
  return wrap(Torch::NoneType::get(unwrap(context)));
}

//===----------------------------------------------------------------------===//
// torch.str type.
//===----------------------------------------------------------------------===//

bool npcompTypeIsATorchString(MlirType t) {
  return unwrap(t).isa<Torch::StringType>();
}

MlirType npcompTorchStringTypeGet(MlirContext context) {
  return wrap(Torch::StringType::get(unwrap(context)));
}
