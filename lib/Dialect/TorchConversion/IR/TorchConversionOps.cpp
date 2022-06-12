//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// ToBuiltinTensorOp
//===----------------------------------------------------------------------===//

LogicalResult ToBuiltinTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto resultType =
      operands[0].getType().cast<Torch::ValueTensorType>().toBuiltinTensor();
  if (!resultType)
    return failure();
  inferredReturnTypes.push_back(resultType);
  return success();
}

OpFoldResult FromI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

OpFoldResult ToI64Op::fold(llvm::ArrayRef<mlir::Attribute> operands) {
  auto attr = operands[0].dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    return attr;
  } else {
    return nullptr;
  }
}

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.cpp.inc"
