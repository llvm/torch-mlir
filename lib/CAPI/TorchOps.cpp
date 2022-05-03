//===- TorchOps.cpp - C Interface for torch ops ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-c/TorchOps.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

MlirValue torchMlirAdjustStaticInformation(MlirBlock block_,
                                           MlirOperation insertBefore_,
                                           MlirValue value_,
                                           MlirType desiredType_,
                                           bool userAllowsRefinement) {
  Block *block = unwrap(block_);
  Operation *insertBefore = unwrap(insertBefore_);
  OpBuilder builder(unwrap(mlirTypeGetContext(desiredType_)));
  builder.setInsertionPoint(block, insertBefore ? insertBefore->getIterator()
                                                : block->end());

  Value value = unwrap(value_);
  Type type = value.getType();
  Type desiredType = unwrap(desiredType_);

  // If the value is already of the desired type, we're done.
  if (type == desiredType)
    return wrap(value);

  // If the type is a tensor, then adjust the static information.
  if ((type.isa<Torch::ValueTensorType>() &&
       desiredType.isa<Torch::ValueTensorType>()) ||
      (type.isa<Torch::NonValueTensorType>() &&
       desiredType.isa<Torch::NonValueTensorType>())) {
    Value adjusted = builder.create<Torch::TensorStaticInfoCastOp>(
        value.getLoc(), desiredType, value);
    return wrap(adjusted);
  }

  // If the type is a subtype of desiredType, then we need to derefine it to
  // desiredType, unless the user allows refinement.
  if (Torch::isValidSubtype(type, desiredType)) {
    if (!userAllowsRefinement) {
      Value adjusted =
          builder.create<Torch::DerefineOp>(value.getLoc(), desiredType, value);
      return wrap(adjusted);
    } else {
      return wrap(value);
    }
  }

  // If the desiredType is subtype of type, then we assume that the desiredType
  // is dynamically valid, so we do an unchecked cast.
  if (Torch::isValidSubtype(desiredType, type)) {
    Value adjusted = builder.create<Torch::PrimUncheckedCastOp>(
        value.getLoc(), desiredType, value);
    return wrap(adjusted);
  }

  // No known adjustment.
  return {};
}
