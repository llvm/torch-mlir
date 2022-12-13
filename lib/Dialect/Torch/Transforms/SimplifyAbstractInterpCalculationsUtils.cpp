//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "SimplifyAbstractInterpCalculationsUtils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

LogicalResult Torch::updateCalculateOpResultTypes(Operation *calculateOp,
                                                  int resultNum,
                                                  Type newResultType,
                                                  PatternRewriter &rewriter) {
  Location loc = calculateOp->getLoc();
  auto result = calculateOp->getResult(resultNum);
  Type originalResultType = result.getType();
  Type updatedType;
  if (auto originalBaseTensorType =
          originalResultType.template dyn_cast<BaseTensorType>()) {
    // If we didn't get any new information, there is nothing left for us to do.
    updatedType = meetTensorTypes(originalBaseTensorType,
                                  newResultType.cast<BaseTensorType>());
    if (!updatedType || updatedType == originalBaseTensorType)
      return rewriter.notifyMatchFailure(
          calculateOp, "New type information does not refine old type");
  } else if (auto originalResultType =
                 result.getType().template dyn_cast<Torch::NumberType>()) {
    if (!newResultType.isa<Torch::FloatType, Torch::IntType>()) {
      return rewriter.notifyMatchFailure(
          calculateOp,
          "Refinement of `NumberType` must be a `FloatType` or `IntType`");
    }
    updatedType = newResultType;
  } else {
    return rewriter.notifyMatchFailure(calculateOp,
                                       "Unimplemented: Expected result type to "
                                       "be `BaseTensorType` or `NumberType`");
  }

  // Update all the uses of the result type to the new type, if possible. Insert
  // a TensorStaticInfoCastOp for any users that might require the exact
  // previous type.
  Value originalTypedValue;
  for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
    if (use.getOwner()
            ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
      continue;
    }
    if (!originalTypedValue) {
      rewriter.setInsertionPointAfter(calculateOp);
      if (originalResultType.isa<BaseTensorType>()) {
        originalTypedValue = rewriter.create<TensorStaticInfoCastOp>(
            loc, originalResultType, result);
      } else if (originalResultType.isa<Torch::NumberType>()) {
        originalTypedValue =
            rewriter.create<DerefineOp>(loc, originalResultType, result);
      } else {
        return rewriter.notifyMatchFailure(
            calculateOp, "Unimplemented: Expected result type to "
                         "be `BaseTensorType` or `NumberType`");
      }
    }
    use.set(originalTypedValue);
  }
  result.setType(updatedType);

  // Update the value yielded from the body to match the new result type. If we
  // can refine the def in place, do that, otherwise insert a
  // TensorStaticInfoCastOp.
  Operation *yieldValues = calculateOp->getRegion(0).front().getTerminator();
  OpOperand &use = yieldValues->getOpOperand(resultNum);
  Value def = use.get();
  Value newYieldedValue;
  if (def.isa<OpResult>() &&
      def.cast<OpResult>()
          .getDefiningOp()
          ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
    newYieldedValue = def;
  } else {
    rewriter.setInsertionPoint(yieldValues);
    if (updatedType.isa<BaseTensorType>()) {
      newYieldedValue =
          rewriter.create<TensorStaticInfoCastOp>(loc, updatedType, def);
    } else {
      newYieldedValue =
          rewriter.create<PrimUncheckedCastOp>(loc, updatedType, def);
    }
  }
  use.set(newYieldedValue);
  newYieldedValue.setType(updatedType);

  return success();
}
