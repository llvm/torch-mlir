//===- RestructureNonConstantAxes.cpp --------------------------------*-
// C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torch-lower-to-backend-contract"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

template <typename SrcOp>
class ConstantifyDimArgument : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  bool isDimConstant(SrcOp op) const {
    SmallVector<int64_t> dimList;
    int64_t dim;
    return matchPattern(op.getDim(), m_TorchListOfConstantInts(dimList)) ||
           matchPattern(op.getDim(), m_TorchConstantInt(&dim));
  }

  /*
  This function renders the reduction dim constant by reshaping the input tensor
  such that the dim argument is the middle dimension.

  For example, if the input tensor has shape [3,4,5,6,7] and the dim argument is
  -2, the input tensor is reshaped to [3,4,5,6,7] -> [12,5,42], the reduction
  operation is applied, and the result is reshaped back to [3,4,1,6,7].

  Since we don't know the dim argument at compile time, we need to compute the
  arguments to the reshape op at runtime. We do this by computing the new shape
  of the tensor by multiplying the shapes of the tensor before and after the dim
  argument, and then reshaping the tensor to this new shape.
  */
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    Value self = op.getSelf();
    Value dim = op.getDim();

    if (isDimConstant(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "dim argument is already constant");
    }

    if (isa<Torch::NoneType>(dim.getType())) {
      return rewriter.notifyMatchFailure(
          op, "RestructureNonConstantAxes does not support None dim");
    }

    // when keepdim is not constant, check the ranks of the input and output
    // tensors
    ValueTensorType selfTy =
        llvm::cast<ValueTensorType>(op.getSelf().getType());
    ValueTensorType resultTy =
        llvm::cast<ValueTensorType>(op.getResult().getType());
    if (selfTy.hasSizes() && resultTy.hasSizes() &&
        selfTy.getSizes().size() != resultTy.getSizes().size()) {
      return rewriter.notifyMatchFailure(
          op,
          "RestructureNonConstantAxes does not yet support keepdim=false, but "
          "the input and output tensors have different ranks");
    }

    Type intType = rewriter.getType<Torch::IntType>();
    Type boolType = rewriter.getType<Torch::BoolType>();
    auto createInt = [&](int value) {
      return rewriter.create<Torch::ConstantIntOp>(
          loc, intType,
          rewriter.getIntegerAttr(rewriter.getIntegerType(64), value));
    };
    Value zero = createInt(0);
    Value one = createInt(1);

    // handle when dim is a single element list
    bool oldDimIsList = isa<Torch::ListType>(dim.getType());
    if (oldDimIsList) {
      Value len = rewriter.create<Torch::AtenLenTOp>(loc, intType, dim);
      Value dimListIsLengthOne =
          rewriter.create<Torch::AtenEqIntOp>(loc, boolType, len, one);
      rewriter.create<Torch::RuntimeAssertOp>(
          loc, dimListIsLengthOne,
          rewriter.getStringAttr("RestructureNonConstantAxes does not support "
                                 "dim lists with more than one element"));
      dim = rewriter.create<Torch::Aten__Getitem__TOp>(loc, intType, dim, zero);
    }

    // Normalize negative dim
    Value rank = rewriter.create<Torch::AtenDimOp>(loc, intType, self);
    Value isNegative = rewriter.create<Torch::AtenLtIntOp>(loc, dim, zero);
    Value rankOffset = rewriter.create<Torch::AtenMulIntOp>(
        loc, intType,
        rewriter.create<Torch::AtenIntBoolOp>(loc, intType, isNegative), rank);
    dim = rewriter.create<Torch::AtenAddIntOp>(loc, intType, dim, rankOffset);

    auto createConditionalMult = [&](Value self, Value multiplier,
                                     Value condition) {
      // compute:
      // result = codition ? (self * multiplier) : self
      // via
      // result = self * (1 + (multiplier - 1) * condition)
      // which translates to:

      // result = multiplier - 1
      Value result = rewriter.create<Torch::AtenSubIntOp>(
          loc, intType, multiplier, createInt(1));
      // result = result * condition
      result =
          rewriter.create<Torch::AtenMulIntOp>(loc, intType, result, condition);
      // result = result + 1
      result = rewriter.create<Torch::AtenAddIntOp>(loc, intType, result,
                                                    createInt(1));
      // result = self * result
      result = rewriter.create<Torch::AtenMulIntOp>(loc, intType, self, result);
      return result;
    };

    // new shape = [beforeDim, dimSize, afterDim]
    Value beforeProd = createInt(1);
    Value afterProd = createInt(1);
    Value dimSize = createInt(1);

    for (size_t i = 0; i < selfTy.getSizes().size(); ++i) {
      Value idx = createInt(i);
      Value size =
          rewriter.create<Torch::AtenSizeIntOp>(loc, intType, self, idx);

      Value isBeforeDim =
          rewriter.create<Torch::AtenLtIntOp>(loc, boolType, idx, dim);
      isBeforeDim =
          rewriter.create<Torch::AtenIntBoolOp>(loc, intType, isBeforeDim);
      Value isAfterDim =
          rewriter.create<Torch::AtenGtIntOp>(loc, boolType, idx, dim);
      isAfterDim =
          rewriter.create<Torch::AtenIntBoolOp>(loc, intType, isAfterDim);

      Value isEqualToDim =
          rewriter.create<Torch::AtenEqIntOp>(loc, boolType, idx, dim);
      isEqualToDim =
          rewriter.create<Torch::AtenIntBoolOp>(loc, intType, isEqualToDim);
      dimSize = createConditionalMult(dimSize, size, isEqualToDim);

      beforeProd = createConditionalMult(beforeProd, size, isBeforeDim);
      afterProd = createConditionalMult(afterProd, size, isAfterDim);
    }

    Value newShape = rewriter.create<Torch::PrimListConstructOp>(
        loc, rewriter.getType<Torch::ListType>(intType),
        ValueRange{beforeProd, dimSize, afterProd});

    // Reshape input
    auto newSelfTy = selfTy.getWithSizesAndDtype(
        SmallVector<int64_t>{Torch::kUnknownSize, Torch::kUnknownSize,
                             Torch::kUnknownSize},
        selfTy.getDtype());
    Value reshapedSelf =
        rewriter.create<Torch::AtenViewOp>(loc, newSelfTy, self, newShape);

    // construct new operange range where self is replaced with reshapedSelf
    // tensor, and dim is replaced with 1
    Value newDim;
    if (oldDimIsList) {
      newDim = rewriter.create<Torch::PrimListConstructOp>(
          loc, rewriter.getType<Torch::ListType>(intType), ValueRange{one});
    } else {
      newDim = one;
    }
    ValueRange oldOperands = op->getOperands();
    SmallVector<Value> newOperandsVect;
    for (size_t i = 0; i < oldOperands.size(); ++i) {
      if (oldOperands[i] == op.getSelf()) {
        newOperandsVect.push_back(reshapedSelf);
      } else if (oldOperands[i] == op.getDim()) {
        newOperandsVect.push_back(newDim);
      } else {
        newOperandsVect.push_back(oldOperands[i]);
      }
    }
    ValueRange newOperands = ValueRange(newOperandsVect);

    // construct new reduction op result type
    ValueTensorType newResultTy =
        cast<ValueTensorType>(resultTy.getWithSizesAndDtype(
            SmallVector<int64_t>{Torch::kUnknownSize, 1, Torch::kUnknownSize},
            resultTy.getDtype()));

    Value newReductionOp =
        rewriter.create<SrcOp>(loc, newResultTy, newOperands, op->getAttrs());

    // Reshape the result back to original shape
    ValueTensorType oldResultTy =
        cast<ValueTensorType>(op.getResult().getType());
    SmallVector<Value> shapeValues;
    for (auto dim : oldResultTy.getSizes()) {
      shapeValues.push_back(createInt(dim));
    }
    Value originalShape = rewriter.create<Torch::PrimListConstructOp>(
        loc, rewriter.getType<Torch::ListType>(intType), shapeValues);
    Value result = rewriter.create<Torch::AtenViewOp>(
        loc, op->getResult(0).getType(), newReductionOp, originalShape);

    rewriter.replaceOp(op, result);
    return success();
  };
};

template <typename... OpTypes>
void addConstantifyDimArgumentPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  // simple variadic template to sugar up adding the patterns
  (patterns.add<ConstantifyDimArgument<OpTypes>>(context), ...);
}

void populateRestructureNonConstantAxesPattern(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  // these are the reduction ops with a dim argument

  addConstantifyDimArgumentPatterns<
      // not supported because they have multiple results
      // AtenMaxDimOp,
      // AtenMinDimOp,
      AtenSumDimIntListOp, AtenAllDimOp, AtenLinalgVectorNormOp,
      AtenFrobeniusNormDimOp>(patterns, context);
}

class RestructureNonConstantAxesPass
    : public RestructureNonConstantAxesBase<RestructureNonConstantAxesPass> {
public:
  RestructureNonConstantAxesPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    populateRestructureNonConstantAxesPattern(patterns, context);

    // TODO: Debug visitation order to make this more efficient.
    // A single linear scan should suffice.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRestructureNonConstantAxesPass() {
  return std::make_unique<RestructureNonConstantAxesPass>();
}
