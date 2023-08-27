//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/PassDetail.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::torch::TMTensor;

namespace {
class SimplifyNumpyBroadcast : public OpRewritePattern<NumpyBroadcastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NumpyBroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    Location loc = broadcastOp.getLoc();
    Value input = broadcastOp.getInput();
    Value output = broadcastOp.getOutput();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto outputType = output.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = outputType.getRank();
    int64_t diff = outputRank - inputRank;

    Value oneIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    SmallVector<bool> broadcastedStatus;
    for (int64_t i = 0, e = inputRank; i < e; ++i) {
      FailureOr<bool> dimsEqual =
          ValueBoundsConstraintSet::areEqual(input, output, i, i + diff);
      if (succeeded(dimsEqual) && *dimsEqual) {
        broadcastedStatus.push_back(false);
        continue;
      }
      FailureOr<bool> isUnit =
          ValueBoundsConstraintSet::areEqual(input, oneIndex, i, std::nullopt);
      if (succeeded(isUnit) || *isUnit) {
        broadcastedStatus.push_back(true);
        continue;
      }
      // Unable to statically bound all input dims to a broadcast status; bail.
      return failure();
    }

    // If no dims are broadcasted and the rank doesn't change, we can just fold
    // the op away entirely.
    if (!llvm::any_of(broadcastedStatus, [](bool b) { return b; }) &&
        inputRank == outputRank) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(
          broadcastOp, broadcastOp.getResult(0).getType(), input);
      return success();
    }

    SmallVector<AffineExpr> inputExprs;
    for (int64_t i = 0, e = inputRank; i < e; ++i) {
      if (broadcastedStatus[i]) {
        inputExprs.push_back(rewriter.getAffineConstantExpr(0));
        continue;
      }
      inputExprs.push_back(rewriter.getAffineDimExpr(i + diff));
    }

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(outputRank, 0, inputExprs, broadcastOp.getContext()),
        rewriter.getMultiDimIdentityMap(outputRank)};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputRank, utils::IteratorType::parallel);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        broadcastOp, output.getType(), input, output, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });
    return success();
  }
};
} // namespace

/// Pattern rewriter hook to lower a `tm_tensor.npbroadcast` to linalg.
namespace {
class LowerNumpyBroadcastToLinalg : public OpRewritePattern<NumpyBroadcastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(NumpyBroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    Location loc = broadcastOp.getLoc();
    Value input = broadcastOp.getInput();
    Value output = broadcastOp.getOutput();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto outputType = output.getType().cast<RankedTensorType>();
    int64_t diff = outputType.getRank() - inputType.getRank();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    Value zeroIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    Value oneIndex =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(outputType.getRank())};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);
    rewriter
        .replaceOpWithNewOp<linalg::GenericOp>(
            broadcastOp, output.getType(), ValueRange(), output, indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              // `loopIndices` contains IV of the linalg loops which
              // would be used to extract values from the input tensor
              // later on.
              SmallVector<Value> loopIndices;
              for (int64_t i = diff; i < outputType.getRank(); ++i) {
                loopIndices.push_back(b.create<linalg::IndexOp>(loc, i));
              }
              // `inputIndicesToExtract` contains i-th linalg loop IV if
              // the i-th input dimension is not 1, else it contains a
              // zero index.
              SmallVector<Value> inputIndicesToExtract;
              for (size_t i = 0, n = inputShape.size(); i < n; i++) {
                if (inputShape[i] == 1 && outputShape[i + diff] != 1) {
                  inputIndicesToExtract.push_back(zeroIndex);
                } else {
                  Value inputDim = b.createOrFold<tensor::DimOp>(loc, input, i);
                  Value isEqual = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, inputDim, oneIndex);
                  Value select = b.create<arith::SelectOp>(
                      loc, isEqual, zeroIndex, loopIndices[i]);
                  inputIndicesToExtract.push_back(select);
                }
              }
              // Extract and yield the value from input tensor at
              // `inputIndicesToExtract` indices.
              Value result = b.create<tensor::ExtractOp>(loc, input,
                                                         inputIndicesToExtract);
              b.create<linalg::YieldOp>(loc, result);
            })
        .getResult(0);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct TMTensorBroadcastToLinalgPass
    : public TMTensorBroadcastToLinalgBase<TMTensorBroadcastToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.insert<SimplifyNumpyBroadcast>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<LowerNumpyBroadcastToLinalg>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
torch::TMTensor::createTMTensorBroadcastToLinalgPass() {
  return std::make_unique<TMTensorBroadcastToLinalgPass>();
}
