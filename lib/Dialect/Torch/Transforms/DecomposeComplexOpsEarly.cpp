//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class DecomposeAtenSplitSizesOp : public OpRewritePattern<AtenSplitSizesOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSplitSizesOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> splitSizes;
    // TODO: Support non static splitSizes.
    if (!matchPattern(op.split_size(), m_TorchConstantIntList(splitSizes)))
      return failure();

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    auto listType = op.getType().template cast<Torch::ListType>();
    Location loc = op.getLoc();

    // Calculate the end indices of the slices.
    SmallVector<int64_t> endIdx;
    endIdx.push_back(splitSizes[0]);
    for (unsigned i = 1; i < splitSizes.size(); i++) {
      endIdx.push_back(splitSizes[i] + endIdx[i - 1]);
    }
    int64_t start = 0;
    SmallVector<Value> listOfTensors;
    Value stepVal = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    // For each of the split create slice operation to slice the input tensor
    // from `start` to `endIdx`.
    for (unsigned i = 0; i < splitSizes.size(); i++) {
      Value startVal = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(start));
      Value endVal = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(endIdx[i]));
      Type resultType = listType.getContainedType();
      Value slice = rewriter.create<AtenSliceTensorOp>(
          loc, resultType, op.self(), op.dim(), startVal, endVal, stepVal);
      listOfTensors.push_back(slice);
      start = endIdx[i];
    }

    Value result = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(listType.getContainedType()), listOfTensors);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {

template <typename SplitTensorLikeOp>
class DecomposeAtenSplitTensorLikeOp
    : public OpRewritePattern<SplitTensorLikeOp> {
public:
  using OpRewritePattern<SplitTensorLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SplitTensorLikeOp op,
                                PatternRewriter &rewriter) const override {
    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    auto listType = op.getType().template cast<Torch::ListType>();
    Location loc = op.getLoc();

    auto viewOp = dyn_cast<AtenViewOp>(op.self().getDefiningOp());
    if (!viewOp)
      return failure();

    auto sizesListOp =
        dyn_cast<PrimListConstructOp>(viewOp.size().getDefiningOp());
    if (!sizesListOp)
      return failure();
    SmallVector<Value> sizes = sizesListOp.elements();

    auto constantDimSizeOp =
        dyn_cast<ConstantIntOp>(sizes[dim].getDefiningOp());
    if (!constantDimSizeOp)
      return rewriter.notifyMatchFailure(op, "dim size must be constant");

    int64_t dimSize = constantDimSizeOp.value().getSExtValue();
    int64_t splitSize = 0, chunks = 0;
    // TODO: Support non static splitSizes.
    if (isa<AtenSplitTensorOp>(op)) {
      if (!matchPattern(op.getOperand(1), m_TorchConstantInt(&splitSize)))
        return rewriter.notifyMatchFailure(op, "split_size must be constant");
      chunks = (dimSize + splitSize - 1) / splitSize;
    } else if (isa<AtenChunkOp>(op)) {
      if (!matchPattern(op.getOperand(1), m_TorchConstantInt(&chunks)))
        return rewriter.notifyMatchFailure(op,
                                           "number of chunks must be constant");
      splitSize = dimSize / chunks;
    }

    // Calculate the end indices of the slices.
    SmallVector<int64_t> endIdx;
    endIdx.push_back(splitSize * 1);
    for (unsigned i = 1; i < chunks - 1; i++) {
      endIdx.push_back(splitSize * (i + 1));
    }
    endIdx.push_back(dimSize);
    int64_t start = 0;
    SmallVector<Value> listOfTensors;
    Value stepVal = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    // For each of the split create slice operation to slice the input tensor
    // from `start` to `endIdx`.
    for (unsigned i = 0; i < chunks; i++) {
      Value startVal = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(start));
      Value endVal = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(endIdx[i]));
      Type resultType = listType.getContainedType();
      Value slice = rewriter.create<AtenSliceTensorOp>(
          loc, resultType, op.self(), op.dim(), startVal, endVal, stepVal);
      listOfTensors.push_back(slice);
      start = endIdx[i];
    }

    Value result = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(listType.getContainedType()), listOfTensors);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {

template <typename SplitSizesLikeOp>
class DecomposeAtenSplitSizesLikeOp
    : public OpRewritePattern<SplitSizesLikeOp> {
public:
  using OpRewritePattern<SplitSizesLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SplitSizesLikeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenSplitSizesOp>(op, op.getType(), op.self(),
                                                  op.split_sizes(), op.dim());
    return success();
  }
};
} // namespace

namespace {
class DecomposeComplexOpsEarlyPass
    : public DecomposeComplexOpsEarlyBase<DecomposeComplexOpsEarlyPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    patterns.add<DecomposeAtenSplitSizesOp>(context);
    target.addIllegalOp<AtenSplitSizesOp>();
    patterns.add<DecomposeAtenSplitSizesLikeOp<AtenSplitOp>>(context);
    target.addIllegalOp<AtenSplitOp>();
    patterns.add<DecomposeAtenSplitSizesLikeOp<AtenSplitWithSizesOp>>(context);
    target.addIllegalOp<AtenSplitWithSizesOp>();
    patterns.add<DecomposeAtenSplitTensorLikeOp<AtenSplitTensorOp>>(context);
    target.addIllegalOp<AtenSplitTensorOp>();
    patterns.add<DecomposeAtenSplitTensorLikeOp<AtenChunkOp>>(context);
    target.addIllegalOp<AtenChunkOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createDecomposeComplexOpsEarlyPass() {
  return std::make_unique<DecomposeComplexOpsEarlyPass>();
}
