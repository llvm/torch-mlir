//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class RecomposeSliceCopy_ : public OpRewritePattern<AtenCopy_Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCopy_Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getSelf().getDefiningOp() ||
        !isa<AtenSliceTensorOp>(op.getSelf().getDefiningOp()))
      return failure();
    auto sliceOp = cast<AtenSliceTensorOp>(op.getSelf().getDefiningOp());

    // Get indices
    int64_t dim;
    if (!matchPattern(sliceOp.getDim(), m_TorchConstantInt(&dim)))
      return failure();
    int64_t end;
    if (!matchPattern(sliceOp.getEnd(), m_TorchConstantInt(&end)))
      return failure();

    Value newEnd = sliceOp.getEnd();
    if (end < 0) {
      Value dimSize = rewriter.create<AtenSizeIntOp>(
          op.getLoc(), sliceOp.getSelf(), sliceOp.getDim());
      newEnd =
          rewriter.create<AtenAddIntOp>(op.getLoc(), dimSize, sliceOp.getEnd());
    } else if(end == std::numeric_limits<int64_t>::max()) {
      newEnd = rewriter.create<AtenSizeIntOp>(
          op.getLoc(), sliceOp.getSelf(), sliceOp.getDim());
    }

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value falseVal = rewriter.create<ConstantBoolOp>(op.getLoc(), false);

    // Create IndexPut_Op
    BaseTensorType tensorType = op->getResultTypes()[0].cast<BaseTensorType>();
    Value range = rewriter.create<AtenArangeStartStepOp>(
        op.getLoc(), tensorType, sliceOp.getStart(), newEnd, sliceOp.getStep(),
        /*dtype=*/noneVal, /*layout=*/noneVal, /*device=*/noneVal,
        /*pin_memory=*/noneVal);

    SmallVector<Value> indicesVector;
    for (auto i = 0; i < dim - 1; i++)
      indicesVector.push_back(noneVal);
    indicesVector.push_back(range);
    Value indices = rewriter.create<PrimListConstructOp>(
        op.getLoc(),
        Torch::ListType::get(op->getContext(),
                             Torch::OptionalType::get(tensorType)),
        indicesVector);

    rewriter.replaceOpWithNewOp<Aten_IndexPutImpl_Op>(
        op, op->getResultTypes(), sliceOp.getSelf(), indices, op.getSrc(),
        /*accumulate=*/falseVal, /*unsafe=*/falseVal);

    return success();
  }
};

class RecomposeSelectFill_ : public OpRewritePattern<AtenFill_TensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFill_TensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getSelf().getDefiningOp() ||
        !isa<AtenSelectIntOp>(op.getSelf().getDefiningOp()))
      return failure();
    auto selectOp = cast<AtenSelectIntOp>(op.getSelf().getDefiningOp());

    // Get indices
    int64_t dim;
    if (!matchPattern(selectOp.getDim(), m_TorchConstantInt(&dim)))
      return failure();

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value falseVal = rewriter.create<ConstantBoolOp>(op.getLoc(), false);

    // Create IndexPut_Op
    // Convert indexNum to indexTensor for the selectOp
    BaseTensorType selectOutTy =
        selectOp.getType().template cast<BaseTensorType>();
    SmallVector<int64_t> empty;
    auto dtype = getTypeForTorchType(selectOp.getContext(),
                                     selectOp.getIndex().getType());
    Type emptyTensorType =
        selectOutTy.getWithSizesAndDtype(llvm::ArrayRef(empty), dtype);
    Value indexTensor = rewriter.create<PrimNumToTensorScalarOp>(
        selectOp.getLoc(), emptyTensorType, selectOp.getIndex());

    // Create indicesVector for IndexPut_Op by TorchNone and indexTensor
    BaseTensorType tensorType = op->getResultTypes()[0].cast<BaseTensorType>();
    SmallVector<Value> indicesVector(dim - 1, noneVal);
    indicesVector.push_back(indexTensor);

    Value indices = rewriter.create<PrimListConstructOp>(
        op.getLoc(),
        Torch::ListType::get(op->getContext(),
                             Torch::OptionalType::get(tensorType)),
        indicesVector);

    rewriter.replaceOpWithNewOp<Aten_IndexPutImpl_Op>(
        op, op->getResultTypes(), selectOp.getSelf(), indices, op.getValue(),
        /*accumulate=*/falseVal, /*unsafe=*/falseVal);

    return success();
  }
};

class RecomposeUnbindListUnpack : public OpRewritePattern<PrimListUnpackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListUnpackOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenUnbindOp + PrimListUnpackOp to select.int
    auto unbind = dyn_cast<AtenUnbindIntOp>(op.getOperand().getDefiningOp());
    if (!unbind)
      return failure();
    if (isListPotentiallyMutated(unbind.getResult()))
      return failure();
    Value dim = unbind.getDim();
    Value input = unbind.getSelf();
    SmallVector<Value> slices;
    for (int i = 0; i < op.getNumResults(); i++) {
      // rewrite to slice op
      auto resultTy = op.getResult(i).getType();
      auto index = rewriter.create<Torch::ConstantIntOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(i));
      auto newSelect = rewriter.create<AtenSelectIntOp>(op->getLoc(), resultTy,
                                                        input, dim, index);
      slices.push_back(newSelect);
    }
    rewriter.replaceOp(op, slices);
    if (unbind.getResult().use_empty())
      rewriter.eraseOp(unbind);
    return success();
  }
};

class RecomposeUnbindGetItem : public OpRewritePattern<Aten__Getitem__TOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten__Getitem__TOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenUnbindIntOp + __getitem__t to select.int
    auto unbind = dyn_cast<AtenUnbindIntOp>(op.getList().getDefiningOp());
    if (!unbind)
      return failure();
    if (isListPotentiallyMutated(unbind.getResult()))
      return failure();
    int64_t index;
    if (!matchPattern(op.getIdx(), m_TorchConstantInt(&index)))
      return rewriter.notifyMatchFailure(
          op, "Expected `idx` of `Aten__Getitem__TOp` to be a constant int");

    Location loc = op.getLoc();
    Value dim = unbind.getDim();
    Value input = unbind.getSelf();
    // rewrite to slice op
    auto resultTy = op.getResult().getType();
    Value newSelect = rewriter.create<AtenSelectIntOp>(loc, resultTy, input,
                                                       dim, op.getIdx());
    rewriter.replaceOp(op, newSelect);
    if (unbind.getResult().use_empty())
      rewriter.eraseOp(unbind);
    return success();
  }
};
} // namespace

namespace {
class RecomposeComplexOpsPass
    : public RecomposeComplexOpsBase<RecomposeComplexOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // pattern.add calls go here
    patterns.add<RecomposeSliceCopy_>(context);
    patterns.add<RecomposeSelectFill_>(context);
    patterns.add<RecomposeUnbindListUnpack>(context);
    patterns.add<RecomposeUnbindGetItem>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRecomposeComplexOpsPass() {
  return std::make_unique<RecomposeComplexOpsPass>();
}
