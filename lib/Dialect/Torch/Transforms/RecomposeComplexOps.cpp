//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <climits>

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

    // TODO Comparing directly to INT64_MAX/INT64_MIN seems fragile.
    // This is a potential general way of implementing the clamping in terms of other torch ops.
    // https://github.com/llvm/torch-mlir/pull/2005
    //
    // def to_valid_dim(dim, max_dim):
    //  dim = torch.ops.prim.min(dim, max_dim)
    //  dim = torch.ops.prim.max(dim, -max_dim)
    //  is_neg = torch.ops.aten.lt(dim, 0)
    //  is_neg_int = torch.ops.aten.Int.bool(is_neg)
    //  return (dim + max_dim) * is_neg_int + dim * (1 - is_neg_int)
    //
    // dim_size = torch.ops.aten.size.int(slice, slice.get_dim())
    // start = to_valid_dim(slice.get_start(), dim_size)
    // end = to_valid_dim(slice.get_end(), dim_size)
    int64_t end;
    if (sliceOp.getEnd().getType().isa<Torch::NoneType>())
      end = INT64_MAX;
    else if (!matchPattern(sliceOp.getEnd(), m_TorchConstantInt(&end)))
      return failure();
    int64_t start;
    if (sliceOp.getStart().getType().isa<Torch::NoneType>())
      start = INT64_MIN;
    else if (!matchPattern(sliceOp.getStart(), m_TorchConstantInt(&start)))
      return failure();

    Value newEnd = sliceOp.getEnd();
    if (end < 0) {
      Value dimSize = rewriter.create<AtenSizeIntOp>(
          op.getLoc(), sliceOp.getSelf(), sliceOp.getDim());
      newEnd =
          rewriter.create<AtenAddIntOp>(op.getLoc(), dimSize, sliceOp.getEnd());
    } else if (end == INT64_MAX) {
      newEnd = rewriter.create<AtenSizeIntOp>(op.getLoc(), sliceOp.getSelf(),
                                              sliceOp.getDim());
    }

    Value newStart = sliceOp.getStart();
    if (start == INT64_MIN) {
      newStart = rewriter.create<ConstantIntOp>(op.getLoc(),
                                                rewriter.getI64IntegerAttr(0));
    } else if (start < 0) {
      Value dimSize = rewriter.create<AtenSizeIntOp>(
          op.getLoc(), sliceOp.getSelf(), sliceOp.getDim());
      newStart =
          rewriter.create<AtenAddIntOp>(op.getLoc(), dimSize, sliceOp.getEnd());
    }

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value falseVal = rewriter.create<ConstantBoolOp>(op.getLoc(), false);

    // Create IndexPut_Op
    BaseTensorType tensorType = op->getResultTypes()[0].cast<BaseTensorType>();
    Value range = rewriter.create<AtenArangeStartStepOp>(
        op.getLoc(), tensorType, newStart, newEnd, sliceOp.getStep(),
        /*dtype=*/noneVal, /*layout=*/noneVal, /*device=*/noneVal,
        /*pin_memory=*/noneVal);

    SmallVector<Value> indicesVector;
    for (auto i = 0; i < dim; i++)
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
    SmallVector<Value> indicesVector(dim, noneVal);
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
