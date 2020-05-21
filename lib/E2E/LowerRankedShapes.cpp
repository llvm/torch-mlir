//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/E2E/E2E.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

// This has to be a "conversion pattern" since the `operands` argument
// gives access to the post-conversion operands from earlier ops.
namespace {
class LowerShapeBroadcastOp : public OpConversionPattern<shape::BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::BroadcastOp::OperandAdaptor adaptor(operands);
    auto lhs = adaptor.lhs().getDefiningOp<shape::FromExtentsOp>();
    auto rhs = adaptor.rhs().getDefiningOp<shape::FromExtentsOp>();
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(op, "operands not converted");
    // Establish invariant that rank(lhs) >= rank(rhs).
    if (lhs.extents().size() < rhs.extents().size())
      std::swap(lhs, rhs);
    auto rankDiscrepancy = lhs.extents().size() - rhs.extents().size();

    // Helper that creates IR
    // ```
    // abort_if(extent != resultExtent && extent != 1)
    // ```
    // This is the numpy broadcasting legality check.
    auto createAbortIfIllegalBroadcastExtent = [&](Value extent,
                                                   Value resultExtent) {
      auto c1 = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
      auto extentNeMax = rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::ne,
                                                 extent, resultExtent);
      auto extentNeOne =
          rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::ne, extent, c1);
      auto bothTrue =
          rewriter.create<AndOp>(op.getLoc(), extentNeMax, extentNeOne);
      rewriter.create<tcp::AbortIfOp>(op.getLoc(), bothTrue);
    };

    auto resultExtents = llvm::to_vector<6>(lhs.extents());
    for (int i = 0, e = rhs.extents().size(); i < e; i++) {
      auto lhsExtent = lhs.extents()[rankDiscrepancy + i];
      auto rhsExtent = rhs.extents()[i];
      auto ugt = rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::ugt,
                                         lhsExtent, rhsExtent);
      auto max =
          rewriter.create<SelectOp>(op.getLoc(), ugt, lhsExtent, rhsExtent);
      auto &resultExtent = resultExtents[rankDiscrepancy + i];
      resultExtent = max;
      createAbortIfIllegalBroadcastExtent(lhsExtent, resultExtent);
      createAbortIfIllegalBroadcastExtent(rhsExtent, resultExtent);
    }
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(op, resultExtents);
    return success();
  }
};
} // namespace

// Rewrite `get_extent(from_extents(x1,x2,x3), N) -> xN`
//
// TODO: this should be a fold on tcp::GetExtentOp.
// (though then the contract of this pass depends on that set of folds,
// which isn't great)
//
// Also, we use OpConversionPattern to get post-rewrite operands as above.
namespace {
class LowerShapeGetExtentOp : public OpConversionPattern<tcp::GetExtentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::GetExtentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    tcp::GetExtentOp::OperandAdaptor adaptor(operands);
    auto fromExtents = adaptor.shape().getDefiningOp<shape::FromExtentsOp>();
    if (!fromExtents)
      return rewriter.notifyMatchFailure(op, "not a from_extents op");
    int64_t dim = op.dim().getLimitedValue();
    rewriter.replaceOp(op, ValueRange(fromExtents.extents())[dim]);
    return success();
  }
};
} // namespace

namespace {
// Now that we have lowered ranked shapes, which reifies the eager
// error-handling code, the tcp::ShapeObserveErrorOp's are no longer
// needed.
class EraseShapeObserveErrorOp
    : public OpConversionPattern<tcp::ShapeObserveErrorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tcp::ShapeObserveErrorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// Basic invariant of this pass:
// Every def of a !shape.shape type is replaced with a
// `shape.from_extents` op.
// When converting an op, look for the `shape.from_extents` op that
// defined all operands, then do a computation on the extents (i.e.
// operands to the `shape.from_extents` op) and produce a
// `shape.from_extents` op.
//
// We expect that previous passes have inserted a "root" set of
// shape::FromExtentsOp's that allow this process to get started.
//
// We then use this to resolve get_extent ops by using a rewrite
// `get_extent(from_extents(x1,x2,x3), N) -> xN`, which should apply in
// maximally many places due to the above invariant.
//
// This is similar to the approach that is used in IREE. It is basically a
// combination of the ConvertShapeToShapex pass and the
// "ranked_dim(make_ranked_shape(x1, x2), N) -> xN" folding pattern.
//
// This pass depends heavily on ranked shapes, since only ranked shapes can
// be statically expanded to a fixed set of SSA extents.
//
// TODO: This approach doesn't naively work with control flow.
// In the presence of non-cyclic control flow, we can just generalize the
// `getDefiningOp<shape::FromExtentsOp>()` calls into something that will
// look through block arguments and rewrite "phi of shapes -> phi of extents".
// In the presence of cyclic control flow, we need to somehow resolve the
// ranks of use-def cycles ahead of time or optimistically assume that
// backedges will match the rank of forward edges, and somehow be robust
// when that assumption fails.
namespace {
class LowerRankedShapes : public LowerRankedShapesBase<LowerRankedShapes> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<LowerShapeBroadcastOp>(context);
    patterns.insert<LowerShapeGetExtentOp>(context);
    patterns.insert<EraseShapeObserveErrorOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<shape::ShapeOfOp>();
    target.addIllegalOp<shape::BroadcastOp>();
    target.addIllegalOp<tcp::GetExtentOp>();
    target.addLegalOp<shape::FromExtentsOp>();
    target.addLegalOp<tcp::AbortIfOp>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<tcp::ShapeObserveErrorOp>();
    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerRankedShapesPass() {
  return std::make_unique<LowerRankedShapes>();
}
