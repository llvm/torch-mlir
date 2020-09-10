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
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtDialect.h"
#include "npcomp/Dialect/Npcomprt/IR/NpcomprtOps.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class LowerConstShapeOp : public OpConversionPattern<shape::ConstShapeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ConstShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto extents = llvm::to_vector<6>(llvm::map_range(
        op.shape().getValues<int64_t>(), [&](int64_t extent) -> Value {
          return rewriter.create<ConstantIndexOp>(op.getLoc(), extent);
        }));
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(
        op, rewriter.getType<shape::ShapeType>(), extents);
    return success();
  }
};
} // namespace

namespace {

// Given an operand that is either a Shape or Extent Tensor, returns an
// Extent Tensor or nullptr if this cannot be locally determined.
// The return value, if !nullptr, will be a 1D RankedTensorType (with possibly
// unknown element).
Value findExtentsFromShape(Value operand, bool requireKnownRank) {
  if (auto tensorType = operand.getType().dyn_cast<RankedTensorType>()) {
    if (tensorType.getRank() == 1 &&
        (!requireKnownRank || tensorType.hasStaticShape())) {
      return operand;
    }
  }
  return nullptr;
}

class LowerShapeBroadcastOp : public OpConversionPattern<shape::BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::BroadcastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::BroadcastOp::Adaptor adaptor(operands);
    // When the ranks are statically known, generate non-branchy code.
    // TODO: Generate rank-generic code.
    auto lhsExtents = findExtentsFromShape(adaptor.lhs(), true);
    auto rhsExtents = findExtentsFromShape(adaptor.rhs(), true);
    if (!lhsExtents || !rhsExtents)
      return rewriter.notifyMatchFailure(op, "dynamic extents not supported");

    // Establish invariant that rank(lhs) >= rank(rhs).
    auto lhsSize = lhsExtents.getType().cast<RankedTensorType>().getDimSize(0);
    auto rhsSize = rhsExtents.getType().cast<RankedTensorType>().getDimSize(0);
    if (lhsSize < rhsSize) {
      std::swap(lhsExtents, rhsExtents);
      std::swap(lhsSize, rhsSize);
    }
    auto rankDiscrepancy = lhsSize - rhsSize;

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
      // TODO: Should there be a more generic error-handling dialect?
      // It seems a bit awkward to hardcode npcomprt here.
      rewriter.create<npcomprt::AbortIfOp>(op.getLoc(), bothTrue);
    };

    SmallVector<Value, 6> resultExtents;
    for (int i = 0, e = lhsSize; i < e; i++) {
      auto lhsDim = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
      auto lhsExtent = rewriter.create<ExtractElementOp>(
          op.getLoc(), lhsExtents, ValueRange{lhsDim});
      if (i < rankDiscrepancy) {
        // Padded extent.
        resultExtents.push_back(lhsExtent);
        continue;
      }

      // Non-padded extent.
      auto rhsDim =
          rewriter.create<ConstantIndexOp>(op.getLoc(), i - rankDiscrepancy);
      auto rhsExtent = rewriter.create<ExtractElementOp>(
          op.getLoc(), rhsExtents, ValueRange{rhsDim});
      auto ugt = rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::ugt,
                                         lhsExtent, rhsExtent);
      auto resultExtent =
          rewriter.create<SelectOp>(op.getLoc(), ugt, lhsExtent, rhsExtent);
      createAbortIfIllegalBroadcastExtent(lhsExtent, resultExtent);
      createAbortIfIllegalBroadcastExtent(rhsExtent, resultExtent);
      resultExtents.push_back(resultExtent);
    }

    // TODO: Remove the return type once ODS is fixed to do proper inference.
    rewriter.replaceOpWithNewOp<shape::FromExtentsOp>(
        op, shape::ShapeType::get(rewriter.getContext()), resultExtents);
    return success();
  }
};
} // namespace

namespace {
class LowerShapeToExtentTensorOp
    : public OpConversionPattern<shape::ToExtentTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::ToExtentTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::ToExtentTensorOpAdaptor adaptor(operands);
    if (adaptor.input().getType().isa<shape::ShapeType>()) {
      // Convert by matching to a producing FromExtentsOp.
      auto fromExtents = adaptor.input().getDefiningOp<shape::FromExtentsOp>();
      if (!fromExtents) {
        return rewriter.notifyMatchFailure(op, "not a from_extents op");
      }
      rewriter.replaceOpWithNewOp<TensorFromElementsOp>(op,
                                                        fromExtents.extents());
      return success();
    }

    // Assume that it is already an extent tensor.
    // TODO: Since these ops are all multi-type, there should be a utility
    // for switching on the allowable types instead of just assuming that it
    // is an extent tensor.
    rewriter.replaceOp(op, adaptor.input());
    return success();
  }
};

class LowerShapeGetExtentOp : public OpConversionPattern<shape::GetExtentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(shape::GetExtentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    shape::GetExtentOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<ExtractElementOp>(op, adaptor.shape(),
                                                  adaptor.dim());
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
// Every `shape.from_extents` op operating on an extent tensor
// (`tensor<?xindex>`) is replaced by corresponding standard ops and folded
// away (for the ranked case, it should be possible to eliminate these).
//
// We expect that previous passes have inserted a "root" set of
// shape::FromExtentsOp's that allow this process to get started.
//
// This is similar to the approach that is used in IREE. It is basically a
// combination of the ConvertShapeToShapex pass and the
// "ranked_dim(make_ranked_shape(x1, x2), N) -> xN" folding pattern.
// These patterns have to be "conversion patterns" since the `operands` argument
// gives access to the post-conversion operands from earlier ops.
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
//
// TODO: Add in a fold of
// `extract_element(tensor_from_elements(x0, x1, ...), n) -> xn` to restore
// the above invariant without relying on a subsequent canonicalization
// step.
namespace {
class LowerRankedShapes : public LowerRankedShapesBase<LowerRankedShapes> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<npcomprt::NpcomprtDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<LowerConstShapeOp>(context);
    patterns.insert<LowerShapeBroadcastOp>(context);
    patterns.insert<LowerShapeGetExtentOp>(context);
    patterns.insert<LowerShapeToExtentTensorOp>(context);
    patterns.insert<EraseShapeObserveErrorOp>(context);
    ConversionTarget target(*context);
    target.addIllegalOp<shape::ShapeOfOp>();
    target.addIllegalOp<shape::BroadcastOp>();
    target.addIllegalOp<shape::GetExtentOp>();
    target.addLegalOp<shape::FromExtentsOp>();
    target.addIllegalOp<shape::ToExtentTensorOp>();
    target.addLegalOp<npcomprt::AbortIfOp>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<tcp::ShapeObserveErrorOp>();
    if (failed(applyPartialConversion(func, target, patterns))) {
      return signalPassFailure();
    }

    // Erase some stray shape ops from the program. They can't be
    // deleted during conversion because they become unused only after
    // subsequent patterns bypass them.
    auto walkResult = func.walk([](Operation *op) {
      if (!isa<shape::FromExtentsOp>(op))
        return WalkResult::advance();
      if (op->use_empty()) {
        op->erase();
      } else {
        op->emitError("could not be eliminated");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerRankedShapesPass() {
  return std::make_unique<LowerRankedShapes>();
}
