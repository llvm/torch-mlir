//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

static RankedTensorType getExtentTensorType(Builder &builder) {
  return RankedTensorType::get({ShapedType::kDynamicSize},
                               builder.getIndexType());
}

// Non-templated version of the body of ConvertBinaryElementwise to keep things
// simple.
static LogicalResult
matchAndRewriteBinaryElementwise(Operation *op, PatternRewriter &rewriter) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Location loc = op->getLoc();
  Value result = op->getResult(0);

  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType)
    return rewriter.notifyMatchFailure(op, "requires ranked tensors");

  Value lhsShape = rewriter.create<shape::ShapeOfOp>(loc, lhs);
  Value rhsShape = rewriter.create<shape::ShapeOfOp>(loc, rhs);

  // Create the constraints, and the assuming region.
  Value witness =
      rewriter.create<shape::CstrBroadcastableOp>(loc, lhsShape, rhsShape);
  auto assuming = rewriter.create<shape::AssumingOp>(
      loc, ArrayRef<Type>{result.getType()}, witness);

  // Start building the region body.
  rewriter.createBlock(&assuming.doRegion());
  Value broadcastedShape = rewriter.create<shape::BroadcastOp>(
      loc, getExtentTensorType(rewriter), lhsShape, rhsShape,
      /*error=*/nullptr);

  // TODO: It's annoying to do the dynamic broadcast above then
  // do the static transfer function here. Would be nice if they could
  // somehow be unified.
  SmallVector<int64_t, 6> broadcastedStaticShape;
  OpTrait::util::getBroadcastedShape(lhsType.getShape(), rhsType.getShape(),
                                     broadcastedStaticShape);
  auto resultType =
      RankedTensorType::get(broadcastedStaticShape, lhsType.getElementType());
  Value lhsBroadcasted = rewriter.create<tcp::BroadcastToOp>(
      loc, resultType, lhs, broadcastedShape);
  Value rhsBroadcasted = rewriter.create<tcp::BroadcastToOp>(
      loc, resultType, rhs, broadcastedShape);
  Value binaryOpResult;
  if (isa<tcf::AddOp>(op)) {
    binaryOpResult = rewriter.create<tcp::AddOp>(
        loc, result.getType(), lhsBroadcasted, rhsBroadcasted);
  } else if (isa<tcf::MaxOp>(op)) {
    binaryOpResult = rewriter.create<tcp::MaxOp>(
        loc, result.getType(), lhsBroadcasted, rhsBroadcasted);
  } else {
    op->dump();
    llvm::report_fatal_error(
        "unhandled op (see dump above): TCF->TCP binary elementwise");
  }
  rewriter.create<shape::AssumingYieldOp>(loc, binaryOpResult);

  // Finally, replace with the results of the shape.assuming
  rewriter.replaceOp(op, assuming.getResults());
  return success();
}

namespace {
template <typename SourceOp>
class ConvertBinaryElementwise : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteBinaryElementwise(op, rewriter);
  }
};
} // namespace

static LogicalResult
matchAndRewriteUnaryElementwise(Operation *op, PatternRewriter &rewriter) {
  if (isa<tcf::ExpOp>(op)) {
    rewriter.replaceOpWithNewOp<tcp::ExpOp>(op, op->getOperand(0));
  } else if (isa<tcf::TanhOp>(op)) {
    rewriter.replaceOpWithNewOp<tcp::TanhOp>(op, op->getOperand(0));
  } else {
    op->dump();
    llvm::report_fatal_error(
        "unhandled op (see dump above): TCF->TCP unary elementwise");
  }
  return success();

}

namespace {
template <typename SourceOp>
class ConvertUnaryElementwise : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    return matchAndRewriteUnaryElementwise(op, rewriter);
  }
};
} // namespace

namespace {
class ConvertMatmul : public OpRewritePattern<tcf::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcf::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Create the constraints, and the assuming region.
    Value lhsK = rewriter.create<DimOp>(op.getLoc(), op.lhs(), 1);
    Value rhsK = rewriter.create<DimOp>(op.getLoc(), op.rhs(), 0);
    Value matchingK =
        rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, lhsK, rhsK);
    Value witness = rewriter.create<shape::CstrRequireOp>(
        op.getLoc(), matchingK, "mismatching contracting dimension for matmul");
    auto assuming = rewriter.create<shape::AssumingOp>(
        op.getLoc(), ArrayRef<Type>{op.getType()}, witness);

    // Build the region body.
    rewriter.createBlock(&assuming.doRegion());
    Value matmul = rewriter.create<tcp::MatmulOp>(op.getLoc(), op.getType(),
                                                  op.lhs(), op.rhs());
    rewriter.create<shape::AssumingYieldOp>(op.getLoc(), matmul);

    // Finally, replace with the results of the shape.assuming
    rewriter.replaceOp(op, assuming.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTCFToTCP : public ConvertTCFToTCPBase<ConvertTCFToTCP> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, tcp::TCPDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ConvertUnaryElementwise<tcf::ExpOp>,
                    ConvertUnaryElementwise<tcf::TanhOp>>(context);
    patterns.insert<ConvertBinaryElementwise<tcf::AddOp>,
                    ConvertBinaryElementwise<tcf::MaxOp>>(context);
    patterns.insert<ConvertMatmul>(context);
    (void)applyPatternsAndFoldGreedily(module, patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createConvertTCFToTCPPass() {
  return std::make_unique<ConvertTCFToTCP>();
}
