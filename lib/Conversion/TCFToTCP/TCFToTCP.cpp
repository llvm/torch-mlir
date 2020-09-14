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
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {

RankedTensorType getExtentTensorType(Builder &builder) {
  return RankedTensorType::get({ShapedType::kDynamicSize},
                               builder.getIndexType());
}

class ConvertAdd : public OpRewritePattern<tcf::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcf::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = op.lhs().getType().dyn_cast<RankedTensorType>();
    auto rhsType = op.rhs().getType().dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, "requires ranked tensors");
    }
    Value lhsShape = rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.lhs());
    Value rhsShape = rewriter.create<shape::ShapeOfOp>(op.getLoc(), op.rhs());

    // Create the constraints, and the assuming region.
    Value witness = rewriter.create<shape::CstrBroadcastableOp>(
        op.getLoc(), lhsShape, rhsShape);
    auto assuming = rewriter.create<shape::AssumingOp>(
        op.getLoc(), ArrayRef<Type>{op.getType()}, witness);

    // Start building the region body.
    rewriter.createBlock(&assuming.doRegion());
    Value broadcastedShape = rewriter.create<shape::BroadcastOp>(
        op.getLoc(), getExtentTensorType(rewriter), lhsShape, rhsShape,
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
        op.getLoc(), resultType, op.lhs(), broadcastedShape);
    Value rhsBroadcasted = rewriter.create<tcp::BroadcastToOp>(
        op.getLoc(), resultType, op.rhs(), broadcastedShape);
    Value add = rewriter.create<tcp::AddOp>(op.getLoc(), op.getType(),
                                            lhsBroadcasted, rhsBroadcasted);
    rewriter.create<shape::AssumingYieldOp>(op.getLoc(), add);

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
    patterns.insert<ConvertAdd>(context);
    (void)applyPatternsAndFoldGreedily(module, patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createConvertTCFToTCPPass() {
  return std::make_unique<ConvertTCFToTCP>();
}
