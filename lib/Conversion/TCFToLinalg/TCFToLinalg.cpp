//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TCFToLinalg/TCFToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

static SmallVector<Value, 6> bypassResultShapes(Operation *op,
                                                OpBuilder &builder) {

  if (auto matmul = dyn_cast<tcf::MatmulOp>(op)) {
    auto lhsRows = builder.create<DimOp>(op->getLoc(), matmul.lhs(), 0);
    auto rhsCols = builder.create<DimOp>(op->getLoc(), matmul.rhs(), 1);
    auto shape = builder.create<TensorFromElementsOp>(
        op->getLoc(), ValueRange({lhsRows, rhsCols}));
    return {shape};
  }
  // TODO: This only supports the NCHW data format. Consider other formats and lower ranks.
  if (auto conv2dNCHWBias = dyn_cast<tcf::ConvNCHWBiasOp>(op)) {
    auto batch = builder.create<DimOp>(op->getLoc(), conv2dNCHWBias.in(), 0);
    auto height = builder.create<DimOp>(op->getLoc(), conv2dNCHWBias.in(), 2);
    auto width = builder.create<DimOp>(op->getLoc(), conv2dNCHWBias.in(), 3);
    auto filter = builder.create<DimOp>(op->getLoc(), conv2dNCHWBias.filter(), 0);
    auto shape = builder.create<TensorFromElementsOp>(
        op->getLoc(), ValueRange({batch, filter, height, width}));
    return {shape};
  }

  // No shape transfer function.
  return {};
}

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
    // Create the init tensor for the matmul.
    // TODO: Expand supported data types.
    Value c0 =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getF32FloatAttr(0.0));
    Value shape = bypassResultShapes(op, rewriter)[0];
    Value initTensor =
        rewriter.create<tcp::SplattedOp>(op.getLoc(), op.getType(), c0, shape);

    // Create the matmul.
    auto matmul = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), TypeRange(op.getType()), op.getOperands(), ValueRange(),
        ValueRange(initTensor));
    rewriter.create<shape::AssumingYieldOp>(op.getLoc(), matmul.getResult(0));

    // Finally, replace with the results of the shape.assuming
    rewriter.replaceOp(op, assuming.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertConvNCHWBias : public OpRewritePattern<tcf::ConvNCHWBiasOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcf::ConvNCHWBiasOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Create the constraints for conv_2d_nchw_bias.
    // Create the constraints, and the assuming region.
    //Value inK = rewriter.create<DimOp>(op.getLoc(), op.in(), 1);
    //Value filterK = rewriter.create<DimOp>(op.getLoc(), op.filter(), 0);
    //Value matchingK =
    //    rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, inK, inK);
    //Value witness = rewriter.create<shape::CstrRequireOp>(
    //    op.getLoc(), matchingK, "mismatching contracting dimension for conv_2d_nchw_bias");
    Value witness = rewriter.create<shape::ConstWitnessOp>(op.getLoc(), true);
    auto assuming = rewriter.create<shape::AssumingOp>(
        op.getLoc(), ArrayRef<Type>{op.getType()}, witness);

    // Build the region body.
    rewriter.createBlock(&assuming.doRegion());
    // Create the init tensor for the ConvNCHWBias.
    // TODO: Expand supported data types.
    Value c0 =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getF32FloatAttr(0.0));
    Value shape = bypassResultShapes(op, rewriter)[0];
    Value initTensor =
        rewriter.create<tcp::SplattedOp>(op.getLoc(), op.getType(), c0, shape);

    // Create the ConvNCHWBias.
    auto conv2dNCHW = rewriter.create<linalg::ConvNCHWOp>(
        op.getLoc(), TypeRange(op.getType()), ValueRange({op.in(), op.filter()}), ValueRange(),
        ValueRange(initTensor));
    auto conv2dNCHWBias = rewriter.create<AddFOp>(
        op.getLoc(), TypeRange(op.getType()), conv2dNCHW.getResult(0), op.bias());
    rewriter.create<shape::AssumingYieldOp>(op.getLoc(), conv2dNCHWBias.getResult());

    // Finally, replace with the results of the shape.assuming
    rewriter.replaceOp(op, assuming.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTCFToLinalg : public ConvertTCFToLinalgBase<ConvertTCFToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect, tcp::TCPDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<ConvertMatmul>(context);
    patterns.insert<ConvertConvNCHWBias>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTCFToLinalgPass() {
  return std::make_unique<ConvertTCFToLinalg>();
}
