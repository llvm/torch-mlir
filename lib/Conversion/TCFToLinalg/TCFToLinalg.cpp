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
  if (auto conv2dNCHW = dyn_cast<tcf::ConvNCHWOp>(op)) {
    // TODO: Replace hard-coded stride/dilation/padding constant-ops.
    // TODO: Consider migrating this SSA shape-computing graph to a complex op or use the `mlir-linalg-ods-gen` approach and define a `*.tc` spec file.
    auto cI0 = builder.create<ConstantOp>(op->getLoc(), builder.getIntegerAttr(builder.getIndexType(), 0));
    auto cI1 = builder.create<ConstantOp>(op->getLoc(), builder.getIntegerAttr(builder.getIndexType(), 1));
    auto cI2 = builder.create<ConstantOp>(op->getLoc(), builder.getIntegerAttr(builder.getIndexType(), 2));
    auto stride = cI1;
    auto dilation = cI1;
    auto padding = cI0;
    auto strideHeight = stride;
    auto strideWidth = stride;
    auto dilationHeight = dilation;
    auto dilationWidth = dilation;
    auto paddingHeight = padding;
    auto paddingWidth = padding;
    auto batch = builder.create<DimOp>(op->getLoc(), conv2dNCHW.in(), 0);
    auto height = builder.create<DimOp>(op->getLoc(), conv2dNCHW.in(), 2);
    auto width = builder.create<DimOp>(op->getLoc(), conv2dNCHW.in(), 3);
    auto filterOutChannels = builder.create<DimOp>(op->getLoc(), conv2dNCHW.filter(), 0);
    auto filterHeight = builder.create<DimOp>(op->getLoc(), conv2dNCHW.filter(), 2);
    auto filterWidth = builder.create<DimOp>(op->getLoc(), conv2dNCHW.filter(), 3);
    // Output height
    auto twicePaddingHeight = builder.create<MulIOp>(op->getLoc(), paddingHeight, cI2);
    auto heightPlusTwicePadding = builder.create<SubIOp>(op->getLoc(), height, twicePaddingHeight);
    auto filterHeightMinusOne = builder.create<SubIOp>(op->getLoc(), filterHeight, cI1);
    auto dilationFilterHeight = builder.create<MulIOp>(op->getLoc(), dilationHeight, filterHeightMinusOne);
    auto outHeightUnstridedPlusOne = builder.create<SubIOp>(op->getLoc(), heightPlusTwicePadding, dilationFilterHeight);
    auto outHeightUnstrided = builder.create<SubIOp>(op->getLoc(), outHeightUnstridedPlusOne, cI1);
    auto outHeightMinusOne = builder.create<UnsignedDivIOp>(op->getLoc(), outHeightUnstrided, strideHeight);
    auto outHeight = builder.create<AddIOp>(op->getLoc(), outHeightMinusOne, cI1);
    // Output width
    auto twicePaddingWidth = builder.create<MulIOp>(op->getLoc(), paddingWidth, cI2);
    auto widthPlusTwicePadding = builder.create<SubIOp>(op->getLoc(), width, twicePaddingWidth);
    auto filterWidthMinusOne = builder.create<SubIOp>(op->getLoc(), filterWidth, cI1);
    auto dilationFilterWidth = builder.create<MulIOp>(op->getLoc(), dilationWidth, filterWidthMinusOne);
    auto outWidthUnstridedPlusOne = builder.create<SubIOp>(op->getLoc(), widthPlusTwicePadding, dilationFilterWidth);
    auto outWidthUnstrided = builder.create<SubIOp>(op->getLoc(), outWidthUnstridedPlusOne, cI1);
    auto outWidthMinusOne = builder.create<UnsignedDivIOp>(op->getLoc(), outWidthUnstrided, strideWidth);
    auto outWidth = builder.create<AddIOp>(op->getLoc(), outWidthMinusOne, cI1);
    // Output shape
    auto shape = builder.create<TensorFromElementsOp>(
        op->getLoc(), ValueRange({batch, filterOutChannels, outHeight, outWidth}));
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
class ConvertConvNCHW : public OpRewritePattern<tcf::ConvNCHWOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tcf::ConvNCHWOp op,
                                PatternRewriter &rewriter) const override {
    // Create the constraints, and the assuming region.
    Value inputC   = rewriter.create<DimOp>(op.getLoc(), op.in(), 1);
    Value inputH   = rewriter.create<DimOp>(op.getLoc(), op.in(), 2);
    Value inputW   = rewriter.create<DimOp>(op.getLoc(), op.in(), 3);
    Value filterC  = rewriter.create<DimOp>(op.getLoc(), op.filter(), 1);
    Value filterKH = rewriter.create<DimOp>(op.getLoc(), op.filter(), 2);
    Value filterKW = rewriter.create<DimOp>(op.getLoc(), op.filter(), 3);
    Value matchingC =
        rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::eq, inputC, filterC);
    Value validFilterH =
        rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::uge, inputH, filterKH);
    Value validFilterW =
        rewriter.create<CmpIOp>(op.getLoc(), CmpIPredicate::uge, inputW, filterKW);
    Value witnessC = rewriter.create<shape::CstrRequireOp>(
        op.getLoc(), matchingC, "input and filter channels must be equal");
    Value witnessFilterH = rewriter.create<shape::CstrRequireOp>(
        op.getLoc(), validFilterH, "input height must be greater than or equal to filter KH-dimension");
    Value witnessFilterW = rewriter.create<shape::CstrRequireOp>(
        op.getLoc(), validFilterW, "input width must be greater than or equal to filter KW-dimension");
    Value assumingAll = rewriter.create<shape::AssumingAllOp>(
        op.getLoc(), witnessC.getType(), ValueRange({witnessC, witnessFilterH, witnessFilterW}));
    auto assuming = rewriter.create<shape::AssumingOp>(
        op.getLoc(), ArrayRef<Type>{op.getType()}, assumingAll);

    // Build the region body.
    rewriter.createBlock(&assuming.doRegion());
    // Create the init tensor for the ConvNCHW.
    // TODO: Expand supported data types.
    Value c0 =
        rewriter.create<ConstantOp>(op.getLoc(), rewriter.getF32FloatAttr(0.0));
    Value shape = bypassResultShapes(op, rewriter)[0];
    Value initTensor =
        rewriter.create<tcp::SplattedOp>(op.getLoc(), op.getType(), c0, shape);

    // Create the ConvNCHW.
    auto conv2dNCHW = rewriter.create<linalg::ConvNCHWOp>(
        op.getLoc(), TypeRange(op.getType()), ValueRange({op.in(), op.filter()}), ValueRange(),
        ValueRange(initTensor));
    rewriter.create<shape::AssumingYieldOp>(op.getLoc(), conv2dNCHW.getResults());

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
    patterns.insert<ConvertConvNCHW>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTCFToLinalgPass() {
  return std::make_unique<ConvertTCFToLinalg>();
}
