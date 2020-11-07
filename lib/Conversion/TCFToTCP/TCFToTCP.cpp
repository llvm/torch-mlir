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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

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
    (void)applyPatternsAndFoldGreedily(module, getPatterns());
  }

  FrozenRewritePatternList getPatterns() {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<ConvertMatmul>(context);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::createConvertTCFToTCPPass() {
  return std::make_unique<ConvertTCFToTCP>();
}
