//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToStd/TorchToStd.h"

#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"


using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.

// Note: Confusingly, ATen's "dim" means "number of dimensions" which is what
// MLIR calls "rank".
LogicalResult convertDimOp(AtenDimOp op, PatternRewriter &rewriter) {
  if (!op.getOperand().getType().isa<TensorType>())
    return rewriter.notifyMatchFailure(op, "must be tensor only");
  auto rank = rewriter.create<RankOp>(op->getLoc(), op.getOperand());
  rewriter.replaceOpWithNewOp<IndexCastOp>(op, op.getType(), rank);
  return success();
}

LogicalResult convertNeIntOp(AtenNeIntOp op, PatternRewriter &rewriter) {
  auto i1 = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::ne,
                                    op->getOperand(0), op->getOperand(1));
  rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(op, op.getType(), i1);
  return success();
}

LogicalResult convertGtIntOp(AtenGtIntOp op, PatternRewriter &rewriter) {
  auto i1 = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::sgt,
                                    op->getOperand(0), op->getOperand(1));
  rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(op, op.getType(), i1);
  return success();
}

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToStd : public ConvertTorchToStdBase<ConvertTorchToStd> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, Basicpy::BasicpyDialect>();
  }

  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation(), getPatterns());
  }

  FrozenRewritePatternSet getPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add(convertDimOp);
    patterns.add(convertNeIntOp);
    patterns.add(convertGtIntOp);
    return std::move(patterns);
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToStdPass() {
  return std::make_unique<ConvertTorchToStd>();
}
