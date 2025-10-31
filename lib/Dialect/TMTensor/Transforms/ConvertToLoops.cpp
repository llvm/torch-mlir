//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/PassDetail.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::torch::TMTensor;

/// Recursive method that lowers one dimension of the `ScalarLoopOpInterface` to
/// scalar loops at a time.
static LogicalResult lowerToLoopsImpl(OpBuilder &builder,
                                      ScalarLoopOpInterface scalarLoopOp,
                                      ArrayRef<Range> loopRanges,
                                      unsigned loopDepth,
                                      SmallVectorImpl<Value> &ivs) {
  Location loc = scalarLoopOp.getLoc();
  if (loopDepth == loopRanges.size()) {
    return scalarLoopOp.generateScalarImplementation(builder, loc, ivs);
  }
  LogicalResult status = success();
  Value offset = getValueOrCreateConstantIndexOp(builder, loc,
                                                 loopRanges[loopDepth].offset);
  Value size =
      getValueOrCreateConstantIndexOp(builder, loc, loopRanges[loopDepth].size);
  Value stride = getValueOrCreateConstantIndexOp(builder, loc,
                                                 loopRanges[loopDepth].stride);
  scf::ForOp::create(
      builder, loc, offset, size, stride, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        ivs.push_back(iv);
        status =
            lowerToLoopsImpl(b, scalarLoopOp, loopRanges, loopDepth + 1, ivs);
        scf::YieldOp::create(b, loc);
      });
  return status;
}

/// Main entry point for lowering `ScalarLoopOpInterface` op to loops.
static LogicalResult lowerToLoops(OpBuilder &builder,
                                  ScalarLoopOpInterface scalarLoopOp) {
  SmallVector<Range> loopBounds = scalarLoopOp.getIterationDomain(builder);
  SmallVector<Value> ivs;
  return lowerToLoopsImpl(builder, scalarLoopOp, loopBounds, 0, ivs);
}

/// Pattern rewriter hook to lower a `ScalarLoopOpInterface` to loops.
namespace {
struct ScalarLoopOpInterfaceLowerToLoopsPattern : public RewritePattern {
  ScalarLoopOpInterfaceLowerToLoopsPattern(MLIRContext *context,
                                           PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto scalarLoopOp = dyn_cast<ScalarLoopOpInterface>(op);
    if (!scalarLoopOp) {
      return failure();
    }
    if (llvm::any_of(scalarLoopOp->getResults(),
                     [&](Value v) { return isa<ShapedType>(v.getType()); })) {
      return rewriter.notifyMatchFailure(
          scalarLoopOp, "lower to loops needs to have tensor semantics");
    }
    if (failed(lowerToLoops(rewriter, scalarLoopOp))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct TMTensorToLoopsPass : public TMTensorToLoopsBase<TMTensorToLoopsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    mlir::arith::ArithDialect, math::MathDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<ScalarLoopOpInterfaceLowerToLoopsPattern>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
torch::TMTensor::createTMTensorToLoopsPass() {
  return std::make_unique<TMTensorToLoopsPass>();
}
