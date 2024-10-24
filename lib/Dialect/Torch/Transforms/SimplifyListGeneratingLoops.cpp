//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "SimplificationUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class SimplifyListGeneratingLoopsPass
    : public SimplifyListGeneratingLoopsBase<SimplifyListGeneratingLoopsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    populateFullyUnrollPrimLoopOpPattern(patterns, context);
    populateFoldListAppendChainWithinABlockPattern(patterns, context);
    /// populateFoldPrimUncheckedCastOpPattern(patterns, context);

    // PrimIfOp::getCanonicalizationPatterns(patterns, context);
    // Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    //  PrimTupleUnpackOp::getCanonicalizationPatterns(patterns, context);

    // TODO: Debug visitation order to make this more efficient.
    // A single linear scan should suffice.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 2;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createSimplifyListGeneratingLoopsPass() {
  return std::make_unique<SimplifyListGeneratingLoopsPass>();
}
