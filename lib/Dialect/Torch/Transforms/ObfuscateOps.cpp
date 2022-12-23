//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ObfuscateMM : public OpRewritePattern<AtenMmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input0 = op.getOperand(0);
    Value input1 = op.getOperand(1);
    BaseTensorType resultTensorType = op.getType().cast<BaseTensorType>();
    Value add = rewriter.create<AtenDivTensorOp>(loc, resultTensorType, input0,
                                            input0);
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTensorType,
                                                        add);
    return success();
  }
};
} // namespace
namespace {
class ObfuscateOpsPass
    : public ObfuscateOpsBase<ObfuscateOpsPass> {
public:
  ObfuscateOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
//    patterns.add<DecomposeAten_SoftmaxOp>(context);
    patterns.add<ObfuscateMM>(context);
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoIterationLimit;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createObfuscateOpsPass() {
  return std::make_unique<ObfuscateOpsPass>();
}
