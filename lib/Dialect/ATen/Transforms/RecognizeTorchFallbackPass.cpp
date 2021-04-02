//===- RecognizeTorchFallback.cpp ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/ATen/Transforms/Passes.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/Torch/IR/OpInterfaces.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aten-recognize-torch-fallback"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::aten;
using namespace mlir::NPCOMP::Basicpy;
using namespace mlir::NPCOMP::Torch;

namespace {

class EncapsulateKernelCallOpPattern : public OpRewritePattern<Torch::KernelCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Torch::KernelCallOp kernelCall,
                                PatternRewriter &rewriter) const override {
    Operation *kcOp = kernelCall.getOperation();
    // Change nothing if this KernelCallOp does not descend directly from a FuncOp.
    if (!isa<FuncOp>(kcOp->getParentOp())) {
      return failure();
    }
    // New Torch fallback region.
    auto torchfb = rewriter.create<refback::TorchFallbackOp>(
        kernelCall.getLoc(), kernelCall.results().getType(), kernelCall.args());
    // Build the region body.
    rewriter.createBlock(&torchfb.doRegion());
    //Operation *encapKcOp = rewriter.clone(*kcOp);
    SmallVector<Value, 6> encapArgs;
    for (auto arg : kernelCall.args()) {
      Operation *op = arg.getDefiningOp();
      if (op) {
        auto encapOp = rewriter.clone(*op);
        //op->replaceAllUsesWith(encapOp);
        encapArgs.push_back(encapOp->getResult(0));
      }
      else {
        encapArgs.push_back(arg);
      }
    }
    auto encapKcOp = rewriter.create<Torch::KernelCallOp>(kernelCall.getLoc(),
        kernelCall.results().getType(),
        kernelCall.kernelName(),
        ValueRange(encapArgs),
        kernelCall.sigArgTypes(),
        kernelCall.sigRetTypes(),
        kernelCall.sigIsVararg(),
        kernelCall.sigIsVarret(),
        kernelCall.sigIsMutable()
        );
    rewriter.create<refback::TorchFallbackYieldOp>(kernelCall.getLoc(), encapKcOp->getResults());
    // Finally, replace with the results of the shape.assuming
    rewriter.replaceOp(kernelCall, torchfb.getResults());
    return success();
  }
};

class ATenRecognizeTorchFallbackPass
    : public ATenRecognizeTorchFallbackBase<ATenRecognizeTorchFallbackPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ATenDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto &context = getContext();
    OwningRewritePatternList patterns(&context);
    patterns.insert<EncapsulateKernelCallOpPattern>(&context);
    if (failed(
          applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::aten::createRecognizeTorchFallbackPass() {
  return std::make_unique<ATenRecognizeTorchFallbackPass>();
}
