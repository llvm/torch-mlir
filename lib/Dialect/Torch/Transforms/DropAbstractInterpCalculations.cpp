//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
template <typename CalculateOp>
class DropCalculateOp : public OpConversionPattern<CalculateOp> {
public:
  using OpConversionPattern<CalculateOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CalculateOp op, typename CalculateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *block = &op.getBody().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, op);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
    return success();
  }
};
} // namespace

namespace {
class DropAbstractInterpCalculationsPass
    : public DropAbstractInterpCalculationsBase<
          DropAbstractInterpCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<DropCalculateOp<DtypeCalculateOp>>(context);
    patterns.insert<DropCalculateOp<ShapeCalculateOp>>(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();
    target.addIllegalOp<DtypeCalculateOp, ShapeCalculateOp>();
    target.addLegalOp<func::FuncOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createDropAbstractInterpCalculationsPass() {
  return std::make_unique<DropAbstractInterpCalculationsPass>();
}
