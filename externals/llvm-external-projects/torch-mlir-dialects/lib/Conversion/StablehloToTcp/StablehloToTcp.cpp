//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/StablehloToTcp/StablehloToTcp.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOLINALG
#include "torch-mlir-dialects/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class TanhOpConverter : public OpRewritePattern<stablehlo::TanhOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TanhOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<tcp::TanhOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

void populateStablehloToTcpConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<TanhOpConverter>(patterns->getContext());
}

class ConvertStablehloToTcp
    : public ConvertStablehloToTcpBase<ConvertStablehloToTcp> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addIllegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<tcp::TcpDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    populateStablehloToTcpConversionPatterns(&patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertStablehloToTcpPass() {
  return std::make_unique<ConvertStablehloToTcp>();
}

}  // namespace tcp
}  // namespace mlir
