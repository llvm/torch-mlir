//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToArith/TcpToArith.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOLINALG
#include "torch-mlir-dialects/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class ConstOpConverter : public OpRewritePattern<ConstOp> {
public:
  using OpRewritePattern<ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

void populateTcpToArithConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<ConstOpConverter>(patterns->getContext());
}

class ConvertTcpToArith : public ConvertTcpToArithBase<ConvertTcpToArith> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);
    populateTcpToArithConversionPatterns(&patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTcpToArithPass() {
  return std::make_unique<ConvertTcpToArith>();
}

} // namespace tcp
} // namespace mlir
