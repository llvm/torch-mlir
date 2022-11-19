//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

class ConvertTcpToLinalg : public ConvertTcpToLinalgBase<ConvertTcpToLinalg> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, math::MathDialect,
                           tensor::TensorDialect, arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(context);

    TcpToLinalg::populateElementwisePatternsAndLegality(typeConverter, patterns,
                                                        target);
    TcpToLinalg::populateMiscPatternsAndLegality(typeConverter, patterns,
                                                 target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertTcpToLinalgPass() {
  return std::make_unique<ConvertTcpToLinalg>();
}

} // namespace tcp
} // namespace mlir
