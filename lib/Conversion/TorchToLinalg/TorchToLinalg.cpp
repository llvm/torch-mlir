//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------
// Patterns for individual ops should live in one of the other files, and
// added via the relevant `populate*PatternsAndLegality` functions.
// This file is just for the pass definition itself.

namespace {
class ConvertTorchToLinalg
    : public ConvertTorchToLinalgBase<ConvertTorchToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<cf::ControlFlowDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           tensor::TensorDialect, arith::ArithmeticDialect>();
    target.addLegalOp<TorchConversion::GetNextSeedOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    torch_to_linalg::populateTensorScalarInteropPatternsAndLegality(
        typeConverter, patterns, target);
    torch_to_linalg::populateLinearPatternsAndLegality(typeConverter, patterns,
                                                       target);
    torch_to_linalg::populatePoolingPatternsAndLegality(typeConverter, patterns,
                                                        target);
    torch_to_linalg::populateRandomPatternsAndLegality(typeConverter, patterns,
                                                       target);
    torch_to_linalg::populateUncategorizedPatternsAndLegality(typeConverter,
                                                              patterns, target);
    torch_to_linalg::populateReductionPatternsAndLegality(typeConverter,
                                                          patterns, target);
    torch_to_linalg::populateDataMovementPatternsAndLegality(typeConverter,
                                                             patterns, target);
    torch_to_linalg::populateIndirectDataMovementPatternsAndLegality(
        typeConverter, patterns, target);
    torch_to_linalg::populateTensorConstructorsPatternsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
