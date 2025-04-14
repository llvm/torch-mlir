//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertTorchToStablehlo
    : public ConvertTorchToStablehloBase<ConvertTorchToStablehlo> {
public:
  ConvertTorchToStablehlo() = default;
  ConvertTorchToStablehlo(bool enableStaticShape, bool enableI32Index) {
    this->enableStaticShape = enableStaticShape;
    this->enableI32Index = enableI32Index;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::ChloDialect>();
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<quant::QuantDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<chlo::ChloDialect, stablehlo::StablehloDialect,
                           tensor::TensorDialect, arith::ArithDialect,
                           shape::ShapeDialect, quant::QuantDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                            typeConverter);

    RewritePatternSet patterns(context);

    torch_to_stablehlo::TorchToStablehloOptions options{
        enableStaticShape, enableI32Index ? 32u : 64u};
    torch_to_stablehlo::populateBasicOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateViewLikeOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateGatherScatterOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateReductionOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateLinearOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populatePoolingOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateRngOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_stablehlo::populateUncategorizedPatternsAndLegality(
        typeConverter, patterns, target, options);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToStablehloPass() {
  return std::make_unique<ConvertTorchToStablehlo>(false, false);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToStablehloPass(bool enableStaticShape,
                                               bool enableI32Index) {
  return std::make_unique<ConvertTorchToStablehlo>(enableStaticShape,
                                                   enableI32Index);
}
