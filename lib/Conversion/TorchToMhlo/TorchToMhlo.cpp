//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "../PassDetail.h"
#include "./PopulatePatterns.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertTorchToMhlo : public ConvertTorchToMhloBase<ConvertTorchToMhlo> {
public:
  ConvertTorchToMhlo() = default;
  ConvertTorchToMhlo(bool enableStaticShape, bool enableI32Index) {
    this->enableStaticShape = enableStaticShape;
    this->enableI32Index = enableI32Index;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<chlo::ChloDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<chlo::ChloDialect, mhlo::MhloDialect,
                           tensor::TensorDialect, arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    torch_to_mhlo::TorchToMhloOptions options{enableStaticShape,
                                              enableI32Index ? 32u : 64u};
    torch_to_mhlo::populateBasicOpPatternsAndLegality(typeConverter, patterns,
                                                      target, options);
    torch_to_mhlo::populateViewLikeOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_mhlo::populateGatherOpPatternsAndLegality(typeConverter, patterns,
                                                       target, options);
    torch_to_mhlo::populateReductionOpPatternsAndLegality(
        typeConverter, patterns, target, options);
    torch_to_mhlo::populateLinearOpPatternsAndLegality(typeConverter, patterns,
                                                       target, options);
    torch_to_mhlo::populatePoolingOpPatternsAndLegality(typeConverter, patterns,
                                                        target, options);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToMhloPass() {
  return std::make_unique<ConvertTorchToMhlo>(false, false);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToMhloPass(bool enableStaticShape,
                                          bool enableI32Index) {
  return std::make_unique<ConvertTorchToMhlo>(enableStaticShape,
                                              enableI32Index);
}
