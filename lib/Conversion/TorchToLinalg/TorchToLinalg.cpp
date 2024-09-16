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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    registry.insert<arith::ArithDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<complex::ComplexDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<
        linalg::LinalgDialect, func::FuncDialect, cf::ControlFlowDialect,
        math::MathDialect, scf::SCFDialect, sparse_tensor::SparseTensorDialect,
        tensor::TensorDialect, arith::ArithDialect, complex::ComplexDialect>();
    target.addLegalOp<TorchConversion::GetNextSeedOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    torch::populateTorchToLinalgOnTensorsPatterns(typeConverter, patterns);
    torch::populateTorchToLinalgOnTensorsOpsLegality(target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

//  Add any new populate*PatternsAndLegality functionality here
void mlir::torch::populateTorchToLinalgOnTensorsPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  torch_to_linalg::populateTensorScalarInteropPatterns(typeConverter, patterns);
  torch_to_linalg::populateLinearPatterns(typeConverter, patterns);
  torch_to_linalg::populatePoolingPatterns(typeConverter, patterns);
  torch_to_linalg::populateRandomPatterns(typeConverter, patterns);
  torch_to_linalg::populateUncategorizedPatterns(typeConverter, patterns);
  torch_to_linalg::populateReductionPatterns(typeConverter, patterns);
  torch_to_linalg::populateDataMovementPatterns(typeConverter, patterns);
  torch_to_linalg::populateIndirectDataMovementPatterns(typeConverter,
                                                        patterns);
  torch_to_linalg::populateTensorConstructorsPatterns(typeConverter, patterns);
}

void mlir::torch::populateTorchToLinalgOnTensorsOpsLegality(
    ConversionTarget &target) {
  torch_to_linalg::populateTensorScalarInteropOpsLegality(target);
  torch_to_linalg::populateLinearOpsLegality(target);
  torch_to_linalg::populatePoolingOpsLegality(target);
  torch_to_linalg::populateRandomOpsLegality(target);
  torch_to_linalg::populateUncategorizedOpsLegality(target);
  torch_to_linalg::populateReductionOpsLegality(target);
  torch_to_linalg::populateDataMovementOpsLegality(target);
  torch_to_linalg::populateIndirectDataMovementOpsLegality(target);
  torch_to_linalg::populateTensorConstructorsOpsLegality(target);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
