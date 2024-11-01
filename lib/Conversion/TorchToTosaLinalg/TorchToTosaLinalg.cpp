//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosaLinalg/TorchToTosaLinalg.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosaLinalg
    : public ConvertTorchToTosaLinalgBase<ConvertTorchToTosaLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<tosa::TosaDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<complex::ComplexDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           scf::SCFDialect, sparse_tensor::SparseTensorDialect,
                           tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithDialect, complex::ComplexDialect>();

    target.addLegalOp<TorchConversion::GetNextSeedOp>();
    torch::populateTorchToTosaConversionLegalOps(target);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    auto illegalOps = populateTorchToTosaConversionPatternsAndIllegalOps(
        typeConverter, patterns);

    // for (auto op : illegalOps) {
    //   target.addIllegalOp(OperationName(op, context));
    // }

    torch::populateTorchToLinalgOnTensorsPatternsAndLegality(typeConverter,
                                                             patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaLinalgPass() {
  return std::make_unique<ConvertTorchToTosaLinalg>();
}
