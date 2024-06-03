//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

namespace {
class VerifyTosaBackendContractPass
    : public VerifyTosaBackendContractBase<VerifyTosaBackendContractPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();
    TypeConverter converter;
    converter.addConversion([](TensorType type) -> Type {
      if (BaseMemRefType::isValidElementType(type.getElementType()))
        return type;
      return nullptr;
    });

    auto opHasLegalTypes = [&](Operation *op) { return converter.isLegal(op); };

    ConversionTarget target(*context);

    // Structural operations.
    target.addDynamicallyLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>(
        opHasLegalTypes);
    // Basic scalar operations.
    target.addLegalDialect<tosa::TosaDialect>();
    target.addDynamicallyLegalOp<tensor::CastOp>(opHasLegalTypes);
    target.addDynamicallyLegalOp<arith::ExtSIOp>(opHasLegalTypes);
    target.addDynamicallyLegalOp<arith::ConstantOp>(opHasLegalTypes);

    RewritePatternSet patterns(context);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      // We avoid `module.emitError()` so that mlir-print-op-on-diagnostics
      // doesn't unnecessarily spew out the entire module.
      emitError(module.getLoc())
          << "Module does not conform to the TOSA backend contract. "
             "See dialect conversion legality information above.";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createVerifyTosaBackendContractPass() {
  return std::make_unique<VerifyTosaBackendContractPass>();
}
