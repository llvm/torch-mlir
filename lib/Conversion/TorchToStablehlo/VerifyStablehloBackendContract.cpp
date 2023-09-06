//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "./PassDetail.h"
#include "torch-mlir/Conversion/TorchToStablehlo/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::torch;

namespace {
class VerifyStablehloBackendContractPass
    : public VerifyStablehloBackendContractBase<
          VerifyStablehloBackendContractPass> {
  void runOnOperation() override {
    TypeConverter converter;
    converter.addConversion([](Type type) -> Type {
      auto elemTy = type;
      if (isa<TensorType>(type))
        elemTy = type.cast<TensorType>().getElementType();
      if (BaseMemRefType::isValidElementType(elemTy))
        return type;
      return nullptr;
    });

    auto opHasLegalTypes = [&](Operation *op) { return converter.isLegal(op); };

    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Structural operations.
    target.addDynamicallyLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>(
        opHasLegalTypes);
    // Shape operations.
    target.addDynamicallyLegalOp<shape::ShapeOfOp>(opHasLegalTypes);

    target.addLegalDialect<chlo::ChloDialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::createVerifyStablehloBackendContractPass() {
  return std::make_unique<VerifyStablehloBackendContractPass>();
}
