//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMLProgram/TorchToMLProgram.h"

#include "../PassDetail.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;

namespace {
class ConvertTorchToMLProgram
    : public ConvertTorchToMLProgramBase<ConvertTorchToMLProgram> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();
    ConversionTarget dummyTarget(*context);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(dummyTarget, typeConverter);

    auto globalBuilder =
        OpBuilder::atBlockBegin(&*module.getBodyRegion().begin());
    module.walk([&](Torch::ResourceValueTensorLiteralOp op) {
      auto type = typeConverter.convertType(op.getType());
      globalBuilder.create<ml_program::GlobalOp>(
          op.getLoc(), op.getSymNameAttr().getAttr(), type,
          /*is_mutable=*/true, // Just to enable generator
          /*value=*/op.getValue(),
          globalBuilder.getStringAttr("public"));
      OpBuilder builder(op);
      auto loadConst = builder.create<ml_program::GlobalLoadOp>(
          op.getLoc(), type, op.getSymNameAttr());
      Value torchTensor = builder.create<TorchConversion::FromBuiltinTensorOp>(
          op.getLoc(), op.getType(), loadConst);
      op.replaceAllUsesWith(torchTensor);
      op.erase();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::createConvertTorchToMLProgramPass() {
  return std::make_unique<ConvertTorchToMLProgram>();
}