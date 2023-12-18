//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v3.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-1.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"

#include "../PassDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

class ConvertAtenShapeToTensorPatternOp
    : public OpConversionPattern<Aten_ShapeAsTensorOp> {
public:
  using OpConversionPattern<Aten_ShapeAsTensorOp>::OpConversionPattern;
  using OpAdaptor = typename Aten_ShapeAsTensorOp::Adaptor;
  LogicalResult
  matchAndRewrite(Aten_ShapeAsTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto operand = adaptor.getOperands()[0];
    auto operandTy = operand.getType().cast<RankedTensorType>();
    auto resultTy =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

    int64_t rank = operandTy.getRank();
    SmallVector<Value> dims;
    for (int i = 0; i < rank; ++i) {
      Value dim = rewriter.createOrFold<tensor::DimOp>(loc, operand, i);
      dim = rewriter.createOrFold<arith::IndexCastOp>(
          loc, resultTy.getElementType(), dim);
      dims.push_back(dim);
    }

    Value tensor =
        rewriter.createOrFold<tensor::FromElementsOp>(op.getLoc(), dims);
    rewriter.replaceOp(op, tensor);
    return success();
  }
};

class ConvertTorchToTensor
    : public ConvertTorchToTensorBase<ConvertTorchToTensor> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addIllegalOp<Torch::Aten_ShapeAsTensorOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenShapeToTensorPatternOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTensorPass() {
  return std::make_unique<ConvertTorchToTensor>();
}
