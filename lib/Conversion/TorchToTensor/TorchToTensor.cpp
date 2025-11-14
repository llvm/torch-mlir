//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v3.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-1.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOTENSOR
#include "torch-mlir/Conversion/Passes.h.inc"

namespace {

class ConvertAtenItemOp : public OpConversionPattern<AtenItemOp> {
public:
  using OpConversionPattern<AtenItemOp>::OpConversionPattern;
  using OpAdaptor = typename AtenItemOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenItemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operand = adaptor.getOperands()[0];
    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto torchDTy = cast<ValueTensorType>(op.getOperand().getType()).getDtype();

    if (operandTy.getNumElements() != 1)
      return rewriter.notifyMatchFailure(op, "expected only one item");

    auto zeroIdx = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto rank = operandTy.getRank();
    llvm::SmallVector<Value> indices(rank, zeroIdx);

    Value extract = tensor::ExtractOp::create(
        rewriter, op.getLoc(), operandTy.getElementType(), operand, indices);
    auto extractTy = extract.getType();
    if (isa<mlir::IntegerType>(extractTy) && !extractTy.isInteger(64)) {
      if (torchDTy.isUnsignedInteger()) {
        extract = arith::ExtUIOp::create(rewriter, op.getLoc(),
                                         rewriter.getIntegerType(64), extract);
      } else {
        extract = arith::ExtSIOp::create(rewriter, op.getLoc(),
                                         rewriter.getIntegerType(64), extract);
      }
    }

    if (isa<mlir::FloatType>(extractTy) && !extractTy.isF64()) {
      extract = arith::ExtFOp::create(rewriter, op.getLoc(),
                                      rewriter.getF64Type(), extract);
    }

    rewriter.replaceOp(op, extract);
    return success();
  }
};

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
    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto resultTy =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    int64_t rank = operandTy.getRank();
    if (rank == 0) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, resultTy.getShape(),
                                                   resultTy.getElementType());
      return success();
    }

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

class ConvertAtenTensorOpPattern : public OpConversionPattern<AtenTensorOp> {
public:
  using OpConversionPattern<AtenTensorOp>::OpConversionPattern;
  using OpAdaptor = typename AtenTensorOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto list = op.getData().getDefiningOp<Torch::PrimListConstructOp>();
    if (!list)
      return failure();

    auto typeConverter = getTypeConverter();
    auto resultTy = cast<ShapedType>(typeConverter->convertType(op.getType()));
    auto resultETy = resultTy.getElementType();

    SmallVector<Value> values;
    for (Value operand : list.getOperands()) {
      Value value = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(operand.getType()),
          operand);

      if (isa<mlir::IntegerType>(resultETy) && value.getType() != resultETy)
        value = arith::TruncIOp::create(rewriter, loc, resultETy, value);

      if (isa<mlir::FloatType>(resultETy) && value.getType() != resultETy)
        value = arith::TruncFOp::create(rewriter, loc, resultETy, value);

      values.push_back(value);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, values);

    return success();
  }
};

class ConvertTorchToTensor
    : public impl::ConvertTorchToTensorBase<ConvertTorchToTensor> {
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
    target.addIllegalOp<Torch::AtenItemOp>();
    target.addIllegalOp<Torch::AtenTensorOp>();
    target.addIllegalOp<Torch::Aten_ShapeAsTensorOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenShapeToTensorPatternOp, ConvertAtenItemOp,
                 ConvertAtenTensorOpPattern>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToTensorPass() {
  return std::make_unique<ConvertTorchToTensor>();
}

} // namespace mlir::torch
