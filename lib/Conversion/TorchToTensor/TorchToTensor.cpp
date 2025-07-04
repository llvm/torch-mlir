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
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

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

    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto rank = operandTy.getRank();
    llvm::SmallVector<Value> indices(rank, zeroIdx);

    Value extract = rewriter.create<tensor::ExtractOp>(
        op.getLoc(), operandTy.getElementType(), operand, indices);
    auto extractTy = extract.getType();
    if (isa<mlir::IntegerType>(extractTy) && !extractTy.isInteger(64)) {
      if (torchDTy.isUnsignedInteger()) {
        extract = rewriter.create<arith::ExtUIOp>(
            op.getLoc(), rewriter.getIntegerType(64), extract);
      } else {
        extract = rewriter.create<arith::ExtSIOp>(
            op.getLoc(), rewriter.getIntegerType(64), extract);
      }
    }

    if (isa<mlir::FloatType>(extractTy) && !extractTy.isF64()) {
      extract = rewriter.create<arith::ExtFOp>(op.getLoc(),
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
        value = rewriter.create<arith::TruncIOp>(loc, resultETy, value);

      if (isa<mlir::FloatType>(resultETy) && value.getType() != resultETy)
        value = rewriter.create<arith::TruncFOp>(loc, resultETy, value);

      values.push_back(value);
    }

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy, values);

    return success();
  }
};

class ConvertAtenAsStridedOp : public OpConversionPattern<AtenAsStridedOp> {
public:
  using OpConversionPattern<AtenAsStridedOp>::OpConversionPattern;
  using OpAdaptor = typename AtenAsStridedOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenAsStridedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // In some cases AtenAsStridedOp is equivalent to a Tensor ExtractSliceOp.
    // We will try to match those cases here.
    auto inputShape =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getShape();
    auto outputShape =
        cast<BaseTensorType>(op.getResult().getType()).getSizes();
    auto resultTy =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    // If the output shape is strictly larger than the input shape at any
    // dimension than this AtenAsStridedOp is not equivalent to a slice.
    for (uint64_t i = 0; i < outputShape.size(); ++i) {
      if (outputShape[i] > inputShape[i])
        return failure();
    }

    // Calculate what the strides attribute should be if the input tensor is
    // contiguous.
    SmallVector<int64_t> contiguousStrides(inputShape.size(), 1);
    for (int i = inputShape.size() - 2; i >= 0; --i) {
      contiguousStrides[i] = contiguousStrides[i + 1] * inputShape[i + 1];
    }

    SmallVector<Value> outSizeValues, opStridesValues;
    if (!getListConstructElements(adaptor.getStride(), opStridesValues))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");

    if (!getListConstructElements(adaptor.getSize(), outSizeValues))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");

    // Get storage offset
    int64_t offset;
    if (!matchPattern(op.getStorageOffset(), m_TorchConstantInt(&offset)))
      offset = 0;

    APInt size;
    SmallVector<int64_t> outSize(inputShape.size(), 0);
    for (uint64_t i = 0; i < outSizeValues.size(); ++i) {
      if (!matchPattern(outSizeValues[i], m_Op<TorchConversion::FromI64Op>(
                                              m_ConstantInt(&size))) ||
          !size.isSignedIntN(64))
        return failure();
      outSize[i] = size.getSExtValue();
    }
    APInt stride;
    SmallVector<int64_t> opStrides(inputShape.size(), 0);
    for (uint64_t i = 0; i < opStridesValues.size(); ++i) {
      if (!matchPattern(opStridesValues[i], m_Op<TorchConversion::FromI64Op>(
                                                m_ConstantInt(&stride))) ||
          !stride.isSignedIntN(64))
        return failure();
      opStrides[i] = stride.getSExtValue();
    }

    // Slice dims are the dims where the input and output shapes are not equal.
    SmallVector<int64_t> sliceDims;
    for (uint64_t i = 0; i < inputShape.size(); ++i) {
      if (outSize[i] != inputShape[i])
        sliceDims.push_back(i);
    }

    // If there are no slice dims, then the AtenAsStridedOp is equivalent to the
    // input tensor.
    if (sliceDims.empty()) {
      rewriter.replaceOp(op, adaptor.getSelf());
      return success();
    }

    SmallVector<int64_t> sliceOffsets(inputShape.size(), 0);
    SmallVector<int64_t> sliceStrides(opStrides.size(), 1);
    for (auto dim : sliceDims) {
      sliceOffsets[dim] = offset / contiguousStrides[dim];
      sliceStrides[dim] = opStrides[dim] / contiguousStrides[dim];
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultTy, adaptor.getSelf(), ValueRange(), ValueRange(),
        ValueRange(), sliceOffsets, outSize, sliceStrides);
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
    target.addIllegalOp<Torch::AtenItemOp>();
    target.addIllegalOp<Torch::AtenTensorOp>();
    target.addIllegalOp<Torch::Aten_ShapeAsTensorOp>();
    target.addIllegalOp<Torch::AtenAsStridedOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenShapeToTensorPatternOp, ConvertAtenItemOp,
                 ConvertAtenTensorOpPattern, ConvertAtenAsStridedOp>(
        typeConverter, context);

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
