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
#include "Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenConstantPadNdOp
    : public OpConversionPattern<AtenConstantPadNdOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConstantPadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    auto type = self.getType().cast<RankedTensorType>();
    int64_t rank = type.getRank();

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t> padInts;
    if (!matchPattern(op.pad(), m_TorchConstantIntList(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");
    uint64_t padRank = padInts.size() / 2;
    if (padRank * 2 != padInts.size())
      return rewriter.notifyMatchFailure(op, "pad range size is not even");
    if (rank < 0 || padRank > (uint64_t)rank)
      return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

    // Initialize low/high paddings with the dims that should not be padded.
    SmallVector<int64_t, 4> lowPadding(/*Size=*/rank - padRank, /*Value=*/0);
    SmallVector<int64_t, 4> highPadding(/*Size=*/rank - padRank, /*Value=*/0);
    // Add the requested padding - note op.pad() is highest dim first ordered
    // pairs of low,high.
    for (uint64_t i = padRank; i > 0; --i) {
      lowPadding.push_back(padInts[i * 2 - 2]);
      highPadding.push_back(padInts[i * 2 - 1]);
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<RankedTensorType>().getElementType();
    Value castedValue =
        convertScalarToDtype(rewriter, loc, adaptor.value(), elementType);
    Value paddedInput = torch_to_linalg::getPaddedTensor(
        op, rewriter, self, lowPadding, highPadding, castedValue);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, paddedInput);
    return success();
  }
};
} // namespace

namespace {
class ConvertValsemVariantAtenFillScalarOp
    : public OpConversionPattern<ValsemVariantAtenFillScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ValsemVariantAtenFillScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    Value initVal = adaptor.value();
    auto tensorType = self.getType().cast<RankedTensorType>();
    RankedTensorType resultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

    Value initValCasted = convertScalarToDtype(rewriter, loc, initVal,
                                               tensorType.getElementType());
    Value result =
        createInitTensor(rewriter, loc, getTensorSizes(rewriter, loc, self),
                         tensorType.getElementType(), initValCasted);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

namespace {
// Converts constant tensor allocation like ops.
template <typename OpTy, int fillVal>
class ConvertConstantTensorAllocOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support for layout, pin_memory features.
    // Only `none` layout is supported.
    // At this point all tensors should have value semantics, and hence the
    // `layout` check can be ignored.

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.size(), resultSizeTorchInt)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size must be constructed using ListConstruct");
    }
    resultSize = getTypeConvertedValues(rewriter, loc, typeConverter,
                                        resultSizeTorchInt);
    for (auto size : resultSize)
      resultSizeIndex.push_back(castIntToIndex(rewriter, loc, size));

    auto resultType = typeConverter->convertType(op.getType())
                          .template cast<RankedTensorType>();
    Type resultElementType;
    if (op.dtype().getType().template isa<Torch::NoneType>()) {
      resultElementType = resultType.getElementType();
    } else {
      int64_t dtypeInt;
      if (!matchPattern(op.dtype(), m_TorchConstantInt(&dtypeInt)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dtype must be a constant integer or none");
      resultElementType = getTypeForScalarType(
          op->getContext(), (torch_upstream::ScalarType)dtypeInt,
          IntegerType::Signless);
    }

    // Create an uninitialized tensor of `resultSize` shape and fill it with
    // value `fillVal`.
    Value constVal = getConstant(rewriter, loc, fillVal, resultElementType);
    Value outputTensor =
        createInitTensor(rewriter, loc, resultSizeIndex, resultElementType, constVal);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, outputTensor);
    return success();
  }
};
} // namespace

namespace {
// Converts `aten.empty` to `linalg.init_tensor` op.
class ConvertAtenEmptyMemoryFormatOp
    : public OpConversionPattern<AtenEmptyMemoryFormatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenEmptyMemoryFormatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support pin_memory and memory_format features.
    // At this point all tensors should have value semantics, and hence the
    // `layout` check can be ignored.

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");

    // Only `none`, `contiguous` and `preserve` memory_format is supported.
    if (!op.memory_format().getType().isa<Torch::NoneType>()) {
      int64_t memoryFormat;
      if (!matchPattern(op.memory_format(), m_TorchConstantInt(&memoryFormat)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the memory format should be specified in "
                "an integer constant");
      if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::Preserve)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only none, contiguous and preserve "
                "memory_format is supported");
    }

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.size(), resultSizeTorchInt)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size must be constructed using ListConstruct");
    }
    resultSize = getTypeConvertedValues(rewriter, loc, typeConverter,
                                        resultSizeTorchInt);
    for (auto size : resultSize)
      resultSizeIndex.push_back(castIntToIndex(rewriter, loc, size));

    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    Type resultElementType;
    if (op.dtype().getType().isa<Torch::NoneType>()) {
      resultElementType = resultType.getElementType();
    } else {
      int64_t dtypeInt;
      if (!matchPattern(op.dtype(), m_TorchConstantInt(&dtypeInt)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dtype must be a constant integer or none");
      resultElementType = getTypeForScalarType(
          op->getContext(), (torch_upstream::ScalarType)dtypeInt,
          IntegerType::Signless);
    }

    // Create an uninitialized tensor of `resultSize` shape.
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultSizeIndex, resultElementType);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, initTensor);
    return success();
  }
};
} // namespace

namespace {
// Let's say the result of the `aten.arange.start_step` is `output` which is a
// 1-d output tensor. The approach used for generating the output tensor is as
// follows:
//    for i in range(ceil((end-start)/step))
//          output[i] = start + (i * step)
class ConvertAtenArangeStartStepOp
    : public OpConversionPattern<AtenArangeStartStepOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenArangeStartStepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support for pin_memory features.
    // At this point all tensors should have value semantics, and hence the
    // `layout` check can be ignored.

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();
    Type dtype = resultType.getElementType();
    Value start = convertScalarToDtype(rewriter, loc, adaptor.start(), dtype);
    Value end = convertScalarToDtype(rewriter, loc, adaptor.end(), dtype);
    Value step = convertScalarToDtype(rewriter, loc, adaptor.step(), dtype);

    // The result will always be a 1-d tensor.
    // The size of the result is calculated as follows:
    //          ceil((end - start)/step)
    Value resultShape;
    if (dtype.isa<mlir::IntegerType>()) {
      Value subOut = rewriter.create<arith::SubIOp>(loc, end, start);
      resultShape = rewriter.create<arith::CeilDivSIOp>(loc, subOut, step);
    } else {
      Value subOut = rewriter.create<arith::SubFOp>(loc, end, start);
      Value divOut = rewriter.create<arith::DivFOp>(loc, subOut, step);
      Value ceilOut = rewriter.create<math::CeilOp>(loc, divOut);
      resultShape =
          rewriter.create<arith::FPToUIOp>(loc, rewriter.getI64Type(), ceilOut);
    }
    resultShape = castIntToIndex(rewriter, loc, resultShape);

    Value resultTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, dtype);

    StringRef iteratorType = getParallelIteratorTypeName();
    AffineMap indexingMap =
        AffineMap::getMultiDimIdentityMap(1, op->getContext());

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, /*resultTensorTypes=*/resultTensor.getType(),
                /*inputs=*/ValueRange({}),
                /*outputs=*/resultTensor,
                /*indexingMaps=*/indexingMap,
                /*iteratorTypes=*/iteratorType,
                [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
                  Value index = b.create<linalg::IndexOp>(loc, 0);
                  index = castIndexToInt64(b, loc, index);
                  index = convertScalarToDtype(b, loc, index, dtype);
                  Value mulOut, result;
                  if (dtype.isa<mlir::FloatType>()) {
                    mulOut = b.create<arith::MulFOp>(loc, step, index);
                    result = b.create<arith::AddFOp>(loc, start, mulOut);
                  } else {
                    mulOut = b.create<arith::MulIOp>(loc, step, index);
                    result = b.create<arith::AddIOp>(loc, start, mulOut);
                  }
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::
    populateTensorConstructorsPatternsAndLegality(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns,
                                                  ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenConstantPadNdOp>();
  patterns.add<ConvertAtenConstantPadNdOp>(typeConverter, context);
  target.addIllegalOp<ValsemVariantAtenFillScalarOp>();
  patterns.add<ConvertValsemVariantAtenFillScalarOp>(typeConverter, context);
  target.addIllegalOp<AtenZerosOp, AtenOnesOp>();
  patterns.add<ConvertConstantTensorAllocOp<AtenZerosOp, 0>>(typeConverter,
                                                             context);
  patterns.add<ConvertConstantTensorAllocOp<AtenOnesOp, 1>>(typeConverter,
                                                            context);
  target.addIllegalOp<AtenEmptyMemoryFormatOp>();
  patterns.add<ConvertAtenEmptyMemoryFormatOp>(typeConverter, context);
  patterns.add<ConvertAtenArangeStartStepOp>(typeConverter, context);
  target.addIllegalOp<AtenArangeStartStepOp>();
}
