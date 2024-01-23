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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <numeric>

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
    Value self = adaptor.getSelf();
    auto type = self.getType().cast<RankedTensorType>();
    int64_t rank = type.getRank();

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPad(), m_TorchListOfConstantInts(padInts)))
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
        convertScalarToDtype(rewriter, loc, adaptor.getValue(), elementType);
    Value paddedInput = torch_to_linalg::getPaddedTensor(
        op, rewriter, self, lowPadding, highPadding, castedValue);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, paddedInput);
    return success();
  }
};
} // namespace

namespace {

  // Lower aten.replication_pad2d operator into a sequence of
  // tensor.extract_slice and tensor.concat operations.

  class ConvertAtenReplicationPad2dOp 
      : public OpConversionPattern<AtenReplicationPad2dOp> {
  public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult
    matchAndRewrite(AtenReplicationPad2dOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
        return failure();

      Location loc = op->getLoc();
      Value input = adaptor.getSelf();
      auto inputType = llvm::cast<RankedTensorType>(input.getType());
      int64_t inputRank = inputType.getRank();
      unsigned numDims = inputType.getRank();
        assert(numDims >= 2 && "Not enough input dimensions");

      SmallVector<int64_t> padInts;
      if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
        return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");
      uint64_t padRank = padInts.size() / 2;
      if (padRank * 2 != padInts.size())
        return rewriter.notifyMatchFailure(op, "pad range size is not even");
      if (inputRank < 0 || padRank > (uint64_t)inputRank)
        return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

      SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
      int64_t hDim = numDims - 1;
      int64_t vDim = numDims - 2;
      Value hDimSize = inputShape[hDim];
      Value vDimSize = inputShape[vDim];

      enum tileHLoc { LEFT = 0, HCENTER = 1, RIGHT = 2 };
      enum tileVLoc { TOP = 0, VCENTER = 2, BOTTOM = 1, };
      // vTile denotes the vertical size of the tile
      // hTile denotes the horizontal size of the tile
      // The padding results are composed of following tiles:
      // vTile[TOP]hTile[LEFT], vTile[TOP]hTile[HCENTER], vTile[TOP]hTile[RIGHT]
      // vTile[VCENTER]hTile[LEFT], vTile[VCENTER]hTile[HCENTER], vTile[VCENTER]hTile[RIGHT]
      // vTile[BOTTOM]hTile[LEFT], vTile[BOTTOM]hTile[HCENTER], vTile[BOTTOM]hTile[RIGHT]
      // vTile[VCENTER]hTile[HCENTER] is the original input tensor
      Type indexType = rewriter.getIndexType();
      Value vTile[3];
      Value hTile[3];
      vTile[VCENTER] = vDimSize;
      hTile[HCENTER] = hDimSize;
      vTile[TOP] = getConstant(rewriter, loc, padInts[2], indexType);
      vTile[BOTTOM] = getConstant(rewriter, loc, padInts[3], indexType);
      hTile[LEFT] = getConstant(rewriter, loc, padInts[0], indexType);
      hTile[RIGHT] = getConstant(rewriter, loc, padInts[1], indexType);

      bool hasLeftPadding = false;
      bool hasRightPadding = false;
      bool hasTopPadding = false;
      bool hasBottomPadding = false;

      for (auto i: {TOP, VCENTER, BOTTOM}){
        for (auto j: {LEFT, HCENTER, RIGHT}) {
          auto constVtile{
          mlir::dyn_cast<mlir::arith::ConstantOp>(vTile[i].getDefiningOp())
              .getValue()
              .dyn_cast_or_null<mlir::IntegerAttr>()};

          auto constHtile{
          mlir::dyn_cast<mlir::arith::ConstantOp>(hTile[j].getDefiningOp())
              .getValue()
              .dyn_cast_or_null<mlir::IntegerAttr>()};
          auto vSize = constVtile.getInt();
          auto hSize = constHtile.getInt();

          if ((i == TOP) && (vSize > 0))
            hasTopPadding = true;
          if ((i == BOTTOM) && (vSize > 0))
            hasBottomPadding = true;
          if ((j == LEFT) && (hSize > 0))
            hasLeftPadding = true;
          if ((j == RIGHT) && (hSize > 0))
            hasRightPadding = true;
        }
      }

      auto createSub = [&](Value x, Value y) {
        return rewriter.create<arith::SubIOp>(loc, x, y);
      };

      // Extract left and right pad tiles.
      Value zero = getConstant(rewriter, loc, 0, indexType);
      Value one = getConstant(rewriter, loc, 1, indexType);
      Value hDimSizeMinusOne = createSub(hDimSize, one);
      Value vDimSizeMinusOne = createSub(vDimSize, one);
      SmallVector<Value> allOneStrides(numDims, one);

      SmallVector<Value> extractOffsetsLT(numDims, zero);
      extractOffsetsLT[hDim] = zero;
      extractOffsetsLT[vDim] = zero;
      SmallVector<Value> extractShapeLR(numDims, one);
      extractShapeLR[hDim] = one;
      extractShapeLR[vDim] = vDimSize;

      SmallVector<Value> extractOffsetsRight(numDims, zero);
      extractOffsetsRight[hDim] = hDimSizeMinusOne;
      extractOffsetsRight[vDim] = zero;

      SmallVector<Value> extractOffsetsBottom(numDims, zero);
      extractOffsetsBottom[hDim] = zero;
      extractOffsetsBottom[vDim] = vDimSizeMinusOne;

      SmallVector<Value> extractShapeTB(numDims, one);
      extractShapeTB[hDim] = hDimSize;
      extractShapeTB[vDim] = one;

      SmallVector<Value> tensorsLeft;
      SmallVector<Value> tensorsRight;
      SmallVector<Value> tensorsCenter;
      Value centerTile;
      SmallVector<Value> tensorsRes;

      if (hasLeftPadding) {
        Value vCenterLeftSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, input, extractOffsetsLT, extractShapeLR, allOneStrides);
        Value vLeftSlice = vCenterLeftSlice;
        if (hasTopPadding) {
          Value topLeftValue = rewriter.create<tensor::ExtractOp>(
              loc, input, ValueRange{zero, zero, zero, zero});
          //pad vCenterLeftSlice on the top
          SmallVector<int64_t> lowPadding(4, 0);
          SmallVector<int64_t> highPadding(4, 0);
          lowPadding[2] = padInts[2];
          vLeftSlice = torch_to_linalg::getPaddedTensor(op, rewriter, vLeftSlice, lowPadding, highPadding, topLeftValue);
        }
        if (hasBottomPadding) {
          Value bottomLeftValue = rewriter.create<tensor::ExtractOp> (loc, input, ValueRange{zero, zero, vDimSizeMinusOne, zero});

          //pad vLeftSlice at the bottom
          SmallVector<int64_t> lowPadding(4, 0);
          SmallVector<int64_t> highPadding(4, 0);
          highPadding[2] = padInts[3];
          vLeftSlice = torch_to_linalg::getPaddedTensor(op, rewriter, vLeftSlice, lowPadding, highPadding, bottomLeftValue);
        }
        for (auto i=0; i<padInts[0]; ++i) {
          tensorsLeft.push_back(vLeftSlice);
        }
        Value leftPadTile =
            rewriter.create<tensor::ConcatOp>(loc, 3, tensorsLeft);
        tensorsRes.push_back(leftPadTile);
      }
      if (hasTopPadding) {
        Value topHcenterSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, input, extractOffsetsLT, extractShapeTB, allOneStrides);
        for (auto i = 0; i < padInts[2]; ++i) {
          tensorsCenter.push_back(topHcenterSlice);
        }
      }
      tensorsCenter.push_back(input);
      if (hasBottomPadding) {
        Value bottomHcenterSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, input, extractOffsetsBottom, extractShapeTB, allOneStrides);
        for (auto i = 0; i < padInts[3]; ++i) {
          tensorsCenter.push_back(bottomHcenterSlice);
        }
      }
      centerTile = rewriter.create<tensor::ConcatOp>(loc, 2, tensorsCenter);
      tensorsRes.push_back(centerTile);

      if (hasRightPadding) {
        Value vCenterRightSlice = rewriter.create<tensor::ExtractSliceOp>(
            loc, input, extractOffsetsRight, extractShapeLR, allOneStrides);
        Value vRightSlice = vCenterRightSlice;
        if (hasTopPadding) {
          Value topRightValue = rewriter.create<tensor::ExtractOp> (loc, input, ValueRange{zero, zero, zero, hDimSizeMinusOne});

          //pad vCenterRightSlice on the top
          SmallVector<int64_t> lowPadding(4, 0);
          SmallVector<int64_t> highPadding(4, 0);
          lowPadding[2] = padInts[2];
          vRightSlice = torch_to_linalg::getPaddedTensor(op, rewriter, vRightSlice, lowPadding, highPadding, topRightValue);
        }
        if (hasBottomPadding) {
          Value bottomRightValue = rewriter.create<tensor::ExtractOp> (loc, input, ValueRange{zero, zero, vDimSizeMinusOne, hDimSizeMinusOne});

          // Pad vCenterRightSlice or vRightTopPaddedSlice at the bottom.
          SmallVector<int64_t> lowPadding(4, 0);
          SmallVector<int64_t> highPadding(4, 0);
          highPadding[2] = padInts[3];
          vRightSlice = torch_to_linalg::getPaddedTensor(op, rewriter, vRightSlice, lowPadding, highPadding, bottomRightValue);
        }
        for (auto i=0; i<padInts[1]; ++i) {
          tensorsRight.push_back(vRightSlice);
        }
        Value rightPadTile = rewriter.create<tensor::ConcatOp>(loc, 3, tensorsRight);
        tensorsRes.push_back(rightPadTile);
      }     
      Value resTensor = rewriter.create<tensor::ConcatOp>(loc, 3, tensorsRes);
      Type newResultType = getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, resTensor);
      return success();
    }
  };
}

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
    if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.getSize(), resultSizeTorchInt)) {
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
    if (op.getDtype().getType().template isa<Torch::NoneType>()) {
      resultElementType = resultType.getElementType();
    } else {
      int64_t dtypeInt;
      if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dtype must be a constant integer or none");
      FailureOr<Type> maybeResultElementType =
          torch_to_linalg::getBackendTypeForScalarType(
              op->getContext(), (torch_upstream::ScalarType)dtypeInt);
      if (failed(maybeResultElementType)) {
        return rewriter.notifyMatchFailure(
            op, "unable to convert `dtypeInt` to builtin type");
      }
      resultElementType = *maybeResultElementType;
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
    if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");

    // Only `none`, `contiguous` and `preserve` memory_format is supported.
    if (!op.getMemoryFormat().getType().isa<Torch::NoneType>()) {
      int64_t memoryFormat;
      if (!matchPattern(op.getMemoryFormat(), m_TorchConstantInt(&memoryFormat)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the memory format should be specified in "
                "an integer constant");
      if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::Preserve)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: only none, contiguous and preserve "
                "memory_format is supported");
    }

    // TODO: Add support for device arg other than cpu.
    if (!op.getDevice().getType().isa<Torch::NoneType>()) {
      std::string device;
      if (!matchPattern(op.getDevice(), m_TorchConstantDevice(device)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: device must be a constant str");
      else if (device != "cpu")
        return rewriter.notifyMatchFailure(
            op, "unimplemented: device is expected to be cpu");
    }

    // TODO: Add support for non-strided layout.
    // torch.layout is by default strided i.e. 0.
    if (!op.getLayout().getType().isa<Torch::NoneType>()) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout must be a constant");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout is expected to be strided");
    }

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.getSize(), resultSizeTorchInt)) {
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
    if (op.getDtype().getType().isa<Torch::NoneType>()) {
      resultElementType = getDefaultDtypeForTorchScalar(
          Torch::FloatType::get(op->getContext()));
    } else {
      int64_t dtypeInt;
      if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dtype must be a constant integer or none");
      FailureOr<Type> maybeResultElementType =
          torch_to_linalg::getBackendTypeForScalarType(
              op->getContext(), (torch_upstream::ScalarType)dtypeInt);
      if (failed(maybeResultElementType)) {
        return rewriter.notifyMatchFailure(
            op, "unable to convert `dtypeInt` to builtin type");
      }
      resultElementType = *maybeResultElementType;
    }

    // Create an uninitialized tensor of `resultSize` shape.
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultSizeIndex), resultElementType);
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
    if (!op.getPinMemory().getType().isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();
    Type dtype = resultType.getElementType();
    Value start = convertScalarToDtype(rewriter, loc, adaptor.getStart(), dtype);
    Value end = convertScalarToDtype(rewriter, loc, adaptor.getEnd(), dtype);
    Value step = convertScalarToDtype(rewriter, loc, adaptor.getStep(), dtype);

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

    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), dtype);

    auto iteratorType = utils::IteratorType::parallel;
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
  target.addIllegalOp<AtenReplicationPad2dOp>();
  patterns.add<ConvertAtenReplicationPad2dOp>(typeConverter, context);
  target.addIllegalOp<AtenConstantPadNdOp>();
  patterns.add<ConvertAtenConstantPadNdOp>(typeConverter, context);
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
