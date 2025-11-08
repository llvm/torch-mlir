//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

enum sliceLoc { START = 0, END = 1 };

static Value extractSlice(ConversionPatternRewriter &rewriter, Location loc,
                          Value input, int64_t dimension, sliceLoc sliceLoc) {
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);

  SmallVector<OpFoldResult> offsets(inputRank, rewriter.getIndexAttr(0));
  if (sliceLoc == END) {
    Value dimSize = inputShape[dimension];
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value endIdx = arith::SubIOp::create(rewriter, loc, dimSize, one);
    offsets[dimension] = getAsOpFoldResult(endIdx);
  }

  SmallVector<OpFoldResult> allOneStrides(inputRank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes(inputRank, rewriter.getIndexAttr(0));
  for (int i = 0; i < inputRank; ++i)
    sizes[i] = (i == dimension) ? rewriter.getIndexAttr(1)
                                : getAsOpFoldResult(inputShape[i]);

  Value extractedSlice = tensor::ExtractSliceOp::create(
      rewriter, loc, input, offsets, sizes, allOneStrides);
  return extractedSlice;
}

static Value createTile(ConversionPatternRewriter &rewriter, Location loc,
                        Value slice, int64_t tileWidth, int64_t dimension) {
  SmallVector<Value> slices(tileWidth, slice);
  if (tileWidth == 1)
    return slice;
  return tensor::ConcatOp::create(rewriter, loc, dimension, slices);
}

static Value replicationPad(ConversionPatternRewriter &rewriter, Location loc,
                            Value input, SmallVector<int64_t> &padInts,
                            int64_t numDims) {
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();

  Value res = input;
  int64_t padIdx = 0;
  for (int64_t dim = inputRank - 1; dim >= inputRank - numDims; dim--) {
    int64_t startTileWidth = padInts[padIdx++];
    int64_t endTileWidth = padInts[padIdx++];

    SmallVector<Value> resultParts;
    if (startTileWidth > 0) {
      Value slice = extractSlice(rewriter, loc, res, dim, sliceLoc::START);
      Value tile = createTile(rewriter, loc, slice, startTileWidth, dim);
      resultParts.push_back(tile);
    }

    resultParts.push_back(res);

    if (endTileWidth > 0) {
      Value slice = extractSlice(rewriter, loc, res, dim, sliceLoc::END);
      Value tile = createTile(rewriter, loc, slice, endTileWidth, dim);
      resultParts.push_back(tile);
    }

    if (resultParts.size() > 1)
      res = tensor::ConcatOp::create(rewriter, loc, dim, resultParts);
  }
  return res;
}

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
    auto type = cast<RankedTensorType>(self.getType());
    int64_t rank = type.getRank();

    auto primList = op.getPad().getDefiningOp<Torch::PrimListConstructOp>();
    if (!primList) {
      return rewriter.notifyMatchFailure(op, "unable to get pad values");
    }

    SmallVector<Value> padVals(primList.getOperands());

    uint64_t padRank = padVals.size() / 2;
    if (padRank * 2 != padVals.size())
      return rewriter.notifyMatchFailure(op, "pad range size is not even");
    if (rank < 0 || padRank > (uint64_t)rank)
      return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

    // Initialize low/high paddings with the dims that should not be padded.
    int64_t noPad = rank - padRank;
    Attribute zero = rewriter.getIndexAttr(0);
    SmallVector<int64_t> staticLow(noPad, 0);
    SmallVector<int64_t> staticHigh(noPad, 0);
    SmallVector<OpFoldResult> lowPad(noPad, zero);
    SmallVector<OpFoldResult> highPad(noPad, zero);

    auto tc = getTypeConverter();

    // Add the requested padding - note op.pad() is highest dim first ordered
    // pairs of low,high.
    for (uint64_t i = padRank; i > 0; --i) {
      int64_t lowi, highi;
      Value lowv = padVals[i * 2 - 2];
      Value highv = padVals[i * 2 - 1];
      if (!matchPattern(lowv, m_TorchConstantInt(&lowi))) {
        Type cty = tc->convertType(lowv.getType());
        lowv = tc->materializeTargetConversion(rewriter, loc, cty, lowv);
        lowv = arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), lowv);
        lowPad.push_back(lowv);
        staticLow.push_back(ShapedType::kDynamic);
      } else {
        lowPad.push_back(rewriter.getIndexAttr(lowi));
        staticLow.push_back(lowi);
      }

      if (!matchPattern(highv, m_TorchConstantInt(&highi))) {
        Type cty = tc->convertType(highv.getType());
        highv = tc->materializeTargetConversion(rewriter, loc, cty, highv);
        highv = arith::IndexCastOp::create(rewriter, loc,
                                           rewriter.getIndexType(), highv);
        highPad.push_back(highv);
        staticHigh.push_back(ShapedType::kDynamic);
      } else {
        highPad.push_back(rewriter.getIndexAttr(highi));
        staticHigh.push_back(highi);
      }
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = cast<RankedTensorType>(newResultType).getElementType();

    auto dstOriginalDtype =
        cast<Torch::ValueTensorType>(op.getType()).getDtype();
    Value castedValue =
        convertScalarToDtype(rewriter, loc, adaptor.getValue(), elementType,
                             std::nullopt, dstOriginalDtype);

    Type padType = tensor::PadOp::inferResultType(
        cast<RankedTensorType>(self.getType()), staticLow, staticHigh);
    Value paddedInput = tensor::PadOp::create(rewriter, loc, padType, self,
                                              lowPad, highPad, castedValue);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, paddedInput);
    return success();
  }
};
} // namespace

namespace {

class ConvertAtenReplicationPad1dOp
    : public OpConversionPattern<AtenReplicationPad1dOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenReplicationPad1dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();

    if (inputRank < 2)
      return rewriter.notifyMatchFailure(op, "input rank must be at least 2");

    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");

    if (padInts.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "pad range must have exactly two values");

    int64_t numSpatialDims = 1;
    Value result =
        replicationPad(rewriter, loc, input, padInts, numSpatialDims);
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);

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
    assert(inputRank >= 2 && "Not enough input dimensions");

    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");
    uint64_t padRank = padInts.size() / 2;
    if (padRank * 2 != padInts.size())
      return rewriter.notifyMatchFailure(op, "pad range size is not even");
    if (inputRank < 0 || padRank > (uint64_t)inputRank)
      return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

    int64_t numSpatialDims = 2;
    Value resTensor =
        replicationPad(rewriter, loc, input, padInts, numSpatialDims);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, resTensor);
    return success();
  }
};
} // namespace

namespace {

// Lower aten.replication_pad3d operator into a sequence of
// tensor.extract_slice and tensor.concat operations.
class ConvertAtenReplicationPad3dOp
    : public OpConversionPattern<AtenReplicationPad3dOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenReplicationPad3dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    Value input = adaptor.getSelf();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    [[maybe_unused]] int64_t inputRank = inputType.getRank();
    assert(inputRank >= 3 && "Not enough input dimensions");

    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");

    if (padInts.size() != 6)
      return rewriter.notifyMatchFailure(
          op, "pad range must have exactly six values");

    int64_t numSpatialDims = 3;
    Value res = replicationPad(rewriter, loc, input, padInts, numSpatialDims);
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, res);
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
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
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

    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    Type resultElementType;
    if (isa<Torch::NoneType>(op.getDtype().getType())) {
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
    Value outputTensor = createInitTensor(rewriter, loc, resultSizeIndex,
                                          resultElementType, constVal);
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
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");

    // Only `none`, `contiguous` and `preserve` memory_format is supported.
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType())) {
      int64_t memoryFormat;
      if (!matchPattern(op.getMemoryFormat(),
                        m_TorchConstantInt(&memoryFormat)))
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
    if (!isa<Torch::NoneType>(op.getDevice().getType())) {
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
    if (!isa<Torch::NoneType>(op.getLayout().getType())) {
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
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    Type resultElementType;
    if (isa<Torch::NoneType>(op.getDtype().getType())) {
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
    Value initTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(resultSizeIndex), resultElementType);
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
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    RankedTensorType resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));
    Type dtype = resultType.getElementType();
    Value start =
        convertScalarToDtype(rewriter, loc, adaptor.getStart(), dtype);
    Value end = convertScalarToDtype(rewriter, loc, adaptor.getEnd(), dtype);
    Value step = convertScalarToDtype(rewriter, loc, adaptor.getStep(), dtype);

    // The result will always be a 1-d tensor.
    // The size of the result is calculated as follows:
    //          ceil((end - start)/step)
    Value resultShape;
    if (isa<mlir::IntegerType>(dtype)) {
      Value subOut = arith::SubIOp::create(rewriter, loc, end, start);
      resultShape = arith::CeilDivSIOp::create(rewriter, loc, subOut, step);
    } else {
      Value subOut = arith::SubFOp::create(rewriter, loc, end, start);
      Value divOut = arith::DivFOp::create(rewriter, loc, subOut, step);
      Value ceilOut = math::CeilOp::create(rewriter, loc, divOut);
      resultShape = arith::FPToUIOp::create(rewriter, loc,
                                            rewriter.getI64Type(), ceilOut);
    }
    resultShape = castIntToIndex(rewriter, loc, resultShape);

    Value resultTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(resultShape), dtype);

    auto iteratorType = utils::IteratorType::parallel;
    AffineMap indexingMap =
        AffineMap::getMultiDimIdentityMap(1, op->getContext());

    Value finalRes =
        linalg::GenericOp::create(
            rewriter, loc, /*resultTensorTypes=*/resultTensor.getType(),
            /*inputs=*/ValueRange({}),
            /*outputs=*/resultTensor,
            /*indexingMaps=*/indexingMap,
            /*iteratorTypes=*/iteratorType,
            [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
              Value index = linalg::IndexOp::create(b, loc, 0);
              index = castIndexToInt64(b, loc, index);
              index = convertScalarToDtype(b, loc, index, dtype);
              Value mulOut, result;
              if (isa<mlir::FloatType>(dtype)) {
                mulOut = arith::MulFOp::create(b, loc, step, index);
                result = arith::AddFOp::create(b, loc, start, mulOut);
              } else {
                mulOut = arith::MulIOp::create(b, loc, step, index);
                result = arith::AddIOp::create(b, loc, start, mulOut);
              }
              linalg::YieldOp::create(b, loc, result);
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
  target.addIllegalOp<AtenReplicationPad3dOp>();
  patterns.add<ConvertAtenReplicationPad3dOp>(typeConverter, context);
  target.addIllegalOp<AtenReplicationPad2dOp>();
  patterns.add<ConvertAtenReplicationPad2dOp>(typeConverter, context);
  target.addIllegalOp<AtenReplicationPad1dOp>();
  patterns.add<ConvertAtenReplicationPad1dOp>(typeConverter, context);
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
