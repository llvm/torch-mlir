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

// Checks the validity of pooling parameters and stores them in the respective
// vector.
template <typename OpTy>
static LogicalResult
checkAndGetPoolingParameters(OpTy op, ConversionPatternRewriter &rewriter,
                             bool &ceilMode,
                             SmallVectorImpl<int64_t> &kernelSizeInts,
                             SmallVectorImpl<int64_t> &strideInts,
                             SmallVectorImpl<int64_t> &paddingInts) {
  // Pattern match against the op's original operands, because otherwise we
  // will get the lowered version of the operands which is harder to pattern
  // match.
  if (!matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSizeInts)))
    return rewriter.notifyMatchFailure(op, "only support kernel size ints");
  if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
    return rewriter.notifyMatchFailure(op, "only support constant int strides");
  if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant int paddings");
  if (!matchPattern(op.ceil_mode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant bool ceil_mode");
  return success();
}

// Creates a pooling operation based on the type specified by `OpTy` and
// arguments passed.
template <typename OpTy>
static LogicalResult createPoolingOp(
    Operation *op, ConversionPatternRewriter &rewriter, Value self,
    bool supportFPInputOnly, bool ceilMode,
    SmallVectorImpl<int64_t> &kernelSizeInts,
    SmallVectorImpl<int64_t> &strideInts, SmallVectorImpl<int64_t> &paddingInts,
    SmallVectorImpl<int64_t> &dilationInts, Attribute initValueAttr,
    SmallVectorImpl<Value> &outTensorShape, Value &paddedInput, Value &result) {
  Location loc = op->getLoc();
  Type elementType = self.getType().cast<RankedTensorType>().getElementType();
  if (!elementType.isa<mlir::FloatType>() && !supportFPInputOnly)
    return op->emitError("unimplemented: non-floating point type");

  SmallVector<int64_t, 4> lowPaddingIncludingNC = {0, 0};
  lowPaddingIncludingNC.append(paddingInts);
  SmallVector<int64_t, 4> highPaddingIncludingNC = lowPaddingIncludingNC;
  if (ceilMode) {
    highPaddingIncludingNC[2] += strideInts[0];
    highPaddingIncludingNC[3] += strideInts[1];
  }
  Value initValue = rewriter.create<arith::ConstantOp>(loc, initValueAttr);
  paddedInput = torch_to_linalg::getPaddedTensor(
      op, rewriter, self, lowPaddingIncludingNC, highPaddingIncludingNC,
      initValue);

  Value N = getDimOp(rewriter, loc, self, 0);
  Value C = getDimOp(rewriter, loc, self, 1);
  Value H = getDimOp(rewriter, loc, self, 2);
  Value W = getDimOp(rewriter, loc, self, 3);

  SmallVector<Value> paddingIntValues =
      getAsConstantIntValues(rewriter, loc, paddingInts);
  SmallVector<Value> dilationIntValues =
      getAsConstantIntValues(rewriter, loc, dilationInts);
  SmallVector<Value> kernelSizeIntValues =
      getAsConstantIntValues(rewriter, loc, kernelSizeInts);
  SmallVector<Value> strideIntValues =
      getAsConstantIntValues(rewriter, loc, strideInts);

  Value hOut = torch_to_linalg::getOutputDimForConvOps(
      rewriter, loc, H, paddingIntValues[0], dilationIntValues[0],
      kernelSizeIntValues[0], strideIntValues[0], ceilMode);
  Value wOut = torch_to_linalg::getOutputDimForConvOps(
      rewriter, loc, W, paddingIntValues[1], dilationIntValues[1],
      kernelSizeIntValues[1], strideIntValues[1], ceilMode);

  // Create output tensor initialized with smallest floating point value.
  outTensorShape.insert(outTensorShape.begin(), {N, C, hOut, wOut});
  Value outTensorInitialized =
      createInitTensor(rewriter, loc, outTensorShape, elementType, initValue);

  auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
  auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
  Value windowTensor = rewriter.create<linalg::InitTensorOp>(
      loc, getAsConstantIndexValues(rewriter, loc, kernelSizeInts),
      elementType);

  result = rewriter
               .create<OpTy>(loc, outTensorInitialized.getType(),
                             ValueRange{paddedInput, windowTensor},
                             outTensorInitialized, stridesAttr, dilationAttr)
               .getResult(0);

  return success();
}

namespace {
class ConvertAtenMaxPool2dOp : public OpConversionPattern<AtenMaxPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Value self = adaptor.self();
    int64_t selfRank = self.getType().cast<RankedTensorType>().getRank();
    // TODO: Add support for 3D inputs.
    if (selfRank == 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only support 4D input");

    SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts,
        dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    bool ceilMode;
    if (failed(checkAndGetPoolingParameters<AtenMaxPool2dOp>(
            op, rewriter, ceilMode, kernelSizeInts, strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    Type elementType = self.getType().cast<RankedTensorType>().getElementType();
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    SmallVector<Value, 4> outTensorShape;
    // `maxpool2d` contains the result of maxpool2d operation over the input.
    Value maxPool2d, paddedInput;
    if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
            op, rewriter, self, /*supportFPInput=*/false, ceilMode,
            kernelSizeInts, strideInts, paddingInts, dilationInts,
            smallestFPValueAttr, outTensorShape, paddedInput, maxPool2d)))
      return rewriter.notifyMatchFailure(op, "unable to compute maxpool2d");
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, maxPool2d);
    return success();
  }
};
} // namespace

namespace {
// Returns the result of maxpool2d over the input tensor. And the corresponding
// indices of the input tensor for the values of the result tensor.
//
// The result of the maxpool2d operation is calculated using the helper function
// written above. For finding the indices, we follow the below method:
//
// Let's say the input tensor is a 4-d tensor. The maxpool2d and indices will
// also be a 4-d tensor. Then:
// for i in range(N):
//     for j in range(C):
//         for m in range(Hout):
//             for n in range(Wout):
//                 for p in range(kH):
//                     for r in range(kW):
//                         indexH = m * stride[0] + p * dilation[0]
//                         indexW = n * stride[0] + r * dilation[0]
//                         if paddedInput[i, j, indexH, indexW] ==
//                            maxPool2d[i, j, m, n]:
//                             indices[i, j, m, n] = (indexH - padding[0]) * W +
//                                                   (indexW - padding[1])
//
class ConvertAtenMaxPool2dWithIndicesOp
    : public OpConversionPattern<AtenMaxPool2dWithIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dWithIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    RankedTensorType selfType = self.getType().cast<RankedTensorType>();
    Type elementType = selfType.getElementType();
    RankedTensorType indicesRankedTensorType =
        getTypeConverter()
            ->convertType(op->getResult(1).getType())
            .cast<RankedTensorType>();

    // TODO: Add support for 3D inputs.
    if (selfType.getRank() == 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only support 4D input");

    SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts,
        dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    bool ceilMode;
    if (failed(checkAndGetPoolingParameters<AtenMaxPool2dWithIndicesOp>(
            op, rewriter, ceilMode, kernelSizeInts, strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    // `maxpool2d` contains the result of maxpool2d operation over the input.
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value maxPool2d, paddedInput;
    SmallVector<Value, 4> outTensorShape;
    if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
            op, rewriter, self, /*supportFPInput=*/false, ceilMode,
            kernelSizeInts, strideInts, paddingInts, dilationInts,
            smallestFPValueAttr, outTensorShape, paddedInput, maxPool2d)))
      return rewriter.notifyMatchFailure(op, "unable to compute maxpool2d");

    Value cstMinusOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(-1));
    Value indicesTensor =
        createInitTensor(rewriter, loc, outTensorShape,
                         indicesRankedTensorType.getElementType(), cstMinusOne);

    SmallVector<Value> kernelSize =
        getAsConstantIndexValues(rewriter, loc, kernelSizeInts);
    SmallVector<Value> padding =
        getAsConstantIndexValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilation =
        getAsConstantIndexValues(rewriter, loc, dilationInts);
    SmallVector<Value> stride =
        getAsConstantIndexValues(rewriter, loc, strideInts);

    Value windowTensor = rewriter.create<linalg::InitTensorOp>(
        loc, kernelSize, indicesRankedTensorType.getElementType());

    SmallVector<AffineExpr> inputExprs, outputExprs, kernelExprs;
    for (unsigned i = 0; i < 4; i++) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    kernelExprs.push_back(rewriter.getAffineDimExpr(4));
    kernelExprs.push_back(rewriter.getAffineDimExpr(5));

    // Here we have six dimensions, each corresponding to N, C, Hout, Wout, kH,
    // and kW, respectively, as described in the algorithm above.
    SmallVector<AffineMap> indexingMaps =
        AffineMap::inferFromExprList({inputExprs, kernelExprs, outputExprs});
    SmallVector<StringRef> iteratorTypes(4, getParallelIteratorTypeName());
    iteratorTypes.push_back(getReductionIteratorTypeName());
    iteratorTypes.push_back(getReductionIteratorTypeName());

    // Input format is : [N, C, H, W]
    Value inputShapeW = getDimOp(rewriter, loc, self, 3);

    Value indicesResult =
        rewriter
            .create<linalg::GenericOp>(
                loc, /*resultTensorTypes=*/indicesTensor.getType(),
                /*inputs=*/ValueRange({maxPool2d, windowTensor}),
                /*outputs=*/indicesTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value maxVal = args[0], res = args[2];

                  Value i = b.create<linalg::IndexOp>(loc, 0);
                  Value j = b.create<linalg::IndexOp>(loc, 1);
                  Value m = b.create<linalg::IndexOp>(loc, 2);
                  Value n = b.create<linalg::IndexOp>(loc, 3);
                  Value p = b.create<linalg::IndexOp>(loc, 4);
                  Value r = b.create<linalg::IndexOp>(loc, 5);

                  Value mTimesStride =
                      b.create<arith::MulIOp>(loc, m, stride[0]);
                  Value pTimesDilation =
                      b.create<arith::MulIOp>(loc, p, dilation[0]);
                  Value indexH = b.create<arith::AddIOp>(loc, mTimesStride,
                                                         pTimesDilation);
                  Value nTimesStride =
                      b.create<arith::MulIOp>(loc, n, stride[1]);
                  Value rTimesDilation =
                      b.create<arith::MulIOp>(loc, r, dilation[1]);
                  Value indexW = b.create<arith::AddIOp>(loc, nTimesStride,
                                                         rTimesDilation);
                  Value input = b.create<tensor::ExtractOp>(
                      loc, paddedInput, ValueRange{i, j, indexH, indexW});
                  Value pred = b.create<arith::CmpFOp>(
                      loc, arith::CmpFPredicate::OEQ, input, maxVal);

                  Value indexHMinusPadding =
                      b.create<arith::SubIOp>(loc, indexH, padding[0]);
                  Value indexWMinusPadding =
                      b.create<arith::SubIOp>(loc, indexW, padding[1]);
                  Value outIndex = b.create<arith::MulIOp>(
                      loc, indexHMinusPadding, inputShapeW);
                  outIndex = b.create<arith::AddIOp>(loc, outIndex,
                                                     indexWMinusPadding);
                  Value result = b.create<arith::SelectOp>(
                      loc, pred, castIndexToInt64(b, loc, outIndex), res);

                  Value predInvalidIndex = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, res, cstMinusOne);
                  Value out = b.create<arith::SelectOp>(loc, predInvalidIndex,
                                                        result, res);

                  b.create<linalg::YieldOp>(loc, out);
                })
            .getResult(0);

    Type maxPool2dResultType =
        getTypeConverter()->convertType(op->getResult(0).getType());
    Type indicesResultType =
        getTypeConverter()->convertType(op->getResult(1).getType());
    Value outMaxpool2d =
        rewriter.create<tensor::CastOp>(loc, maxPool2dResultType, maxPool2d);
    Value outIndices =
        rewriter.create<tensor::CastOp>(loc, indicesResultType, indicesResult);

    rewriter.replaceOp(op, {outMaxpool2d, outIndices});
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAdaptiveAvgPool2dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.self(); /* in form of N*C*H*W */
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type elementType = inputType.getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    auto inputRank = inputType.getRank();
    if (inputRank != 4)
      return rewriter.notifyMatchFailure(op, "input should be rank 4");

    SmallVector<int64_t, 2> expects{1, 1};
    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    if (!isConstantIntListMatching(op.output_size(), expects))
      return rewriter.notifyMatchFailure(
          op, "only support output_size with H and W both equal to constant 1");

    Value N = getDimOp(rewriter, loc, input, 0);
    Value C = getDimOp(rewriter, loc, input, 1);
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, C}, elementType);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
    Value initTensor0 =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);

    SmallVector<AffineExpr, 2> ncExprs;
    ncExprs.push_back(mlir::getAffineDimExpr(0, context));
    ncExprs.push_back(mlir::getAffineDimExpr(1, context));
    auto ncIndexingMap = AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0, ncExprs, context);
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(4), // input
        ncIndexingMap,                      // output
    };
    SmallVector<StringRef, 4> iteratorTypesSum{"parallel", "parallel",
                                               "reduction", "reduction"};
    Value sumPool2d = rewriter
                          .create<linalg::GenericOp>(
                              loc, initTensor0.getType(), input, initTensor0,
                              /*indexingMaps=*/indexingMaps,
                              /*iteratorTypes=*/iteratorTypesSum,
                              [&](OpBuilder &b, Location loc, ValueRange args) {
                                Value input = args[0], sum = args[1];
                                Value result = rewriter.create<arith::AddFOp>(
                                    loc, sum, input);
                                b.create<linalg::YieldOp>(loc, result);
                              })
                          .getResult(0);

    // Calculate H*W so that avg can be got from sum / (H*W)
    Value H = getDimOp(rewriter, loc, input, 2);
    Value W = getDimOp(rewriter, loc, input, 3);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(
          loc, IntegerType::get(context, 64), v);
    };
    Value HtimesW = rewriter.create<arith::MulIOp>(loc, castIndexToInt(H),
                                                   castIndexToInt(W));
    Value HtimesWf =
        rewriter.create<arith::SIToFPOp>(loc, elementType, HtimesW);

    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
    Value outputTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, C, c1Index, c1Index}, elementType);
    SmallVector<AffineMap, 2> indexingMapsAvg{
        ncIndexingMap, rewriter.getMultiDimIdentityMap(4)};
    SmallVector<StringRef, 4> iteratorTypesAvg(4, "parallel");
    Value avgPool2d =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTensor.getType(), sumPool2d, outputTensor,
                /*indexingMaps=*/indexingMapsAvg,
                /*iteratorTypes=*/iteratorTypesAvg,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value avg = b.create<arith::DivFOp>(loc, args[0], HtimesWf);
                  b.create<linalg::YieldOp>(loc, avg);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, avgPool2d);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAvgPool2dOp : public OpConversionPattern<AtenAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();

    Type inputElementType =
        self.getType().cast<RankedTensorType>().getElementType();
    Type resultType = getTypeConverter()->convertType(op.getType());
    Type resultElementType =
        resultType.cast<RankedTensorType>().getElementType();

    SmallVector<int64_t, 2> dilationInts{1, 1};
    SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts;
    bool ceilMode;
    if (failed(checkAndGetPoolingParameters<AtenAvgPool2dOp>(
            op, rewriter, ceilMode, kernelSizeInts, strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    // TODO: Add support for count_include_pad equal to `False`.
    bool countIncludePad;
    if (!matchPattern(op.count_include_pad(),
                      m_TorchConstantBool(&countIncludePad)))
      return rewriter.notifyMatchFailure(
          op, "count_include_pad must be a constant");
    if (!countIncludePad) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: count_include_pad is expected to be true");
    }

    // `sumPool2d` contains the result of sumpool2d operation over the input.
    Value sumPool2d, paddedInput;
    SmallVector<Value, 4> outTensorShape;
    if (failed(createPoolingOp<linalg::PoolingNchwSumOp>(
            op, rewriter, self, /*supportFPInput=*/true, ceilMode,
            kernelSizeInts, strideInts, paddingInts, dilationInts,
            rewriter.getZeroAttr(inputElementType), outTensorShape, paddedInput,
            sumPool2d)))
      return rewriter.notifyMatchFailure(op, "unable to compute sumpool2d");

    SmallVector<Value> kernelSizeIntValues =
        getAsConstantIntValues(rewriter, loc, kernelSizeInts);
    Value kHtimeskW = rewriter.create<arith::MulIOp>(
        loc, kernelSizeIntValues[0], kernelSizeIntValues[1]);
    Value divisor = op.divisor_override().getType().isa<Torch::NoneType>()
                        ? kHtimeskW
                        : adaptor.divisor_override();
    divisor = convertScalarToDtype(rewriter, loc, divisor, resultElementType);

    Value outputTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outTensorShape, resultElementType);
    SmallVector<AffineMap> indexingMapsAvg(2,
                                           rewriter.getMultiDimIdentityMap(4));
    SmallVector<StringRef> iteratorTypesAvg(4, "parallel");

    Value avgPool2d =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTensor.getType(), sumPool2d, outputTensor,
                /*indexingMaps=*/indexingMapsAvg,
                /*iteratorTypes=*/iteratorTypesAvg,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value avg;
                  if (resultElementType.isa<mlir::IntegerType>())
                    avg = b.create<arith::DivSIOp>(loc, args[0], divisor);
                  else if (resultElementType.isa<mlir::FloatType>())
                    avg = b.create<arith::DivFOp>(loc, args[0], divisor);
                  b.create<linalg::YieldOp>(loc, avg);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, avgPool2d);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populatePoolingPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMaxPool2dOp>();
  patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
  target.addIllegalOp<AtenMaxPool2dWithIndicesOp>();
  patterns.add<ConvertAtenMaxPool2dWithIndicesOp>(typeConverter, context);
  target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
  patterns.add<ConvertAtenAdaptiveAvgPool2dOp>(typeConverter, context);
  target.addIllegalOp<AtenAvgPool2dOp>();
  patterns.add<ConvertAtenAvgPool2dOp>(typeConverter, context);
}
