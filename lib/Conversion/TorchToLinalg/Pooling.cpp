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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
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
                             const TypeConverter *typeConverter, bool &ceilMode,
                             SmallVectorImpl<Value> &kernelSizeIntValues,
                             SmallVectorImpl<int64_t> &strideInts,
                             SmallVectorImpl<int64_t> &paddingInts) {
  // Pattern match against the op's original operands, because otherwise we
  // will get the lowered version of the operands which is harder to pattern
  // match.
  SmallVector<Value> kernelSizeTorchInt;
  if (!getListConstructElements(op.getKernelSize(), kernelSizeTorchInt)) {
    return rewriter.notifyMatchFailure(op,
                                       "unimplemented: the kernel size is "
                                       "not constructed from ListConstruct");
  }
  kernelSizeIntValues = getTypeConvertedValues(
      rewriter, op.getLoc(), typeConverter, kernelSizeTorchInt);

  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInts)))
    return rewriter.notifyMatchFailure(op, "only support constant int strides");
  // If `stride` is not specified by the user, it is assigned the value of empty
  // list during import. For such a case, the stride value is the kernel size.
  // See:
  // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
  if (strideInts.empty()) {
    if (!matchPattern(op.getKernelSize(),
                      m_TorchListOfConstantInts(strideInts))) {
      return rewriter.notifyMatchFailure(
          op, "if stride is the empty list, kernel_size must be a list of "
              "constant ints");
    }
  }

  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingInts)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant int paddings");
  if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant bool ceil_mode");
  return success();
}

// Creates a pooling operation based on the type specified by `OpTy` and
// arguments passed.
template <typename OpTy>
static LogicalResult createPoolingOp(
    Operation *op, ConversionPatternRewriter &rewriter, Value self,
    bool supportNonFPInput, bool ceilMode, int64_t dimensionality,
    SmallVectorImpl<Value> &kernelSizeIntValues,
    SmallVectorImpl<int64_t> &strideInts, SmallVectorImpl<int64_t> &paddingInts,
    SmallVectorImpl<int64_t> &dilationInts, Attribute initValueAttr,
    SmallVectorImpl<Value> &outTensorShape, Value &paddedInput, Value &result) {
  Location loc = op->getLoc();
  Type elementType = self.getType().cast<RankedTensorType>().getElementType();
  if (!elementType.isa<mlir::FloatType>() && !supportNonFPInput)
    return op->emitError("unimplemented: non-floating point type");

  SmallVector<int64_t> lowPaddingIncludingNC = {0, 0};
  lowPaddingIncludingNC.append(paddingInts);
  SmallVector<int64_t> highPaddingIncludingNC = lowPaddingIncludingNC;

  if (ceilMode) {
    for (int64_t i = 0; i < dimensionality; ++i) {
      highPaddingIncludingNC[i + 2] += strideInts[i];
    }
  }

  Value initValue =
      rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValueAttr));
  paddedInput = torch_to_linalg::getPaddedTensor(
      op, rewriter, self, lowPaddingIncludingNC, highPaddingIncludingNC,
      initValue);

  Value N = getDimOp(rewriter, loc, self, 0);
  Value C = getDimOp(rewriter, loc, self, 1);

  SmallVector<Value> paddingIntValues =
      getAsConstantIntValues(rewriter, loc, paddingInts);
  SmallVector<Value> dilationIntValues =
      getAsConstantIntValues(rewriter, loc, dilationInts);
  SmallVector<Value> strideIntValues =
      getAsConstantIntValues(rewriter, loc, strideInts);

  // Get dimension size for each dimension and calculate output size
  for (int64_t i = dimensionality - 1; i > -1; --i) {
    Value dimSize = getDimOp(rewriter, loc, self, i + 2);
    Value outDim = torch_to_linalg::getOutputDimForConvOps(
        rewriter, loc, dimSize, paddingIntValues[i], dilationIntValues[i],
        kernelSizeIntValues[i], strideIntValues[i], ceilMode);
    outTensorShape.insert(outTensorShape.begin(), {outDim});
  }

  // Create output tensor initialized with smallest floating point value.
  outTensorShape.insert(outTensorShape.begin(), {N, C});
  Value outTensorInitialized =
      createInitTensor(rewriter, loc, outTensorShape, elementType, initValue);

  auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
  auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
  auto shape = castIntVectorToIndexVector(rewriter, loc, kernelSizeIntValues);
  Value windowTensor = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(shape), elementType);

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

    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
    int64_t selfRank = self.getType().cast<RankedTensorType>().getRank();
    // TODO: Add support for 3D inputs.
    if (selfRank == 3)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only support 4D input");

    bool ceilMode;
    SmallVector<Value, 2> kernelSizeIntValues;
    SmallVector<int64_t, 2> strideInts, paddingInts, dilationInts;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    if (failed(checkAndGetPoolingParameters<AtenMaxPool2dOp>(
            op, rewriter, typeConverter, ceilMode, kernelSizeIntValues,
            strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    Type elementType = self.getType().cast<RankedTensorType>().getElementType();
    TypedAttr smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(elementType.cast<mlir::FloatType>().getFloatSemantics(),
                        /*Negative=*/true));
    SmallVector<Value, 4> outTensorShape;
    // `maxpool2d` contains the result of maxpool2d operation over the input.
    Value maxPool2d, paddedInput;
    if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
            op, rewriter, self, /*supportNonFPInput=*/false, ceilMode,
            /*dimensionality=*/2, kernelSizeIntValues, strideInts, paddingInts,
            dilationInts, smallestFPValueAttr, outTensorShape, paddedInput,
            maxPool2d)))
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
    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
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

    bool ceilMode;
    SmallVector<Value, 2> kernelSizeIntValues;
    SmallVector<int64_t, 2> strideInts, paddingInts, dilationInts;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    if (failed(checkAndGetPoolingParameters<AtenMaxPool2dWithIndicesOp>(
            op, rewriter, typeConverter, ceilMode, kernelSizeIntValues,
            strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    // `maxpool2d` contains the result of maxpool2d operation over the input.
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(elementType.cast<mlir::FloatType>().getFloatSemantics(),
                        /*Negative=*/true));
    Value maxPool2d, paddedInput;
    SmallVector<Value, 4> outTensorShape;
    if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
            op, rewriter, self, /*supportNonFPInput=*/false, ceilMode,
            /*dimensionality=*/2, kernelSizeIntValues, strideInts, paddingInts,
            dilationInts, smallestFPValueAttr, outTensorShape, paddedInput,
            maxPool2d)))
      return rewriter.notifyMatchFailure(op, "unable to compute maxpool2d");

    Value cstMinusOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(-1));
    Value indicesTensor =
        createInitTensor(rewriter, loc, outTensorShape,
                         indicesRankedTensorType.getElementType(), cstMinusOne);

    SmallVector<Value> kernelSize =
        castIntVectorToIndexVector(rewriter, loc, kernelSizeIntValues);
    SmallVector<Value> padding =
        getAsConstantIndexValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilation =
        getAsConstantIndexValues(rewriter, loc, dilationInts);
    SmallVector<Value> stride =
        getAsConstantIndexValues(rewriter, loc, strideInts);

    Value windowTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(kernelSize),
        indicesRankedTensorType.getElementType());

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
    SmallVector<utils::IteratorType> iteratorTypes(
        4, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);
    iteratorTypes.push_back(utils::IteratorType::reduction);

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
template <typename OpTy, typename PoolingOpTy, int Dim>
class ConvertAtenAvgPoolOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();
    Value self = adaptor.getSelf();

    Type inputElementType =
        self.getType().cast<RankedTensorType>().getElementType();
    Type resultType = typeConverter->convertType(op.getType());
    Type resultElementType =
        resultType.cast<RankedTensorType>().getElementType();

    bool ceilMode;
    SmallVector<Value, Dim> kernelSizeIntValues;
    SmallVector<int64_t, Dim> strideInts, paddingInts, dilationInts(Dim, 1);
    if (failed(checkAndGetPoolingParameters<OpTy>(op, rewriter, typeConverter,
                                                  ceilMode, kernelSizeIntValues,
                                                  strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    // TODO: Add support for count_include_pad equal to `False`.
    bool countIncludePad;
    if (!matchPattern(op.getCountIncludePad(),
                      m_TorchConstantBool(&countIncludePad)))
      return rewriter.notifyMatchFailure(
          op, "count_include_pad must be a constant");
    if (!countIncludePad) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: count_include_pad is expected to be true");
    }

    // `sumPool` contains the result of sumpool operation over the input.
    Value sumPool, paddedInput;
    SmallVector<Value, Dim + 2> outTensorShape;
    if (failed(createPoolingOp<PoolingOpTy>(
            op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
            /*dimensionality=*/Dim, kernelSizeIntValues, strideInts,
            paddingInts, dilationInts, rewriter.getZeroAttr(inputElementType),
            outTensorShape, paddedInput, sumPool)))
      return rewriter.notifyMatchFailure(op, "unable to compute sumpool");
    Value divisor;
    if constexpr (std::is_same<OpTy, AtenAvgPool2dOp>()) {
      Value kHtimeskW = rewriter.create<arith::MulIOp>(
          loc, kernelSizeIntValues[0], kernelSizeIntValues[1]);
      divisor =
          op.getDivisorOverride().getType().template isa<Torch::NoneType>()
              ? kHtimeskW
              : adaptor.getDivisorOverride();
    } else {
      divisor = kernelSizeIntValues[0];
    }
    divisor = convertScalarToDtype(rewriter, loc, divisor, resultElementType);

    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outTensorShape), resultElementType);
    SmallVector<AffineMap> indexingMapsAvg(
        2, rewriter.getMultiDimIdentityMap(Dim + 2));
    SmallVector<utils::IteratorType> iteratorTypesAvg(
        Dim + 2, utils::IteratorType::parallel);
    Value avgPool =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTensor.getType(), sumPool, outputTensor,
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

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, avgPool);
    return success();
  }
};
} // namespace

/*
This section is for lowering adaptive pooling ops, which cannot generally be
decomposed into typical pooling ops. Given an input tensor of rank (N,C,Hin) and
an output spatial size Hout, an element of the output tensor at position (n, c,
h) is computed as follows.
    1. compute st(h) = (h*Hin)//Hout
    2. compute en(h) = 1 + ((h+1)*Hin - 1)//Hout
    3. apply the operation (max or avg) over input[n, c, st(h):en(h)]
This is problematic for linalg ops for a few reasons:
    1. The access to the input tensor is not constantly strided
    2. The size of the window itself is not contant: en(h) - st(h) can vary with
h! Although it is a bit like using a hammer to paint, our workaround is to use
tensor.extract to access the elements of the input tensor inside our linalg
generic op's payload.

Current TODO's:
    1. gather most of the boilerplate out of this op and make it into an
adaptive pooling helper function.
    2. figure out what to do with the conflicting decompositions in
DecomposeComplexOps.cpp
    3. Implement more efficient passes for when the kernel-size, input spatial
dims, and output spatial dims are constant.
*/

namespace {
class ConvertAtenAdaptiveAvgPool1dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool1dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool1dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    const TypeConverter *typeConverter = getTypeConverter();

    // get rank of input (same as rank of output)
    int64_t rank =
        adaptor.getSelf().getType().cast<RankedTensorType>().getRank();
    // input operand should be NCH (i.e. rank 3)
    if (rank != 3) {
      return rewriter.notifyMatchFailure(op, "only supports input type NCH");
    }

    // input tensor and output shape
    Value input = adaptor.getSelf();
    Value outputShape = op.getOutputSize();
    SmallVector<Value> outShapeVector;
    getListConstructElements(outputShape, outShapeVector);
    outShapeVector =
        getTypeConvertedValues(rewriter, loc, typeConverter, outShapeVector);
    Value hIn = getDimOp(rewriter, loc, input, 2);
    Value hOut = outShapeVector[0];
    Value hOutIndex = castIntToIndex(rewriter, loc, hOut);
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType outputType =
        typeConverter->convertType(op.getResult().getType())
            .cast<RankedTensorType>();

    // get elementType of input tensor
    Type elementType = inputType.getElementType();

    // make an iteration space of size kMax = 1 + ceildiv (hIn - 1) , hOut
    Type boolType = rewriter.getI1Type();
    Value kIter;
    Value constantOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value hInPlusOne = rewriter.create<arith::SubIOp>(loc, hIn, constantOne);
    Value kMaxMinusOne =
        rewriter.create<arith::CeilDivSIOp>(loc, hInPlusOne, hOutIndex);
    Value kMax = rewriter.create<arith::AddIOp>(loc, constantOne, kMaxMinusOne);
    kIter = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(ValueRange({kMax})), boolType);

    // need to buffer input, else there will possibly be an out of bounds access
    // later buffVal = 0 for avg pooling and -inf for max pooling
    Value buffVal = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));
    SmallVector<int64_t> lowPadding = {0, 0, 0};
    SmallVector<int64_t> highPadding = {0, 0, 1};
    Value buffInput = torch_to_linalg::getPaddedTensor(
        op, rewriter, input, lowPadding, highPadding, buffVal);

    // make a list of outputSizes
    SmallVector<Value> outputSizes;
    for (unsigned i = 0; i < rank - 1; i++) {
      outputSizes.push_back(getDimOp(rewriter, loc, input, i));
    }
    outputSizes.push_back(hOutIndex);

    // initialize a kernel size tensor (only for avg pooling)
    Value kSizeTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(ValueRange({hOutIndex})), elementType);

    // initialize an output tensor
    Value initOutput =
        createInitTensor(rewriter, loc, outputSizes, elementType, buffVal);

    // setup indexing maps and iterator types for linalg generic op
    // for kIter (d0,d1,d2,d3) -> (d3)
    // for output (d0,d1,d2,d3) -> (d0,d1,d2)
    // for kSizeTensor (d0,d1,d2,d3) -> (d2)
    SmallVector<AffineExpr> kIterExprs, outputExprs, kSizeTensorExprs;
    for (unsigned i = 0; i < 3; i++) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    kSizeTensorExprs.push_back(rewriter.getAffineDimExpr(2));
    kIterExprs.push_back(rewriter.getAffineDimExpr(3));
    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {kIterExprs, outputExprs, kSizeTensorExprs});
    SmallVector<utils::IteratorType> iteratorTypes(
        3, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);

    Value indexOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto sumPool = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/
        TypeRange({initOutput.getType(), kSizeTensor.getType()}),
        /*inputs=*/ValueRange({kIter}),
        /*outputs=*/ValueRange({initOutput, kSizeTensor}),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value res = args[1];
          Value ind0 = b.create<linalg::IndexOp>(loc, 0);
          Value ind1 = b.create<linalg::IndexOp>(loc, 1);
          Value ind2 = b.create<linalg::IndexOp>(loc, 2);
          Value ind3 = b.create<linalg::IndexOp>(loc, 3);
          // compute start and end indices
          // st = s1( s0(ind2 * Hin) // Hout )
          Value s0 = b.create<arith::MulIOp>(loc, ind2, hIn);
          Value s1 = b.create<arith::FloorDivSIOp>(loc, s0, hOutIndex);
          // en = e4( 1 + e3( e2( e1( e0(ind2 + 1) * hIn ) - 1 ) // hOut ) )
          Value e0 = b.create<arith::AddIOp>(loc, ind2, indexOne);
          Value e1 = b.create<arith::MulIOp>(loc, e0, hIn);
          Value e2 = b.create<arith::SubIOp>(loc, e1, indexOne);
          Value e3 = b.create<arith::FloorDivSIOp>(loc, e2, hOutIndex);
          Value e4 = b.create<arith::AddIOp>(loc, indexOne, e3);
          // get input element @ st + ind3:
          Value wIndex = b.create<arith::AddIOp>(loc, s1, ind3);
          Value inElt = b.create<tensor::ExtractOp>(
              loc, elementType, buffInput, ValueRange({ind0, ind1, wIndex}));
          // check if we extracted at windex < end index
          Value cond =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate(6), wIndex, e4);
          // if inElt is in bounds, include it in the computation
          // else, use buffVal = 0 (for max pool use -infinity)
          Value out1 = b.create<arith::SelectOp>(loc, cond, inElt, buffVal);
          // compute Kernel size: we store this to kwTensor
          Value kSize = b.create<arith::SubIOp>(loc, e4, s1);
          Value kSizeInt = castIndexToInt64(b, loc, kSize);
          Value kSizeF = b.create<arith::SIToFPOp>(loc, elementType, kSizeInt);
          // accumulate out2 to res = args[1]
          Value out2 = b.create<arith::AddFOp>(loc, res, out1);
          b.create<linalg::YieldOp>(loc, ValueRange({out2, kSizeF}));
        });

    // make a linalg generic to divide each element by the corresponding
    // Kernel Width. This step is only necessary for avg pooling.
    SmallVector<AffineMap> indexingMaps1 =
        AffineMap::inferFromExprList({kSizeTensorExprs, outputExprs});
    SmallVector<utils::IteratorType> iteratorTypes1(
        3, utils::IteratorType::parallel);
    auto output = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/initOutput.getType(),
        /*inputs=*/sumPool.getResultTensors()[1],
        /*outputs=*/sumPool.getResultTensors()[0],
        /*indexingMaps=*/indexingMaps1,
        /*iteratorTypes=*/iteratorTypes1,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value q = b.create<arith::DivFOp>(loc, args[1], args[0]);
          b.create<linalg::YieldOp>(loc, q);
        });

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType,
                                                output.getResultTensors());
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
  target.addIllegalOp<AtenAvgPool1dOp, AtenAvgPool2dOp>();
  patterns
      .add<ConvertAtenAvgPoolOp<AtenAvgPool1dOp, linalg::PoolingNcwSumOp, 1>>(
          typeConverter, context);
  patterns
      .add<ConvertAtenAvgPoolOp<AtenAvgPool2dOp, linalg::PoolingNchwSumOp, 2>>(
          typeConverter, context);
  target.addIllegalOp<AtenAdaptiveAvgPool1dOp>();
  patterns.add<ConvertAtenAdaptiveAvgPool1dOp>(typeConverter, context);
}
