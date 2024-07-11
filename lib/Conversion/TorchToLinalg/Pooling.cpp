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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
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

static Value
computeOutputTensor(Operation *op, ConversionPatternRewriter &rewriter,
                    Value self, int64_t dimensionality, bool ceilMode,
                    SmallVectorImpl<int64_t> &strideInts,
                    SmallVectorImpl<int64_t> &paddingInts,
                    SmallVectorImpl<int64_t> &dilationInts,
                    SmallVectorImpl<Value> &kernelSizeIntValues,
                    SmallVectorImpl<Value> &outTensorShape, Value initValue) {
  Type elementType = cast<RankedTensorType>(self.getType()).getElementType();
  Location loc = op->getLoc();

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
  return createInitTensor(rewriter, loc, outTensorShape, elementType,
                          initValue);
}

static Value padInputTensor(Operation *op, ConversionPatternRewriter &rewriter,
                            Value self, bool ceilMode, int64_t dimensionality,
                            SmallVectorImpl<int64_t> &strideInts,
                            SmallVectorImpl<int64_t> &paddingInts,
                            Value initValue) {
  SmallVector<int64_t> lowPaddingIncludingNC = {0, 0};
  SmallVector<int64_t> highPaddingIncludingNC = {0, 0};

  unsigned selfRank = cast<RankedTensorType>(self.getType()).getRank();
  unsigned paddingIntsSize = paddingInts.size();

  if (paddingIntsSize == 2 * (selfRank - 2)) {
    // This condition being true means that the `paddingInts` contain seperate
    // values for low padding and high padding.
    for (unsigned i = 0; i < paddingIntsSize / 2; i++)
      lowPaddingIncludingNC.push_back(paddingInts[i]);
    for (unsigned i = paddingIntsSize / 2; i < paddingIntsSize; i++)
      highPaddingIncludingNC.push_back(paddingInts[i]);
  } else {
    lowPaddingIncludingNC.append(paddingInts);
    highPaddingIncludingNC = lowPaddingIncludingNC;
  }

  if (ceilMode) {
    for (int64_t i = 0; i < dimensionality; ++i) {
      highPaddingIncludingNC[i + 2] += strideInts[i];
    }
  }

  return torch_to_linalg::getPaddedTensor(op, rewriter, self,
                                          lowPaddingIncludingNC,
                                          highPaddingIncludingNC, initValue);
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
  Type elementType = cast<RankedTensorType>(self.getType()).getElementType();
  if (!isa<mlir::FloatType>(elementType) && !supportNonFPInput)
    return op->emitError("unimplemented: non-floating point type");

  Value initValue =
      rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValueAttr));

  paddedInput = padInputTensor(op, rewriter, self, ceilMode, dimensionality,
                               strideInts, paddingInts, initValue);

  auto outTensorInitialized = computeOutputTensor(
      op, rewriter, self, dimensionality, ceilMode, strideInts, paddingInts,
      dilationInts, kernelSizeIntValues, outTensorShape, initValue);

  auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
  auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
  auto shape = castIntVectorToIndexVector(rewriter, loc, kernelSizeIntValues);
  Value windowTensor = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(shape), elementType);

  Value permutedInput = paddedInput, permutedOutput = outTensorInitialized;
  if (dimensionality == 3) {
    // Permute input and output tensor as follows:
    // (n,c,d,h,w) -> (n,d,h,w,c)
    SmallVector<int64_t> dimensions = {0, 2, 3, 4, 1};
    if (failed(torch_to_linalg::permuteTensor(op, rewriter, op->getLoc(),
                                              dimensions, paddedInput,
                                              permutedInput)))
      return rewriter.notifyMatchFailure(
          op, "failed to perform permutation of tensor");

    if (failed(torch_to_linalg::permuteTensor(op, rewriter, op->getLoc(),
                                              dimensions, outTensorInitialized,
                                              permutedOutput)))
      return rewriter.notifyMatchFailure(
          op, "failed to perform permutation of tensor");
  }

  Value poolingResult =
      rewriter
          .create<OpTy>(loc, permutedOutput.getType(),
                        ValueRange{permutedInput, windowTensor}, permutedOutput,
                        stridesAttr, dilationAttr)
          .getResult(0);

  result = poolingResult;
  if (dimensionality == 3) {
    // Permute output tensor as follows:
    // (n,d,h,w,c) -> (n,c,d,h,w)
    SmallVector<int64_t> dimensions = {0, 4, 1, 2, 3};
    if (failed(torch_to_linalg::permuteTensor(
            op, rewriter, op->getLoc(), dimensions, poolingResult, result)))
      return rewriter.notifyMatchFailure(
          op, "failed to perform permutation of tensor");
  }

  return success();
}

namespace {

template <typename T> struct DimensionTraits {};

template <> struct DimensionTraits<AtenMaxPool1dOp> {
  static constexpr int64_t Dim = 1;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <> struct DimensionTraits<AtenMaxPool2dOp> {
  static constexpr int64_t Dim = 2;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <> struct DimensionTraits<AtenMaxPool3dOp> {
  static constexpr int64_t Dim = 3;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <typename OpTy>
class ConvertAtenMaxPoolOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

private:
  static const int64_t Dim = DimensionTraits<OpTy>::Dim;

  LogicalResult createPoolingMax3D(AtenMaxPool3dOp &op,
                                   typename OpTy::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   SmallVectorImpl<Value> &kernelSizeIntValues,
                                   SmallVectorImpl<int64_t> &strideInts,
                                   SmallVectorImpl<int64_t> &paddingInts,
                                   SmallVectorImpl<int64_t> &dilationInts,
                                   bool ceilMode) const {
    SmallVector<Value, 5> outTensorShape;
    Value self = adaptor.getSelf();
    Type elementType = cast<RankedTensorType>(self.getType()).getElementType();
    TypedAttr smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
                        /*Negative=*/true));
    Value initValue =
        rewriter.create<arith::ConstantOp>(op->getLoc(), smallestFPValueAttr);

    Value paddedInput = padInputTensor(op, rewriter, self, ceilMode, 3,
                                       strideInts, paddingInts, initValue);

    auto outTensorInitialized = computeOutputTensor(
        op, rewriter, self, 3, ceilMode, strideInts, paddingInts, dilationInts,
        kernelSizeIntValues, outTensorShape, initValue);

    auto shape =
        castIntVectorToIndexVector(rewriter, op->getLoc(), kernelSizeIntValues);
    Value windowTensor = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), getAsOpFoldResult(shape), elementType);

    MLIRContext *context = rewriter.getContext();

    auto mapInput = mlir::AffineMap::get(
        8, 0,
        {
            rewriter.getAffineDimExpr(0), // n
            rewriter.getAffineDimExpr(1), // c
            // dim_d * stride_d + kernal_d * dilation_d
            rewriter.getAffineDimExpr(2) *
                    getAffineConstantExpr(strideInts[0], context) +
                rewriter.getAffineDimExpr(5) *
                    getAffineConstantExpr(dilationInts[0], context),
            // dim_h * stride_h + kernal_h * dilation_h
            rewriter.getAffineDimExpr(3) *
                    getAffineConstantExpr(strideInts[1], context) +
                rewriter.getAffineDimExpr(6) *
                    getAffineConstantExpr(dilationInts[1], context),
            // dim_w * stride_w + kernal_w * dilation_w
            rewriter.getAffineDimExpr(4) *
                    getAffineConstantExpr(strideInts[2], context) +
                rewriter.getAffineDimExpr(7) *
                    getAffineConstantExpr(dilationInts[2], context),
        },
        context);
    auto mapKernel =
        mlir::AffineMap::get(8, 0,
                             {
                                 rewriter.getAffineDimExpr(5), // kd
                                 rewriter.getAffineDimExpr(6), // kh
                                 rewriter.getAffineDimExpr(7)  // kw
                             },
                             context);
    auto mapOutput = mlir::AffineMap::get(
        8, 0,
        {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1),
         rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(3),
         rewriter.getAffineDimExpr(4)},
        context);
    auto iteratorTypes =
        SmallVector<utils::IteratorType>(5, utils::IteratorType::parallel);
    iteratorTypes.append(3, utils::IteratorType::reduction);
    SmallVector<AffineMap> indexingMaps = {mapInput, mapKernel, mapOutput};
    Value poolingOp =
        rewriter
            .create<linalg::GenericOp>(
                op->getLoc(),
                /* result types */ outTensorInitialized.getType(),
                /* operands */ ValueRange({paddedInput, windowTensor}),
                /* outputs */ outTensorInitialized,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value currentVal = args[0], accMaxValue = args[2];
                  Value max_result =
                      b.create<arith::MaximumFOp>(loc, currentVal, accMaxValue);
                  ;
                  b.create<linalg::YieldOp>(loc, max_result);
                })
            .getResult(0);
    Type newResultType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, poolingOp);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    const TypeConverter *typeConverter = this->getTypeConverter();
    Value self = adaptor.getSelf();
    int64_t selfRank = cast<RankedTensorType>(self.getType()).getRank();

    if (selfRank != Dim + 2)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: Does not support inputs with rank");

    bool ceilMode;
    SmallVector<Value, Dim> kernelSizeIntValues;
    SmallVector<int64_t, Dim> strideInts, paddingInts, dilationInts;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    if (failed(checkAndGetPoolingParameters<OpTy>(op, rewriter, typeConverter,
                                                  ceilMode, kernelSizeIntValues,
                                                  strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    Type elementType = cast<RankedTensorType>(self.getType()).getElementType();

    if constexpr (Dim == 1) {
      SmallVector<Value, 4> outTensorShape;
      Value maxPool1d, paddedInput;
      TypedAttr smallestFPValueAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(
              cast<mlir::FloatType>(elementType).getFloatSemantics(),
              /*Negative=*/true));
      if (failed(createPoolingOp<linalg::PoolingNcwMaxOp>(
              op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
              /*dimensionality=*/1, kernelSizeIntValues, strideInts,
              paddingInts, dilationInts, smallestFPValueAttr, outTensorShape,
              paddedInput, maxPool1d)))
        return rewriter.notifyMatchFailure(op, "unable to compute maxpool1d");
      Type newResultType = this->getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, maxPool1d);
      return success();
    } else if constexpr (Dim == 2) {
      SmallVector<Value, 4> outTensorShape;
      // `maxpool2d` contains the result of maxpool2d operation over the input.
      Value maxPool2d, paddedInput;
      TypedAttr smallestFPValueAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(
              cast<mlir::FloatType>(elementType).getFloatSemantics(),
              /*Negative=*/true));
      if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
              op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
              /*dimensionality=*/2, kernelSizeIntValues, strideInts,
              paddingInts, dilationInts, smallestFPValueAttr, outTensorShape,
              paddedInput, maxPool2d)))
        return rewriter.notifyMatchFailure(op, "unable to compute maxpool2d");
      Type newResultType = this->getTypeConverter()->convertType(op.getType());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, maxPool2d);
      return success();
    } else {
      return createPoolingMax3D(op, adaptor, rewriter, kernelSizeIntValues,
                                strideInts, paddingInts, dilationInts,
                                ceilMode);
    }
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
    RankedTensorType selfType = cast<RankedTensorType>(self.getType());
    Type elementType = selfType.getElementType();
    RankedTensorType indicesRankedTensorType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(1).getType()));

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
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
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
    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {inputExprs, kernelExprs, outputExprs}, rewriter.getContext());
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
class ConvertAtenMaxUnpool3dOp final
    : public OpConversionPattern<AtenMaxUnpool3dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxUnpool3dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
    auto selfType = cast<RankedTensorType>(self.getType());

    if (ShapedType::isDynamicShape(selfType.getShape().take_back(3)))
      return rewriter.notifyMatchFailure(op, "input type must be of static shape");

    auto resType = typeConverter->convertType<RankedTensorType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "invalid result type");

    if (ShapedType::isDynamicShape(resType.getShape().take_back(3)))
      return rewriter.notifyMatchFailure(op, "output type must be of static shape");

    SmallVector<int64_t> stride;
    SmallVector<int64_t> padding;

    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))
        return rewriter.notifyMatchFailure(op, "only support constant int strides");

    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
        return rewriter.notifyMatchFailure(op, "only support constant int padding");

    if (stride.size() != 3 || padding.size() != 3)
      return rewriter.notifyMatchFailure(op, "stride and padding must be of size 3");

    int64_t outRank = resType.getRank();
    int64_t NC = outRank - 3;

    Type indexType = rewriter.getIndexType();
    SmallVector<Value> outSizePadded;
    for (auto &&[i, size] : llvm::enumerate(resType.getShape())) {
      if (int64_t(i) < NC) {
        outSizePadded.emplace_back(rewriter.create<tensor::DimOp>(loc, self, i));
        continue;
      }
      int64_t pad = padding[i - NC];

      outSizePadded.emplace_back(rewriter.create<arith::ConstantIndexOp>(loc, size + pad));
    }

    Value indices = adaptor.getIndices();

    auto divUp = [](int64_t v1, int64_t v2) -> int64_t {
      return (v1 + v2 - 1) / v2;
    };

    SmallVector<int64_t> expectedInputShape = llvm::to_vector(resType.getShape().drop_back(3));
    for (auto &&[str, pad, resSize] : llvm::zip_equal(stride, padding, resType.getShape().take_back(3)))
      expectedInputShape.emplace_back(divUp(resSize, str) + pad*2);

    auto padBorder = [&](Value src, ArrayRef<OpFoldResult> low, ArrayRef<OpFoldResult> high) -> Value {
      auto srcType = cast<ShapedType>(src.getType());
      SmallVector<int64_t> newShape;
      for (auto &&[i, l, s, h] : llvm::enumerate(low, srcType.getShape(), high)) {
        if (int64_t(i) < NC) {
          newShape.emplace_back(s);
          continue;
        }
        newShape.emplace_back(*getConstantIntValue(l) + s + *getConstantIntValue(h));
      }

      auto resType = srcType.clone(newShape);
      auto pad = rewriter.create<tensor::PadOp>(loc, resType, src, low, high);
      Region &region = pad.getRegion();
      SmallVector<Type> blockArgTypes(outRank, indexType);
      SmallVector<Location> blockArgLocs(outRank, loc);
      OpBuilder::InsertionGuard g(rewriter);
      Block* block = rewriter.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
      ValueRange blockArgs = block->getArguments();
      SmallVector<Value> indices;
      for (auto &&[i, arg, size] : llvm::enumerate(blockArgs, selfType.getShape())) {
        if (int64_t(i) < NC) {
          indices.emplace_back(arg);
          continue;
        }
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value bound = rewriter.create<arith::ConstantIndexOp>(loc, size - 1);
        Value boundIndex = rewriter.create<arith::SubIOp>(loc, arg, getValueOrCreateConstantIndexOp(rewriter, loc, low[i]));
        Value cmp1 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, boundIndex, bound);
        Value cmp2 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, boundIndex, zero);
        boundIndex = rewriter.create<arith::SelectOp>(loc, cmp1, bound, boundIndex);
        boundIndex = rewriter.create<arith::SelectOp>(loc, cmp2, zero, boundIndex);
        indices.emplace_back(boundIndex);
      }

      Value borderVal = rewriter.create<tensor::ExtractOp>(loc, src, indices);
      rewriter.create<tensor::YieldOp>(loc, borderVal);
      return pad.getResult();
    };

    if (expectedInputShape != selfType.getShape()) {
      // Input tensor sizes are smaller than output size, due to overlapping
      // pooling windows, pad inputs with border values.
      // TODO: this is probably expensive, and it may be possible to solve by
      // cleverly constructing affine maps for the next linalg.generic op,
      // but I'm not smart enough to figure this out.

      SmallVector<OpFoldResult> low(outRank, rewriter.getI64IntegerAttr(0));
      SmallVector<OpFoldResult> high;
      for (auto &&[i, inpSize, outSize] : llvm::enumerate(selfType.getShape(), expectedInputShape)) {
        if (int64_t(i) < NC) {
          high.emplace_back(rewriter.getI64IntegerAttr(0));
          continue;
        }
        high.emplace_back(rewriter.getI64IntegerAttr(outSize - inpSize));
      }

      self = padBorder(self, low, high);
      indices = padBorder(indices, low, high);
    }

    Value init = rewriter.create<tensor::EmptyOp>(loc, getAsOpFoldResult(outSizePadded), selfType.getElementType());

    SmallVector<AffineExpr> inputExprs;
    SmallVector<AffineExpr> outputExprs;
    for (auto i : llvm::seq<int64_t>(0, outRank)) {
      AffineExpr dim = rewriter.getAffineDimExpr(i);
      if (i < NC) {
        inputExprs.emplace_back(dim);
      } else {
        int64_t j = i - NC;
        inputExprs.emplace_back(dim.floorDiv(stride[j]));
      }
      outputExprs.emplace_back(dim);
    }

    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {inputExprs, inputExprs, outputExprs}, rewriter.getContext());

    SmallVector<utils::IteratorType> iteratorTypes(
        outRank, utils::IteratorType::parallel);

    auto computeIndex = [&](OpBuilder &b, Location loc) -> Value {
      Value ret;
      for (auto i : llvm::seq<int64_t>(NC, outRank)) {
        Value idx = b.create<linalg::IndexOp>(loc, i);
        // Adjust for the pad
        int64_t pad = padding[i - NC];
        if (pad != 0) {
          Value padVal = b.create<arith::ConstantIndexOp>(loc, pad);
          idx = b.create<arith::SubIOp>(loc, idx, padVal);
        }

        if (!ret) {
          ret = idx;
        } else {
          Value size = b.create<arith::ConstantIndexOp>(loc, resType.getShape()[i]);
          ret = b.create<arith::MulIOp>(loc, ret, size);
          ret = b.create<arith::AddIOp>(loc, ret, idx);
        }
      }
      return ret;
    };

    auto builder = [&](OpBuilder &b, Location loc, ValueRange args) {
      // Compute current output linear index and compare it with the value
      // from indices arg.
      Value input = args[0];
      Value zero = b.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(input.getType()));
      Value index = b.create<arith::IndexCastOp>(loc, indexType, args[1]);
      Value currentIndex = computeIndex(b, loc);
      Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, currentIndex);
      Value out = b.create<arith::SelectOp>(loc, cmp, input, zero);
      b.create<linalg::YieldOp>(loc, out);
    };

    Value result = rewriter.create<linalg::GenericOp>(loc,
      /*resultTensorTypes=*/init.getType(),
      /*inputs=*/ValueRange({self, indices}),
      /*outputs=*/init,
      /*indexingMaps=*/indexingMaps,
      /*iteratorTypes=*/iteratorTypes, builder).getResult(0);

    if (llvm::any_of(padding, [](int64_t v) { return v != 0; })) {
      // MaxPool input was padded, unpad it by taking the slice.
      SmallVector<OpFoldResult> offsetVals(NC, rewriter.getI64IntegerAttr(0));
      for (int64_t pad : padding)
        offsetVals.emplace_back(rewriter.getI64IntegerAttr(pad));

      SmallVector<OpFoldResult> sizeVals;// = getAsIndexOpFoldResult(rewriter.getContext(), resType.getShape());
      for (auto &&[i, dim] : llvm::enumerate(resType.getShape())) {
        if (!ShapedType::isDynamic(dim)) {
          sizeVals.emplace_back(rewriter.getI64IntegerAttr(dim));
          continue;
        }

        sizeVals.emplace_back(rewriter.create<tensor::DimOp>(loc, self, i));
      }
      SmallVector<OpFoldResult> stridesVals(outRank, rewriter.getI64IntegerAttr(1));
      result = rewriter.create<tensor::ExtractSliceOp>(loc, result, offsetVals, sizeVals, stridesVals);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
}

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
        cast<RankedTensorType>(self.getType()).getElementType();
    Type resultType = typeConverter->convertType(op.getType());
    Type resultElementType =
        cast<RankedTensorType>(resultType).getElementType();

    bool ceilMode;
    SmallVector<Value, Dim> kernelSizeIntValues;
    SmallVector<int64_t, Dim> strideInts, paddingInts, dilationInts(Dim, 1);
    if (failed(checkAndGetPoolingParameters<OpTy>(op, rewriter, typeConverter,
                                                  ceilMode, kernelSizeIntValues,
                                                  strideInts, paddingInts)))
      return rewriter.notifyMatchFailure(op, "invalid pooling parameters");

    // Decode strideInts into strideInts and dilation
    if (strideInts.size() == 2 * Dim) {
      for (int i = 0; i < Dim; i++) {
        dilationInts[i] = strideInts[Dim + i];
      }
      for (int i = 0; i < Dim; i++) {
        strideInts.pop_back();
      }
    }

    // TODO: Add support for count_include_pad equal to `False`.
    bool countIncludePad;
    if (!matchPattern(op.getCountIncludePad(),
                      m_TorchConstantBool(&countIncludePad)))
      return rewriter.notifyMatchFailure(
          op, "count_include_pad must be a constant");

    // `sumPool` contains the result of sumpool operation over the input.
    Value sumPool, paddedInput;
    SmallVector<Value, Dim + 2> outTensorShape;
    if (failed(createPoolingOp<PoolingOpTy>(
            op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
            /*dimensionality=*/Dim, kernelSizeIntValues, strideInts,
            paddingInts, dilationInts, rewriter.getZeroAttr(inputElementType),
            outTensorShape, paddedInput, sumPool)))
      return rewriter.notifyMatchFailure(op, "unable to compute sumpool");

    // Compute the average of sumPool.
    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outTensorShape), resultElementType);
    SmallVector<AffineMap> indexingMapsAvg(
        2, rewriter.getMultiDimIdentityMap(Dim + 2));
    SmallVector<utils::IteratorType> iteratorTypesAvg(
        Dim + 2, utils::IteratorType::parallel);
    Value avgPool;
    Value divisor;
    // Case1: AtenAvgPool1d/2dOp with countIncludePad=false support.
    if constexpr (std::is_same<OpTy, AtenAvgPool2dOp>()) {
      auto selfType = cast<RankedTensorType>(self.getType());
      const int64_t selfRank = selfType.getRank();
      int64_t wDim = toPositiveDim(-1, selfRank);
      int64_t hDim = toPositiveDim(-2, selfRank);
      Value inputHeight = getDimOp(rewriter, loc, self, hDim);
      Value inputWidth = getDimOp(rewriter, loc, self, wDim);
      RankedTensorType sumPoolType = cast<RankedTensorType>(sumPool.getType());
      const int64_t rank = sumPoolType.getRank();
      int dimH = toPositiveDim(-2, rank);
      int dimW = toPositiveDim(-1, rank);
      avgPool =
          rewriter
              .create<linalg::GenericOp>(
                  loc, outputTensor.getType(), sumPool, outputTensor,
                  /*indexingMaps=*/indexingMapsAvg,
                  /*iteratorTypes=*/iteratorTypesAvg,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    // The algorithm for computing the divisor with
                    // count_include_pad is manily based on pytorch
                    // implementation. The following code is comment
                    // with pytorch code.
                    // https://github.com/pytorch/pytorch/blob/4a6dfbe4806b361c43210dfd56db64c4097c66bb/aten/src/ATen/native/cpu/AvgPoolKernel.cpp#L78
                    Value indexOh =
                        b.create<linalg::IndexOp>(loc, /*value=*/dimH);
                    Value oh = castIndexToInt64(b, loc, indexOh);
                    Value indexOw =
                        b.create<linalg::IndexOp>(loc, /*value=*/dimW);
                    Value ow = castIndexToInt64(b, loc, indexOw);

                    // int64_t ih0 = oh * dH - padH;
                    Value dH = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(strideInts[0]));
                    Value padH = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(paddingInts[0]));
                    Value ohDH = b.create<arith::MulIOp>(loc, oh, dH);
                    Value ih0 = b.create<arith::SubIOp>(loc, ohDH, padH);
                    // int64_t iw0 = ow * dW - padW;
                    Value dW = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(strideInts[1]));
                    Value padW = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(paddingInts[1]));
                    Value owDW = b.create<arith::MulIOp>(loc, ow, dW);
                    Value iw0 = b.create<arith::SubIOp>(loc, owDW, padW);
                    // int64_t ih1 = std::min(ih0 + kH, input_height + padH);
                    Value ih = castIndexToInt64(b, loc, inputHeight);
                    Value ih0KH = b.create<arith::AddIOp>(
                        loc, ih0, kernelSizeIntValues[0]);
                    Value ihPadH = b.create<arith::AddIOp>(loc, ih, padH);
                    Value ih1 = b.create<arith::MinSIOp>(loc, ih0KH, ihPadH);
                    // int64_t iw1 = std::min(iw0 + kW, input_width + padW);
                    Value iw = castIndexToInt64(b, loc, inputWidth);
                    Value iw0KW = b.create<arith::AddIOp>(
                        loc, iw0, kernelSizeIntValues[1]);
                    Value iwPadW = b.create<arith::AddIOp>(loc, iw, padW);
                    Value iw1 = b.create<arith::MinSIOp>(loc, iw0KW, iwPadW);
                    // int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
                    Value ih1Ih0 = b.create<arith::SubIOp>(loc, ih1, ih0);
                    Value iw1Iw0 = b.create<arith::SubIOp>(loc, iw1, iw0);
                    Value poolSize =
                        b.create<arith::MulIOp>(loc, ih1Ih0, iw1Iw0);
                    // ih0 = std::max(ih0, 0);
                    Value cstZero = rewriter.create<arith::ConstantOp>(
                        loc, rewriter.getI64IntegerAttr(0));
                    Value ih0Clamped =
                        b.create<arith::MaxSIOp>(loc, ih0, cstZero);
                    // iw0 = std::max(iw0, 0);
                    Value iw0Clamped =
                        b.create<arith::MaxSIOp>(loc, iw0, cstZero);
                    // ih1 = std::min(ih1, input_height);
                    Value ih1Clamped = b.create<arith::MinSIOp>(loc, ih1, ih);
                    // iw1 = std::min(iw1, input_width);
                    Value iw1Clamped = b.create<arith::MinSIOp>(loc, iw1, iw);
                    // if (divisor_override.has_value()) {
                    //   divisor = divisor_override.value();
                    // } else {
                    //   if(count_include_pad) {
                    //     divisor = pool_size;
                    //   } else {
                    //     divisor = (ih1 - ih0) * (iw1 - iw0);
                    //   }
                    // }
                    if (countIncludePad) {
                      divisor = convertScalarToDtype(b, loc, poolSize,
                                                     resultElementType);
                    } else {
                      Value ih1_ih0 =
                          b.create<arith::SubIOp>(loc, ih1Clamped, ih0Clamped);
                      Value iw1_iw0 =
                          b.create<arith::SubIOp>(loc, iw1Clamped, iw0Clamped);
                      divisor = b.create<arith::MulIOp>(loc, ih1_ih0, iw1_iw0);
                    }
                    // AtenAvgPool2/3dOp has an optional divisor_override
                    // attribute while AtenAvgPool1dOp does not.
                    if constexpr (std::is_same<OpTy, AtenAvgPool2dOp>()) {
                      if (!isa<Torch::NoneType>(
                              op.getDivisorOverride().getType()))
                        divisor = adaptor.getDivisorOverride();
                    }

                    divisor = convertScalarToDtype(b, loc, divisor,
                                                   resultElementType);
                    Value avg;
                    if (isa<mlir::IntegerType>(resultElementType))
                      avg = b.create<arith::DivSIOp>(loc, args[0], divisor);
                    else if (isa<mlir::FloatType>(resultElementType))
                      avg = b.create<arith::DivFOp>(loc, args[0], divisor);
                    b.create<linalg::YieldOp>(loc, avg);
                  })
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, avgPool);
      return success();
    }

    // TODO: Add support for count_include_pad equal to `False` in
    // AtenAvgPool1/3dOp.
    if (!countIncludePad &&
        !llvm::all_of(paddingInts, [](int64_t p) { return p == 0; })) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: count_include_pad is expected to be true for "
              "AtenAvgPool3dOp");
    }

    // Case2: AtenAvgPool1/3dOp without count_include_pad equal to `False`.
    divisor = kernelSizeIntValues[0];
    for (uint32_t i = 1; i < kernelSizeIntValues.size(); i++) {
      divisor =
          rewriter.create<arith::MulIOp>(loc, divisor, kernelSizeIntValues[i]);
    }
    if constexpr (!std::is_same<OpTy, AtenAvgPool1dOp>()) {
      divisor = isa<Torch::NoneType>(op.getDivisorOverride().getType())
                    ? divisor
                    : adaptor.getDivisorOverride();
    }
    divisor = convertScalarToDtype(rewriter, loc, divisor, resultElementType);
    avgPool = rewriter
                  .create<linalg::GenericOp>(
                      loc, outputTensor.getType(), sumPool, outputTensor,
                      /*indexingMaps=*/indexingMapsAvg,
                      /*iteratorTypes=*/iteratorTypesAvg,
                      [&](OpBuilder &b, Location loc, ValueRange args) {
                        Value avg;
                        if (isa<mlir::IntegerType>(resultElementType))
                          avg = b.create<arith::DivSIOp>(loc, args[0], divisor);
                        else if (isa<mlir::FloatType>(resultElementType))
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
*/

namespace {

class AdaptivePoolingHelper {
public:
  AdaptivePoolingHelper(ConversionPatternRewriter &cpr, int64_t rnk,
                        int64_t nsp, Type elt)
      : rewriter(cpr), rank(rnk), nonSpatial(nsp), elementType(elt) {}

  // Variables that are used in various helper functions in the derived classes
  // are stored as members of the base class (to reduce the number of arguments
  // passed to helper functions).
  ConversionPatternRewriter &rewriter;
  const int64_t rank;
  const int64_t nonSpatial;
  Type elementType;
};

// The following two derived helper classes are used to store the differing
// logic between adaptive avg pooling and adaptive max pooling.
// 1. auxTensorSetup initializes a tensor for storing either indices (max) or
// kernel volumes (avg)
// 2. payloadCustomization customizes those features of the main linalg generic
// op that are not generically "AdaptivePooling". Specifically, for switching
// between sum/max and writing the code for computing the aux tensor elements.
// 3. customizedOpReplacement finishes the op replacement. In the adaptive avg
// case, it includes an additional generic op to divide the sum pool by the
// kernel volume.
// To access these helper functions in the conversion pattern, we
// have an AdaptivePoolingOpTraits class that stores the number of dimensions
// and aliases the associated helper class to a more generic name.

template <class OpTy>
class AdaptiveMaxPoolingHelper : public AdaptivePoolingHelper {

  // This member variable is templated, so I've chosen not to make it part of
  // the base class (to keep the base class non-templated).
  const OpConversionPattern<OpTy> &opConversionPattern;

public:
  // Constructor for AdaptiveMaxPoolingHelper. Just forwards all arguments
  // (except the OpConversionPattern) to the base class constructor.
  template <typename... Args>
  AdaptiveMaxPoolingHelper(const OpConversionPattern<OpTy> &ocp, Args &&...args)
      : AdaptivePoolingHelper(std::forward<Args>(args)...),
        opConversionPattern(ocp) {}

  LogicalResult auxTensorSetup(OpTy op, const SmallVector<Value> &outputSizes,
                               const SmallVector<Value> &outShapeIndexVector,
                               RankedTensorType &outputType,
                               RankedTensorType &auxTensorType, Value &buffVal,
                               Value &auxTensor,
                               SmallVector<AffineExpr> &auxTensorExprs) {

    Location loc = op->getLoc();
    const TypeConverter *typeConverter = opConversionPattern.getTypeConverter();
    outputType = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult0().getType()));
    auxTensorType = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult1().getType()));
    Type auxTensorElementType = auxTensorType.getElementType();
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
                        /*Negative=*/true));
    buffVal = rewriter.create<arith::ConstantOp>(loc, elementType,
                                                 smallestFPValueAttr);
    auxTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outputSizes), auxTensorElementType);
    for (unsigned i = 0; i < rank; i++) {
      auxTensorExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    return success();
  }

  LogicalResult payloadCustomization(
      OpBuilder &b, Location loc, const Value &inElt, const Value &res,
      const Value &maxIndex, const SmallVector<Value> &inputElementIndices,
      const SmallVector<Value> &inputSpatialSizes, const Value &indexOne,
      const SmallVector<Value> &starts, const SmallVector<Value> &ends,
      Value &out2, Value &auxOut) {
    // compute max using select, since cond1 will be used for indices
    Value cond1 =
        b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, inElt, res);
    out2 = b.create<arith::SelectOp>(loc, cond1, inElt, res);
    // index in different dims (n x c x d x h x w)
    // 1d: (iw)
    // 2d: (ih*W + iw)
    // 3d: (id*H*W + ih*W + iw)
    Value currIndex = inputElementIndices[nonSpatial];
    for (unsigned i = 0; i < rank - nonSpatial - 1; i++) {
      Value prevTimesNewSize =
          b.create<arith::MulIOp>(loc, currIndex, inputSpatialSizes[i + 1]);
      currIndex = b.create<arith::AddIOp>(
          loc, prevTimesNewSize, inputElementIndices[nonSpatial + i + 1]);
    }
    Value indexOut1Int = castIndexToInt64(b, loc, currIndex);
    auxOut = b.create<arith::SelectOp>(loc, cond1, indexOut1Int, maxIndex);
    return success();
  }

  LogicalResult
  customizedOpReplacement(OpTy op, const RankedTensorType &outputType,
                          const RankedTensorType &auxTensorType,
                          const Value &adaptivePoolOutput,
                          const Value &auxTensorReturn,
                          const SmallVector<AffineExpr> &auxTensorExprs,
                          const SmallVector<AffineExpr> &outputExprs) {
    Location loc = op->getLoc();
    Value maxValues =
        rewriter.create<tensor::CastOp>(loc, outputType, adaptivePoolOutput);
    Value outputIndices =
        rewriter.create<tensor::CastOp>(loc, auxTensorType, auxTensorReturn);
    rewriter.replaceOp(op, {maxValues, outputIndices});
    return success();
  }
};

template <class OpTy>
class AdaptiveAvgPoolingHelper : public AdaptivePoolingHelper {

  const OpConversionPattern<OpTy> &opConversionPattern;

public:
  template <typename... Args>
  AdaptiveAvgPoolingHelper(const OpConversionPattern<OpTy> &ocp, Args &&...args)
      : AdaptivePoolingHelper(std::forward<Args>(args)...),
        opConversionPattern(ocp) {}

  LogicalResult auxTensorSetup(OpTy op, const SmallVector<Value> &outputSizes,
                               const SmallVector<Value> &outShapeIndexVector,
                               RankedTensorType &outputType,
                               RankedTensorType &auxTensorType, Value &buffVal,
                               Value &auxTensor,
                               SmallVector<AffineExpr> &auxTensorExprs) {

    Location loc = op->getLoc();
    const TypeConverter *typeConverter = opConversionPattern.getTypeConverter();
    outputType = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult().getType()));
    buffVal = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));
    auxTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outShapeIndexVector), elementType);
    for (unsigned i = nonSpatial; i < rank; i++) {
      auxTensorExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    return success();
  }

  LogicalResult payloadCustomization(
      OpBuilder &b, Location loc, const Value &inElt, const Value &res,
      const Value &maxIndex, const SmallVector<Value> &inputElementIndices,
      const SmallVector<Value> &inputSpatialSizes, const Value &indexOne,
      const SmallVector<Value> &starts, const SmallVector<Value> &ends,
      Value &out2, Value &auxOut) {
    out2 = b.create<arith::AddFOp>(loc, inElt, res);
    Value kernelVolume = indexOne;
    for (unsigned i = 0; i < rank - nonSpatial; i++) {
      Value currSize = b.create<arith::SubIOp>(loc, ends[i], starts[i]);
      kernelVolume = b.create<arith::MulIOp>(loc, kernelVolume, currSize);
    }
    Value auxOutSI = castIndexToInt64(b, loc, kernelVolume);
    auxOut = b.create<arith::SIToFPOp>(loc, elementType, auxOutSI);
    return success();
  }

  LogicalResult
  customizedOpReplacement(OpTy op, const RankedTensorType &outputType,
                          const RankedTensorType &auxTensorType,
                          const Value &adaptivePoolOutput,
                          const Value &auxTensorReturn,
                          const SmallVector<AffineExpr> &auxTensorExprs,
                          const SmallVector<AffineExpr> &outputExprs) {

    Location loc = op->getLoc();
    SmallVector<AffineMap> indexingMaps1 = AffineMap::inferFromExprList(
        {auxTensorExprs, outputExprs}, op.getContext());
    SmallVector<utils::IteratorType> iteratorTypes1(
        rank, utils::IteratorType::parallel);
    auto output = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/adaptivePoolOutput.getType(),
        /*inputs=*/auxTensorReturn,
        /*outputs=*/adaptivePoolOutput,
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

// stores Dim = spatial dims and aliases helper class to a generic name
template <typename T> struct AdaptivePoolingOpTraits {};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveMaxPool1dOp> {
  static constexpr int64_t Dim = 1;
  using AdaptivePoolingHelper =
      AdaptiveMaxPoolingHelper<AtenAdaptiveMaxPool1dOp>;
};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveMaxPool2dOp> {
  static constexpr int64_t Dim = 2;
  using AdaptivePoolingHelper =
      AdaptiveMaxPoolingHelper<AtenAdaptiveMaxPool2dOp>;
};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveMaxPool3dOp> {
  static constexpr int64_t Dim = 3;
  using AdaptivePoolingHelper =
      AdaptiveMaxPoolingHelper<AtenAdaptiveMaxPool3dOp>;
};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveAvgPool1dOp> {
  static constexpr int64_t Dim = 1;
  using AdaptivePoolingHelper =
      AdaptiveAvgPoolingHelper<AtenAdaptiveAvgPool1dOp>;
};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveAvgPool2dOp> {
  static constexpr int64_t Dim = 2;
  using AdaptivePoolingHelper =
      AdaptiveAvgPoolingHelper<AtenAdaptiveAvgPool2dOp>;
};

template <> struct AdaptivePoolingOpTraits<AtenAdaptiveAvgPool3dOp> {
  static constexpr int64_t Dim = 3;
  using AdaptivePoolingHelper =
      AdaptiveAvgPoolingHelper<AtenAdaptiveAvgPool3dOp>;
};

template <> struct AdaptivePoolingOpTraits<Aten_AdaptiveAvgPool3dOp> {
  static constexpr int64_t Dim = 3;
  using AdaptivePoolingHelper =
      AdaptiveAvgPoolingHelper<Aten_AdaptiveAvgPool3dOp>;
};

template <typename OpTy>
class ConvertAtenAdaptivePoolOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

private:
  static const int64_t Dim = AdaptivePoolingOpTraits<OpTy>::Dim;

public:
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    const TypeConverter *typeConverter = this->getTypeConverter();

    Value input = adaptor.getSelf();
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    const Type elementType = inputType.getElementType();

    // get rank of input (same as rank of output)
    const int64_t rank = inputType.getRank();
    // get number of non-spatial dims
    const int64_t nonSpatial = rank - Dim;
    if (nonSpatial < 0) {
      return rewriter.notifyMatchFailure(op,
                                         "input has insufficient spatial dims");
    }

    typename AdaptivePoolingOpTraits<OpTy>::AdaptivePoolingHelper
        adaptivePoolingHelper(*this, rewriter, rank, nonSpatial, elementType);

    // get input and output spatial dimensions as index values
    Value outputShape = op.getOutputSize();
    SmallVector<Value> outShapeVector;
    getListConstructElements(outputShape, outShapeVector);
    outShapeVector =
        getTypeConvertedValues(rewriter, loc, typeConverter, outShapeVector);
    SmallVector<Value> inputSpatialSizes;
    for (unsigned i = nonSpatial; i < rank; i++) {
      inputSpatialSizes.push_back(getDimOp(rewriter, loc, input, i));
    }
    SmallVector<Value> outShapeIndexVector;
    for (auto v : outShapeVector) {
      outShapeIndexVector.push_back(castIntToIndex(rewriter, loc, v));
    }

    // make an iteration space of size kMax = 1 + ceildiv (hIn - 1) , hOut
    Type boolType = rewriter.getI1Type();
    SmallVector<Value> kIterSizeVector;
    Value constantOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    for (int i = 0; i < rank - nonSpatial; i++) {
      Value hInPlusOne = rewriter.create<arith::SubIOp>(
          loc, inputSpatialSizes[i], constantOne);
      Value kMaxMinusOne = rewriter.create<arith::CeilDivSIOp>(
          loc, hInPlusOne, outShapeIndexVector[i]);
      Value kMax =
          rewriter.create<arith::AddIOp>(loc, constantOne, kMaxMinusOne);
      kIterSizeVector.push_back(kMax);
    }
    Value kIter = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(kIterSizeVector), boolType);

    // get output sizes used for initializing some tensors
    SmallVector<Value> outputSizes;
    for (unsigned i = 0; i < nonSpatial; i++) {
      outputSizes.push_back(getDimOp(rewriter, loc, input, i));
    }
    for (unsigned i = 0; i < rank - nonSpatial; i++) {
      outputSizes.push_back(outShapeIndexVector[i]);
    }

    // get outputType and initialize an auxTensor
    // the auxTensor is customizable:
    // avg pooling -> auxTensor = kernelVolumes
    // max pooling -> auxTensor = indices
    RankedTensorType outputType, auxTensorType;
    Value buffVal, auxTensor;
    SmallVector<AffineExpr> auxTensorExprs;
    if (failed(adaptivePoolingHelper.auxTensorSetup(
            op, outputSizes, outShapeIndexVector, outputType, auxTensorType,
            buffVal, auxTensor, auxTensorExprs))) {
      return rewriter.notifyMatchFailure(op, "failed auxTensor setup");
    }

    // initialize output tensor
    Value initOutput =
        createInitTensor(rewriter, loc, outputSizes, elementType, buffVal);

    // pad the input with buffVal = 0 (avg) or -inf (max)
    SmallVector<int64_t> lowPadding(rank, 0);
    SmallVector<int64_t> highPadding(nonSpatial, 0);
    for (int i = 0; i < rank - nonSpatial; i++) {
      highPadding.push_back(1);
    }
    Value buffInput = torch_to_linalg::getPaddedTensor(
        op, rewriter, input, lowPadding, highPadding, buffVal);

    // setup indexing maps and iterator types for linalg generic op
    // for example, with rank = 4 and nonSpatial = 2:
    // kIter (d0,d1,d2,d3,d4,d5) -> (d4,d5)
    // output (d0,d1,d2,d3,d4,d5) -> (d0,d1,d2,d3)
    SmallVector<AffineExpr> kIterExprs, outputExprs;
    // batch + channel + output spatial dims
    for (unsigned i = 0; i < rank; i++) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    // kIter covers last rank-2 indices
    for (unsigned i = rank; i < 2 * rank - nonSpatial; i++) {
      kIterExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {kIterExprs, outputExprs, auxTensorExprs}, rewriter.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    for (unsigned i = 0; i < rank - nonSpatial; i++) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
    }
    Value indexOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    bool failedCustomization = false;
    // adaptive pooling generic op
    auto adaptivePool = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/
        TypeRange({initOutput.getType(), auxTensor.getType()}),
        /*inputs=*/ValueRange({kIter}),
        /*outputs=*/ValueRange({initOutput, auxTensor}),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value res = args[1];
          Value maxIndex = args[2];
          SmallVector<Value> ind;
          for (unsigned i = 0; i < 2 * rank - nonSpatial; i++) {
            ind.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          // compute start and end indices
          // st = s1( s0(ind2 * Hin) // Hout )
          SmallVector<Value> starts;
          SmallVector<Value> ends;
          for (unsigned i = nonSpatial; i < rank; i++) {
            Value s0 = b.create<arith::MulIOp>(
                loc, ind[i], inputSpatialSizes[i - nonSpatial]);
            Value s1 = b.create<arith::FloorDivSIOp>(
                loc, s0, outShapeIndexVector[i - nonSpatial]);
            starts.push_back(s1);
            // en = e4( 1 + e3( e2( e1( e0(ind2 + 1) * hIn ) - 1 ) // hOut ) )
            Value e0 = b.create<arith::AddIOp>(loc, ind[i], indexOne);
            Value e1 = b.create<arith::MulIOp>(
                loc, e0, inputSpatialSizes[i - nonSpatial]);
            Value e2 = b.create<arith::SubIOp>(loc, e1, indexOne);
            Value e3 = b.create<arith::FloorDivSIOp>(
                loc, e2, outShapeIndexVector[i - nonSpatial]);
            Value e4 = b.create<arith::AddIOp>(loc, indexOne, e3);
            ends.push_back(e4);
          }
          // extract input element
          SmallVector<Value> inputElementIndices;
          for (unsigned i = 0; i < nonSpatial; i++) {
            inputElementIndices.push_back(ind[i]);
          }
          for (unsigned i = nonSpatial; i < rank; i++) {
            inputElementIndices.push_back(b.create<arith::AddIOp>(
                loc, starts[i - nonSpatial], ind[rank - nonSpatial + i]));
          }
          Value inElt = b.create<tensor::ExtractOp>(loc, elementType, buffInput,
                                                    inputElementIndices);
          // check if we extracted at windex < end index
          for (unsigned i = 0; i < rank - nonSpatial; i++) {
            Value cond = b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate(6),
                inputElementIndices[i + nonSpatial], ends[i]);
            // if out-of-bounds, replace the extracted element with buffVal
            inElt = b.create<arith::SelectOp>(loc, cond, inElt, buffVal);
          }
          Value out2, auxOut;
          // customize for max vs. avg:
          if (failed(adaptivePoolingHelper.payloadCustomization(
                  b, loc, inElt, res, maxIndex, inputElementIndices,
                  inputSpatialSizes, indexOne, starts, ends, out2, auxOut))) {
            failedCustomization = true;
          }
          b.create<linalg::YieldOp>(loc, ValueRange({out2, auxOut}));
        });

    if (failedCustomization) {
      return rewriter.notifyMatchFailure(
          op, "failed linalg generic payload customization.");
    }
    Value adaptivePoolOutput = adaptivePool.getResultTensors()[0];
    Value auxTensorReturn = adaptivePool.getResultTensors()[1];

    if (failed(adaptivePoolingHelper.customizedOpReplacement(
            op, outputType, auxTensorType, adaptivePoolOutput, auxTensorReturn,
            auxTensorExprs, outputExprs))) {
      return rewriter.notifyMatchFailure(op, "failed customizedOpReplacement.");
    }
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populatePoolingPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMaxPool1dOp>();
  target.addIllegalOp<AtenMaxPool2dOp>();
  target.addIllegalOp<AtenMaxPool3dOp>();
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool1dOp>>(typeConverter, context);
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool2dOp>>(typeConverter, context);
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool3dOp>>(typeConverter, context);

  target.addIllegalOp<AtenMaxPool2dWithIndicesOp>();
  patterns.add<ConvertAtenMaxPool2dWithIndicesOp>(typeConverter, context);

  target.addIllegalOp<AtenMaxUnpool3dOp>();
  patterns.add<ConvertAtenMaxUnpool3dOp>(typeConverter, context);

  target.addIllegalOp<AtenAvgPool1dOp, AtenAvgPool2dOp, AtenAvgPool3dOp>();
  patterns
      .add<ConvertAtenAvgPoolOp<AtenAvgPool1dOp, linalg::PoolingNcwSumOp, 1>>(
          typeConverter, context);
  patterns
      .add<ConvertAtenAvgPoolOp<AtenAvgPool2dOp, linalg::PoolingNchwSumOp, 2>>(
          typeConverter, context);
  patterns
      .add<ConvertAtenAvgPoolOp<AtenAvgPool3dOp, linalg::PoolingNdhwcSumOp, 3>>(
          typeConverter, context);
  target.addIllegalOp<AtenAdaptiveAvgPool1dOp, AtenAdaptiveAvgPool2dOp,
                      AtenAdaptiveAvgPool3dOp, Aten_AdaptiveAvgPool3dOp>();
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveAvgPool1dOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveAvgPool2dOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveAvgPool3dOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenAdaptivePoolOp<Aten_AdaptiveAvgPool3dOp>>(
      typeConverter, context);
  target.addIllegalOp<AtenAdaptiveMaxPool1dOp, AtenAdaptiveMaxPool2dOp,
                      AtenAdaptiveMaxPool3dOp>();
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveMaxPool1dOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveMaxPool2dOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenAdaptivePoolOp<AtenAdaptiveMaxPool3dOp>>(
      typeConverter, context);
}
