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
#include <optional>

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

  SmallVector<Value> dilationIntValues =
      getAsConstantIntValues(rewriter, loc, dilationInts);
  SmallVector<Value> strideIntValues =
      getAsConstantIntValues(rewriter, loc, strideInts);

  // Get dimension size for each dimension and calculate output size
  for (int64_t i = dimensionality - 1; i > -1; --i) {
    // In case of asymmetric padding the total padding value would be the sum of
    // low and high padding. And, in case of symmetric padding it would just be
    // the double of padding value for the corresponding dimension.
    int64_t totalPadding = paddingInts[i] * 2;
    if ((int64_t)paddingInts.size() == 2 * dimensionality)
      totalPadding = paddingInts[i] + paddingInts[i + dimensionality];

    Value dimSize = getDimOp(rewriter, loc, self, i + 2);
    Value outDim = torch_to_linalg::getOutputDimForPoolOps(
        rewriter, loc, dimSize, /*totalPadding=*/totalPadding,
        /*leftPadding=*/paddingInts[i], dilationIntValues[i],
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
      arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(initValueAttr));

  paddedInput = padInputTensor(op, rewriter, self, ceilMode, dimensionality,
                               strideInts, paddingInts, initValue);

  auto outTensorInitialized = computeOutputTensor(
      op, rewriter, self, dimensionality, ceilMode, strideInts, paddingInts,
      dilationInts, kernelSizeIntValues, outTensorShape, initValue);

  auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
  auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
  auto shape = castIntVectorToIndexVector(rewriter, loc, kernelSizeIntValues);
  Value windowTensor = tensor::EmptyOp::create(
      rewriter, loc, getAsOpFoldResult(shape), elementType);

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

  Value poolingResult = OpTy::create(rewriter, loc, permutedOutput.getType(),
                                     ValueRange{permutedInput, windowTensor},
                                     permutedOutput, stridesAttr, dilationAttr)
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

static Value createMaxUnpoolOp(Operation *op, int64_t poolingDimensionality,
                               ConversionPatternRewriter &rewriter,
                               const TypeConverter *typeConverter, Value self,
                               Value indices, ArrayRef<int64_t> inputSize,
                               ArrayRef<int64_t> inferredOutSize,
                               SmallVector<int64_t> &stride,
                               SmallVector<int64_t> &padding,
                               RankedTensorType resType) {

  Location loc = op->getLoc();
  Type indexType = rewriter.getIndexType();

  int64_t outRank = resType.getRank();
  int64_t NC = outRank - poolingDimensionality;

  auto selfType = cast<RankedTensorType>(self.getType());
  auto indicesType = cast<RankedTensorType>(indices.getType());

  SmallVector<Value> outSizePadded;
  for (auto &&[i, size] : llvm::enumerate(resType.getShape())) {
    if (int64_t(i) < NC) {
      outSizePadded.emplace_back(tensor::DimOp::create(rewriter, loc, self, i));
      continue;
    }
    int64_t pad = padding[i - NC];

    outSizePadded.emplace_back(
        arith::ConstantIndexOp::create(rewriter, loc, size + pad));
  }

  // In case if input tensor size is not divisible by stride
  // (e.g. pooling_input_size=5, kernel_size=2, stride=2, output_size=2)
  // pad self and indices tensors to avoid out of bounds access.
  SmallVector<int64_t> expectedInputShape =
      llvm::to_vector(resType.getShape().drop_back(poolingDimensionality));
  for (auto &&[str, pad, resSize] :
       llvm::zip_equal(stride, padding, inferredOutSize))
    expectedInputShape.emplace_back((resSize + str - 1) / str + pad * 2);

  if (expectedInputShape != selfType.getShape()) {
    // TODO: this is probably expensive, and it may be possible to solve by
    // cleverly constructing affine maps for the next linalg.generic op,
    // but I'm not smart enough to figure this out.

    SmallVector<int64_t> low(outRank, 0);
    SmallVector<int64_t> high(NC, 0);
    for (auto &&[inpSize, outSize] : llvm::zip_equal(
             inputSize,
             ArrayRef(expectedInputShape).take_back(poolingDimensionality))) {
      high.emplace_back(outSize - inpSize);
    }

    // Pad the indices tensor with a value which cannot appear in real data
    // (-1) so it will never match. In this case we can pad self with any
    // value, as it will never affect the output.
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(selfType.getElementType()));
    Value invalidIdx = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getIntegerAttr(indicesType.getElementType(), -1));
    self =
        torch_to_linalg::getPaddedTensor(op, rewriter, self, low, high, zero);
    indices = torch_to_linalg::getPaddedTensor(op, rewriter, indices, low, high,
                                               invalidIdx);
  }

  Value init =
      tensor::EmptyOp::create(rewriter, loc, getAsOpFoldResult(outSizePadded),
                              selfType.getElementType());

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

  SmallVector<utils::IteratorType> iteratorTypes(outRank,
                                                 utils::IteratorType::parallel);

  auto computeIndex = [&](OpBuilder &b, Location loc) -> Value {
    // Next linalg.generic uses identity mapping for the unpooled tensor,
    // compute linear index for output element, which we will the compare with
    // values which came from indices tensor.
    Value ret;
    for (auto i : llvm::seq<int64_t>(NC, outRank)) {
      Value idx = linalg::IndexOp::create(b, loc, i);
      // If pool input was padded, adjust indices so they start at 0 in the
      // non-padded area. Indices outside non-padded area will make no sense,
      // but it doesnt matter as we will cut the padded area later by
      // extract_slice.
      int64_t pad = padding[i - NC];
      if (pad != 0) {
        Value padVal = arith::ConstantIndexOp::create(b, loc, pad);
        idx = arith::SubIOp::create(b, loc, idx, padVal);
      }

      if (!ret) {
        ret = idx;
      } else {
        Value size =
            arith::ConstantIndexOp::create(b, loc, resType.getShape()[i]);
        ret = arith::MulIOp::create(b, loc, ret, size);
        ret = arith::AddIOp::create(b, loc, ret, idx);
      }
    }
    return ret;
  };

  auto builder = [&](OpBuilder &b, Location loc, ValueRange args) {
    // Compute current output linear index and compare it with the value
    // from indices arg.
    Value input = args[0];
    Value zero = arith::ConstantOp::create(
        b, loc, rewriter.getZeroAttr(input.getType()));
    Value index = arith::IndexCastOp::create(b, loc, indexType, args[1]);
    Value currentIndex = computeIndex(b, loc);
    Value cmp = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, index,
                                      currentIndex);
    Value out = arith::SelectOp::create(b, loc, cmp, input, zero);
    linalg::YieldOp::create(b, loc, out);
  };

  Value result =
      linalg::GenericOp::create(rewriter, loc,
                                /*resultTensorTypes=*/init.getType(),
                                /*inputs=*/ValueRange({self, indices}),
                                /*outputs=*/init,
                                /*indexingMaps=*/indexingMaps,
                                /*iteratorTypes=*/iteratorTypes, builder)
          .getResult(0);

  if (llvm::any_of(padding, [](int64_t v) { return v != 0; })) {
    // MaxPool input was padded, unpad it by taking the slice.
    SmallVector<OpFoldResult> offsetVals(NC, rewriter.getI64IntegerAttr(0));
    for (int64_t pad : padding)
      offsetVals.emplace_back(rewriter.getI64IntegerAttr(pad));

    SmallVector<OpFoldResult> sizeVals;
    for (auto &&[i, dim] : llvm::enumerate(resType.getShape())) {
      if (!ShapedType::isDynamic(dim)) {
        sizeVals.emplace_back(rewriter.getI64IntegerAttr(dim));
        continue;
      }

      sizeVals.emplace_back(tensor::DimOp::create(rewriter, loc, self, i));
    }
    SmallVector<OpFoldResult> stridesVals(outRank,
                                          rewriter.getI64IntegerAttr(1));
    result = tensor::ExtractSliceOp::create(rewriter, loc, result, offsetVals,
                                            sizeVals, stridesVals);
  }

  if (result.getType() != resType)
    result = tensor::CastOp::create(rewriter, loc, resType, result);

  return result;
}

namespace {

template <typename T> struct DimensionTraits {};

template <> struct DimensionTraits<AtenMaxPool1dOp> {
  static constexpr int64_t Dim = 1;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <>
struct DimensionTraits<AtenMaxPool1dWithIndicesOp>
    : DimensionTraits<AtenMaxPool1dOp> {};

template <> struct DimensionTraits<AtenMaxPool2dOp> {
  static constexpr int64_t Dim = 2;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <>
struct DimensionTraits<AtenMaxPool2dWithIndicesOp>
    : DimensionTraits<AtenMaxPool2dOp> {};

template <> struct DimensionTraits<AtenMaxPool3dOp> {
  static constexpr int64_t Dim = 3;
  // unused const variable warning suppression:
  static_assert(Dim == Dim);
};

template <>
struct DimensionTraits<AtenMaxPool3dWithIndicesOp>
    : DimensionTraits<AtenMaxPool3dOp> {};

template <typename OpTy>
class ConvertAtenMaxPoolOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static const bool withIndices =
      llvm::is_one_of<OpTy, AtenMaxPool1dWithIndicesOp,
                      AtenMaxPool2dWithIndicesOp,
                      AtenMaxPool3dWithIndicesOp>::value;

private:
  static const int64_t Dim = DimensionTraits<OpTy>::Dim;

  LogicalResult createPoolingMax3D(OpTy &op, typename OpTy::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   SmallVectorImpl<Value> &kernelSizeIntValues,
                                   SmallVectorImpl<int64_t> &strideInts,
                                   SmallVectorImpl<int64_t> &paddingInts,
                                   SmallVectorImpl<int64_t> &dilationInts,
                                   bool ceilMode,
                                   SmallVectorImpl<Value> &outTensorShape,
                                   Value &paddedInput, Value &poolingOp) const {
    static_assert(Dim == 3, "op must be MaxPool3d or MaxPool3dWithIndices");
    Value self = adaptor.getSelf();
    Type elementType = cast<RankedTensorType>(self.getType()).getElementType();
    TypedAttr smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
                        /*Negative=*/true));
    Value initValue =
        arith::ConstantOp::create(rewriter, op->getLoc(), smallestFPValueAttr);

    paddedInput = padInputTensor(op, rewriter, self, ceilMode, 3, strideInts,
                                 paddingInts, initValue);

    auto outTensorInitialized = computeOutputTensor(
        op, rewriter, self, 3, ceilMode, strideInts, paddingInts, dilationInts,
        kernelSizeIntValues, outTensorShape, initValue);

    auto shape =
        castIntVectorToIndexVector(rewriter, op->getLoc(), kernelSizeIntValues);
    Value windowTensor = tensor::EmptyOp::create(
        rewriter, op->getLoc(), getAsOpFoldResult(shape), elementType);

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
    poolingOp = linalg::GenericOp::create(
                    rewriter, op->getLoc(),
                    /* result types */ outTensorInitialized.getType(),
                    /* operands */ ValueRange({paddedInput, windowTensor}),
                    /* outputs */ outTensorInitialized,
                    /*indexingMaps=*/indexingMaps,
                    /*iteratorTypes=*/iteratorTypes,
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                      Value currentVal = args[0], accMaxValue = args[2];
                      Value max_result = arith::MaximumFOp::create(
                          b, loc, currentVal, accMaxValue);
                      linalg::YieldOp::create(b, loc, max_result);
                    })
                    .getResult(0);

    return success();
  }

  // Returns the corresponding indices of the input tensor for the max pooling
  // result tensor.
  //
  // For finding the indices, we follow the below method:
  //
  // Take maxpool2d as an example to illustrate. Let's say the input tensor is a
  // 4-d tensor. The maxpool2d and indices will also be a 4-d tensor. Then:
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
  //                              indices[i, j, m, n] =
  //                                (indexH - padding[0]) * W +
  //                                (indexW - padding[1])
  //
  LogicalResult
  computeMaxPoolingIndices(Value maxPool, Value paddedInput, OpTy &op,
                           typename OpTy::Adaptor adaptor,
                           ConversionPatternRewriter &rewriter,
                           SmallVectorImpl<Value> &outTensorShape,
                           SmallVectorImpl<Value> &kernelSizeIntValues,
                           SmallVectorImpl<int64_t> &strideInts,
                           SmallVectorImpl<int64_t> &paddingInts,
                           SmallVectorImpl<int64_t> &dilationInts, int64_t rank,
                           Value &indicesResult) const {
    Location loc = op->getLoc();
    RankedTensorType indicesRankedTensorType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op->getResult(1).getType()));
    Value cstMinusOne = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(-1));
    Value indicesTensor =
        createInitTensor(rewriter, loc, outTensorShape,
                         indicesRankedTensorType.getElementType(), cstMinusOne);

    SmallVector<Value> kernelSize =
        castIntVectorToIndexVector(rewriter, loc, kernelSizeIntValues);
    SmallVector<Value> padding =
        getAsConstantIndexValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilation =
        getAsConstantIndexValues(rewriter, loc, dilationInts);
    SmallVector<Value> kernelStride =
        getAsConstantIndexValues(rewriter, loc, strideInts);

    Value windowTensor =
        tensor::EmptyOp::create(rewriter, loc, getAsOpFoldResult(kernelSize),
                                indicesRankedTensorType.getElementType());

    SmallVector<AffineExpr> inputExprs, outputExprs, kernelExprs;
    for (unsigned i = 0; i < rank; i++) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (unsigned i = 0; i < rank - 2; i++) {
      kernelExprs.push_back(rewriter.getAffineDimExpr(i + rank));
    }

    // If computing indices for maxpool2d, we have six dimensions here. Each
    // corresponding to N, C, Hout, Wout, kH, and kW, respectively, as described
    // in the algorithm above.
    SmallVector<AffineMap> indexingMaps = AffineMap::inferFromExprList(
        {inputExprs, kernelExprs, outputExprs}, rewriter.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    iteratorTypes.append(rank - 2, utils::IteratorType::reduction);

    // Extract pooling dimensions of input shape.
    SmallVector<Value> inputSubShape;
    for (unsigned i = 0; i < rank - 2; i++) {
      inputSubShape.push_back(
          getDimOp(rewriter, loc, adaptor.getSelf(), i + 2));
    }

    indicesResult =
        linalg::GenericOp::create(
            rewriter, loc, /*resultTensorTypes=*/indicesTensor.getType(),
            /*inputs=*/ValueRange({maxPool, windowTensor}),
            /*outputs=*/indicesTensor,
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value maxVal = args[0], res = args[2];

              SmallVector<Value> inputDims;
              inputDims.append({linalg::IndexOp::create(b, loc, 0),
                                linalg::IndexOp::create(b, loc, 1)});
              for (unsigned i = 2; i < rank; i++) {
                Value mainIndex = linalg::IndexOp::create(b, loc, i);
                Value subIndex = linalg::IndexOp::create(b, loc, i + rank - 2);
                Value origin = arith::MulIOp::create(b, loc, mainIndex,
                                                     kernelStride[i - 2]);
                Value offset =
                    arith::MulIOp::create(b, loc, subIndex, dilation[i - 2]);
                inputDims.push_back(
                    arith::AddIOp::create(b, loc, origin, offset));
              }

              Value input =
                  tensor::ExtractOp::create(b, loc, paddedInput, inputDims);
              Value pred = arith::CmpFOp::create(
                  b, loc, arith::CmpFPredicate::OEQ, input, maxVal);

              Value outIndex =
                  arith::ConstantOp::create(b, loc, b.getIndexAttr(0));
              Value curInputStride =
                  arith::ConstantOp::create(b, loc, b.getIndexAttr(1));
              for (unsigned i = 0; i < rank - 2; i++) {
                Value minusPadding = arith::SubIOp::create(
                    b, loc, inputDims[rank - 1 - i], padding[rank - 3 - i]);
                Value timesStride =
                    arith::MulIOp::create(b, loc, minusPadding, curInputStride);
                outIndex = arith::AddIOp::create(b, loc, outIndex, timesStride);
                curInputStride = arith::MulIOp::create(
                    b, loc, curInputStride, inputSubShape[rank - 3 - i]);
              }
              Value result = arith::SelectOp::create(
                  b, loc, pred, castIndexToInt64(b, loc, outIndex), res);

              Value predInvalidIndex = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::eq, res, cstMinusOne);
              Value out = arith::SelectOp::create(b, loc, predInvalidIndex,
                                                  result, res);

              linalg::YieldOp::create(b, loc, out);
            })
            .getResult(0);

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

    TypedAttr smallestValueAttr;

    if (auto fpty = dyn_cast<mlir::FloatType>(elementType)) {
      smallestValueAttr = rewriter.getFloatAttr(
          elementType,
          APFloat::getInf(fpty.getFloatSemantics(), /*Negative=*/true));
    } else if (auto intTy = dyn_cast<mlir::IntegerType>(elementType)) {
      int64_t bw = intTy.getIntOrFloatBitWidth();
      smallestValueAttr = rewriter.getIntegerAttr(
          elementType, intTy.isUnsigned() ? APInt::getMinValue(bw)
                                          : APInt::getSignedMinValue(bw));
    }

    if (!smallestValueAttr)
      return rewriter.notifyMatchFailure(op, "invalid element type");

    // `maxPool` contains the result of maxpool 1d/2d/3d operation over the
    // input, `paddedInput` means the padded result of input tensor.
    Value maxPool, paddedInput;
    Type maxPoolResultType =
        typeConverter->convertType(op->getResult(0).getType());
    SmallVector<Value, 5> outTensorShape;
    if constexpr (Dim == 1) {
      if (failed(createPoolingOp<linalg::PoolingNcwMaxOp>(
              op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
              /*dimensionality=*/1, kernelSizeIntValues, strideInts,
              paddingInts, dilationInts, smallestValueAttr, outTensorShape,
              paddedInput, maxPool)))
        return rewriter.notifyMatchFailure(op, "unable to compute maxpool1d");
    } else if constexpr (Dim == 2) {
      if (failed(createPoolingOp<linalg::PoolingNchwMaxOp>(
              op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
              /*dimensionality=*/2, kernelSizeIntValues, strideInts,
              paddingInts, dilationInts, smallestValueAttr, outTensorShape,
              paddedInput, maxPool)))
        return rewriter.notifyMatchFailure(op, "unable to compute maxpool2d");
    } else {
      if (failed(createPoolingMax3D(op, adaptor, rewriter, kernelSizeIntValues,
                                    strideInts, paddingInts, dilationInts,
                                    ceilMode, outTensorShape, paddedInput,
                                    maxPool)))
        return rewriter.notifyMatchFailure(op, "unable to compute maxpool3d");
    }

    Value outMaxPool = tensor::CastOp::create(rewriter, op->getLoc(),
                                              maxPoolResultType, maxPool);
    SmallVector<Value> outResult({outMaxPool});
    if (withIndices) {
      Value indicesResult;
      if (failed(computeMaxPoolingIndices(
              maxPool, paddedInput, op, adaptor, rewriter, outTensorShape,
              kernelSizeIntValues, strideInts, paddingInts, dilationInts,
              selfRank, indicesResult)))
        return rewriter.notifyMatchFailure(op,
                                           "unable to compute maxpool indices");
      Type indicesResultType =
          typeConverter->convertType(op->getResult(1).getType());
      Value outIndices = tensor::CastOp::create(
          rewriter, op->getLoc(), indicesResultType, indicesResult);
      outResult.push_back(outIndices);
    }
    rewriter.replaceOp(op, outResult);

    return success();
  }
};
} // namespace

namespace {
// Max unpooling operation, takes result of max_pooling op and indices and
// tries to reconstructs original pooling input by filling out values by either
// values from self or zero.
// Upstream CPU implementation use parallel loop over the indices array to fill
// out tensor but such approach requires random access writes, which is tricky
// to represent in linalg.
// Instead we are using a different method: we are mapping each input/index
// value to multiple output values via affine maps in linalg.generic, then,
// inside the body of generic, we compute out index and compare it with expected
// index we got from input, returning either input or zero.
// This method only works if we have non-overlapping pooling windows.
// In case of overlap (e.g. kernel_size=2, stride=1) we need to map many-to-many
// input to output values and do a reduction. To construct such mapping we need
// to know original Kernel size, but it doesn't encoded in aten op. We cannot
// reconstruct kernel_size either as such reconstruction is ambiguous (e.g. for
// input_size=2, output_size=5 and stride=2, kernel_size can be either 2 or 3).
// What worse, without knowing kernel size we cannot even reliably detect such
// cases and this conversion will just return invalid values.
class ConvertAtenMaxUnpool3dOp final
    : public OpConversionPattern<AtenMaxUnpool3dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxUnpool3dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
    auto selfType = cast<RankedTensorType>(self.getType());

    ArrayRef<int64_t> spatialInputShape = selfType.getShape().take_back(3);
    if (ShapedType::isDynamicShape(spatialInputShape))
      return rewriter.notifyMatchFailure(op,
                                         "input type must be of static shape");

    Value indices = adaptor.getIndices();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    if (spatialInputShape != indicesType.getShape().take_back(3))
      return rewriter.notifyMatchFailure(op, "input/indices shape mismatch");

    auto resType = typeConverter->convertType<RankedTensorType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "invalid result type");

    ArrayRef<int64_t> inferredOutSize = resType.getShape().take_back(3);
    if (ShapedType::isDynamicShape(inferredOutSize))
      return rewriter.notifyMatchFailure(op,
                                         "output type must be of static shape");

    {
      SmallVector<int64_t> output;
      if (!matchPattern(op.getOutputSize(), m_TorchListOfConstantInts(output)))
        return rewriter.notifyMatchFailure(op,
                                           "only support constant int output");

      if (inferredOutSize != ArrayRef(output))
        return rewriter.notifyMatchFailure(op, "Invalid output size");
    }
    SmallVector<int64_t> stride;
    SmallVector<int64_t> padding;

    if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int padding");

    // TODO: add support for asymmetric padding coming from "onnx.MaxUnpool"
    // (padding.size() == 6).
    if (stride.size() != 3 || padding.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "stride and padding must be of size 3");

    for (auto &&[inDim, outDim, str, pad] :
         llvm::zip_equal(spatialInputShape, inferredOutSize, stride, padding)) {
      // Kernel size computation is ambiguous, this formula will return the
      // biggest possible kernel size. As there is no way to know actual kernel
      // size we have to treat it conservatively and always bail if kernel size
      // potentially bigger than stride.
      int64_t kernelSize = outDim - (inDim - 1) * str + 2 * pad;
      if (kernelSize > str)
        return rewriter.notifyMatchFailure(
            op, "potential pooling windows overlapping is detected, this case "
                "is not supported yet");
    }

    int64_t poolingDimensionality = 3;
    Value result = createMaxUnpoolOp(
        op, poolingDimensionality, rewriter, typeConverter, self, indices,
        spatialInputShape, inferredOutSize, stride, padding, resType);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// Max unpooling operation, takes result of max_pooling op and indices and
// tries to reconstructs original pooling input by filling out values by either
// values from self or zero.
class ConvertAtenMaxUnpool2dOp final
    : public OpConversionPattern<AtenMaxUnpool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxUnpool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    const TypeConverter *typeConverter = getTypeConverter();
    Value self = adaptor.getSelf();
    auto selfType = cast<RankedTensorType>(self.getType());
    int64_t poolingDimensionality = 2;

    ArrayRef<int64_t> inputSize =
        selfType.getShape().take_back(poolingDimensionality);
    if (ShapedType::isDynamicShape(inputSize))
      return rewriter.notifyMatchFailure(op,
                                         "input type must be of static shape");

    Value indices = adaptor.getIndices();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    if (inputSize != indicesType.getShape().take_back(poolingDimensionality))
      return rewriter.notifyMatchFailure(op, "input/indices shape mismatch");

    auto resType = typeConverter->convertType<RankedTensorType>(op.getType());

    ArrayRef<int64_t> inferredOutSize =
        resType.getShape().take_back(poolingDimensionality);
    if (ShapedType::isDynamicShape(inferredOutSize))
      return rewriter.notifyMatchFailure(op,
                                         "output type must be of static shape");

    {
      SmallVector<int64_t> output;
      if (!matchPattern(op.getOutputSize(), m_TorchListOfConstantInts(output)))
        return rewriter.notifyMatchFailure(op,
                                           "only support constant int output");

      if (inferredOutSize != ArrayRef(output))
        return rewriter.notifyMatchFailure(op, "Invalid output size");
    }

    // MaxUnpool2d currently supports only default stride and padding
    SmallVector<int64_t> stride(poolingDimensionality, poolingDimensionality);
    SmallVector<int64_t> padding(poolingDimensionality, 0);

    Value result = createMaxUnpoolOp(op, poolingDimensionality, rewriter,
                                     typeConverter, self, indices, inputSize,
                                     inferredOutSize, stride, padding, resType);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
// The following structures and the getNumOfDims method
// are used to get the number of dimensions from the
// average pooling type at compile time.
template <typename OpTy> struct AtenAvgPoolTypeNumOfDims {
  static constexpr int getNumOfDims() { return -1; }
};
template <> struct AtenAvgPoolTypeNumOfDims<AtenAvgPool1dOp> {
  static constexpr int getNumOfDims() { return 1; }
};
template <> struct AtenAvgPoolTypeNumOfDims<AtenAvgPool2dOp> {
  static constexpr int getNumOfDims() { return 2; }
};
template <> struct AtenAvgPoolTypeNumOfDims<AtenAvgPool3dOp> {
  static constexpr int getNumOfDims() { return 3; }
};
template <typename OpTy> constexpr int getAvgPoolNumOfDims() {
  return AtenAvgPoolTypeNumOfDims<OpTy>::getNumOfDims();
}
} // namespace

namespace {
// This is a helper class to create the pooling size value
// used in the divisor of the average pooling operator.
template <int NumOfDims> class PoolSizeCalculator {
public:
  PoolSizeCalculator(Value self, Value sumPool, bool countIncludePad,
                     ConversionPatternRewriter &rewriter, Location loc);

  // The algorithm for computing the divisor with
  // count_include_pad equal is mainly based on pytorch
  // implementation. The following code is comment
  // with pytorch code.
  // https://github.com/pytorch/pytorch/blob/4a6dfbe4806b361c43210dfd56db64c4097c66bb/aten/src/ATen/native/cpu/AvgPoolKernel.cpp#L78
  // Dim below stands for spatial dimension. It replaces the
  // height and width labels in variables.
  Value getPoolSize(OpBuilder &b, SmallVectorImpl<Value> &kernelSizeIntValues,
                    SmallVectorImpl<int64_t> &strideInts,
                    SmallVectorImpl<int64_t> &paddingInts);

private:
  int64_t SumPoolTypeDimIndex[NumOfDims];
  Value InputSpatialDimSizes[NumOfDims];
  Location location;
  bool isCountIncludePad;
};

} // namespace

template <int NumOfDims>
PoolSizeCalculator<NumOfDims>::PoolSizeCalculator(
    Value self, Value sumPool, bool countIncludePad,
    ConversionPatternRewriter &rewriter, Location loc)
    : location(loc), isCountIncludePad(countIncludePad) {
  auto selfType = cast<RankedTensorType>(self.getType());
  const int64_t selfRank = selfType.getRank();
  RankedTensorType sumPoolType = cast<RankedTensorType>(sumPool.getType());
  const int64_t rank = sumPoolType.getRank();

  // Store dimensions in this order:
  // 0 => depth, 1 => height, 2 => width
  for (int i = 0; i < NumOfDims; ++i) {
    int64_t inputSpatialDimIndex = toPositiveDim(-(i + 1), selfRank);
    InputSpatialDimSizes[NumOfDims - i - 1] =
        getDimOp(rewriter, location, self, inputSpatialDimIndex);
    SumPoolTypeDimIndex[NumOfDims - i - 1] = toPositiveDim(-(i + 1), rank);
  }
}

template <int NumOfDims>
Value PoolSizeCalculator<NumOfDims>::getPoolSize(
    OpBuilder &b, SmallVectorImpl<Value> &kernelDimSizes,
    SmallVectorImpl<int64_t> &strideInts,
    SmallVectorImpl<int64_t> &paddingInts) {
  Value poolSize;

  Value cstZero =
      b.createOrFold<arith::ConstantOp>(location, b.getI64IntegerAttr(0));

  for (int i = 0; i < NumOfDims; ++i) {
    // See the link below for the PyTorch implementation where this is
    // derived from:
    // https://github.com/pytorch/pytorch/blob/4a6dfbe4806b361c43210dfd56db64c4097c66bb/aten/src/ATen/native/cpu/AvgPoolKernel.cpp#L78
    // Dim below stands for spatial dimension. Prior to the February 2025
    // change, these variables used "height" and "width" (or "h" and "w")
    // in these intermediate variables instead of "Dim".

    Value IndexODim = linalg::IndexOp::create(b, location,
                                              /*value=*/SumPoolTypeDimIndex[i]);
    Value ODim = castIndexToInt64(b, location, IndexODim);
    Value DDim = b.createOrFold<arith::ConstantOp>(
        location, b.getI64IntegerAttr(strideInts[i]));
    Value PadDim = b.createOrFold<arith::ConstantOp>(
        location, b.getI64IntegerAttr(paddingInts[i]));
    Value ODimDDim = b.createOrFold<arith::MulIOp>(location, ODim, DDim);
    Value IDim0 = b.createOrFold<arith::SubIOp>(location, ODimDDim, PadDim);
    Value IDim = castIndexToInt64(b, location, InputSpatialDimSizes[i]);
    Value IDim0KDim =
        b.createOrFold<arith::AddIOp>(location, IDim0, kernelDimSizes[i]);
    Value IDimPadDim = b.createOrFold<arith::AddIOp>(location, IDim, PadDim);
    Value IDim1 =
        b.createOrFold<arith::MinSIOp>(location, IDim0KDim, IDimPadDim);

    Value IDim0Clamped =
        b.createOrFold<arith::MaxSIOp>(location, IDim0, cstZero);
    Value IDim1Clamped = b.createOrFold<arith::MinSIOp>(location, IDim1, IDim);
    Value IDim1_IDim0_Clamped =
        b.createOrFold<arith::SubIOp>(location, IDim1Clamped, IDim0Clamped);

    Value poolSizeDim =
        !isCountIncludePad
            ? IDim1_IDim0_Clamped
            : b.createOrFold<arith::SubIOp>(location, IDim1, IDim0);
    if (i == 0) {
      poolSize = poolSizeDim;
    } else {
      poolSize = b.createOrFold<arith::MulIOp>(location, poolSize, poolSizeDim);
    }
  }
  return poolSize;
}

namespace {
template <typename OpTy, typename PoolingOpTy, int Dim>
class ConvertAtenAvgPoolOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  // If the condition below is true, the divisor total must subtract the
  // elements not counted (clamped divisor count). If false, the divisor
  // is just the product of kernel dimensions.
  static bool
  doesAvgPoolDivisorNeedsClamping(bool ceilMode, bool countIncludePad,
                                  SmallVectorImpl<int64_t> &strideInts,
                                  SmallVectorImpl<int64_t> &paddingInts);

  // Creates the average pooling operation value with a clamped
  // divisor. The clamped divisor is the product of kernel
  // dimensions minus the elements not counted; e.g., padding
  // and ceiling mode implicit padding.
  static LogicalResult createAveragePoolValueWithClampedDivisor(
      bool ceilMode, bool countIncludePad, OpTy op,
      typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter,
      Value self, Value sumPool, Value outputTensor, Type resultType,
      SmallVectorImpl<Value> &kernelDimSizes,
      SmallVectorImpl<int64_t> &strideInts,
      SmallVectorImpl<int64_t> &paddingInts,
      SmallVector<AffineMap> &indexingMapsAvg,
      SmallVector<utils::IteratorType> &iteratorTypesAvg);

  // Creates the average pooling operation value with a
  // regular divisor; i.e., the product of kernel dimensions.
  static LogicalResult createAveragePoolValueWithRegularDivisor(
      OpTy op, typename OpTy::Adaptor &adaptor,
      ConversionPatternRewriter &rewriter, Value self, Value sumPool,
      Value outputTensor, Type resultType,
      SmallVectorImpl<Value> &kernelDimSizes,
      SmallVector<AffineMap> &indexingMapsAvg,
      SmallVector<utils::IteratorType> &iteratorTypesAvg);
};
} // namespace

template <typename OpTy, typename PoolingOpTy, int Dim>
LogicalResult ConvertAtenAvgPoolOp<OpTy, PoolingOpTy, Dim>::matchAndRewrite(
    OpTy op, typename OpTy::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
    return failure();

  Location loc = op->getLoc();
  const TypeConverter *typeConverter = this->getTypeConverter();
  Value self = adaptor.getSelf();

  Type inputElementType =
      cast<RankedTensorType>(self.getType()).getElementType();
  Type resultType = typeConverter->convertType(op.getType());
  Type resultElementType = cast<RankedTensorType>(resultType).getElementType();

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

  bool countIncludePad;
  if (!matchPattern(op.getCountIncludePad(),
                    m_TorchConstantBool(&countIncludePad)))
    return rewriter.notifyMatchFailure(op,
                                       "count_include_pad must be a constant");

  // `sumPool` contains the result of sumpool operation over the input.
  Value sumPool, paddedInput;
  SmallVector<Value, Dim + 2> outTensorShape;
  if (failed(createPoolingOp<PoolingOpTy>(
          op, rewriter, self, /*supportNonFPInput=*/true, ceilMode,
          /*dimensionality=*/Dim, kernelSizeIntValues, strideInts, paddingInts,
          dilationInts, rewriter.getZeroAttr(inputElementType), outTensorShape,
          paddedInput, sumPool)))
    return rewriter.notifyMatchFailure(op, "unable to compute sumpool");

  // Compute the average of sumPool.
  Value outputTensor = tensor::EmptyOp::create(
      rewriter, loc, getAsOpFoldResult(outTensorShape), resultElementType);
  SmallVector<AffineMap> indexingMapsAvg(
      2, rewriter.getMultiDimIdentityMap(Dim + 2));
  SmallVector<utils::IteratorType> iteratorTypesAvg(
      Dim + 2, utils::IteratorType::parallel);

  if (doesAvgPoolDivisorNeedsClamping(ceilMode, countIncludePad, strideInts,
                                      paddingInts)) {
    return createAveragePoolValueWithClampedDivisor(
        ceilMode, countIncludePad, op, adaptor, rewriter, self, sumPool,
        outputTensor, resultType, kernelSizeIntValues, strideInts, paddingInts,
        indexingMapsAvg, iteratorTypesAvg);
  }

  return createAveragePoolValueWithRegularDivisor(
      op, adaptor, rewriter, self, sumPool, outputTensor, resultType,
      kernelSizeIntValues, indexingMapsAvg, iteratorTypesAvg);
}

template <typename OpTy, typename PoolingOpTy, int Dim>
bool ConvertAtenAvgPoolOp<OpTy, PoolingOpTy, Dim>::
    doesAvgPoolDivisorNeedsClamping(bool ceilMode, bool countIncludePad,
                                    SmallVectorImpl<int64_t> &strideInts,
                                    SmallVectorImpl<int64_t> &paddingInts) {
  // Determines whether the average pooling divisor needs to be clamped
  // (i.e., adjusted to exclude padded or out-of-bounds elements).
  //
  // There are two primary cases where clamping is needed:
  // 1. Padding with count_include_pad == false:
  //    - If padding is applied (padding != 0) and count_include_pad is false,
  //      then padding elements are *excluded* from the divisor, effectively
  //      clamping the divisor to the number of valid input elements.
  //
  // 2. Ceil mode with non-unit stride:
  //    - When ceil_mode is enabled, output dimensions are rounded up,
  //    potentially
  //      creating pooling windows that extend beyond the input tensor bounds.
  //      PyTorch handles this by implicitly adding zero-padding outside the
  //      tensor, but these extra (implicit) padded elements are *not* included
  //      in the divisor. This behavior is independent of the count_include_pad
  //      flag.
  //    - If all strides are 1, ceil_mode will not produce fractional divisions,
  //      so the windows will not extend beyond bounds, and no clamping occurs.
  //
  // Reference: PyTorch AvgPool2d documentation and formula for H_out/W_out:
  // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
  //
  // See torch.nn.AvgPool2d E2E tests for comprehensive coverage.

  bool hasPadding =
      !llvm::all_of(paddingInts, [](int64_t p) { return p == 0; });
  bool allStridesUnitary =
      llvm::all_of(strideInts, [](int64_t s) { return s == 1; });

  return (!countIncludePad && hasPadding) || (ceilMode && !allStridesUnitary);
}

template <typename OpTy, typename PoolingOpTy, int Dim>
LogicalResult ConvertAtenAvgPoolOp<OpTy, PoolingOpTy, Dim>::
    createAveragePoolValueWithClampedDivisor(
        bool ceilMode, bool countIncludePad, OpTy op,
        typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter,
        Value self, Value sumPool, Value outputTensor, Type resultType,
        SmallVectorImpl<Value> &kernelDimSizes,
        SmallVectorImpl<int64_t> &strideInts,
        SmallVectorImpl<int64_t> &paddingInts,
        SmallVector<AffineMap> &indexingMapsAvg,
        SmallVector<utils::IteratorType> &iteratorTypesAvg) {
  Location loc = op->getLoc();

  constexpr int avgPoolDims = getAvgPoolNumOfDims<OpTy>();

  if (avgPoolDims < 1) {
    return rewriter.notifyMatchFailure(
        op, "Unexpected type. Only expected AtenAvgPool1dOp, AtenAvgPool2dOp, "
            "and AtenAvgPool3dOp.");
  }

  Type resultElementType = cast<RankedTensorType>(resultType).getElementType();

  PoolSizeCalculator<avgPoolDims> poolSizeCalculator(
      self, sumPool, countIncludePad, rewriter, loc);

  // AtenAvgPool2/3dOp has an optional divisor_override
  // attribute while AtenAvgPool1dOp does not.
  // We evaluate the constexpr avgPoolDims outside of the lambda capture below
  // for wider compiler support: https://github.com/llvm/torch-mlir/issues/4085.
  Value poolSize = nullptr;
  if constexpr (avgPoolDims > 1) {
    if (!isa<Torch::NoneType>(op.getDivisorOverride().getType()))
      poolSize = adaptor.getDivisorOverride();
  }

  Value avgPool =
      linalg::GenericOp::create(
          rewriter, loc, outputTensor.getType(), sumPool, outputTensor,
          /*indexingMaps=*/indexingMapsAvg,
          /*iteratorTypes=*/iteratorTypesAvg,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            if (!poolSize) {
              poolSize = poolSizeCalculator.getPoolSize(
                  b, kernelDimSizes, strideInts, paddingInts);
            }
            Value divisor =
                convertScalarToDtype(b, loc, poolSize, resultElementType);
            Value avg;
            if (isa<mlir::IntegerType>(resultElementType))
              avg = b.createOrFold<arith::DivSIOp>(loc, args[0], divisor);
            else if (isa<mlir::FloatType>(resultElementType))
              avg = b.createOrFold<arith::DivFOp>(loc, args[0], divisor);
            b.createOrFold<linalg::YieldOp>(loc, avg);
          })
          .getResult(0);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, avgPool);
  return success();
}

template <typename OpTy, typename PoolingOpTy, int Dim>
LogicalResult ConvertAtenAvgPoolOp<OpTy, PoolingOpTy, Dim>::
    createAveragePoolValueWithRegularDivisor(
        OpTy op, typename OpTy::Adaptor &adaptor,
        ConversionPatternRewriter &rewriter, Value self, Value sumPool,
        Value outputTensor, Type resultType,
        SmallVectorImpl<Value> &kernelDimSizes,
        SmallVector<AffineMap> &indexingMapsAvg,
        SmallVector<utils::IteratorType> &iteratorTypesAvg) {
  Location loc = op->getLoc();

  Type resultElementType = cast<RankedTensorType>(resultType).getElementType();

  Value divisor = kernelDimSizes[0];
  for (uint32_t i = 1; i < kernelDimSizes.size(); ++i) {
    divisor =
        rewriter.createOrFold<arith::MulIOp>(loc, divisor, kernelDimSizes[i]);
  }
  // Only average pooling 2D/3D have optional divisor override.
  if constexpr (!std::is_same<OpTy, AtenAvgPool1dOp>()) {
    divisor = isa<Torch::NoneType>(op.getDivisorOverride().getType())
                  ? divisor
                  : adaptor.getDivisorOverride();
  }
  divisor = convertScalarToDtype(rewriter, loc, divisor, resultElementType);

  Value avgPool =
      linalg::GenericOp::create(
          rewriter, loc, outputTensor.getType(), sumPool, outputTensor,
          /*indexingMaps=*/indexingMapsAvg,
          /*iteratorTypes=*/iteratorTypesAvg,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value avg;
            if (isa<mlir::IntegerType>(resultElementType))
              avg = arith::DivSIOp::create(b, loc, args[0], divisor);
            else if (isa<mlir::FloatType>(resultElementType))
              avg = arith::DivFOp::create(b, loc, args[0], divisor);
            linalg::YieldOp::create(b, loc, avg);
          })
          .getResult(0);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, avgPool);
  return success();
}

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
    buffVal = arith::ConstantOp::create(rewriter, loc, elementType,
                                        smallestFPValueAttr);
    auxTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(outputSizes), auxTensorElementType);
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
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, inElt, res);
    out2 = arith::SelectOp::create(b, loc, cond1, inElt, res);
    // index in different dims (n x c x d x h x w)
    // 1d: (iw)
    // 2d: (ih*W + iw)
    // 3d: (id*H*W + ih*W + iw)
    Value currIndex = inputElementIndices[nonSpatial];
    for (unsigned i = 0; i < rank - nonSpatial - 1; i++) {
      Value prevTimesNewSize =
          arith::MulIOp::create(b, loc, currIndex, inputSpatialSizes[i + 1]);
      currIndex = arith::AddIOp::create(
          b, loc, prevTimesNewSize, inputElementIndices[nonSpatial + i + 1]);
    }
    Value indexOut1Int = castIndexToInt64(b, loc, currIndex);
    auxOut = arith::SelectOp::create(b, loc, cond1, indexOut1Int, maxIndex);
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
        tensor::CastOp::create(rewriter, loc, outputType, adaptivePoolOutput);
    Value outputIndices =
        tensor::CastOp::create(rewriter, loc, auxTensorType, auxTensorReturn);
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
    buffVal = arith::ConstantOp::create(rewriter, loc, elementType,
                                        rewriter.getFloatAttr(elementType, 0));
    auxTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(outShapeIndexVector), elementType);
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
    out2 = arith::AddFOp::create(b, loc, inElt, res);
    Value kernelVolume = indexOne;
    for (unsigned i = 0; i < rank - nonSpatial; i++) {
      Value currSize = arith::SubIOp::create(b, loc, ends[i], starts[i]);
      kernelVolume = arith::MulIOp::create(b, loc, kernelVolume, currSize);
    }
    Value auxOutSI = castIndexToInt64(b, loc, kernelVolume);
    auxOut = arith::SIToFPOp::create(b, loc, elementType, auxOutSI);
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
    auto output = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/adaptivePoolOutput.getType(),
        /*inputs=*/auxTensorReturn,
        /*outputs=*/adaptivePoolOutput,
        /*indexingMaps=*/indexingMaps1,
        /*iteratorTypes=*/iteratorTypes1,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value q = arith::DivFOp::create(b, loc, args[1], args[0]);
          linalg::YieldOp::create(b, loc, q);
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
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
    for (int i = 0; i < rank - nonSpatial; i++) {
      Value hInPlusOne = arith::SubIOp::create(
          rewriter, loc, inputSpatialSizes[i], constantOne);
      Value kMaxMinusOne = arith::CeilDivSIOp::create(rewriter, loc, hInPlusOne,
                                                      outShapeIndexVector[i]);
      Value kMax =
          arith::AddIOp::create(rewriter, loc, constantOne, kMaxMinusOne);
      kIterSizeVector.push_back(kMax);
    }
    Value kIter = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(kIterSizeVector), boolType);

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
    Value indexOne = arith::ConstantIndexOp::create(rewriter, loc, 1);

    bool failedCustomization = false;
    // adaptive pooling generic op
    auto adaptivePool = linalg::GenericOp::create(
        rewriter, loc, /*resultTensorTypes=*/
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
            ind.push_back(linalg::IndexOp::create(b, loc, i));
          }
          // compute start and end indices
          // st = s1( s0(ind2 * Hin) // Hout )
          SmallVector<Value> starts;
          SmallVector<Value> ends;
          for (unsigned i = nonSpatial; i < rank; i++) {
            Value s0 = arith::MulIOp::create(b, loc, ind[i],
                                             inputSpatialSizes[i - nonSpatial]);
            Value s1 = arith::FloorDivSIOp::create(
                b, loc, s0, outShapeIndexVector[i - nonSpatial]);
            starts.push_back(s1);
            // en = e4( 1 + e3( e2( e1( e0(ind2 + 1) * hIn ) - 1 ) // hOut ) )
            Value e0 = arith::AddIOp::create(b, loc, ind[i], indexOne);
            Value e1 = arith::MulIOp::create(b, loc, e0,
                                             inputSpatialSizes[i - nonSpatial]);
            Value e2 = arith::SubIOp::create(b, loc, e1, indexOne);
            Value e3 = arith::FloorDivSIOp::create(
                b, loc, e2, outShapeIndexVector[i - nonSpatial]);
            Value e4 = arith::AddIOp::create(b, loc, indexOne, e3);
            ends.push_back(e4);
          }
          // extract input element
          SmallVector<Value> inputElementIndices;
          for (unsigned i = 0; i < nonSpatial; i++) {
            inputElementIndices.push_back(ind[i]);
          }
          for (unsigned i = nonSpatial; i < rank; i++) {
            inputElementIndices.push_back(arith::AddIOp::create(
                b, loc, starts[i - nonSpatial], ind[rank - nonSpatial + i]));
          }
          Value inElt = tensor::ExtractOp::create(
              b, loc, elementType, buffInput, inputElementIndices);
          // check if we extracted at windex < end index
          for (unsigned i = 0; i < rank - nonSpatial; i++) {
            Value cond = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate(6),
                inputElementIndices[i + nonSpatial], ends[i]);
            // if out-of-bounds, replace the extracted element with buffVal
            inElt = arith::SelectOp::create(b, loc, cond, inElt, buffVal);
          }
          Value out2, auxOut;
          // customize for max vs. avg:
          if (failed(adaptivePoolingHelper.payloadCustomization(
                  b, loc, inElt, res, maxIndex, inputElementIndices,
                  inputSpatialSizes, indexOne, starts, ends, out2, auxOut))) {
            failedCustomization = true;
          }
          linalg::YieldOp::create(b, loc, ValueRange({out2, auxOut}));
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

  target.addIllegalOp<AtenMaxPool1dWithIndicesOp>();
  target.addIllegalOp<AtenMaxPool2dWithIndicesOp>();
  target.addIllegalOp<AtenMaxPool3dWithIndicesOp>();
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool1dWithIndicesOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool2dWithIndicesOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenMaxPoolOp<AtenMaxPool3dWithIndicesOp>>(typeConverter,
                                                                 context);

  target.addIllegalOp<AtenMaxUnpool3dOp>();
  patterns.add<ConvertAtenMaxUnpool3dOp>(typeConverter, context);

  target.addIllegalOp<AtenMaxUnpool2dOp>();
  patterns.add<ConvertAtenMaxUnpool2dOp>(typeConverter, context);

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
