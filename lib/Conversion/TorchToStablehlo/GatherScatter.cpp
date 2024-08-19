//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

namespace {
static Value createInitialValueForGatherScatterOp(Operation *op,
                                                  RankedTensorType constType,
                                                  PatternRewriter &rewriter) {
  if (!constType.hasStaticShape()) {
    return nullptr;
  }
  auto elementTy = constType.getElementType();
  if (isa<AtenEmbeddingBagPaddingIdxOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (isa<mlir::IntegerType>(elementTy) &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  op->emitError("unimplemented lowering in "
                "createInitialValueForGatherScatterOp");
  return nullptr;
}

Value gatherTensorAlongSingleAxis(PatternRewriter &rewriter, Operation *op,
                                  Value input, Value indices, int64_t axis,
                                  size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  Type intType = rewriter.getIntegerType(dimSizeIndexBits);
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));

  // sliceSizes
  auto inputRankTy = dyn_cast<RankedTensorType>(input.getType());
  auto inputRank = inputRankTy.getRank();
  SmallVector<Value, 4> sliceSizes;
  sliceSizes.reserve(inputRank);
  for (int64_t r = 0; r < inputRank; ++r) {
    if (r == axis) {
      sliceSizes.push_back(one);
    } else {
      sliceSizes.push_back(rewriter.create<arith::IndexCastOp>(
          loc, intType, rewriter.create<tensor::DimOp>(loc, input, r)));
    }
  }
  auto sliceSizesTensor =
      rewriter.create<tensor::FromElementsOp>(loc, sliceSizes);

  // offsetDims
  SmallVector<int64_t, 4> offsetDims;
  offsetDims.reserve(inputRank);
  for (int64_t r = 0; r < axis; ++r) {
    offsetDims.push_back(r);
  }
  auto indicesRankTy = dyn_cast<RankedTensorType>(indices.getType());
  auto indicesRank = indicesRankTy.getRank();
  for (int64_t r = axis + 1; r < inputRank; ++r) {
    offsetDims.push_back(r + indicesRank - 1);
  }

  // collapsedSliceDims
  SmallVector<int64_t, 4> collapsedSliceDims(1, axis);
  // startIndexMap
  SmallVector<int64_t, 4> startIndexMap(1, axis);
  // indexVecDim
  int64_t indexVecDim = indicesRank;
  auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedSliceDims,
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  // outputShape = input.shape[:axis] + indices.shape +
  //                input.shape[axis + 1:]
  auto inputShape = inputRankTy.getShape();
  auto indicesShape = indicesRankTy.getShape();
  SmallVector<int64_t, 4> outputShape(inputShape.begin(),
                                      inputShape.begin() + axis);
  outputShape.insert(outputShape.end(), indicesShape.begin(),
                     indicesShape.end());
  outputShape.insert(outputShape.end(), inputShape.begin() + axis + 1,
                     inputShape.end());

  // create output tensor type
  auto outputTy =
      RankedTensorType::get(outputShape, inputRankTy.getElementType());
  return rewriter
      .create<stablehlo::DynamicGatherOp>(loc, outputTy, input, indices,
                                          sliceSizesTensor, dimsAttr)
      .getResult();
}

template <typename OpTy, typename OpAdaptor>
LogicalResult prepareArgumentsForSlicingOp(OpTy op, OpAdaptor adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           SmallVector<Value> &resultShape,
                                           SmallVector<Value> &offsets,
                                           SmallVector<Value> &strides) {
  Location loc = op.getLoc();
  auto input = adaptor.getSelf();
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return op->emitError("unimplemented: dim is not constant");

  int64_t inputRank = inputType.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
  Value dimSize = inputShape[dim];

  Value torchTypeStart = op.getStart();
  Value torchTypeEnd = op.getEnd();
  Value builtinTypeStart = adaptor.getStart();
  Value builtinTypeEnd = adaptor.getEnd();

  if (isa<OptionalType>(torchTypeStart.getType()) ||
      isa<OptionalType>(torchTypeEnd.getType()))
    return rewriter.notifyMatchFailure(op, "unimplemented optional type arg");

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step))) {
    if (!isa<Torch::NoneType>(op.getStep().getType()))
      return op->emitError("unimplemented: step is not constant");
    step = 1;
  }

  Value start = toPositiveValidDim(rewriter, loc, torchTypeStart,
                                   builtinTypeStart, zero, dimSize);
  Value end = toPositiveValidDim(rewriter, loc, torchTypeEnd, builtinTypeEnd,
                                 dimSize, dimSize);

  // end >= start ? end : start
  Value endSgeStart = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, end, start);
  end = rewriter.create<arith::SelectOp>(loc, endSgeStart, end, start);
  Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);

  // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
  resultShape = getTensorSizes(rewriter, loc, input);
  Value len = rewriter.create<arith::SubIOp>(loc, end, start);
  Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
  resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, one);
  resultSize = rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);
  resultShape[dim] = resultSize;

  strides.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);

  offsets[dim] = start;
  strides[dim] = rewriter.create<arith::MulIOp>(loc, strides[dim], stepIndex);
  return success();
}
} // namespace

namespace {
// A helper function used to generate stablehlo's ScatterIndices or
// GatherIndices from torch's indices, usually appear in torch ops, like
// aten.index.Tensor or aten.input_put A usage example is as follow: Input: [[1,
// 2, 3],
//         [4, 5, 6],
//         [7, 8, 9]]
// Indices[0]: [[0, 0, 0],
//              [2, 2, 0]]
// Indices[1]: [[2],
//              [1]]
// Step 1: broadcast indices tensors
// Indices[0]: [[0, 0, 0],
//              [2, 2, 0]]
// Indices[1]: [[2, 2, 2],
//              [1, 1, 1]]
// Step 2: concat index tensors at a unsqueezed -1 dimension.
// Indices: [[[0, 2], [0, 2], [0, 2]],
//           [[2, 1], [2, 1], [0, 1]]]
FailureOr<Value> broadcastAndConcatIndices(Operation *op,
                                           ConversionPatternRewriter &rewriter,
                                           SmallVector<Value> indexTensors,
                                           llvm::ArrayRef<int64_t> inputShape,
                                           size_t dimSizeIndexBits,
                                           int &maxIndexRank) {
  // Step 1: broadcast indices tensors
  SmallVector<int64_t> indicesShape;
  SmallVector<int64_t> expandShape;
  SmallVector<int64_t> concatShape;

  bool allIndexStaticShape = true;
  Value bcastSizeTensor;

  // concat index tensor into to indices tensor for concat
  for (size_t i = 0; i < indexTensors.size(); i++) {
    auto indexTensor = indexTensors[i];
    auto indexTensorType = cast<RankedTensorType>(indexTensor.getType());
    for (int64_t size : makeShapeTorchCompatible(indexTensorType.getShape())) {
      if (size == kUnknownSize)
        allIndexStaticShape = false;
    }
    maxIndexRank = std::max(maxIndexRank, (int)indexTensorType.getRank());
  }

  if (!allIndexStaticShape) {
    auto bcastSizeTensorInfo = hlo::getBroadcastResultShape(
        rewriter, op, indexTensors, dimSizeIndexBits);
    if (failed(bcastSizeTensorInfo)) {
      return failure();
    }
    bcastSizeTensor = *bcastSizeTensorInfo;
  }

  for (int i = 0; i < maxIndexRank; i++) {
    indicesShape.push_back(inputShape[i]);
    expandShape.push_back(inputShape[i]);
    concatShape.push_back(inputShape[i]);
  }
  expandShape.push_back(1);
  concatShape.push_back(indexTensors.size());

  SmallVector<Value> broadcastedIndices;
  Type indexElemTy = rewriter.getI64Type();
  RankedTensorType bcastIndexType =
      RankedTensorType::get(indicesShape, indexElemTy);
  for (auto indexTensor : indexTensors) {
    Value bcastVal;
    RankedTensorType reshapeType =
        RankedTensorType::get(expandShape, indexElemTy);
    if (allIndexStaticShape) {
      bcastVal = hlo::promoteAndBroadcast(rewriter, indexTensor, bcastIndexType,
                                          std::nullopt);
      bcastVal = rewriter.create<stablehlo::ReshapeOp>(op->getLoc(),
                                                       reshapeType, bcastVal);
    } else {
      bcastVal = hlo::promoteAndBroadcast(rewriter, indexTensor, bcastIndexType,
                                          bcastSizeTensor);
      auto bcastValShapeTensorVec =
          *hlo::getDimSizesOfTensor(rewriter, op, bcastVal, dimSizeIndexBits);
      bcastValShapeTensorVec.push_back(rewriter.create<mlir::arith::ConstantOp>(
          op->getLoc(), rewriter.getIntegerAttr(
                            rewriter.getIntegerType(dimSizeIndexBits), 1)));
      Value bcastValShapeTensor = rewriter
                                      .create<tensor::FromElementsOp>(
                                          op->getLoc(), bcastValShapeTensorVec)
                                      .getResult();
      bcastVal = rewriter.create<stablehlo::DynamicReshapeOp>(
          op->getLoc(), reshapeType, bcastVal, bcastValShapeTensor);
    }
    broadcastedIndices.push_back(bcastVal);
  }

  // Step 2: concat index tensors at a unsqueezed -1 dimension.
  Value finalIndexTensor = broadcastedIndices[0];
  if (broadcastedIndices.size() > 1) {
    RankedTensorType concatTy = RankedTensorType::get(concatShape, indexElemTy);
    finalIndexTensor = rewriter.create<stablehlo::ConcatenateOp>(
        op->getLoc(), concatTy, ValueRange(broadcastedIndices),
        concatShape.size() - 1);
  }
  return finalIndexTensor;
}
} // namespace

// Ref:
// https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
// padding_idx (int, optional)
//  – If specified, the entries at padding_idx do not contribute to the
//  gradient; therefore, the embedding vector at padding_idx is not updated
//  during training, i.e. it remains as a fixed “pad”.
// scale_grad_by_freq (boolean, optional)
//  – If given, this will scale gradients by the inverse of frequency of the
//  words in the mini-batch. Default False.
// sparse (bool, optional)
//  – If True, gradient w.r.t. weight matrix will be a sparse tensor.
template <>
LogicalResult ConvertAtenOp<AtenEmbeddingOp>::matchAndRewrite(
    AtenEmbeddingOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto weight = adaptor.getWeight();
  auto weightTy = cast<RankedTensorType>(weight.getType());
  if (!weightTy)
    return op.emitError("only ranked tensor types are supported");

  int64_t padding_idx;
  if (!matchPattern(op.getPaddingIdx(), m_TorchConstantInt(&padding_idx)))
    return rewriter.notifyMatchFailure(
        op, "only constant padding_idx is currently supported");

  bool scale_grad_by_freq;
  if (!matchPattern(op.getScaleGradByFreq(),
                    m_TorchConstantBool(&scale_grad_by_freq)))
    return rewriter.notifyMatchFailure(
        op, "only constant scale_grad_by_freq is currently supported");
  if (scale_grad_by_freq)
    return rewriter.notifyMatchFailure(
        op, "scale gradients is currently not supported");
  bool sparse;
  if (!matchPattern(op.getSparse(), m_TorchConstantBool(&sparse)))
    return rewriter.notifyMatchFailure(
        op, "only constant sparse is currently supported");
  if (sparse)
    return rewriter.notifyMatchFailure(
        op, "sparse gradients is currently not supported");

  Value output = gatherTensorAlongSingleAxis(
      rewriter, op, weight, adaptor.getIndices(), 0, options.dimSizeIndexBits);
  rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), output);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenEmbeddingBagPaddingIdxOp>::matchAndRewrite(
    AtenEmbeddingBagPaddingIdxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value weight = adaptor.getWeight();
  Value indices = adaptor.getIndices();
  Value offsets = adaptor.getOffsets();

  auto weightTy = cast<RankedTensorType>(weight.getType());
  if (weightTy && weightTy.hasStaticShape() && weightTy.getRank() != 2)
    return rewriter.notifyMatchFailure(
        op, "weight must be rank 2 tensor with static shapes");

  auto indicesTy = cast<RankedTensorType>(indices.getType());
  if (indicesTy && indicesTy.hasStaticShape() && indicesTy.getRank() != 1)
    return rewriter.notifyMatchFailure(
        op, "indices must be a vector with static shapes");

  auto offsetsTy = cast<RankedTensorType>(offsets.getType());
  if (offsetsTy && offsetsTy.getRank() != 1 && offsetsTy.hasStaticShape() &&
      offsetsTy.getShape()[0] == 1)
    return rewriter.notifyMatchFailure(
        op, "offsets must be a vector with static shape equal to 1");

  if (!isa<Torch::NoneType>(op.getPaddingIdx().getType()))
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: padding_idx should be none");

  if (!isa<Torch::NoneType>(op.getPerSampleWeights().getType()))
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: per_sample_weights should be none");

  bool includeLastOffset;
  if (!matchPattern(op.getIncludeLastOffset(),
                    m_TorchConstantBool(&includeLastOffset))) {
    return rewriter.notifyMatchFailure(
        op, "include_last_offset is expected to be a constant boolean value.");
  }
  if (includeLastOffset)
    return rewriter.notifyMatchFailure(
        op, "include_last_offset is currently not supported");

  bool scaleGradByFreq;
  if (!matchPattern(op.getScaleGradByFreq(),
                    m_TorchConstantBool(&scaleGradByFreq)))
    return rewriter.notifyMatchFailure(
        op, "only constant scale_grad_by_freq is currently supported");
  if (scaleGradByFreq)
    return rewriter.notifyMatchFailure(
        op, "scale gradients is currently not supported");

  bool sparse;
  if (!matchPattern(op.getSparse(), m_TorchConstantBool(&sparse)))
    return rewriter.notifyMatchFailure(
        op, "only constant sparse is currently supported");
  if (sparse)
    return rewriter.notifyMatchFailure(
        op, "sparse gradients is currently not supported");

  int64_t modeInt;
  if (!matchPattern(op.getMode(), m_TorchConstantInt(&modeInt))) {
    return rewriter.notifyMatchFailure(
        op, "mode is expected to be a constant integer value.");
  }
  if (modeInt != torch_upstream::EmbeddingBagMode::MODE_SUM) {
    return rewriter.notifyMatchFailure(op,
                                       "Unimplemented: Mean and Max mode are "
                                       "not supported yet for EmbeddingBag.");
  }

  const auto &options =
      ConvertAtenOp<AtenEmbeddingBagPaddingIdxOp>::getOptions();
  auto weightDimSizes =
      *hlo::getDimSizesOfTensor(rewriter, op, weight, options.dimSizeIndexBits);
  auto indicesDimSizes = *hlo::getDimSizesOfTensor(rewriter, op, indices,
                                                   options.dimSizeIndexBits);
  auto offsetsDimSizes = *hlo::getDimSizesOfTensor(rewriter, op, offsets,
                                                   options.dimSizeIndexBits);

  Value gatherOutput = gatherTensorAlongSingleAxis(
      rewriter, op, weight, indices, 0, options.dimSizeIndexBits);

  Type elementTy = weightTy.getElementType();
  auto constType = RankedTensorType::get({}, elementTy);
  Value initValue =
      createInitialValueForGatherScatterOp(op, constType, rewriter);
  if (!initValue)
    return failure();

  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op.getLoc(), gatherOutput, initValue, rewriter.getDenseI64ArrayAttr({0}),
      elementTy);

  Region &region = stablehloReduceOp.getBody();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, elementTy);

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult);
  }

  auto outShapeInfo = hlo::getDimIndexOfTensor(rewriter, op, weight);
  if (failed(outShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto outShapeVec = *outShapeInfo;
  auto one = rewriter.create<mlir::arith::ConstantOp>(
      op->getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  outShapeVec[0] = one;
  auto outShapeTensor =
      rewriter.create<mlir::tensor::FromElementsOp>(op->getLoc(), outShapeVec);
  auto resultA = rewriter.create<stablehlo::DynamicReshapeOp>(
      loc, getTypeConverter()->convertType(op.getType(0)),
      stablehloReduceOp.getResult(0), outShapeTensor);

  RankedTensorType resultType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(1).getType()));
  Value resultB =
      createInitialValueForGatherScatterOp(op, resultType, rewriter);
  if (!resultB)
    return failure();

  resultType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(2).getType()));
  Value resultC =
      createInitialValueForGatherScatterOp(op, resultType, rewriter);
  if (!resultC)
    return failure();

  resultType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(3).getType()));
  Value resultD =
      createInitialValueForGatherScatterOp(op, resultType, rewriter);
  if (!resultD)
    return failure();

  rewriter.replaceOp(op, {resultA, resultB, resultC, resultD});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");
  int64_t inputRank = selfTy.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  Value output = gatherTensorAlongSingleAxis(
      rewriter, op, self, adaptor.getIndex(), dim, options.dimSizeIndexBits);

  rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), output);

  return success();
}

// AtenGatherOp
template <>
LogicalResult ConvertAtenOp<AtenGatherOp>::matchAndRewrite(
    AtenGatherOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  Value index = adaptor.getIndex();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto indexType = cast<RankedTensorType>(index.getType());
  auto indexElemType = indexType.getElementType();

  if (indexType.getRank() != inputType.getRank()) {
    return op.emitError("`index` and `input` param should have the same rank");
  }
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(
        op, "only constant int `dim` param supported");
  }
  dim = toPositiveDim(dim, inputType.getRank());
  if (!isValidDim(dim, inputType.getRank())) {
    return rewriter.notifyMatchFailure(op, "invalid `dim` param detected");
  }

  bool sparseGrad = false;
  if (!matchPattern(op.getSparseGrad(), m_TorchConstantBool(&sparseGrad))) {
    return rewriter.notifyMatchFailure(
        op, "only constant boolean `sparse_grad` param supported");
  }

  auto indexShapeInfo = hlo::getDimIndexOfTensor(rewriter, op, index);
  if (failed(indexShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dim sizes of `index` param");
  }
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  auto toConcatIndexShapeValueVec = *indexShapeInfo;
  toConcatIndexShapeValueVec.push_back(one);
  auto toConcatIndexShape =
      rewriter.create<tensor::FromElementsOp>(loc, toConcatIndexShapeValueVec);

  auto indexShape = indexType.getShape();
  SmallVector<int64_t> toConcatIndexShapeVec(indexShape.begin(),
                                             indexShape.end());
  toConcatIndexShapeVec.push_back(1);
  RankedTensorType toConcatIndexType =
      RankedTensorType::get(toConcatIndexShapeVec, indexElemType);

  SmallVector<Value> toConcat;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (i == dim) {
      toConcat.push_back(rewriter.create<stablehlo::DynamicReshapeOp>(
          loc, toConcatIndexType, index, toConcatIndexShape));
    } else {
      toConcat.push_back(rewriter.create<stablehlo::DynamicIotaOp>(
          loc, toConcatIndexType, toConcatIndexShape,
          rewriter.getI64IntegerAttr(i)));
    }
  }
  auto gatherIndicies = rewriter.create<stablehlo::ConcatenateOp>(
      loc, toConcat, static_cast<uint64_t>(inputType.getRank()));
  SmallVector<int64_t> sliceSizes(inputType.getRank(), 1);

  int64_t indexVecDim = inputType.getRank();
  SmallVector<int64_t> collapsedDims;
  SmallVector<int64_t> startIndexMap;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    collapsedDims.push_back(i);
    startIndexMap.push_back(i);
  }

  auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/{},
      /*collapsedSliceDims=*/collapsedDims,
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
      op, input, gatherIndicies, dimsAttr,
      rewriter.getDenseI64ArrayAttr(sliceSizes));
  return success();
}

// AtenSliceScatterOp
template <>
LogicalResult ConvertAtenOp<AtenSliceScatterOp>::matchAndRewrite(
    AtenSliceScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
    return failure();

  Location loc = op.getLoc();
  const TypeConverter *typeConverter = getTypeConverter();

  auto input = adaptor.getSelf();
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());

  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return op->emitError("unimplemented: dim is not constant");
  }

  int64_t inputRank = inputType.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank)) {
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");
  }

  auto inputShape = inputType.getShape();
  auto dimSize = inputShape[dim];
  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step))) {
    return op->emitError("unimplemented: step is not constant");
  }

  int64_t start;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start))) {
    return op->emitError("unimplemented: start is not constant");
  } else if (ShapedType::isDynamic(dimSize) and start < 0) {
    return op->emitError("unimplemented: not support dynamic dimSize when "
                         "start smaller than 0.");
  }
  start = start >= 0 ? start : dimSize + start;

  int64_t end;
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end))) {
    return op->emitError("unimplemented: end is not constant");
  } else if (ShapedType::isDynamic(dimSize) and end < 0) {
    return op->emitError(
        "unimplemented: not support dynamic dimSize when end smaller than 0.");
  }
  end = end >= 0 ? end : dimSize + end;

  int64_t size = 0;
  std::vector<int64_t> indicesVec;
  for (int64_t i = start; i < end; i += step) {
    indicesVec.push_back(i);
    ++size;
  }
  ArrayRef<int64_t> indices(indicesVec);
  std::vector<int64_t> tmp_shape = {size, 1};
  ArrayRef<int64_t> shape(tmp_shape);
  RankedTensorType constType =
      RankedTensorType::get(shape, rewriter.getIntegerType(64));
  auto constAttr = DenseElementsAttr::get(
      RankedTensorType::get(shape, rewriter.getIntegerType(64)), indices);
  auto const_op =
      rewriter.create<stablehlo::ConstantOp>(loc, constType, constAttr);
  Value scatterIndices = const_op.getResult();

  SmallVector<int64_t> updateWindowDims;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }
    updateWindowDims.push_back(i);
  }

  auto scatterArgs = stablehlo::ScatterDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*updateWindowDims=*/updateWindowDims,
      /*insertedWindowDims=*/{dim},
      /*inputBatchingDims=*/{},
      /*scatterIndicesBatchingDims=*/{},
      /*scatterDimsToOperandDim=*/{dim},
      /*indexVectorDim=*/1);

  Value src = adaptor.getSrc();
  auto scatterOp = rewriter.create<stablehlo::ScatterOp>(
      loc, resultType, input, scatterIndices, src, scatterArgs, false, false);

  Block &block = scatterOp.getUpdateComputation().emplaceBlock();
  auto blockArgumentType =
      RankedTensorType::get({}, inputType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);

  auto *lhs = block.args_begin();
  auto *rhs = std::next(lhs);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    rewriter.create<stablehlo::ReturnOp>(loc, *rhs);
  }

  rewriter.replaceOp(op, scatterOp.getResults());

  return success();
}

template <typename AtenOpT, int reduceType>
class ConvertAtenScatterOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value input = adaptor.getSelf();
    Value index = adaptor.getIndex();
    Value src = adaptor.getSrc();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto indexType = cast<RankedTensorType>(index.getType());
    auto srcType = cast<RankedTensorType>(src.getType());
    auto indexElemType = indexType.getElementType();

    if (indexType.getRank() != inputType.getRank() ||
        inputType.getRank() != srcType.getRank()) {
      return op.emitError(
          "`index`, `input` and `src` param should have the same rank");
    }
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(
          op, "only constant int `dim` param supported");
    }
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank())) {
      return rewriter.notifyMatchFailure(op, "invalid `dim` param detected");
    }

    auto indexShapeInfo = hlo::getDimIndexOfTensor(rewriter, op, index);
    if (failed(indexShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dim sizes of `index` param");
    }

    // slice src tensor to have the same shape bound of index tensor in the
    // leading dimensions. PyTorch has guaranteed that src tensor size will not
    // be smaller than that of index tensor. REF:
    // https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    SmallVector<Value> sliceIndicies(srcType.getRank(), zero);
    SmallVector<Value> sliceStrides(srcType.getRank(), one);

    auto sliceIndiciesValue =
        rewriter.create<tensor::FromElementsOp>(loc, sliceIndicies);
    auto sliceStridesValue =
        rewriter.create<tensor::FromElementsOp>(loc, sliceStrides);
    auto sliceLimitIndiciesValue =
        rewriter.create<tensor::FromElementsOp>(loc, *indexShapeInfo);

    auto newSrcType =
        RankedTensorType::get(indexType.getShape(), srcType.getElementType());
    src = rewriter.create<stablehlo::RealDynamicSliceOp>(
        loc, newSrcType, src, sliceIndiciesValue, sliceLimitIndiciesValue,
        sliceStridesValue);

    // generate scatter indicies for stablehlo::Scatter op.
    auto toConcatIndexShapeValueVec = *indexShapeInfo;
    toConcatIndexShapeValueVec.push_back(one);
    auto toConcatIndexShape = rewriter.create<tensor::FromElementsOp>(
        loc, toConcatIndexShapeValueVec);

    auto indexShape = indexType.getShape();
    SmallVector<int64_t> toConcatIndexShapeVec(indexShape.begin(),
                                               indexShape.end());
    toConcatIndexShapeVec.push_back(1);
    RankedTensorType toConcatIndexType =
        RankedTensorType::get(toConcatIndexShapeVec, indexElemType);

    SmallVector<Value> toConcat;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      if (i == dim) {
        toConcat.push_back(rewriter.create<stablehlo::DynamicReshapeOp>(
            loc, toConcatIndexType, index, toConcatIndexShape));
      } else {
        toConcat.push_back(rewriter.create<stablehlo::DynamicIotaOp>(
            loc, toConcatIndexType, toConcatIndexShape,
            rewriter.getI64IntegerAttr(i)));
      }
    }

    auto scatterIndicies = rewriter.create<stablehlo::ConcatenateOp>(
        loc, toConcat, static_cast<uint64_t>(inputType.getRank()));
    SmallVector<int64_t> sliceSizes(inputType.getRank(), 1);

    // generate ScatterDimensionNumbers for stablehlo::Scatter op.
    int64_t indexVecDim = inputType.getRank();
    SmallVector<int64_t> scatterDimOperandDimMap;
    SmallVector<int64_t> insertedWindowDims;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      scatterDimOperandDimMap.push_back(i);
      insertedWindowDims.push_back(i);
    }
    auto scatterDimensionNumbers = stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*updateWindowDims=*/{},
        /*insertedWindowDims=*/insertedWindowDims,
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDim=*/scatterDimOperandDimMap,
        /*indexVectorDim=*/indexVecDim);

    auto stablehloScatterOp = rewriter.create<stablehlo::ScatterOp>(
        loc, inputType, input, scatterIndicies, src, scatterDimensionNumbers,
        false, false);

    // config update computation function: just return the element from src.
    Block &block = stablehloScatterOp.getUpdateComputation().emplaceBlock();
    // add block arguments
    auto blockArgumentType =
        RankedTensorType::get({}, inputType.getElementType());
    block.addArgument(blockArgumentType, loc);
    block.addArgument(blockArgumentType, loc);

    auto *lhsArg = block.args_begin();
    auto *rhsArg = std::next(lhsArg);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      if (reduceType == 0) {
        rewriter.create<stablehlo::ReturnOp>(loc, *rhsArg);
      } else if (reduceType == 1) {
        Value res = rewriter.create<stablehlo::AddOp>(loc, blockArgumentType,
                                                      *lhsArg, *rhsArg);
        rewriter.create<stablehlo::ReturnOp>(loc, res);
      }
    }

    rewriter.replaceOp(op, stablehloScatterOp.getResults());
    return success();
  }
};

// AtenIndexTensorOp
// Convert to StableHlo::GatherOp.
template <>
LogicalResult ConvertAtenOp<AtenIndexTensorHackedTwinOp>::matchAndRewrite(
    AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  auto inputTensorType = cast<RankedTensorType>(input.getType());
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  auto outShape = outType.getShape();
  Value indexList = op.getIndices();
  SmallVector<Value> indicesTorchType;
  if (!getListConstructElements(indexList, indicesTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");

  auto indexTensors = getTypeConvertedValues(rewriter, loc, getTypeConverter(),
                                             indicesTorchType);

  int maxIndexRank = -1;
  auto gatherIndicesInfo =
      broadcastAndConcatIndices(op, rewriter, indexTensors, outShape,
                                options.dimSizeIndexBits, maxIndexRank);
  if (failed(gatherIndicesInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to generate broadcasted indices");
  }
  auto gatherIndices = *gatherIndicesInfo;

  int64_t numIndicesDim = indexTensors.size();
  int64_t indexVecDim = maxIndexRank;

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedDims;
  SmallVector<int64_t> startIndexMap;
  for (int64_t i = 0; i < numIndicesDim; ++i) {
    collapsedDims.push_back(i);
    startIndexMap.push_back(i);
  }
  for (int64_t i = numIndicesDim; i < inputTensorType.getRank(); i++) {
    offsetDims.push_back(i + maxIndexRank - numIndicesDim);
  }
  auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedDims,
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  SmallVector<int64_t> sliceSizes;
  auto inputShape = makeShapeTorchCompatible(inputTensorType.getShape());
  for (int64_t i = 0; i < inputTensorType.getRank(); ++i) {
    if (i < numIndicesDim) {
      sliceSizes.push_back(1);
    } else {
      sliceSizes.push_back(inputShape[i]);
    }
  }

  rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
      op, outType, input, gatherIndices, dimsAttr,
      rewriter.getDenseI64ArrayAttr(sliceSizes));
  return success();
}

// AtenIndexPutHackedTwinOP
// Convert to stablehlo::ScatterOp
template <>
LogicalResult ConvertAtenOp<AtenIndexPutHackedTwinOp>::matchAndRewrite(
    AtenIndexPutHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  Value values = adaptor.getValues();
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  auto inputType = cast<RankedTensorType>(input.getType());
  auto valuesType = cast<RankedTensorType>(values.getType());
  int64_t valueRank = valuesType.getRank();
  auto valuesShape = valuesType.getShape();
  bool accumulate;
  if (!matchPattern(op.getAccumulate(), m_TorchConstantBool(&accumulate))) {
    return rewriter.notifyMatchFailure(op,
                                       "accumulate should be a constant bool");
  }
  Value indexList = op.getIndices();
  SmallVector<Value> indicesTorchType;
  if (!getListConstructElements(indexList, indicesTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");
  int64_t indexCnt = indicesTorchType.size();

  auto indexTensors = getTypeConvertedValues(rewriter, loc, getTypeConverter(),
                                             indicesTorchType);

  int maxIndexRank = -1;
  auto scatterIndicesInfo =
      broadcastAndConcatIndices(op, rewriter, indexTensors, valuesShape,
                                options.dimSizeIndexBits, maxIndexRank);
  if (failed(scatterIndicesInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to generate broadcasted indices");
  }
  auto scatterIndices = *scatterIndicesInfo;

  // create stablehlo::ScatterOp
  int64_t indexVecDim = maxIndexRank;
  SmallVector<int64_t> scatterDimOperandDimMap;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> updateWindowDims;
  for (int64_t i = 0; i < indexCnt; ++i) {
    scatterDimOperandDimMap.push_back(i);
    insertedWindowDims.push_back(i);
  }
  for (int64_t i = maxIndexRank; i < valueRank; ++i) {
    updateWindowDims.push_back(i);
  }
  auto scatterDimensionNumbers = stablehlo::ScatterDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*updateWindowDims=*/updateWindowDims,
      /*insertedWindowDims=*/insertedWindowDims,
      /*inputBatchingDims=*/{},
      /*scatterIndicesBatchingDims=*/{},
      /*scatterDimsToOperandDim=*/scatterDimOperandDimMap,
      /*indexVectorDim=*/indexVecDim);

  auto stablehloScatterOp = rewriter.create<stablehlo::ScatterOp>(
      loc, outType, input, scatterIndices, values, scatterDimensionNumbers,
      false, false);

  // configure update computation function.
  Block &block = stablehloScatterOp.getUpdateComputation().emplaceBlock();
  // add block arguments
  auto blockArgumentType =
      RankedTensorType::get({}, inputType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);

  auto *lhsArg = block.args_begin();
  auto *rhsArg = std::next(lhsArg);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    if (!accumulate) {
      rewriter.create<stablehlo::ReturnOp>(loc, *rhsArg);
    } else {
      Value out = rewriter.create<stablehlo::AddOp>(loc, blockArgumentType,
                                                    *lhsArg, *rhsArg);
      rewriter.create<stablehlo::ReturnOp>(loc, out);
    }
  }

  rewriter.replaceOp(op, stablehloScatterOp.getResults());
  return success();
}

// AtenGridSamplerOp
// See
// https://github.com/pytorch/pytorch/blob/ec58f1f74ebcec744d2ab90ad34abd09c1018e92/torch/_decomp/decompositions.py#L3923-L4086
namespace {
template <typename T>
static Value getConstantLike(OpBuilder &b, Location loc, T constant,
                             Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  auto getAttr = [&]() -> Attribute {
    if (isa<mlir::IntegerType>(ty))
      return b.getIntegerAttr(ty, constant);
    if (isa<mlir::FloatType>(ty))
      return b.getFloatAttr(ty, constant);
    if (auto complexTy = dyn_cast<mlir::ComplexType>(ty))
      return complex::NumberAttr::get(complexTy, constant, 0);
    llvm_unreachable("unhandled element type");
  };
  return b.create<mlir::chlo::ConstantLikeOp>(loc, cast<TypedAttr>(getAttr()),
                                              val);
}

template <typename T>
static Value getConstTensor(ConversionPatternRewriter &rewriter, Operation *op,
                            ArrayRef<T> values, ArrayRef<int64_t> shape,
                            Type ty) {
  Location loc = op->getLoc();
  RankedTensorType valueType = RankedTensorType::get(shape, ty);
  auto valueAttr = DenseElementsAttr::get(valueType, values);
  return rewriter.create<stablehlo::ConstantOp>(loc, valueType, valueAttr);
}

template <typename T>
static Value getConstScalarTensor(ConversionPatternRewriter &rewriter,
                                  Operation *op, T value, Type ty) {
  return getConstTensor(rewriter, op, ArrayRef<T>{value}, {}, ty);
}

// Helper function to lower AtenGridSamplerOp.
static Value unnormalize(ConversionPatternRewriter &rewriter, Operation *op,
                         Value coords, int64_t size, Type elemTy,
                         bool alignCorners) {
  Location loc = op->getLoc();
  APFloat pointFive(cast<mlir::FloatType>(elemTy).getFloatSemantics(), "0.5");
  APFloat sizeFloat =
      APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), size);
  APFloat one = APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 1);
  APFloat zero = APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 0);

  // double mul = alignCorners ? (size * 0.5 - 0.5) : (size * 0.5);
  // double ofs = size * 0.5 - 0.5;
  APFloat mul =
      alignCorners ? sizeFloat * pointFive - pointFive : sizeFloat * pointFive;
  APFloat ofs = sizeFloat * pointFive - pointFive;
  Value constMul = getConstScalarTensor(rewriter, op, mul, elemTy);
  Value constOfs = getConstScalarTensor(rewriter, op, ofs, elemTy);

  // use chlo::BroadcastMulOp to multiply constMul with coords.
  DenseI64ArrayAttr bcastDimensions;
  Value mulResult = rewriter.create<chlo::BroadcastMulOp>(loc, coords, constMul,
                                                          bcastDimensions);
  // use chlo::BroadcastAddOp to add constOfs to mulResult.
  Value result = rewriter.create<chlo::BroadcastAddOp>(loc, mulResult, constOfs,
                                                       bcastDimensions);
  return result;
}

static Value computeCoordinates(ConversionPatternRewriter &rewriter,
                                Operation *op, Value coords, int64_t size,
                                Type elemTy, int64_t padding_mode) {
  // TODO: add support for padding_mode 1 and 2.
  return coords;
}

static Value computeSourceIndex(ConversionPatternRewriter &rewriter,
                                Operation *op, Value coords, int64_t size,
                                Type elemTy, int64_t padding_mode,
                                bool alignCorners) {
  Value coordsUn =
      unnormalize(rewriter, op, coords, size, elemTy, alignCorners);
  return computeCoordinates(rewriter, op, coordsUn, size, elemTy, padding_mode);
}

// def in_bounds_cond(xs: Tensor, ys: Tensor) -> Tensor:
//     return torch.logical_and(
//         0 <= xs, torch.logical_and(xs < iW, torch.logical_and(0 <= ys, ys
//         < iH))
//     )
static Value inBoundsCond(ConversionPatternRewriter &rewriter, Operation *op,
                          Value xs, Value ys, int64_t ih, int64_t iw,
                          Type elemTy) {
  Location loc = op->getLoc();
  APFloat zeroFloat =
      APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 0);
  Value zero = getConstScalarTensor(rewriter, op, zeroFloat, elemTy);
  APFloat iwFloat =
      APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), iw);
  APFloat ihFloat =
      APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), ih);

  Value iwFloatValue = getConstScalarTensor(rewriter, op, iwFloat, elemTy);
  Value ihFloatValue = getConstScalarTensor(rewriter, op, ihFloat, elemTy);

  chlo::ComparisonTypeAttr compareTypeAttr = chlo::ComparisonTypeAttr::get(
      rewriter.getContext(), chlo::ComparisonType::FLOAT);
  chlo::ComparisonDirectionAttr compareLTAttr =
      chlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                         chlo::ComparisonDirection::LT);
  chlo::ComparisonDirectionAttr compareGEAttr =
      chlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                         chlo::ComparisonDirection::GE);
  DenseI64ArrayAttr bcastDimensions;
  Value cond1 = rewriter.create<chlo::BroadcastCompareOp>(
      loc, xs, zero, bcastDimensions, compareGEAttr, compareTypeAttr);
  Value cond2 = rewriter.create<chlo::BroadcastCompareOp>(
      loc, xs, iwFloatValue, bcastDimensions, compareLTAttr, compareTypeAttr);
  Value cond3 = rewriter.create<chlo::BroadcastCompareOp>(
      loc, ys, zero, bcastDimensions, compareGEAttr, compareTypeAttr);
  Value cond4 = rewriter.create<chlo::BroadcastCompareOp>(
      loc, ys, ihFloatValue, bcastDimensions, compareLTAttr, compareTypeAttr);
  Value cond5 =
      rewriter.create<chlo::BroadcastAndOp>(loc, cond1, cond2, bcastDimensions);
  Value cond6 =
      rewriter.create<chlo::BroadcastAndOp>(loc, cond3, cond4, bcastDimensions);
  return rewriter.create<chlo::BroadcastAndOp>(loc, cond5, cond6,
                                               bcastDimensions);
}
// def clip(xs: Tensor, ys: Tensor, ws: Tensor) -> TensorSequenceType:
//     cond = in_bounds_cond(xs, ys)
//     # To clip to inside valid coordinates, we map the coordinates
//     # to (x, y) = (0, 0) and also set the weight to 0
//     # We also change the shape of the tensor to the appropriate one for
//     # broadcasting with N_idx, C_idx for the purposes of advanced
//     indexing c = C if _expand_grid else 1
//     return tuple(
//         torch.where(cond, t, 0).view(N, c, oH, oW)
//         for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
//     )
SmallVector<Value> clip(ConversionPatternRewriter &rewriter, Operation *op,
                        Value xs, Value ys, Value ws, int64_t N, int64_t oH,
                        int64_t oW, int64_t iH, int64_t iW, Type elemTy) {
  Location loc = op->getLoc();
  auto indexElemTy = rewriter.getI64Type();
  auto indexTy = RankedTensorType::get(mlir::ArrayRef<int64_t>{1}, indexElemTy);

  Value zeroIntValue = rewriter.create<stablehlo::ConstantOp>(
      loc, indexTy, DenseIntElementsAttr::get(indexTy, ArrayRef<int64_t>{0}));

  APFloat zeroAPFloat =
      APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 0);
  Value zeroFloatValue =
      getConstScalarTensor(rewriter, op, zeroAPFloat, elemTy);
  Value cond = inBoundsCond(rewriter, op, xs, ys, iH, iW, elemTy);
  Value xsInt = rewriter.create<stablehlo::ConvertOp>(loc, xs, indexElemTy);
  Value ysInt = rewriter.create<stablehlo::ConvertOp>(loc, ys, indexElemTy);

  Value selectXs = rewriter.create<chlo::BroadcastSelectOp>(
      loc, ArrayRef<Value>{cond, xsInt, zeroIntValue});
  Value selectYs = rewriter.create<chlo::BroadcastSelectOp>(
      loc, ArrayRef<Value>{cond, ysInt, zeroIntValue});
  Value selectWs = rewriter.create<chlo::BroadcastSelectOp>(
      loc, ArrayRef<Value>{cond, ws, zeroFloatValue});

  SmallVector<int64_t> sizes = {N, 1, oH, oW};
  Value reshapedXs = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(sizes, indexElemTy), selectXs);
  Value reshapedYs = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(sizes, indexElemTy), selectYs);
  Value reshapedWs = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(sizes, elemTy), selectWs);
  return SmallVector<Value>{reshapedXs, reshapedYs, reshapedWs};
}

Value getSummand(ConversionPatternRewriter &rewriter, Operation *op,
                 Value input, Value ix, Value iy, Value w, int64_t N,
                 int64_t oH, int64_t oW, int64_t iH, int64_t iW, Value Nidx,
                 Value CIdx, RankedTensorType outType, Type elemTy,
                 size_t dimSizeIndexBits) {
  Location loc = op->getLoc();
  auto inputTensorType = cast<RankedTensorType>(input.getType());
  SmallVector<Value> clipValues =
      clip(rewriter, op, ix, iy, w, N, oH, oW, iH, iW, elemTy);
  Value idxX = clipValues[0];
  Value idxY = clipValues[1];
  Value idxW = clipValues[2];
  SmallVector<Value> indexTensors{Nidx, CIdx, idxY, idxX};

  int maxIndexRank = -1;
  auto gatherIndicesInfo = broadcastAndConcatIndices(
      input.getDefiningOp(), rewriter, indexTensors, outType.getShape(),
      dimSizeIndexBits, maxIndexRank);
  auto gatherIndices = *gatherIndicesInfo;
  int64_t numIndicesDim = indexTensors.size();
  int64_t indexVecDim = maxIndexRank;

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedDims;
  SmallVector<int64_t> startIndexMap;
  for (int64_t i = 0; i < numIndicesDim; ++i) {
    collapsedDims.push_back(i);
    startIndexMap.push_back(i);
  }
  for (int64_t i = numIndicesDim; i < inputTensorType.getRank(); i++) {
    offsetDims.push_back(i + maxIndexRank - numIndicesDim);
  }
  auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedDims,
      /*operandBatchingDims=*/{},
      /*startIndicesBatchingDims=*/{},
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  SmallVector<int64_t> sliceSizes;
  auto inputShape = makeShapeTorchCompatible(inputTensorType.getShape());
  for (int64_t i = 0; i < inputTensorType.getRank(); ++i) {
    if (i < numIndicesDim) {
      sliceSizes.push_back(1);
    } else {
      sliceSizes.push_back(inputShape[i]);
    }
  }

  Value gather = rewriter.create<stablehlo::GatherOp>(
      loc, input, gatherIndices, dimsAttr,
      rewriter.getDenseI64ArrayAttr(sliceSizes));
  // use chlo::BroadcastMulOp to multiply idxW with gather.
  DenseI64ArrayAttr bcastDimensions;
  return rewriter.create<chlo::BroadcastMulOp>(loc, gather, idxW,
                                               bcastDimensions);
}

} // namespace
template <>
LogicalResult ConvertAtenOp<AtenGridSamplerOp>::matchAndRewrite(
    AtenGridSamplerOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getInput();
  Value grid = adaptor.getGrid();

  int64_t interpolationMode;
  if (!matchPattern(op.getInterpolationMode(),
                    m_TorchConstantInt(&interpolationMode)))
    return rewriter.notifyMatchFailure(
        op, "interpolation_mode must be an integer constant");
  int64_t paddingMode;
  if (!matchPattern(op.getPaddingMode(), m_TorchConstantInt(&paddingMode)))
    return rewriter.notifyMatchFailure(
        op, "padding_mode must be an integer constant");

  if (interpolationMode != 0 && interpolationMode != 1)
    return rewriter.notifyMatchFailure(
        op, "only support interpolation_mode = 0 (bilinear) or 1(nearest)");

  if (paddingMode != 0)
    return rewriter.notifyMatchFailure(op,
                                       "only support paddingMode = 0 (Zero)");

  bool alignCorners = false;
  if (!matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCorners)))
    return rewriter.notifyMatchFailure(
        op, "alignCorners must be a boolean constant");

  RankedTensorType inputTy = cast<RankedTensorType>(input.getType());
  RankedTensorType gridTy = cast<RankedTensorType>(grid.getType());
  RankedTensorType outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  Type elemTy = inputTy.getElementType();
  if (inputTy.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "input must be a 4D tensor");
  if (gridTy.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "grid must be a 4D tensor");

  auto inputSize = inputTy.getShape();
  auto gridSize = gridTy.getShape();
  int64_t N = inputSize[0];
  int64_t C = inputSize[1];
  int64_t iH = inputSize[2];
  int64_t iW = inputSize[3];
  int64_t oH = gridSize[1];
  int64_t oW = gridSize[2];
  // grid is a 4D tensor with shape (N, oH, oW, 2)

  Type indexElemTy = rewriter.getI64Type();
  RankedTensorType indexTy =
      RankedTensorType::get(mlir::ArrayRef<int64_t>{1}, indexElemTy);
  Value constN = rewriter.create<stablehlo::ConstantOp>(
      loc, indexTy, DenseIntElementsAttr::get(indexTy, {N}));
  Value constC = rewriter.create<stablehlo::ConstantOp>(
      loc, indexTy, DenseIntElementsAttr::get(indexTy, {C}));
  APFloat one = APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 1);
  APFloat zero = APFloat(cast<mlir::FloatType>(elemTy).getFloatSemantics(), 0);

  Value constOneFloat = getConstScalarTensor(rewriter, op, one, elemTy);

  auto NidxFlatten = rewriter.create<stablehlo::DynamicIotaOp>(
      loc, RankedTensorType::get(mlir::ArrayRef<int64_t>{N}, indexElemTy),
      constN, 0);
  auto CidxFlatten = rewriter.create<stablehlo::DynamicIotaOp>(
      loc, RankedTensorType::get(mlir::ArrayRef<int64_t>{C}, indexElemTy),
      constC, 0);

  // Reshape NidxFlatten to 4D tensor (N, 1, 1, 1)
  auto NidxSizes = mlir::SmallVector<int64_t>{N, 1, 1, 1};
  auto Nidx = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(NidxSizes, indexElemTy), NidxFlatten);

  // Reshape CidxFlatten to 4D tensor (1, C, 1, 1)
  auto CidxSizes = mlir::SmallVector<int64_t>{1, C, 1, 1};
  auto Cidx = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(CidxSizes, indexElemTy), CidxFlatten);

  llvm::SmallVector<int64_t> stride(4, 1);
  auto gridX = rewriter.create<stablehlo::SliceOp>(
      loc,
      RankedTensorType::get(mlir::SmallVector<int64_t>{N, oH, oW, 1},
                            gridTy.getElementType()),
      grid, mlir::SmallVector<int64_t>{0, 0, 0, 0},
      mlir::SmallVector<int64_t>{N, oH, oW, 1}, stride);
  auto gridY = rewriter.create<stablehlo::SliceOp>(
      loc,
      RankedTensorType::get(mlir::SmallVector<int64_t>{N, oH, oW, 1},
                            gridTy.getElementType()),
      grid, mlir::SmallVector<int64_t>{0, 0, 0, 1},
      mlir::SmallVector<int64_t>{N, oH, oW, 2}, stride);
  // squeeze last dimension
  auto gridXshape = mlir::SmallVector<int64_t>{N, oH, oW};

  auto gridXReshape = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(gridXshape, gridTy.getElementType()), gridX);
  auto gridYReshape = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(gridXshape, gridTy.getElementType()), gridY);

  if (interpolationMode == 0) {
    Value ix = computeSourceIndex(rewriter, op, gridXReshape, iW, elemTy,
                                  paddingMode, alignCorners);
    Value iy = computeSourceIndex(rewriter, op, gridYReshape, iH, elemTy,
                                  paddingMode, alignCorners);
    Value ix_nw = rewriter.create<stablehlo::FloorOp>(loc, ix);
    Value iy_nw = rewriter.create<stablehlo::FloorOp>(loc, iy);

    DenseI64ArrayAttr bcastDimensions;
    Value ix_ne = rewriter.create<chlo::BroadcastAddOp>(
        loc, ix_nw, constOneFloat, bcastDimensions);
    Value iy_ne = iy_nw;
    Value ix_sw = ix_nw;
    Value iy_sw = rewriter.create<chlo::BroadcastAddOp>(
        loc, iy_nw, constOneFloat, bcastDimensions);
    Value ix_se = ix_ne;
    Value iy_se = iy_sw;

    // w_nw = (ix_se - ix) * (iy_se - iy)
    // w_ne = (ix - ix_sw) * (iy_sw - iy)
    // w_sw = (ix_ne - ix) * (iy - iy_ne)
    // w_se = (ix - ix_nw) * (iy - iy_nw)
    Value w_nw = rewriter.create<chlo::BroadcastMulOp>(
        loc,
        rewriter.create<chlo::BroadcastSubOp>(loc, ix_se, ix, bcastDimensions),
        rewriter.create<chlo::BroadcastSubOp>(loc, iy_se, iy, bcastDimensions),
        bcastDimensions);
    Value w_ne = rewriter.create<chlo::BroadcastMulOp>(
        loc,
        rewriter.create<chlo::BroadcastSubOp>(loc, ix, ix_sw, bcastDimensions),
        rewriter.create<chlo::BroadcastSubOp>(loc, iy_sw, iy, bcastDimensions),
        bcastDimensions);
    Value w_sw = rewriter.create<chlo::BroadcastMulOp>(
        loc,
        rewriter.create<chlo::BroadcastSubOp>(loc, ix_ne, ix, bcastDimensions),
        rewriter.create<chlo::BroadcastSubOp>(loc, iy, iy_ne, bcastDimensions),
        bcastDimensions);
    Value w_se = rewriter.create<chlo::BroadcastMulOp>(
        loc,
        rewriter.create<chlo::BroadcastSubOp>(loc, ix, ix_nw, bcastDimensions),
        rewriter.create<chlo::BroadcastSubOp>(loc, iy, iy_nw, bcastDimensions),
        bcastDimensions);

    Value summand_nw =
        getSummand(rewriter, op, input, ix_nw, iy_nw, w_nw, N, oH, oW, iH, iW,
                   Nidx, Cidx, outTy, elemTy, options.dimSizeIndexBits);
    Value summand_ne =
        getSummand(rewriter, op, input, ix_ne, iy_ne, w_ne, N, oH, oW, iH, iW,
                   Nidx, Cidx, outTy, elemTy, options.dimSizeIndexBits);
    Value summand_sw =
        getSummand(rewriter, op, input, ix_sw, iy_sw, w_sw, N, oH, oW, iH, iW,
                   Nidx, Cidx, outTy, elemTy, options.dimSizeIndexBits);
    Value summand_se =
        getSummand(rewriter, op, input, ix_se, iy_se, w_se, N, oH, oW, iH, iW,
                   Nidx, Cidx, outTy, elemTy, options.dimSizeIndexBits);

    // summand_nw + summand_ne + summand_sw + summand_se
    Value sum = rewriter.create<stablehlo::AddOp>(loc, summand_nw, summand_ne);
    sum = rewriter.create<stablehlo::AddOp>(loc, sum, summand_sw);
    sum = rewriter.create<stablehlo::AddOp>(loc, sum, summand_se);
    rewriter.replaceOp(op, sum);
  } else if (interpolationMode == 1) {
    Value ix = computeSourceIndex(rewriter, op, gridXReshape, iW, elemTy,
                                  paddingMode, alignCorners);
    Value iy = computeSourceIndex(rewriter, op, gridYReshape, iH, elemTy,
                                  paddingMode, alignCorners);
    Value ix_round = rewriter.create<stablehlo::RoundOp>(loc, ix);
    Value iy_round = rewriter.create<stablehlo::RoundOp>(loc, iy);
    Value oneTensor = getConstantLike(rewriter, loc, 1.0, ix_round);
    Value summand = getSummand(rewriter, op, input, ix_round, iy_round,
                               oneTensor, N, oH, oW, iH, iW, Nidx, Cidx, outTy,
                               elemTy, options.dimSizeIndexBits);
    rewriter.replaceOp(op, summand);
  }
  return success();
}

void mlir::torch::torch_to_stablehlo::
    populateGatherScatterOpPatternsAndLegality(
        TypeConverter &typeConverter, RewritePatternSet &patterns,
        ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATENOP_PATTERN(AtenEmbeddingOp);
  INSERT_ATENOP_PATTERN(AtenEmbeddingBagPaddingIdxOp);
  INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
  INSERT_ATENOP_PATTERN(AtenGatherOp);
  INSERT_ATENOP_PATTERN(AtenSliceScatterOp);
  INSERT_ATENOP_PATTERN(AtenIndexTensorHackedTwinOp);
  INSERT_ATENOP_PATTERN(AtenIndexPutHackedTwinOp);
  INSERT_ATENOP_PATTERN(AtenGridSamplerOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_ATEN_SCATTER_PATTERN(AtenOp, reduceType)                        \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenScatterOp<AtenOp, reduceType>>(typeConverter,        \
                                                         context, options)
  INSERT_ATEN_SCATTER_PATTERN(AtenScatterSrcOp, 0); // 0 for None reduce op
  INSERT_ATEN_SCATTER_PATTERN(AtenScatterAddOp, 1); // 1 for Add reduce op
#undef INSERT_ATEN_SCATTER_PATTERN
}
