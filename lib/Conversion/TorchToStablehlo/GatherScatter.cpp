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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

namespace {
Value gatherTensorAlongSingleAxis(PatternRewriter &rewriter, Operation *op,
                                  Value input, Value indices, int64_t axis,
                                  size_t dimSizeIndexBits) {
  auto loc = op->getLoc();
  Type intType = rewriter.getIntegerType(dimSizeIndexBits);
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));

  // sliceSizes
  auto inputRankTy = input.getType().dyn_cast<RankedTensorType>();
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
  auto indicesRankTy = indices.getType().dyn_cast<RankedTensorType>();
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
  RankedTensorType inputType =
      input.getType().template cast<RankedTensorType>();

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

  if (torchTypeStart.getType().isa<OptionalType>() ||
      torchTypeEnd.getType().isa<OptionalType>())
    return rewriter.notifyMatchFailure(op, "unimplemented optional type arg");

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step))) {
    if (!op.getStep().getType().template isa<Torch::NoneType>())
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
  auto weightTy = weight.getType().cast<RankedTensorType>();
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
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
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
  auto inputType = input.getType().cast<RankedTensorType>();
  auto indexType = index.getType().cast<RankedTensorType>();
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

  auto options = getOptions();
  auto indexShapeInfo =
      hlo::getDimSizesOfTensor(rewriter, op, index, options.dimSizeIndexBits);
  if (failed(indexShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dim sizes of `index` param");
  }
  auto intType = rewriter.getIntegerType(options.dimSizeIndexBits);
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));
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
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
      op, input, gatherIndicies, dimsAttr,
      rewriter.getI64TensorAttr(sliceSizes));
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

  RankedTensorType resultType =
      typeConverter->convertType(op->getResult(0).getType())
          .cast<RankedTensorType>();

  SmallVector<Value> resultShape;
  SmallVector<Value> offsets;
  SmallVector<Value> strides;
  if (failed(prepareArgumentsForSlicingOp<AtenSliceScatterOp,
                                          AtenSliceScatterOpAdaptor>(
          op, adaptor, rewriter, resultShape, offsets, strides))) {
    return failure();
  }

  Value src = adaptor.getSrc();
  auto srcType = src.getType().cast<RankedTensorType>();
  int64_t srcRank = srcType.getRank();
  SmallVector<int64_t> srcAbstractSizes(srcRank, kUnknownSize);
  auto abstractSrcType = RankedTensorType::get(
      makeShapeLLVMCompatible(srcAbstractSizes), srcType.getElementType());
  Value abstractSrc =
      rewriter.create<tensor::CastOp>(loc, abstractSrcType, src);

  Value result = rewriter.create<tensor::InsertSliceOp>(
      loc, abstractSrc, input, offsets, resultShape, strides);

  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);

  return success();
}

// AtenScatterSrcOp
template <>
LogicalResult ConvertAtenOp<AtenScatterSrcOp>::matchAndRewrite(
    AtenScatterSrcOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  Value index = adaptor.getIndex();
  Value src = adaptor.getSrc();
  auto inputType = input.getType().cast<RankedTensorType>();
  auto indexType = index.getType().cast<RankedTensorType>();
  auto srcType = src.getType().cast<RankedTensorType>();
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

  auto options = getOptions();

  auto indexShapeInfo =
      hlo::getDimSizesOfTensor(rewriter, op, index, options.dimSizeIndexBits);
  if (failed(indexShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dim sizes of `index` param");
  }
  auto intType = rewriter.getIntegerType(options.dimSizeIndexBits);

  // slice src tensor to have the same shape bound of index tensor in the
  // leading dimensions. PyTorch has guaranteed that src tensor size will not be
  // smaller than that of index tensor. REF:
  // https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 0));
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));
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
      /*scatterDimsToOperandDim=*/scatterDimOperandDimMap,
      /*indexVectorDim=*/indexVecDim);

  auto stablehloScatterOp = rewriter.create<stablehlo::ScatterOp>(
      loc, input, scatterIndicies, src, scatterDimensionNumbers, false, false);

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
    rewriter.create<stablehlo::ReturnOp>(loc, *rhsArg);
  }

  rewriter.replaceOp(op, stablehloScatterOp.getResults());
  return success();
}

// Aten_IndexPutImplOp
template <>
LogicalResult ConvertAtenOp<Aten_IndexPutImplOp>::matchAndRewrite(
    Aten_IndexPutImplOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  // For understanding of the algorithm, I comment the code with this example.
  // a = torch.tensor([[0, 1, 2, 3]])
  // a[..., 1:] = torch.tensor([4, 5, 6])
  // = a[..., 1:4] = torch.tensor([4, 5, 6])
  // = a[[0, 0, 0], [1, 2, 3]] = torch.tensor([4, 5, 6]) # tensor([[0, 4, 5,
  // 6]]) = torch.ops.aten.index_put(torch.tensor([[0, 1, 2, 3]]), # input
  //                            (torch.tensor([0, 0, 0]), torch.tensor([1, 2,
  //                            3])), # indicies torch.tensor([4, 5, 6])) #
  //                            value
  // = torch.ops.aten.index_put(torch.tensor([[0, 1, 2, 3]]), # input
  //                            (None, torch.tensor([1, 2, 3]),),# indicies
  //                            torch.tensor([4, 5, 6])) # value

  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = adaptor.getSelf().getType().dyn_cast<TensorType>();
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  auto fillValues = adaptor.getValues();
  auto fillValuesType = adaptor.getValues().getType().dyn_cast<TensorType>();
  if (!fillValuesType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  bool accumulate;
  if (!matchPattern(op.getAccumulate(), m_TorchConstantBool(&accumulate)))
    return rewriter.notifyMatchFailure(
        op, "only constant sparse is currently supported");
  if (accumulate)
    return rewriter.notifyMatchFailure(
        op, "Only support false accumulate right now.");

 bool unsafe;
  if (!matchPattern(op.getUnsafe(), m_TorchConstantBool(&unsafe)))
    return rewriter.notifyMatchFailure(
        op, "only constant sparse is currently supported");
  if (unsafe)
    return rewriter.notifyMatchFailure(
        op, "Only support false unsafe right now.");
  
  // Deal with torch.prim.ListConstruct of non const value to get the index
  auto tensorList = op.getIndices();
  SmallVector<Value> tensorsTorchType;
  if (!getListConstructElements(tensorList, tensorsTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");
  auto indexTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), tensorsTorchType);

  // convert list of indices with none into indices tensor  without none
  // indexTensors (none,[1,2,3]) -> ([0,0,0],[1,2,3])
  // ([[0],[0],[0]],[[1],[2],[3]])-> [[0,1],[0,2], [0,3]]
  if (indexTensors.size() <= 1) {
    return rewriter.notifyMatchFailure(
        op, "Only support indexput with multiple index.");
  }
  SmallVector<Value> indicesConcatTensors;
  SmallVector<int64_t> indexesRank;
  SmallVector<SmallVector<int64_t>> indexesShape;

  // concat index tensor into to indices tensor for concat
  for (size_t i = 0; i < indexTensors.size(); i++) {
    auto index = indexTensors[i];
    auto indexTorch = tensorsTorchType[i];
    // TODO add support for none index other than i==0, like (index0, None)
    // (None, index1)
    if (i == 0 && indexTorch.getType().isa<Torch::NoneType>()) {
      // convert None to [0,0,0]
      auto indexNext = indexTensors[i + 1];
      auto indexNextTorch = tensorsTorchType[i + 1];
      if (indexNextTorch.getType().isa<Torch::NoneType>()) {
        return rewriter.notifyMatchFailure(
            op, "Multiple None index is not support for now.");
      }
      auto indexNextType = indexNext.getType().dyn_cast<RankedTensorType>();
      auto indexNextShape = indexNextType.getShape();

      int64_t size = 1;
      for (auto s : indexNextShape)
        size *= s;
      SmallVector<int64_t> values(size, i);
      index = hlo::getConstTensor<int64_t>(rewriter, op, values, indexNextShape)
                  .value();
    }

    auto indexType = index.getType().dyn_cast<RankedTensorType>();
    // Reshape the zero rank index to rank 1
    if (indexType.getRank() == 0) {
      // [] -> [0]
      SmallVector<int64_t, 1> oneShape({1}); // {1}
      RankedTensorType reshapeType =
          RankedTensorType::get(oneShape, indexType.getElementType());
      auto indexOneReshapeOp =
          rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, fillValues);
      index = indexOneReshapeOp.getResult();
      indexType = index.getType().dyn_cast<RankedTensorType>();
    }
    auto indexShape = indexType.getShape();
    indexesShape.push_back(makeShapeTorchCompatible(indexShape));
    indexesRank.push_back(indexType.getRank());

    // Expand last dim of index to indices [3] -> [3,1]
    // convert [0,0,0]  to [[0],[0],[0]]
    SmallVector<int64_t> indiceShapeOneDim(indexShape.begin(), indexShape.end());
    indiceShapeOneDim.push_back(1);
    auto indexElemTy = indexType.getElementType();
    RankedTensorType reshapeType =
        RankedTensorType::get(indiceShapeOneDim, indexElemTy);
    auto indicesOneDim =
        rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, index);

    // create concat tensor for indices
    // ([[0],[0],[0]], [[1],[2],[3]])
    indicesConcatTensors.push_back(indicesOneDim.getResult());
  }

  // Right now only support multiple indexes with same shape
  // TODO for different shape multiple indexes, add broadcast_to for small
  // shape
  for (auto indexShapeOneDim : indexesShape) {
    if (!llvm::equal(indexesShape[0], indexShapeOneDim)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: Only support multi indexes with same shape");
    }
  }

  // concat each indices into indices: shape ([3,1],[3,1]) -> [3,2]
  // ([0,0,0],[1,2,3]) -> [[0,1],[0,2], [0,3]]
  auto indicesShapeConcat = indexesShape[0];
  uint64_t lastDim = indexesRank[0];
  indicesShapeConcat.push_back(indicesConcatTensors.size());
  auto indices = rewriter.create<stablehlo::ConcatenateOp>(
      loc,
      GetTypeFromTensorShape(indicesShapeConcat, rewriter.getIntegerType(64)),
      ValueRange(indicesConcatTensors), lastDim);

  if (!indices) {
    return rewriter.notifyMatchFailure(op,
                                       "Convert TorchIndex To Indices fail.");
  }

  // Reshape the fillValues to the requirement shape in %updates of scatter op
  auto indicesType = indices.getResult().getType().dyn_cast<RankedTensorType>();
  auto indicesElemTy = indicesType.getElementType();
  int inputRank = inputType.getShape().size();     // 2
  int indicesRank = indicesType.getShape().size(); // 2

  int N = 1, W = 1, fillK = 1, C = 1, ND = 1;
  //  ND: indices.shape[-1]
  ND = indicesType.getShape()[indicesRank - 1]; // 2 depth of input

  // Calculate N, K, W, C.  (N is always 1)
  // number of indices/selected value in each batch product(indices.shape[0:-1])
  // (all but the last dimension) W = 1*3 = 3
  for (int i = 0; i < (indicesRank - 1); i++) {
    W *= indicesType.getShape()[i];
  }

  // C: number of channels for each index : numbers of values inside each
  // input(chould be scatter) C = product(params.shape[ND:] ND = 2, paramsRank,
  // C = 1
  for (int i = ND; i < inputRank; i++) {
    C *= inputType.getShape()[i];
  }

  // Preprocess fill value.
  // There are 2 cases of fillValues,
  // Let's replace the example [4,5,6] with [0,0,0] here for easy understanding
  // and comparation of the 2 cases.
  // 1. !torch.vtensor<[1,3],si64>
  //  [[0,0,0]] -> [[[0], [0], [0]]]
  // 2. !torch.vtensor<[],si64>
  //    reshape(1) tile(3)  reshape(1,3)   reshape(1,3,1)
  //  [] -> [0] -> [0,0,0] -> [[0,0,0]] -> [[[0], [0], [0]]]
  //    reshape to [1] and then tile to same number of indicesValue.shape[0],
  //    [1,1,1]
  if (fillValuesType.getRank() == 0) {
    // Reshape the zero rank value into rank 1
    // [] -> [0]
    SmallVector<int64_t, 1> oneShape({1}); // {1}
    RankedTensorType reshapeType =
        RankedTensorType::get(oneShape, indicesElemTy);
    auto fillValuesOneReshapeOp =
        rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, fillValues);

    // [0] -> [0,0,0]
    SmallVector<int64_t, 1> bcastShape({W}); // {3}
    RankedTensorType bcastIndicesType =
        RankedTensorType::get(bcastShape, indicesElemTy);
    Value bcastVal = hlo::promoteAndBroadcast(
        rewriter, fillValuesOneReshapeOp.getResult(), bcastIndicesType);

    // [0,0,0] -> [[0,0,0]]
    SmallVector<int64_t, 2> newFillValuesShape({N, W}); // {1,3}
    reshapeType = RankedTensorType::get(newFillValuesShape, indicesElemTy);
    auto newFillValuesReshapeOp =
        rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, bcastVal);
    fillValues = newFillValuesReshapeOp.getResult();
    fillValuesType = fillValues.getType().dyn_cast<RankedTensorType>();
  }

  // fillK: range of each index, total number of fillInput(could be scatter)
  // after flattened k = 1*1*3 = 3
  for (int i = 0; i < ND; i++) {
    if(i < (int)fillValuesType.getShape().size()){
      fillK *= fillValuesType.getShape()[i];
    }
  }
  SmallVector<int64_t, 2> scatterUpdatesShape({fillK, C}); // {3,1}

  RankedTensorType reshapeType =
      RankedTensorType::get(scatterUpdatesShape, indicesElemTy);
  auto scatterUpdatesReshapeOp =
      rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, fillValues);

  // Generate ScatterDimensionNumbers for stablehlo::Scatter op.
  // index_vector_dim = 1 means the values in scatter_indices like [0,2] is
  // started from dim=1 of scatter_indices. In python, we will use : to slice
  // all the value at index_vector_dim once all the other dims are given, like
  // scatter_indices[1,:]. In current setting, the last dim is where is where
  // indices value exist.
  int64_t indexVecDim = indicesRank - 1; // 1

  //  scatter_dims_to_operand_dims exists because the value in scatter_indices
  //  might represent different dim in input operand. Let's say the value
  //  [0,2] in scatter_indices, since scatter_dims_to_operand_dims=[0,1], so the
  //  indices 2 represent the indice that need to be selected on the input's
  //  dim 1.
  //	If we change the scatter_dims_to_operand_dims=[1,0], so the indices 2
  // represent the indice that need to be selected on the input's dim 0. The
  // result full_start_index = [2,0]. The scatter_dims_to_operand_dims=[0,1] is
  // more intuitive in most cases.
  SmallVector<int64_t> scatterDimOperandDimMap;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    scatterDimOperandDimMap.push_back(i);
  }

  // The update_window_dims means in the %update/fillValues, only the dim in
  // update_window_dims has the actual value that will be used. The other dims
  // will only be used to index it. In this examples, we always set the last
  // dim.
  SmallVector<int64_t> updateWindowDims;
  updateWindowDims.push_back(reshapeType.getRank() - 1); // [1]

  // Since scatter spec (C2) rank(inputs[0]) = size(update_window_dims) +
  // size(inserted_window_dims). We can derive the insertedWindowDims from
  // updateWindowDims size. The dim start from 0.
  SmallVector<int64_t> insertedWindowDims; // [0]
  int64_t offset = inputType.getRank() - updateWindowDims.size();
  for (int64_t i = 0; i < offset; i++) {
    insertedWindowDims.push_back(i);
  }

  auto scatterDimensionNumbers = stablehlo::ScatterDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*updateWindowDims=*/updateWindowDims,
      /*insertedWindowDims=*/insertedWindowDims,
      /*scatterDimsToOperandDim=*/scatterDimOperandDimMap,
      /*indexVectorDim=*/indexVecDim);

  auto stablehloScatterOp = rewriter.create<stablehlo::ScatterOp>(
      loc, input, indices.getResult(), scatterUpdatesReshapeOp.getResult(),
      scatterDimensionNumbers, false, false);

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
    rewriter.create<stablehlo::ReturnOp>(loc, *rhsArg);
  }

  rewriter.replaceOp(op, stablehloScatterOp.getResults());
  return success();
}

// AtenIndexTensorOp
// Convert AtenIndexTensorOp to StableHlo::GatherOp
// Step 1: broadcast indices to the same shape
// Step 2: reshape broadcasted indices to have extra last dimension and concat
// Step 3: Create StableHlo::GatherOp with input tensor and indices
//
// Example:
// Input: [[1, 2, 3],
//         [4, 5, 6],
//         [7, 8, 9]]
// Indices[0]: [[0, 0, 0],
//              [2, 2, 0]]
// Indices[1]: [[2],
//              [1]]
// Step 1:
// Indices[0]: [[0, 0, 0],
//              [2, 2, 0]]
// Indices[1]: [[2, 2, 2],
//              [1, 1, 1]]
// Step 2:
// Indices: [[[0, 2], [0, 2], [0, 2]],
//           [[2, 1], [2, 1], [0, 1]]]
// Step 3:
// Output: [[3, 3, 3],
//          [8, 8, 2]]
template <>
LogicalResult ConvertAtenOp<AtenIndexTensorHackedTwinOp>::matchAndRewrite(
    AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();
  Value input = adaptor.getSelf();
  auto inputTensorType = input.getType().dyn_cast<RankedTensorType>();
  // Check input is a tensor type.
  if (!inputTensorType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");
  Value indexList = op.getIndices();
  SmallVector<Value> indicesTorchType;
  if (!getListConstructElements(indexList, indicesTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");

  auto indexTensors = getTypeConvertedValues(rewriter, loc, getTypeConverter(),
                                             indicesTorchType);

  // Step 1: broadcast indices tensors
  int maxRank = -1;
  SmallVector<int64_t> indicesShape;
  SmallVector<int64_t> expandShape;
  SmallVector<int64_t> concatShape;
  // concat index tensor into to indices tensor for concat
  for (size_t i = 0; i < indexTensors.size(); i++) {
    auto indexTensor = indexTensors[i];
    auto indexTensorType = indexTensor.getType().cast<RankedTensorType>();
    for (int64_t size : makeShapeTorchCompatible(indexTensorType.getShape())) {
      if (size == kUnknownSize)
        return rewriter.notifyMatchFailure(op, "Dynamic index support TBD");
    }
    maxRank = std::max(maxRank, (int)indexTensorType.getRank());
  }

  RankedTensorType resultType =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  SmallVector<int64_t> refinedResultShape =
      makeShapeTorchCompatible(resultType.getShape());
  for (int64_t size : refinedResultShape) {
    if (size == kUnknownSize)
      return rewriter.notifyMatchFailure(op, "Dynamic index support TBD");
  }
  for (int i = 0; i < maxRank; i++) {
    indicesShape.push_back(refinedResultShape[i]);
    expandShape.push_back(refinedResultShape[i]);
    concatShape.push_back(refinedResultShape[i]);
  }
  if (indexTensors.size() > 1) {
    expandShape.push_back(1);
    concatShape.push_back(indexTensors.size());
  }

  SmallVector<Value> broadcastedIndices;
  Type indexElemTy =
      indexTensors[0].getType().cast<RankedTensorType>().getElementType();
  RankedTensorType bcastIndexType =
      RankedTensorType::get(indicesShape, indexElemTy);
  for (auto indexTensor : indexTensors) {
    Value bcastVal =
        hlo::promoteAndBroadcast(rewriter, indexTensor, bcastIndexType);
    if (indexTensors.size() > 1) {
      RankedTensorType reshapeType =
          RankedTensorType::get(expandShape, indexElemTy);
      bcastVal =
          rewriter.create<stablehlo::ReshapeOp>(loc, reshapeType, bcastVal);
    }
    broadcastedIndices.push_back(bcastVal);
  }

  // Step 2: concat index tensors
  Value finalIndexTensor = broadcastedIndices[0];
  if (broadcastedIndices.size() > 1) {
    RankedTensorType concatTy = RankedTensorType::get(concatShape, indexElemTy);
    finalIndexTensor = rewriter.create<stablehlo::ConcatenateOp>(
        loc, concatTy, ValueRange(broadcastedIndices), concatShape.size() - 1);
  }

  // Step 3: create stablehlo::GatherOp
  RankedTensorType finalIndexTy =
      finalIndexTensor.getType().cast<RankedTensorType>();
  int64_t indicesRank = finalIndexTy.getRank();
  int64_t numIndicesDim = broadcastedIndices.size();
  int64_t indexVecDim = numIndicesDim > 1 ? indicesRank - 1 : indicesRank;

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedDims;
  SmallVector<int64_t> startIndexMap;
  for (int64_t i = 0; i < numIndicesDim; ++i) {
    collapsedDims.push_back(i);
    startIndexMap.push_back(i);
  }
  for (int64_t i = numIndicesDim; i < inputTensorType.getRank(); i++) {
    if (numIndicesDim > 1) {
      offsetDims.push_back(i + indicesRank - 1 - numIndicesDim);
    } else {
      offsetDims.push_back(i + indicesRank - numIndicesDim);
    }
  }
  auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedDims,
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
      op, resultType, input, finalIndexTensor, dimsAttr,
      rewriter.getI64TensorAttr(sliceSizes));
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
  INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
  INSERT_ATENOP_PATTERN(AtenGatherOp);
  INSERT_ATENOP_PATTERN(AtenSliceScatterOp);
  INSERT_ATENOP_PATTERN(AtenIndexTensorHackedTwinOp);
  INSERT_ATENOP_PATTERN(Aten_IndexPutImplOp);
  INSERT_ATENOP_PATTERN(AtenScatterSrcOp);
#undef INSERT_ATENOP_PATTERN
}
