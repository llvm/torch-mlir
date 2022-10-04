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
#include "mlir/Dialect/Arith/IR/Arith.h"
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

static void createLinalgPayloadCalculationForGatherOps(
    OpBuilder &b, Location loc, Value input, int64_t inputRank, Value index,
    int64_t dim, int64_t outputRank) {
  SmallVector<Value> indices;
  for (int i = 0; i < inputRank; i++) {
    if (i == dim) {
      indices.push_back(castIntToIndex(b, loc, index));
    } else {
      // `outputRank` might be larger than `inputRank`. The `linalg::IndexOp`
      // takes in the dimension of the output. Add `inputDimOffset` to
      // related to the correct dimension of the output for dimension larger
      // than the given `dim`.
      int64_t inputDimOffset = i < dim ? 0 : outputRank - inputRank;
      indices.push_back(b.create<linalg::IndexOp>(loc, i + inputDimOffset));
    }
  }

  // Assert index < input.sizes[dim]
  Value indexLTInputDim = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, castIntToIndex(b, loc, index),
      getDimOp(b, loc, input, dim));
  b.create<cf::AssertOp>(
      loc, indexLTInputDim,
      b.getStringAttr("index must be smaller than dim size"));

  // Assert index >= 0
  Value cst0 = b.create<arith::ConstantOp>(loc, b.getZeroAttr(index.getType()));
  Value indexGEThanZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, cst0);
  b.create<cf::AssertOp>(loc, indexGEThanZero,
                         b.getStringAttr("index must be larger or equal to 0"));

  Value extract = b.create<tensor::ExtractOp>(loc, input, indices);
  b.create<linalg::YieldOp>(loc, extract);
}

namespace {
class ConvertAtenGatherOp : public OpConversionPattern<AtenGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();

    Value dimValue = op.dim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");

    Value indices = adaptor.index();
    Value self = adaptor.self();
    RankedTensorType newResultTy =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    int64_t rank = newResultTy.getRank();

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, indices);
    Value result = createZeroInitTensor(rewriter, loc, sizes,
                                        newResultTy.getElementType());

    SmallVector<AffineMap, 2> affineMaps(2,
                                         rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iteratorTypes(rank, getParallelIteratorTypeName());
    auto genericOp = rewriter
                         .create<linalg::GenericOp>(
                             loc, result.getType(), indices, result, affineMaps,
                             iteratorTypes,
                             [&](OpBuilder &b, Location loc, ValueRange args) {
                               auto index = args[0];
                               createLinalgPayloadCalculationForGatherOps(
                                   b, loc, self, rank, index, dim, rank);
                             })
                         .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultTy, genericOp);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenEmbeddingOp : public OpConversionPattern<AtenEmbeddingOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenEmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value weight = adaptor.weight();
    Value indices = adaptor.indices();
    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    auto weightTy = weight.getType().cast<RankedTensorType>();
    if (weightTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "weight must be rank 2");
    Value embeddingDim = getDimOp(rewriter, loc, weight, 1);
    Type elemTy = weightTy.getElementType();

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, indices);
    sizes.push_back(embeddingDim);
    int64_t resultRank = sizes.size();

    auto indicesTy = indices.getType().cast<RankedTensorType>();
    int64_t indicesRank = indicesTy.getRank();
    SmallVector<AffineExpr> indicesExprs;
    for (int i = 0; i < indicesRank; i++)
      indicesExprs.push_back(rewriter.getAffineDimExpr(i));
    auto indicesAffineMap = AffineMap::get(
        /*dimCount=*/resultRank,
        /*symbolCount=*/0, indicesExprs, op->getContext());
    SmallVector<AffineMap, 2> indexingMaps = {
        indicesAffineMap,
        rewriter.getMultiDimIdentityMap(resultRank),
    };
    SmallVector<StringRef> iteratorTypes(sizes.size(),
                                         getParallelIteratorTypeName());
    Value initTensor =
        rewriter.create<linalg::InitTensorOp>(loc, sizes, elemTy);
    Value embeddingResult =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), indices, initTensor,
                /*indexingMaps=*/indexingMaps, /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = args[0];
                  createLinalgPayloadCalculationForGatherOps(
                      b, loc, weight, weightTy.getRank(), index, /*dim=*/0,
                      resultRank);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                embeddingResult);
    return success();
  }
};
} // namespace

namespace {
// AtenEmbeddingPaddingIdxOp
// SUM mode == integer 0
// Sums bags of embeddings together from a weight tensor based on an index and
// offset Vector. Example arguments weight = [[1, 3, 5, 3],
//           [3, 4, 2, 1],
//           [2, 2, 3, 2],
//           [0, 4, 2, 1]]
//
// indices = [0, 2, 3, 1, 2, 3, 2, 1, 0, 1]
// offsets = [0, 3, 5]
//
// output_tensor = initZeroTensor(offsets_length, embedding_size)
//
// for i in range(offsets_length):         <- dim0
//     for j in range(indices_length):     <- dim1
//         for k in range(embedding_size): <- dim2
//             if(offsets[i] <= j and j < offsets[i+1]):
//                 output_tensor[i][k] = output_tensor[i][k] +
//                 weight[indices[j]][k]
//             else:
//                 break
//
// Indexing maps for linalg::Generic ops
//
//
// indices_indexing_map  = (d0, d1, d2) -> (d1)
// offset_indexing_map   = (d0, d1, d2) -> (d0)
// output_indexing_map   = (d0, d1, d2) -> (d0, d2)
//
// TODO: Find an optimal lowering.
//       current lowering is not optimal for bags of large embeddings.
//       Since it traverses the output tensor multiple times. 
//      
//

class ConvertAtenEmbeddingBagPaddingIdxOp
    : public OpConversionPattern<AtenEmbeddingBagPaddingIdxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenEmbeddingBagPaddingIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    auto context = op->getContext();
    Value weight = adaptor.weight();
    Value indices = adaptor.indices();
    Value offsets = adaptor.offsets();
    Value scaleGradByFreq = op.scale_grad_by_freq();
    Value mode = op.mode();
    Value sparse = op.sparse();
    Value includeLastOffset = op.include_last_offset();

    bool scaleGradByFreqBool;
    if (!matchPattern(scaleGradByFreq,
                      m_TorchConstantBool(&scaleGradByFreqBool))) {
      return rewriter.notifyMatchFailure(
          op, "scale_grad_by_freq is expected to be a constant boolean value.");
    }

    if (scaleGradByFreqBool) {
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: scale_grad_by_freq=True.");
    }

    int64_t modeInt;
    if (!matchPattern(mode, m_TorchConstantInt(&modeInt))) {
      return rewriter.notifyMatchFailure(
          op, "mode is expected to be a constant integer value.");
    }

    if (modeInt != torch_upstream::EmbeddingBagMode::MODE_SUM) {
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Mean and Max mode are not supported yet for EmbeddingBag.");
    }

    bool isSparse;
    if (!matchPattern(sparse, m_TorchConstantBool(&isSparse))) {
      return rewriter.notifyMatchFailure(
          op, "sparse is expected to be a constant boolean value.");
    }

    if (isSparse) {
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Sparse mode is not supported yet for EmbeddingBag.");
    }

    bool discardLastOffset;
    if (!matchPattern(includeLastOffset,
                      m_TorchConstantBool(&discardLastOffset))) {
      return rewriter.notifyMatchFailure(
          op,
          "include_last_offset is expected to be a constant boolean value.");
    }

    auto weightTy = weight.getType().cast<RankedTensorType>();
    if (weightTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "weight must be rank 2");

    auto indicesTy = indices.getType().cast<RankedTensorType>();
    if (indicesTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "indices must be a vector");

    auto offsetsTy = offsets.getType().cast<RankedTensorType>();
    if (offsetsTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "offsets much be a vector");

    Type weightElemTy = weightTy.getElementType();

    int64_t iterationMapDimension = weightTy.getRank() + indicesTy.getRank();
    SmallVector<AffineExpr> indicesExpr;
    indicesExpr.push_back(mlir::getAffineDimExpr(1, context));
    auto indicesIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         indicesExpr, context);

    SmallVector<AffineExpr> offsetsExpr;
    offsetsExpr.push_back(mlir::getAffineDimExpr(0, context));

    auto offsetIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         offsetsExpr, context);

    SmallVector<AffineExpr> outputExpr;
    outputExpr.push_back(mlir::getAffineDimExpr(0, context));
    outputExpr.push_back(mlir::getAffineDimExpr(2, context));

    auto outputIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         outputExpr, context);

    SmallVector<AffineMap, 3> indexingMaps = {
          indicesIndexingMap,
          offsetIndexingMap,
          outputIndexingMap,
    };

    // Reduce along the indices dim
    SmallVector<StringRef> iteratorTypes({getParallelIteratorTypeName(),
                                          getReductionIteratorTypeName(),
                                          getParallelIteratorTypeName()});

    Value embeddingDim = getDimOp(rewriter, loc, weight, 1);
    Value initTensor;
    Value offsetsLength;
    Value indicesLength;
    if (!discardLastOffset) {
      SmallVector<Value> sizes{getDimOp(rewriter, loc, offsets, 0),
                                 embeddingDim};

      initTensor = createZeroInitTensor(rewriter, loc, sizes, weightElemTy);
      offsetsLength = getDimOp(rewriter, loc, offsets, 0);
      indicesLength = getDimOp(rewriter, loc, indices, 0);
    } else {
      return rewriter.notifyMatchFailure(
            op, "Unimplemented: include last offset is not yet "
                "supported for EmbeddingBag.");
    }

    Value embeddingBagResult =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), ValueRange{indices, offsets},
                initTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value indexInIndices = args[0];
                  Value offsetsI = args[1];
                  Value initTensorElem = args[2];

                  Value indexI = b.create<linalg::IndexOp>(loc, /*value=*/0);
                  Value indexIToInt = castIndexToInt64(b, loc, indexI);
                  Value one = getConstant(
                      b, loc, 1,
                      mlir::IntegerType::get(getContext(), 64,
                                             IntegerType::Signless));
                  Value offsetIndexPlusOneInt =
                      b.create<arith::AddIOp>(loc, indexIToInt, one);

                  Value offsetIndexPlusOne =
                      castIntToIndex(b, loc, offsetIndexPlusOneInt);
                  Value checkLast = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq,
                      castIndexToInt64(b, loc, offsetsLength),
                      offsetIndexPlusOneInt);
                  Value nextOffset = b.create<arith::SelectOp>(
                      loc, checkLast, castIndexToInt64(b, loc, indicesLength),
                      b.create<tensor::ExtractOp>(loc, offsets,
                                                  offsetIndexPlusOne));

                  Value indicesIndex = castIndexToInt64(
                      b, loc, b.create<linalg::IndexOp>(loc, /*value=*/1));

                  Value offsetLessThanIndicesIndex = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::slt, offsetsI, indicesIndex);
                  Value offsetEqualToIndicesIndex = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, offsetsI, indicesIndex);
                  Value offsetLessThanOrEqualToIndicesIndex =
                      b.create<arith::OrIOp>(loc, offsetLessThanIndicesIndex,
                                               offsetEqualToIndicesIndex);

                  Value indicesIndexLessThanNextOffset =
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              indicesIndex, nextOffset);

                  Value indicesIndexWithinBounds = b.create<arith::AndIOp>(
                      loc, offsetLessThanOrEqualToIndicesIndex,
                      indicesIndexLessThanNextOffset);

                  SmallVector<Value> indexIntoWeight;
                  indexIntoWeight.push_back(
                      castIntToIndex(b, loc, indexInIndices));
                  indexIntoWeight.push_back(
                      b.create<linalg::IndexOp>(loc, /*value=*/2));
                  Value weightElem = b.create<tensor::ExtractOp>(
                      loc, weight, indexIntoWeight);

                  Value addResult = b.create<arith::AddFOp>(loc, weightElem,
                                                            initTensorElem);
                  Value select =
                      b.create<arith::SelectOp>(loc, indicesIndexWithinBounds,
                                                  addResult, initTensorElem);
                  b.create<linalg::YieldOp>(loc, select);
              })
          .getResult(0);

      // cast outputType.
    auto restulType0 = typeConverter->convertType(op->getResult(0).getType());
    Value castedEmbeddingBagResult =
        rewriter.create<tensor::CastOp>(loc, restulType0, embeddingBagResult);

    // offset2 tensor, this should be an empty tensor for the sum mode
    SmallVector<Value> offsetResultSize;
    Type offsetElemTy = offsetsTy.getElementType();
    Value zeroDim = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/0);
    offsetResultSize.push_back(zeroDim);
    Value offsetResult = rewriter.create<linalg::InitTensorOp>(
        loc, offsetResultSize, offsetElemTy);
    auto resultType1 = typeConverter->convertType(op->getResult(1).getType());
    Value castedOffsetResult =
        rewriter.create<tensor::CastOp>(loc, resultType1, offsetResult);

    SmallVector<Value> offsetSize = getTensorSizes(rewriter, loc, offsets);
    // bagsize, vector of size offset with zeros, I think this is always just
    // a vector of zeros in the sum mode
    Value bagSize =
        createZeroInitTensor(rewriter, loc, offsetSize, offsetElemTy);
    auto resultType2 = typeConverter->convertType(op->getResult(2).getType());
    Value castedBagSizeResult =
        rewriter.create<tensor::CastOp>(loc, resultType2, bagSize);

    // max indices, vector of size offset with zeros, this is also always a
    // vector of zeros in the sum mode. Its mainly used in the max mode.
    Value indicesOut =
        createZeroInitTensor(rewriter, loc, offsetSize, offsetElemTy);
    auto resultType3 = typeConverter->convertType(op->getResult(3).getType());
    Value castedMaxIndices =
        rewriter.create<tensor::CastOp>(loc, resultType3, indicesOut);

    rewriter.replaceOp(op, {castedEmbeddingBagResult, castedOffsetResult,
                              castedBagSizeResult, castedMaxIndices});

    return success();
  }
};
} // namespace

namespace {
// Let's say we have an input tensor: initialized with some random values of
// size [4, 5, 6]. An index tensor (always 1-d): [0, 2] of size [2], and an
// integer argument dim = 1. The size of the output tensor will be [4, 2, 6].
// The approach is as follows:
//
// for i in range(input.size[0])
//    for j in range(index.size[0])
//       for k in range(input.size[2])
//          indexValue = index[j]
//          output[i,j,k] = input[i,indexValue,k]

class ConvertAtenIndexSelectOp : public OpConversionPattern<AtenIndexSelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.self();
    Value indices = adaptor.index();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    unsigned inputRank = inputType.getRank();

    int64_t dimInt;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dimInt)))
      return op->emitError("unimplemented: dim is not constant");

    SmallVector<Value> resultShape = getTensorSizes(rewriter, loc, input);
    resultShape[dimInt] = getTensorSizes(rewriter, loc, indices)[0];
    Value initTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, elementType);

    SmallVector<AffineExpr> resultExpr;
    AffineExpr indicesExpr = rewriter.getAffineDimExpr(dimInt);
    SmallVector<StringRef> iteratorTypes;

    for (unsigned i = 0; i < inputRank; i++) {
      resultExpr.push_back(rewriter.getAffineDimExpr(i));
      iteratorTypes.push_back(getParallelIteratorTypeName());
    }

    auto indexingMaps = AffineMap::inferFromExprList({indicesExpr, resultExpr});

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), ValueRange{indices}, initTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), args[0]);
                  SmallVector<Value> indexTarget;
                  for (unsigned i = 0; i < inputRank; i++)
                    indexTarget.push_back(b.create<linalg::IndexOp>(loc, i));
                  indexTarget[dimInt] = index;
                  Value extractedElement =
                      b.create<tensor::ExtractOp>(loc, input, indexTarget);
                  b.create<linalg::YieldOp>(loc, extractedElement);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

// IndexTensor for multiple input tensors broadcasts their shapes to a common
// shape and then replaces the indexed dims with the indices given by the
// indexing tensors:
// x[i_1, i_2, ..., i_M] = result
// result[...] = x[i_1[...], i_2[...], ..., i_M[...]]
//
// where the result shape is computed as follows:
// 1. broadcast i_1, i_2, ..., i_M to a common shape
// 2. if i_1, i_2, ..., i_M is not contiguous, transpose the broadcasted
//    shape to the beginning of the result shape, while removing the
//    unchanged dims (marked by None)
// 3. Otherwise replace the indexed dims with the broadcasted shape
//
// e.g. x: [2, 3]
//      x[[4], [6, 1]] -> x[6, 4]
namespace {
class ConvertAtenIndexTensorOp : public OpConversionPattern<AtenIndexTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIndexTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.self();
    Value indices = op.indices();
    SmallVector<Value> indicesTuple;
    if (!getListConstructElements(indices, indicesTuple)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the indices list is not from a list construct");
    }

    SmallVector<Value> indicesVal =
        getTypeConvertedValues(rewriter, loc, getTypeConverter(), indicesTuple);

    // Identify the indices with non-None index tensors and determine if they
    // are contiguous within the input list.
    SmallVector<int> indexTensorDims;
    SmallVector<Value> indexTensors;
    bool contiguous = true;
    for (auto i : llvm::seq(0, (int)indicesVal.size())) {
      Value index = indicesVal[i];
      if (!index || failed(checkNotNone(rewriter, op, index)))
        continue;
      if (!indexTensorDims.empty() && indexTensorDims.back() != i - 1)
        contiguous = false;
      indexTensorDims.push_back(i);
      indexTensors.push_back(index);
    }

    if (indexTensors.empty()) {
      return rewriter.notifyMatchFailure(
          op, "aten.index.Tensor: index tensor must not be None");
    }

    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    int inputRank = inputType.getRank();
    int resultRank = resultType.getRank();
    int firstIndexDim = indexTensorDims[0];
    int replacedIndexCount = indexTensorDims.size();
    int64_t startIndex = contiguous ? firstIndexDim : 0;

    // Currently we only support statically sized index tensors or dynamic size
    // index tensors without overlapping dynamic dims when there is more than
    // one index tensor.
    // TODO: Add support for dynamic size index tensors with overlapping
    // dynamic dims.
    SmallVector<Value> broadcastedIndexShape;
    if (indexTensors.size() > 1) {
      int maxRank = -1;
      for (auto indexTensor : indexTensors) {
        RankedTensorType indexTensorType =
            indexTensor.getType().cast<RankedTensorType>();
        maxRank = std::max(maxRank, (int)indexTensorType.getRank());
      }

      // Because we are assuming static shapes, we can get the shape of the
      // broadcasted index tensors from the shape refinement pass
      auto refinedResultShape = resultType.getShape();
      for (auto i : llvm::seq(startIndex, startIndex + maxRank)) {
        auto resultDimSize = refinedResultShape[i];
        if (ShapedType::isDynamic(resultDimSize)) {
          SmallVector<Value> dynamicDims;
          int64_t staticDimSize = -1;
          for (auto indexTensor : indexTensors) {
            RankedTensorType indexTensorType =
                indexTensor.getType().cast<RankedTensorType>();
            int64_t indexTensorRank = indexTensorType.getRank();
            if ((maxRank - indexTensorRank) > (i - startIndex))
              continue;
            int64_t dim = i - startIndex - maxRank + indexTensorRank;
            if (ShapedType::isDynamic(indexTensorType.getShape()[dim]))
              dynamicDims.push_back(getDimOp(rewriter, loc, indexTensor, dim));
            else
              staticDimSize =
                  std::max(staticDimSize, indexTensorType.getShape()[dim]);
          }
          if (dynamicDims.size() >= 2)
            return rewriter.notifyMatchFailure(
                op,
                "unimplemented: index tensors with overlapping dynamic dims");
          if (staticDimSize > 1) {
            Value cstStaticDimSize = getConstant(rewriter, loc, staticDimSize,
                                                 rewriter.getIndexType());
            auto equalToRunning = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::eq, cstStaticDimSize,
                dynamicDims[0]);
            rewriter.create<cf::AssertOp>(loc, equalToRunning,
                                          "mismatched size for broadcast");
          }
          broadcastedIndexShape.push_back(dynamicDims[0]);
        } else {
          broadcastedIndexShape.push_back(getConstant(
              rewriter, loc, resultDimSize, rewriter.getIndexType()));
        }
      }
    } else {
      // For a single indexing tensor we can simply use its (dynamic) sizes
      broadcastedIndexShape =
          getTensorSizes(rewriter, loc, indexTensors.front());
    }

    // This result shape calculation assumes that there is only one
    // index tensor, or all of the index tensors are statically shaped.
    int broadcastRank = broadcastedIndexShape.size();

    SmallVector<Value> resultShape;
    if (contiguous) {
      for (auto i : llvm::seq(0, firstIndexDim)) {
        resultShape.push_back(getDimOp(rewriter, loc, input, i));
      }
      resultShape.append(broadcastedIndexShape);
      for (auto i : llvm::seq((int)resultShape.size(), resultRank)) {
        resultShape.push_back(getDimOp(rewriter, loc, input,
                                       i - broadcastRank + replacedIndexCount));
      }
    } else {
      resultShape.append(broadcastedIndexShape);
      int j = 0;
      for (auto i : llvm::seq(0, inputRank)) {
        if (j < replacedIndexCount && i == indexTensorDims[j]) {
          j++;
          continue;
        }
        resultShape.push_back(getDimOp(rewriter, loc, input, i));
      }
    }

    // Initialize the indexing maps for the generic op. Because we are assuming
    // static shapes for the indexing tensors when there are more than 1, we can
    // safely map all size 1 dims to 0 in the corresponding affine maps.
    // TODO: For dynamic shapes, we have to either broadcast the index tensors
    // to a common shape or introduce some form of control flow.
    Value initTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, elementType);
    SmallVector<AffineMap> indexingMaps;
    SmallVector<StringRef> iteratorTypes;

    for (auto indexTensor : indexTensors) {
      RankedTensorType indexTensorType =
          indexTensor.getType().cast<RankedTensorType>();
      auto indexTensorShape = indexTensorType.getShape();
      int rank = indexTensorShape.size();
      SmallVector<AffineExpr> indicesExpr;
      for (auto dim : llvm::seq(0, rank)) {
        if (indexTensorShape[dim] == 1) {
          indicesExpr.push_back(rewriter.getAffineConstantExpr(0));
          continue;
        }
        indicesExpr.push_back(
            rewriter.getAffineDimExpr(startIndex + broadcastRank - rank + dim));
      }
      indexingMaps.push_back(
          AffineMap::get(resultRank, 0, indicesExpr, op->getContext()));
    }

    SmallVector<AffineExpr> resultExpr;
    for (auto i : llvm::seq(0, resultRank)) {
      resultExpr.push_back(rewriter.getAffineDimExpr(i));
      iteratorTypes.push_back(getParallelIteratorTypeName());
    }

    indexingMaps.push_back(
        AffineMap::get(resultRank, 0, resultExpr, op->getContext()));

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), indexTensors, initTensor,
                indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> extractionIndices;
                  if (contiguous) {
                    for (auto i : llvm::seq(0, firstIndexDim)) {
                      extractionIndices.push_back(
                          b.create<linalg::IndexOp>(loc, i));
                    }
                    for (auto i : llvm::seq(0, (int)indexTensorDims.size())) {
                      extractionIndices.push_back(
                          castIntToIndex(b, loc, args[i]));
                    }
                    for (auto i :
                         llvm::seq((int)extractionIndices.size(), inputRank)) {
                      extractionIndices.push_back(b.create<linalg::IndexOp>(
                          loc, i + broadcastRank - replacedIndexCount));
                    }
                  } else {
                    int indexCount = 0, unchanged = 0;
                    for (auto i : llvm::seq(0, inputRank)) {
                      if (indexCount < replacedIndexCount &&
                          i == indexTensorDims[indexCount]) {
                        extractionIndices.push_back(
                            castIntToIndex(b, loc, args[indexCount++]));
                        continue;
                      }
                      extractionIndices.push_back(b.create<linalg::IndexOp>(
                          loc, broadcastRank + unchanged));
                      unchanged++;
                    }
                  }
                  Value extractedElement = b.create<tensor::ExtractOp>(
                      loc, input, extractionIndices);
                  b.create<linalg::YieldOp>(loc, extractedElement);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

// `getScaledDims` scales the `dim` value with a scale factor `ScaleFactor`.
// The `dim` and `scaleFactor` are assumed to be of index and float type
// respectively. `scaledDim = int(floor(float(dim) * scaleFactor))`.
static Value getScaledDims(OpBuilder &builder, Location loc, Value dim,
                           Value scaleFactor) {

  Value dimInt = castIndexToInt64(builder, loc, dim);
  Value dimFp =
      builder.create<arith::SIToFPOp>(loc, scaleFactor.getType(), dimInt);
  Value scaleDim = builder.create<arith::MulFOp>(loc, dimFp, scaleFactor);
  Value floorDim = builder.create<math::FloorOp>(loc, scaleDim);
  Value scaledDimToIndex = castIntToIndex(
      builder, loc,
      builder.create<arith::FPToSIOp>(loc, dimInt.getType(), floorDim));

  return scaledDimToIndex;
}

// `getScaleFactor` returns the scale factor from input to output dimension.
// The `dim` and `scaledDim` are assumed to be of index and int64 type
// respectively. scale_factor = (scaled_dim // dim).
static Value getScaleFactor(OpBuilder &builder, Location loc, Value dim,
                            Value scaledDim) {
  Value dimInt = castIndexToInt64(builder, loc, dim);
  Value scaleFactorInt =
      builder.create<arith::CeilDivSIOp>(loc, scaledDim, dimInt);
  return scaleFactorInt;
}

// N, C, H, W = input_tensor.shape
// N, C, H_scaled, W_scaled = out_tensor.shape
// H_factor, W_factor = H_scaled/H, W_scaled/W

// for i in range(N):
//    for j in range(C):
//      for k in range(H_scaled):
//          for l in range(W_scaled):
//              out_tensor[i, j, k, l] = input[i, j, k//H_factor, l//W_factor]

namespace {
class ConvertAtenUpsampleNearest2dVecOp
    : public OpConversionPattern<AtenUpsampleNearest2dVecOp> {

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUpsampleNearest2dVecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    Value input = adaptor.input();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto inputType = input.getType().cast<RankedTensorType>();
    auto inputRank = inputType.getRank();
    Type elementType = inputType.getElementType();

    SmallVector<Value> dims = getTensorSizes(rewriter, loc, input);
    SmallVector<Value, 2> scaleFactorsInt;

    // The dimension at which the scaling starts.
    unsigned hDimOffset = 2;

    if (!adaptor.scale_factors().getType().isa<Torch::NoneType>()) {
      SmallVector<Value, 2> scaleFactorsTorchFloat;
      if (!getListConstructElements(op.scale_factors(), scaleFactorsTorchFloat))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the scale_factors is not constructed from "
                "ListConstruct");
      SmallVector<Value, 2> scaleFactorsFloatValues;
      scaleFactorsFloatValues = getTypeConvertedValues(
          rewriter, loc, getTypeConverter(), scaleFactorsTorchFloat);
      // Convert float values to int values.
      // int_value = (int64_t)ceil(float_value)
      for (auto floatValue : scaleFactorsFloatValues) {
        Value ceilVal = rewriter.create<math::CeilOp>(loc, floatValue);
        Value intVal = rewriter.create<arith::FPToSIOp>(
            loc, rewriter.getI64Type(), ceilVal);
        scaleFactorsInt.push_back(intVal);
      }

      for (unsigned i = 0; i < scaleFactorsFloatValues.size(); i++)
        dims[hDimOffset + i] = getScaledDims(
            rewriter, loc, dims[hDimOffset + i], scaleFactorsFloatValues[i]);

    } else {

      SmallVector<Value, 2> outputSizeTorchInt;
      if (!getListConstructElements(op.output_size(), outputSizeTorchInt))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the output_size is not constructed from "
                "ListConstruct");
      SmallVector<Value, 2> outputSizeIntValues;
      outputSizeIntValues = getTypeConvertedValues(
          rewriter, loc, getTypeConverter(), outputSizeTorchInt);

      for (unsigned i = 0; i < outputSizeTorchInt.size(); i++) {
        auto scaleFactorVal = getScaleFactor(
            rewriter, loc, dims[hDimOffset + i], outputSizeIntValues[i]);
        scaleFactorsInt.push_back(scaleFactorVal);
        dims[hDimOffset + i] =
            castIntToIndex(rewriter, loc, outputSizeIntValues[i]);
      }
    }

    Value outTensor =
        rewriter.create<linalg::InitTensorOp>(loc, dims, elementType);

    AffineMap idMap = rewriter.getMultiDimIdentityMap(inputRank);
    SmallVector<StringRef> iteratorTypes(inputRank,
                                         getParallelIteratorTypeName());

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, outTensor.getType(), ValueRange{}, outTensor,
                /*indexingMaps=*/idMap,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> indices;
                  for (unsigned i = 0; i < inputRank; i++)
                    indices.push_back(b.create<linalg::IndexOp>(loc, i));

                  for (unsigned i = 0; i < (inputRank - hDimOffset); i++)
                    indices[i + hDimOffset] = b.create<arith::FloorDivSIOp>(
                        loc, indices[i + hDimOffset],
                        castIntToIndex(rewriter, loc, scaleFactorsInt[i]));

                  Value retVal =
                      b.create<tensor::ExtractOp>(loc, input, indices);
                  b.create<linalg::YieldOp>(loc, retVal);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::
    populateIndirectDataMovementPatternsAndLegality(
        TypeConverter &typeConverter, RewritePatternSet &patterns,
        ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenGatherOp>();
  patterns.add<ConvertAtenGatherOp>(typeConverter, context);
  target.addIllegalOp<AtenEmbeddingOp>();
  patterns.add<ConvertAtenEmbeddingOp>(typeConverter, context);
  target.addIllegalOp<AtenIndexSelectOp>();
  patterns.add<ConvertAtenIndexSelectOp>(typeConverter, context);
  target.addIllegalOp<AtenIndexTensorOp>();
  patterns.add<ConvertAtenIndexTensorOp>(typeConverter, context);
  target.addIllegalOp<AtenEmbeddingBagPaddingIdxOp>();
  patterns.add<ConvertAtenEmbeddingBagPaddingIdxOp>(typeConverter, context);
  target.addIllegalOp<AtenUpsampleNearest2dVecOp>();
  patterns.add<ConvertAtenUpsampleNearest2dVecOp>(typeConverter, context);
}
