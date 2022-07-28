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

    // Currently we only support statically sized index tensors
    // when there is more than one index tensor.
    // TODO: Add support for dynamic size index tensors. This will probably
    // require broadcasting the index tensors to a common shape.
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
          return rewriter.notifyMatchFailure(
              op, "unimplemented: index tensors must have static shape if "
                  "there is more than one index tensor");
        }
        broadcastedIndexShape.push_back(
            getConstant(rewriter, loc, resultDimSize, rewriter.getIndexType()));
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
}
