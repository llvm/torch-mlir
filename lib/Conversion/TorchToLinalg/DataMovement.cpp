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

#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenFlattenUsingIntsOp
    : public OpConversionPattern<AtenFlattenUsingIntsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t startDim;
    if (!matchPattern(op.start_dim(), m_TorchConstantInt(&startDim)))
      return rewriter.notifyMatchFailure(op, "start_dim must be constant");
    int64_t endDim;
    if (!matchPattern(op.end_dim(), m_TorchConstantInt(&endDim)))
      return rewriter.notifyMatchFailure(op, "end_dim must be constant");
    auto type = adaptor.self().getType().cast<RankedTensorType>();
    auto inputRank = type.getRank();
    auto resultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    if (startDim < 0)
      startDim += inputRank;
    if (endDim < 0)
      endDim += inputRank;

    if (inputRank == 0) {
      SmallVector<ReassociationIndices> reassociation;
      if (!(startDim >= -1 && startDim <= 0 && endDim >= -1 && endDim <= 0))
        return rewriter.notifyMatchFailure(
            op, "start_dim and end_dim must be in [-1, 0] when inputRank is 0");
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          op, resultType, adaptor.self(), reassociation);
      return success();
    }

    if (startDim < 0 || startDim >= inputRank || endDim < 0 ||
        endDim >= inputRank || startDim > endDim)
      return rewriter.notifyMatchFailure(
          op, "statically invalid flattening dim range");

    SmallVector<ReassociationIndices> reassociation(resultType.getRank());
    int j = 0;
    for (auto i : llvm::seq<int64_t>(0, inputRank)) {
      reassociation[j].push_back(i);
      if (i < startDim || i >= endDim)
        j++;
    }
    Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
        op->getLoc(), adaptor.self(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                collapsedTensor);
    return success();
  }
};
} // namespace

namespace {
/// The `ConvertAtenViewOp` conversion pattern converts `aten.View` op to
/// `linalg.TensorExpandShape` op only when one or multiple static dimensions
/// are expanded. All the other cases of `aten.View` op need to be handled.
/// TODO: Handle all the other cases of `aten.View` op.
class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.self();
    auto inputType = input.getType().cast<RankedTensorType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t inputRank = inputType.getRank();
    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();
    if (resultRank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "result shape of rank 0 is invalid");

    // TODO: add support for case inputRank 0 expanded to size 1
    if (inputRank == 0)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input rank 0 is not supported");

    bool isCollapse = inputRank > resultRank ? true : false;
    int64_t collapsedRank = isCollapse ? resultRank : inputRank;
    int64_t expandedRank = isCollapse ? inputRank : resultRank;

    // Extract the desired output size as a list of integers. This list should
    // have been created using the operation `torch.prim.ListConstruct`.
    SmallVector<Value> outputSizeTorchInt;
    if (!getListConstructElements(op.size(), outputSizeTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: the target size is "
                                         "not constructed from ListConstruct");
    }
    SmallVector<Value> outputSizeInt = getTypeConvertedValues(
        rewriter, loc, typeConverter, outputSizeTorchInt);
    if (resultRank != (int64_t)outputSizeInt.size()) {
      return rewriter.notifyMatchFailure(
          op, "desired size list length mismatches with the result type rank");
    }

    SmallVector<Value> inputSize = getTensorSizes(rewriter, loc, input);
    ArrayRef<Value> expandedShapeInt =
        llvm::makeArrayRef(isCollapse ? inputSize : outputSizeInt);
    ArrayRef<Value> collapsedShapeInt =
        llvm::makeArrayRef(isCollapse ? outputSizeInt : inputSize);

    // Currently, we only handle the expanding or collapsing cases or the
    // identity cases where the rank and shape of the input and result are
    // equal, and the input itself is the result. We do not handle expanding And
    // collapsing happening at the same time or cases where it's neither
    // collapsing nor expanding like view of [2,3] for 3x2 tensor.
    // TODO: For the expanding And collapsing case, we will need to identify
    // which dimensions are collapsing and which are expanding and do it in two
    // steps.
    // TODO: For neither collapsing nor expanding, we could find a intermediate
    // shape to collapse and then expanded to the target shape. Like [2,3] =>
    // [6] => [3, 2].
    if (inputRank == resultRank) {
      for (unsigned i = 0; i < inputRank; i++)
        checkDimEqualHelper(rewriter, loc, inputSize[i], outputSizeInt[i]);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
      return success();
    }

    // Iterate through the view op size list to do the following:
    //
    // 1. Combine output size list and input tensor type info to get the most
    // static outputShape.
    //
    // 2. Fill in the reassociation for size list item where the output dim size
    // is got from `torch.aten.size.int(inputTensor, inputDim)`. We naively
    // assume this means the corresponding dimension is not expanded or
    // collapsed. Note this may technically not always be true.
    // TODO: think of a way better way to at least detect when this assumption
    // is violated.
    SmallVector<int64_t> outputShape(resultRank, kUnknownSize);
    SmallVector<ReassociationIndices> reassociation(collapsedRank);
    llvm::Optional<int64_t> inferredDimension;
    for (auto en : llvm::enumerate(outputSizeTorchInt)) {
      int64_t inputDim;
      int64_t size;
      int64_t outputDim = en.index();
      // Match torch.aten.size.int(inputTensor, inputDim) with constant inputDim
      if (matchPattern(en.value(),
                       m_TorchTensorSizeInt(op.self(), &inputDim))) {
        auto collapsedDim = isCollapse ? outputDim : inputDim;
        auto expandedDim = isCollapse ? inputDim : outputDim;
        reassociation[collapsedDim].push_back(expandedDim);
        if (!inputType.isDynamicDim(inputDim)) {
          outputShape[outputDim] = inputShape[inputDim];
          continue;
        }
      } else if (matchPattern(en.value(), m_TorchConstantInt(&size))) {
        if (size != -1) {
          outputShape[outputDim] = size;
          continue;
        }

        if (inferredDimension.hasValue()) {
          return rewriter.notifyMatchFailure(
              op, "at most one element in size list is allowed to be -1");
        }
        inferredDimension = outputDim;
      }
    }

    // Use static information of input tensor to determine size of inferred
    // dimension in output shape.
    //
    // If there is an inferred dimension and that is the only dimension
    // in the output shape (i.e. the tensor is getting fully flattened),
    // then we don't need to analyze the static information of the input
    // shape since the reassociation of dimensions only requires rank
    // information.
    if (inferredDimension.hasValue() && outputShape.size() > 1) {
      if (llvm::count(outputShape, kUnknownSize) != 1 ||
          llvm::count(inputShape, kUnknownSize) != 0) {
        return rewriter.notifyMatchFailure(
            op,
            "unimplemented: an inferred dimension is only supported when there "
            "is enough static shape information to determine its size, or when "
            "the input tensor is being flattened to a single dimension");
      }

      auto productReduceKnownSizes = [](const ArrayRef<int64_t> sizes) {
        auto knownSizes = llvm::make_filter_range(
            sizes, [](int64_t val) { return val != kUnknownSize; });
        return std::accumulate(knownSizes.begin(), knownSizes.end(), /*init=*/1,
                               std::multiplies<int64_t>());
      };

      int64_t numOfElements = productReduceKnownSizes(inputShape);
      int64_t outputKnownNumOfElements = productReduceKnownSizes(outputShape);
      if (numOfElements % outputKnownNumOfElements != 0) {
        return rewriter.notifyMatchFailure(
            op, "number of elements in input tensor must be divisible by "
                "product of non-inferred dimensions in size list");
      }
      outputShape[*inferredDimension] =
          numOfElements / outputKnownNumOfElements;
    }

    SmallVector<int64_t> collapsedShape =
        isCollapse ? outputShape : llvm::to_vector(inputShape);
    SmallVector<int64_t> expandedShape =
        isCollapse ? llvm::to_vector(inputShape) : outputShape;

    // The while loop does the following:
    // 1. Fill in the reassociation indices for dimensions that are expanded.
    // Check the interval dimensions between two unchanged dims in the
    // collapsedShape. If the interval is size 1, associate all the dims
    // in the expandedShape shape until the next unchanged dim. If the interval
    // is larger than size 1, figure out the associations with assumptions that
    // dynamic dimensions are not splitted.
    // 2. Set collapsedShape and expandedShape following the requirements by
    // tensor.expand_shape verification code:
    //    a. As long as one or more of the related dimensions in the expanded
    //    shape is dynamic the collapsed dimension is dynamic.
    //    b. If all of the related dimensions are static, the collapsed
    //    dimension must be static. In other words, if a collapsed dimension is
    //    dynamic, at least one of the related dimensions need to be dynamic.
    int64_t collapsedDim = 0, expandedDim = 0;
    while (collapsedDim < collapsedRank && expandedDim < expandedRank) {
      // Not empty means the associations has been filled in and the dimension
      // is unchanged.
      if (!reassociation[collapsedDim].empty()) {
        if (expandedDim != reassociation[collapsedDim][0])
          return op.emitOpError("Unsupported: expanded dims are off from the "
                                "expected dim got from reassociation");
        collapsedDim++;
        expandedDim++;
        continue;
      }

      // Collect the dims that are collapsed until hitting the next dim that's
      // unchanged.
      SmallVector<int64_t> collapsedDims;
      while (collapsedDim < collapsedRank &&
             reassociation[collapsedDim].empty()) {
        collapsedDims.push_back(collapsedDim);
        collapsedDim++;
      }
      // the next reassociation is for a dim that's unchanged.
      int64_t expandedDimNext = collapsedDim != collapsedRank
                                    ? reassociation[collapsedDim][0]
                                    : expandedRank;
      if (collapsedDims.size() == 1) {
        int64_t collapsedDimSize = 1;
        int64_t collapsedDim = collapsedDims[0];
        for (auto i : llvm::seq<int64_t>(expandedDim, expandedDimNext)) {
          reassociation[collapsedDim].push_back(i);
          if (collapsedDimSize == kUnknownSize)
            continue;

          int64_t expandedDimSize = expandedShape[i];
          if (expandedDimSize == kUnknownSize) {
            collapsedDimSize = kUnknownSize;
            continue;
          }
          collapsedDimSize *= expandedShape[i];
        }
        // To meet both requirements from tensor.expand_shape verification code.
        collapsedShape[collapsedDim] = collapsedDimSize;
        expandedDim = expandedDimNext;
        continue;
      }

      // collpasedDims are expanded to [expandedDim, expandedDimNext)
      if (expandedDimNext - expandedDim < (int64_t)collapsedDims.size())
        op.emitError("unimplemented: mixed of expanding and collapsing "
                     "operations for view");
      for (auto collapsedDim : collapsedDims) {
        if (collapsedShape[collapsedDim] == kUnknownSize) {
          if (expandedDim >= expandedDimNext) {
            return rewriter.notifyMatchFailure(
                op,
                "desired size is not compatible with the input tensor size");
          }
          checkDimEqualHelper(rewriter, loc, collapsedShapeInt[collapsedDim],
                              expandedShapeInt[expandedDim]);
          // To meet the second requirement from tensor.expand_shape
          // verification code.
          expandedShape[expandedDim] = kUnknownSize;
          reassociation[collapsedDim].push_back(expandedDim++);
        } else {
          int64_t remainingSizeToExpand = collapsedShape[collapsedDim];
          // A do-while loop is used here to handle the cases where the
          // collapsed shape tensor has a dimension of size 1.
          do {
            int64_t expandedDimSize = expandedShape[expandedDim];
            if (expandedDim >= expandedDimNext ||
                expandedShape[expandedDim] == kUnknownSize ||
                remainingSizeToExpand % expandedDimSize != 0) {
              return rewriter.notifyMatchFailure(
                  op, "total number of elements mismatch in the expansion");
            }
            reassociation[collapsedDim].push_back(expandedDim++);
            remainingSizeToExpand /= expandedDimSize;
          } while (remainingSizeToExpand != 1);
        }
      }
    }

    if (collapsedDim != collapsedRank || expandedDim != expandedRank)
      return rewriter.notifyMatchFailure(op, "view shape is not supported");
    Type adjustedResultType =
        RankedTensorType::get(isCollapse ? collapsedShape : expandedShape,
                              resultType.getElementType());
    Type adjustedInputType =
        RankedTensorType::get(isCollapse ? expandedShape : collapsedShape,
                              resultType.getElementType());
    Value castedInput =
        rewriter.create<tensor::CastOp>(loc, adjustedInputType, input);
    Value result =
        isCollapse
            ? rewriter
                  .create<tensor::CollapseShapeOp>(loc, adjustedResultType,
                                                   castedInput, reassociation)
                  .result()
            : rewriter
                  .create<tensor::ExpandShapeOp>(loc, adjustedResultType,
                                                 castedInput, reassociation)
                  .result();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSqueezeOp : public OpConversionPattern<AtenSqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.self();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    // In case the operand tensor type is statically shaped with all dimensions
    // being unit extent, it will be collapsed to a 0-D tensor.
    if (resultRank == 0) {
      SmallVector<ReassociationIndices> reassociation;
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, input, reassociation);
      return success();
    }

    // All the static size-1 dimensions at the beginning(going from higher to
    // lower dimensions) will be collapsed into the first dynamic or first non
    // size-1 static dimension. All the other static size-1 dimensions will be
    // collapsed into its previous dynamic or non size-1 static dimension.
    SmallVector<ReassociationIndices> reassociation(resultRank);
    bool isSqueezed = false;
    int64_t headOnesCount = 0;
    while (headOnesCount < inputRank &&
           inputType.getDimSize(headOnesCount) == 1) {
      isSqueezed = true;
      reassociation[0].push_back(headOnesCount++);
    }

    // TODO: Add support for size-1 dynamic dimensions.
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    int64_t j = -1;
    for (auto i : llvm::seq<int64_t>(headOnesCount, inputRank)) {
      if (inputType.isDynamicDim(i)) {
        // Make sure that size-1 dynamic dimension does not exist.
        Value dimSize = getDimOp(rewriter, loc, input, i);
        Value dimSizeNotOne = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, dimSize, one);
        rewriter.create<cf::AssertOp>(
            loc, dimSizeNotOne,
            rewriter.getStringAttr(
                "unimplemented: size 1 dynamic dimension is not supported"));
        ++j;
      } else if (inputType.getDimSize(i) != 1) {
        ++j;
      } else {
        // `isSqueezed` checks if the operand tensor type contains at least one
        // unit dimension.
        isSqueezed = true;
      }
      if (j == resultRank)
        break;
      reassociation[j].push_back(i);
    }

    // Make sure that result type rank is compatible with the squeezed size.
    if (j != resultRank - 1)
      return rewriter.notifyMatchFailure(
          op, "expected output size mismatches with the result type rank");

    if (isSqueezed) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, input, reassociation);

    } else {
      // If the operand tensor type does not have any unit dimension,
      // `aten.squeeze` will behave as an identity operation.
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
    }
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSqueezeDimOp : public OpConversionPattern<AtenSqueezeDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSqueezeDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value input = adaptor.self();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

    // TODO: Handle the case where the dim(th) dimension is dynamic.
    if (inputType.isDynamicDim(dim)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dim(th) dimension is not expected to be dynamic");
    }

    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();

    // If the dim(th) dimension of operand tensor type is not statically unit,
    // `aten.squeeze` will behave as an identity operation.
    if (inputType.getDimSize(dim) != 1) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
      return success();
    }

    SmallVector<ReassociationIndices> reassociationMap(resultRank);
    bool alreadyCrossedSqueezedDim = false;
    for (int i = 0; i != resultRank; i++) {
      if (alreadyCrossedSqueezedDim) {
        reassociationMap[i].push_back(i + 1);
      } else {
        reassociationMap[i].push_back(i);
        if (dim != 0 && i != dim - 1)
          continue;

        alreadyCrossedSqueezedDim = true;
        if (dim == 0)
          reassociationMap[0].push_back(1);
        if (i == dim - 1)
          reassociationMap[i].push_back(dim);
      }
    }
    // Note: In case the operand tensor type is of unit rank and is statically
    // shaped with unit dimension, the `reassociationMap` will be empty and the
    // input will be collapsed to a 0-D tensor.
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, resultType, input,
                                                         reassociationMap);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenUnsqueezeOp : public OpConversionPattern<AtenUnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    auto inputRank =
        adaptor.self().getType().cast<RankedTensorType>().getRank();
    if (dim < 0)
      dim += inputRank + 1;
    if (!(0 <= dim && dim <= inputRank))
      return rewriter.notifyMatchFailure(op, "statically invalid");

    SmallVector<ReassociationIndices> reassociationMap(inputRank);
    // From the perspective of the reassociation map, the situation of
    // unsqueezing before or after the last dimension is symmetrical.
    // Normalize it to the "before" case.
    // The 0 case is special here, since there is no last dimension to insert
    // before -- we simply rely on the loop below iterating 0 times.
    if (dim == inputRank && inputRank != 0)
      dim = inputRank - 1;
    bool alreadyCrossedExpandedDim = false;
    for (int i = 0; i != inputRank; i++) {
      if (alreadyCrossedExpandedDim) {
        reassociationMap[i].push_back(i + 1);
      } else {
        reassociationMap[i].push_back(i);
        if (i == dim) {
          reassociationMap[i].push_back(i + 1);
          alreadyCrossedExpandedDim = true;
        }
      }
    }
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, resultType, adaptor.self(), reassociationMap);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenTransposeIntOp
    : public OpConversionPattern<AtenTransposeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTransposeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    int64_t dim0;
    if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0)))
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    int64_t dim1;
    if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");

    auto inVector = adaptor.self();
    auto inType = inVector.getType().cast<RankedTensorType>();
    auto inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    auto elementType = inType.getElementType();

    dim0 = toPositiveDim(dim0, inputRank);
    if (!isValidDim(dim0, inputRank))
      return rewriter.notifyMatchFailure(op, "dim0 out of range");
    dim1 = toPositiveDim(dim1, inputRank);
    if (!isValidDim(dim1, inputRank))
      return rewriter.notifyMatchFailure(op, "dim1 out of range");

    auto loc = op.getLoc();

    SmallVector<Value> outputDims;
    for (auto i = 0; i < inputRank; i++)
      outputDims.push_back(getDimOp(rewriter, loc, adaptor.self(), i));
    std::swap(outputDims[dim0], outputDims[dim1]);

    Value outVector =
        rewriter.create<linalg::InitTensorOp>(loc, outputDims, elementType);
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (auto i = 0; i < inputRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (auto i = 0; i < inputRank; i++) {
      if (i == dim0)
        swapExprs.push_back(idExprs[dim1]);
      else if (i == dim1)
        swapExprs.push_back(idExprs[dim0]);
      else
        swapExprs.push_back(idExprs[i]);
    }

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(inputRank, 0, idExprs, op.getContext()),
        AffineMap::get(inputRank, 0, swapExprs, op.getContext())};
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
    auto transpose = rewriter
                         .create<linalg::GenericOp>(
                             loc, outVector.getType(), inVector, outVector,
                             indexingMaps, iteratorTypes,
                             [](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             })
                         .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, transpose);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenPermuteOp : public OpConversionPattern<AtenPermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    SmallVector<int64_t> dimensions;
    if (!matchPattern(op.dims(), m_TorchConstantIntList(dimensions)))
      return rewriter.notifyMatchFailure(op, "all dimensions must be constant");

    Value inVector = adaptor.self();
    auto inType = inVector.getType().cast<RankedTensorType>();
    int64_t inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    Type elementType = inType.getElementType();

    // Check if the dimensions are a valid constants.
    int64_t numDimensions = dimensions.size();
    if (inputRank != numDimensions)
      return rewriter.notifyMatchFailure(
          op, "size of `dims` must be equal to the rank of the input");
    for (unsigned i = 0; i < numDimensions; i++) {
      if (dimensions[i] < 0)
        dimensions[i] = toPositiveDim(dimensions[i], inputRank);
      if (!isValidDim(dimensions[i], inputRank))
        return rewriter.notifyMatchFailure(op, "dimension out of range");
    }

    Location loc = op.getLoc();

    SmallVector<Value> outputDims;
    for (unsigned i = 0; i < inputRank; i++)
      outputDims.push_back(getDimOp(rewriter, loc, inVector, dimensions[i]));

    Value outVector =
        rewriter.create<linalg::InitTensorOp>(loc, outputDims, elementType);
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (unsigned i = 0; i < inputRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (unsigned i = 0; i < inputRank; i++)
      swapExprs.push_back(idExprs[dimensions[i]]);

    SmallVector<AffineMap> indexingMaps =
        AffineMap::inferFromExprList({idExprs, swapExprs});
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
    auto transpose = rewriter
                         .create<linalg::GenericOp>(
                             loc, outVector.getType(), inVector, outVector,
                             indexingMaps, iteratorTypes,
                             [](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             })
                         .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, transpose);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    auto input = adaptor.self();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return op->emitError("unimplemented: dim is not constant");

    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
    Value dimSize = inputShape[dim];

    auto adjustStartOrEnd = [&](Value startOrEndTorchType,
                                Value startOrEndBuiltin, Value valueForNone) {
      if (startOrEndTorchType.getType().isa<Torch::NoneType>())
        return valueForNone;
      auto dimSizeAsInt = castIndexToInt64(rewriter, loc, dimSize);
      Value startOrEndToPositive =
          toPositiveDimDynamic(rewriter, loc, startOrEndBuiltin, dimSizeAsInt);
      // startOrEnd < 0 ? 0 : startOrEnd
      Value cst0 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(dimSizeAsInt.getType()));
      Value predDimSltZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, startOrEndToPositive, cst0);
      Value startOrEndAtLeastZero = rewriter.create<arith::SelectOp>(
          loc, predDimSltZero, cst0, startOrEndToPositive);
      // startOrEnd > dimSizeAsInt ? dimSizeAsInt : startOrEnd
      Value startOrEndSgtDimSize = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, startOrEndAtLeastZero, dimSizeAsInt);
      Value startOrEndBoundedByDimSize = rewriter.create<arith::SelectOp>(
          loc, startOrEndSgtDimSize, dimSizeAsInt, startOrEndAtLeastZero);

      return castIntToIndex(rewriter, loc, startOrEndBoundedByDimSize);
    };

    if (op.start().getType().isa<OptionalType>() ||
        op.end().getType().isa<OptionalType>())
      return rewriter.notifyMatchFailure(op, "unimplemented optional type arg");
    Value start = adjustStartOrEnd(op.start(), adaptor.start(), zero);
    Value end = adjustStartOrEnd(op.end(), adaptor.end(), dimSize);

    // end >= start ? end : start
    Value endSgeStart = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, end, start);
    end = rewriter.create<arith::SelectOp>(loc, endSgeStart, end, start);

    int64_t step;
    if (!matchPattern(op.step(), m_TorchConstantInt(&step))) {
      if (!op.step().getType().isa<Torch::NoneType>())
        return op->emitError("unimplemented: step is not constant");
      step = 1;
    }

    // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
    Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);
    Value len = rewriter.create<arith::SubIOp>(loc, end, start);
    Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
    resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, one);
    resultSize =
        rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);

    SmallVector<Value> resultShape = getTensorSizes(rewriter, loc, input);
    resultShape[dim] = resultSize;

    SmallVector<Value> offsets(inputType.getRank(), zero);
    SmallVector<Value> strides(inputType.getRank(), one);
    offsets[dim] = start;
    strides[dim] = rewriter.create<arith::MulIOp>(loc, strides[dim], stepIndex);

    Value result = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, offsets, resultShape, strides);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    Value dimValue = op.dim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");

    // Collect all the tensors to be concatenated.
    auto tensorList = op.tensors();
    SmallVector<Value> tensorsTorchType;
    if (!getListConstructElements(tensorList, tensorsTorchType))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    auto tensors =
        getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int rank = newResultType.getRank();
    SmallVector<Value> offsets, sizes, strides;
    sizes.reserve(rank);
    strides.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    offsets.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));

    for (int i = 0; i < rank; ++i)
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(loc, tensors[0], i));

    // Calculate the size of the `dim` result dimension by adding the dim size
    // of each tensor together.
    Value resultDimSize = sizes[dim];

    Value dimIndex = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dim));
    for (auto tensor : makeArrayRef(tensors).drop_front()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, tensor, dimIndex);
      resultDimSize =
          rewriter.createOrFold<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;

    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op)
        return v;
      return op.getValue();
    };

    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, sizes, newResultType.getElementType());
    for (auto tensor : tensors) {
      SmallVector<Value> sizes = getTensorSizes(rewriter, loc, tensor);
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, tensor, result,
          llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
      offsets[dim] =
          rewriter.createOrFold<arith::AddIOp>(loc, offsets[dim], sizes[dim]);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, result);
    return success();
  }
};
} // namespace

// Broadcasts input tensor based on the broadcastToShape.
static LogicalResult broadcastToGivenShape(Operation *op,
                                           ConversionPatternRewriter &rewriter,
                                           Value input,
                                           SmallVector<Value> broadcastToShape,
                                           Value &result) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (broadcastToShape.size() < inputShape.size()) {
    return rewriter.notifyMatchFailure(
        op, "invalid shape: broadcastToShape size must not be smaller than the "
            "size of the input shape");
  }

  Type elementType = inputType.getElementType();
  Location loc = op->getLoc();
  MLIRContext *context = op->getContext();
  SmallVector<Value> outShape;

  // Create affine map and shapes for tensor initialization.
  SmallVector<AffineExpr> outExpr;
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  size_t diff = broadcastToShape.size() - inputShape.size();
  for (size_t i = 0; i < broadcastToShape.size(); i++) {
    Value shapeValue = broadcastToShape[i];
    size_t j = i - diff;
    if (i < diff) {
      Value isValid = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, shapeValue, zero);
      rewriter.create<cf::AssertOp>(
          loc, isValid,
          rewriter.getStringAttr(
              "negative values not allowed in new dimensions"));
      outShape.push_back(castIntToIndex(rewriter, loc, shapeValue));
      continue;
    }
    if (inputShape[j] == 1) {
      // Broadcast singleton dimension
      Value one =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value isNegative = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, shapeValue, zero);
      Value select = rewriter.create<arith::SelectOp>(
          loc, isNegative, one, castIntToIndex(rewriter, loc, shapeValue));
      outShape.push_back(select);
      outExpr.push_back(mlir::getAffineConstantExpr(0, context));
      continue;
    }
    // Non-broadcast case
    Value dim = getDimOp(rewriter, loc, input, j);
    Value isNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, shapeValue, zero);
    Value isEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, castIndexToInt64(rewriter, loc, dim),
        shapeValue);
    Value isValid = rewriter.create<arith::OrIOp>(loc, isNegative, isEqual);
    rewriter.create<cf::AssertOp>(
        loc, isValid,
        rewriter.getStringAttr(
            "only broadcasting singleton dimensions supported"));
    outShape.push_back(dim);
    outExpr.push_back(mlir::getAffineDimExpr(i, context));
  }

  Value outTensor =
      rewriter.create<linalg::InitTensorOp>(loc, outShape, elementType);

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(broadcastToShape.size(), 0, outExpr, context),
      rewriter.getMultiDimIdentityMap(broadcastToShape.size())};
  SmallVector<StringRef> iteratorTypes(broadcastToShape.size(), "parallel");
  result = rewriter
               .create<linalg::GenericOp>(
                   loc, outTensor.getType(), input, outTensor, indexingMaps,
                   iteratorTypes,
                   [](OpBuilder &b, Location loc, ValueRange args) {
                     b.create<linalg::YieldOp>(loc, args[0]);
                   })
               .getResult(0);

  return success();
}

namespace {
class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value self = adaptor.self();

    SmallVector<Value> inShape;
    if (!getListConstructElements(adaptor.size(), inShape)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the size list is not from list construct");
    }
    SmallVector<Value> inShapeConverted = getTypeConvertedValues(
        rewriter, op.getLoc(), getTypeConverter(), inShape);

    Value result;
    if (failed(broadcastToGivenShape(op, rewriter, self, inShapeConverted,
                                     result))) {
      return rewriter.notifyMatchFailure(
          op, "unable to perform broadcast operation");
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertValsemVariantAtenCopyOp
    : public OpConversionPattern<ValsemVariantAtenCopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ValsemVariantAtenCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value self = adaptor.self();
    Value src = adaptor.src();
    RankedTensorType selfType = self.getType().cast<RankedTensorType>();

    // The non_blocking should be a constant `False`.
    bool nonBlocking;
    if (!matchPattern(op.non_blocking(), m_TorchConstantBool(&nonBlocking))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non_blocking must be a constant");
    } else if (nonBlocking) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non_blocking is expected to be false");
    }

    // The size of the src tensor can be different from the self but should be
    // broadcastable. Therefore, broadcasting the src tensor to match the size
    // of the self tensor.
    SmallVector<Value> selfSizes = getTensorSizes(rewriter, loc, self);
    for (unsigned i = 0; i < selfSizes.size(); i++)
      selfSizes[i] = castIndexToInt64(rewriter, loc, selfSizes[i]);
    Value broadcastedSrc;
    if (failed(broadcastToGivenShape(op, rewriter, src, selfSizes,
                                     broadcastedSrc))) {
      return rewriter.notifyMatchFailure(
          op, "unable to perform broadcast operation");
    }

    AffineMap id = AffineMap::getMultiDimIdentityMap(selfType.getRank(),
                                                     rewriter.getContext());
    SmallVector<StringRef> iteratorTypes(selfType.getRank(),
                                         getParallelIteratorTypeName());
    Value result = rewriter
                       .create<linalg::GenericOp>(
                           loc,
                           /*resultType=*/selfType,
                           /*inputs=*/broadcastedSrc,
                           /*outputs=*/self,
                           /*indexingMaps=*/llvm::makeArrayRef({id, id}),
                           /*iteratorTypes=*/iteratorTypes,
                           [](OpBuilder &b, Location loc, ValueRange args) {
                             Value result = args[0];
                             if (args[0].getType() != args[1].getType()) {
                               result = convertScalarToDtype(b, loc, args[0],
                                                             args[1].getType());
                             }
                             b.create<linalg::YieldOp>(loc, result);
                           })
                       ->getResult(0);

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenFlattenUsingIntsOp>();
  patterns.add<ConvertAtenFlattenUsingIntsOp>(typeConverter, context);
  target.addIllegalOp<AtenViewOp>();
  patterns.add<ConvertAtenViewOp>(typeConverter, context);
  target.addIllegalOp<AtenSqueezeOp>();
  patterns.add<ConvertAtenSqueezeOp>(typeConverter, context);
  target.addIllegalOp<AtenSqueezeDimOp>();
  patterns.add<ConvertAtenSqueezeDimOp>(typeConverter, context);
  target.addIllegalOp<AtenUnsqueezeOp>();
  patterns.add<ConvertAtenUnsqueezeOp>(typeConverter, context);
  target.addIllegalOp<AtenTransposeIntOp>();
  patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);
  target.addIllegalOp<AtenPermuteOp>();
  patterns.add<ConvertAtenPermuteOp>(typeConverter, context);
  target.addIllegalOp<AtenSliceTensorOp>();
  patterns.add<ConvertAtenSliceTensorOp>(typeConverter, context);
  target.addIllegalOp<AtenCatOp>();
  patterns.add<ConvertAtenCatOp>(typeConverter, context);
  target.addIllegalOp<AtenBroadcastToOp>();
  patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);
  target.addIllegalOp<ValsemVariantAtenCopyOp>();
  patterns.add<ConvertValsemVariantAtenCopyOp>(typeConverter, context);
}
