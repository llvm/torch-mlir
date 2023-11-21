//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static int64_t productReduce(ArrayRef<int64_t> a) {
  return accumulate(a.begin(), a.end(), /*init=*/1, std::multiplies<int64_t>());
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
    if (!matchPattern(op.getStartDim(), m_TorchConstantInt(&startDim)))
      return rewriter.notifyMatchFailure(op, "start_dim must be constant");
    int64_t endDim;
    if (!matchPattern(op.getEndDim(), m_TorchConstantInt(&endDim)))
      return rewriter.notifyMatchFailure(op, "end_dim must be constant");
    auto type = adaptor.getSelf().getType().cast<RankedTensorType>();
    auto inputRank = type.getRank();
    if (inputRank == 1) {
      // If input rank is equal to 1, then there's no scope for flattening the
      // input tensor.
      rewriter.replaceOp(op, adaptor.getSelf());
      return success();
    }

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
          op, resultType, adaptor.getSelf(), reassociation);
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
        op->getLoc(), adaptor.getSelf(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                collapsedTensor);
    return success();
  }
};
} // namespace

namespace {

// Convert aten.view to a tensor.collapse_shape followed by a
// tensor.expand_shape, where possible. Cases where this is not possible:
//
// 1) when the resulting expand or collapse has multiple dynamic reassociation
// dimensions. This is because MLIR currently does not support this.
// 2) view's which are squeezes to or unsqueezes from a rank-0 tensor.
//
// In the cases above, the conversion fails.
class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
private:
  using OpConversionPattern::OpConversionPattern;

  // Try to fill in unknown dimensions in `to` using the statically known
  // dimensions in `from`.
  static void tryFillOneDirection(MutableArrayRef<int64_t> from,
                                  MutableArrayRef<int64_t> to) {

    auto fromProduct = productReduce(from);

    SmallVector<uint64_t> unknowns;
    for (auto [i, dim] : llvm::enumerate(to)) {
      if (dim == kUnknownSize)
        unknowns.push_back(i);
    }

    auto nUnknowns = unknowns.size();

    auto toProduct =
        productReduce(to) / std::pow<int64_t>(kUnknownSize, nUnknowns);

    if (fromProduct == toProduct) {
      for (auto i : unknowns) {
        to[i] = 1;
      }
    }

    else if (nUnknowns == 1) {
      to[unknowns[0]] = fromProduct / toProduct;
    }
  }

  // Calculate sizes of dynamic dimensions from static dimensions, where
  // possible, replacing dynamic dimemsions inplace.
  //
  // This function assumes that all the dimensions in `inShape` map to all the
  // dimensions in `outShape`.
  static void tryFillBothDirections(MutableArrayRef<int64_t> inShape,
                                    MutableArrayRef<int64_t> outShape) {

    int64_t inDynamicDimCount = llvm::count(inShape, kUnknownSize);
    int64_t outDynamicDimCount = llvm::count(outShape, kUnknownSize);

    if (inDynamicDimCount == 0 && outDynamicDimCount == 0)
      return;

    // Try and fill in the gaps in `outShape` using `inShape`.
    else if (inDynamicDimCount == 0)
      tryFillOneDirection(inShape, outShape);

    // Try and fill in the gaps in `inShape` using `outShape`.
    else if (outDynamicDimCount == 0)
      tryFillOneDirection(outShape, inShape);
  }

  // Get the shapes of the input and output tensors, making a best-effort
  // attempt to extract static shape information given the inputs to
  // `aten.view`. That is, try to have as few kUnknownSize entries as possible.
  static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
  getInputAndOutputShape(Value inTorchTensor,
                         const SmallVector<Value> &outSizeTorchInt) {

    SmallVector<int64_t> inShape(
        inTorchTensor.getType().cast<BaseTensorType>().getSizes());

    SmallVector<int64_t> outShape(outSizeTorchInt.size(), kUnknownSize);

    for (auto [outDim, outDimSize] : llvm::enumerate(outSizeTorchInt)) {

      int64_t vMatch;

      // if "outDimSize = inTorchTensor.shape[inDim]" then
      // set the integer value of the output dimension directly.
      if (matchPattern(outDimSize,
                       m_TorchTensorSizeInt(inTorchTensor, &vMatch))) {
        outShape[outDim] = inShape[vMatch];
      }

      // if "outputDimSize = result of constant operation" then
      // set the integer value of the output dimension directly.
      else if (matchPattern(outDimSize, m_TorchConstantInt(&vMatch))) {
        outShape[outDim] = vMatch;
      }
    }

    tryFillBothDirections(inShape, outShape);
    return std::make_pair(inShape, outShape);
  }

  LogicalResult
  matchAndRewrite(AtenViewOp viewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(viewOp, rewriter)))
      return failure();
    Location loc = viewOp.getLoc();

    const TypeConverter *typeConverter = getTypeConverter();

    Value input = adaptor.getSelf();
    auto inType = input.getType().cast<RankedTensorType>();
    SmallVector<Value> inSize = getTensorSizes(rewriter, loc, input);
    int64_t inRank = inType.getRank();

    auto outType =
        typeConverter->convertType(viewOp.getType()).cast<RankedTensorType>();
    int64_t outRank = outType.getRank();
    SmallVector<Value> outSizeTorchInt;
    if (!getListConstructElements(viewOp.getSize(), outSizeTorchInt)) {
      return rewriter.notifyMatchFailure(viewOp,
                                         "unimplemented: the target size is "
                                         "not constructed from ListConstruct");
    }
    SmallVector<Value> outSizeInt =
        getTypeConvertedValues(rewriter, loc, typeConverter, outSizeTorchInt);
    if (outRank != static_cast<int64_t>(outSizeInt.size())) {
      return rewriter.notifyMatchFailure(
          viewOp, "size list length mismatches with the result type rank");
    }

    if (inRank == 0 || outRank == 0) {
      return rewriter.notifyMatchFailure(
          viewOp, "It is impossible to expand from or collapse to a scalar");
    }

    auto [inShape, outShape] =
        getInputAndOutputShape(viewOp.getSelf(), outSizeTorchInt);

    DenseSet<std::tuple<int64_t, int64_t>> sameDynDims;
    for (auto [outDim, outDimSize] : llvm::enumerate(outSizeTorchInt)) {
      int64_t inDim;
      // Match torch.aten.size.int(inTensor, inDim) with constant inDim
      if (matchPattern(outDimSize,
                       m_TorchTensorSizeInt(viewOp.getSelf(), &inDim))) {
        sameDynDims.insert({inDim, static_cast<int64_t>(outDim)});
      }
    }

    // As association barrier is the edge between association groups for the
    // expand_shape and collapse_shape ops.
    struct AssociationBarrier {
      int64_t input;
      int64_t output;
    };

    // Consecutive AssociationBarriers define how dimensions of the input and
    // output get reassociated. For example AssociationBarrier({1,2}) followed
    // by AssociationBarrier({4,3}) means that indices in the range [1,4) of the
    // input being reassociated to indices [2,3) in the output.
    SmallVector<AssociationBarrier> barriers;

    // The algorithm for computing the barriers (and therefore the reassociation
    // groups) works as follows: Starting at index 0 in both the input and
    // output shapes, keep track of the cumulative product of dimensions as you
    // increment the index. If at some point the cumulative product of
    // dimensions is the same in the input and output shapes, then a barrier
    // has been met. If the cumulative product of the input shape is less than
    // that of the output shape, then increment the input index (similarly for
    // the output index). Bail when an unrecognized dynamic dimension is met.
    //
    // Here is a working example. Suppose input has shape [2,3,6] and output has
    // shape [6,3,2].
    //
    // Iteration 0:
    // fwdIn = 0, fwdOut = 0, inProd = 1, outProd = 1
    // ===> insert {fwdIn = 0, fwdOut = 0} into barriers (as inProd == outProd)
    // ===> increment fwdIn and fwdOut
    //
    // Iteration 1:
    // fwdIn = 1, fwdOut = 1, inProd = 2, outProd = 6
    // ====> increment fwdIn (as inProd < outProd)
    //
    // Iteration 2:
    // fwdIn = 2, fwdOut = 1, inProd = 6, outProd = 6
    // =====> insert {fwdIn = 2, fwdOut = 1} into barriers (as inProd ==
    // outProd)
    // =====> increment fwdIn and fwdOut
    //
    // Iteartion 3:
    // fwdIn = 3, fwdOut = 2, inProd = 36, outProd = 18
    // =====> increment fwdOut (as inProd > outProd)
    //
    // Iteration 4:
    // fwdIn = 3, fwdOut = 3, inProd = 36, outProd = 36
    // =====> insert {fwdIn = 3, fwdOut = 3} into barriers (as inProd ==
    // outProd)
    // =====> increment fwdIn and fwdOut
    //
    // Iteration 5:
    // Terminate as fwdIn == inRank and fwdOut == outRank
    //
    // Barriers = {{0,0}, {2,1}, {3,3}}
    //
    // Special attention is paid to dynamic dimensions: if a dynamic dimension
    // is encountered, the algorithm will bail, and try again from the end of
    // shapes, in reverse.
    //
    // An exception to this bailing rule is if 2 dynamic dimensions are
    // encountered in the input and output shapes at the same time, and they are
    // known to be the same -- in this case, the algorithm will continue. This
    // is the logic covered by the 'isSameDynDim' function.
    //
    // Temporary variables used to calculate the barriers.
    int64_t fwdIn{0};
    int64_t fwdOut{0};
    int64_t inProd{1};
    int64_t outProd{1};

    auto isSameDynDim = [&](int64_t inDim, int64_t outDim) {
      return sameDynDims.count({inDim, outDim}) != 0;
    };

    while (fwdIn < inRank && fwdOut < outRank && inProd > 0 && outProd > 0) {

      if (inProd < outProd) {
        inProd *= inShape[fwdIn++];
      } else if (outProd < inProd) {
        outProd *= outShape[fwdOut++];
      } else /* if products the same */ {
        barriers.push_back({fwdIn, fwdOut});
        if (!isSameDynDim(fwdIn, fwdOut)) {
          inProd *= inShape[fwdIn];
          outProd *= outShape[fwdOut];
        }
        ++fwdIn;
        ++fwdOut;
      }
    }

    auto cast = [&](Location loc, Type t, Value v) -> Value {
      return rewriter.createOrFold<tensor::CastOp>(loc, t, v);
    };

    // Same algorithm, but from the back of the shapes.
    auto nFwd = barriers.size();
    auto bwdIn = inRank;
    auto bwdOut = outRank;
    inProd = 1;
    outProd = 1;
    while (bwdIn >= fwdIn && bwdOut >= fwdOut && inProd > 0 && outProd > 0) {

      if (inProd < outProd) {
        inProd *= inShape[--bwdIn];
      } else if (outProd < inProd) {
        outProd *= outShape[--bwdOut];
      } else {
        barriers.push_back({bwdIn, bwdOut});
        --bwdIn;
        --bwdOut;
        if (!isSameDynDim(bwdIn, bwdOut)) {
          outProd *= outShape[bwdOut];
          inProd *= inShape[bwdIn];
        }
      }
    }
    std::reverse(barriers.begin() + nFwd, barriers.end());

    SmallVector<ReassociationIndices> inAssociations;
    SmallVector<ReassociationIndices> outAssociations;
    SmallVector<int64_t> collapsedShape;

    for (uint64_t i = 1; i < barriers.size(); ++i) {

      const auto b0 = barriers[i - 1];
      const auto b1 = barriers[i];

      auto inRange = llvm::seq<int64_t>(b0.input, b1.input);
      inAssociations.push_back({inRange.begin(), inRange.end()});

      // Update the intermediate (collapsed) shape with the product of the input
      // dimensions in the range [b0.input, b1.input), or kUnknownSize if any of
      // the dimensions in that range is kUnknownSize.
      ArrayRef<int64_t> inPart(inShape.begin() + b0.input,
                               inShape.begin() + b1.input);
      auto allKnown = std::all_of(inPart.begin(), inPart.end(),
                                  [](auto &&x) { return x != kUnknownSize; });
      collapsedShape.push_back(allKnown ? productReduce(inPart) : kUnknownSize);

      // The collapse_shape operation allows inputs with multiple dynamic
      // dimensions. The expand_shape operation does not allow outputs with
      // multiple dynamic dimensions.

      auto outRange = llvm::seq<int64_t>(b0.output, b1.output);
      outAssociations.push_back({outRange.begin(), outRange.end()});

      ArrayRef<int64_t> outPart(outShape.begin() + b0.output,
                                outShape.begin() + b1.output);
      if (std::count(outPart.begin(), outPart.end(), kUnknownSize) > 1) {
        return rewriter.notifyMatchFailure(
            viewOp, "unimplemented: more than one unknown dimension in the "
                    "collapse association.");
      }
    }

    auto elType = inType.getElementType();

    // update the types.
    //
    // 1) Convert between different conventions on what value denotes a dynamic
    // size (kUnknownSize vs ShapedType::kDynamic) for the input, collapsed, and
    // outputs.
    //
    // 2) use the refined shapes (some dynamic dimension values may have been
    // calculated).

    Type adjustedInType =
        RankedTensorType::get(makeShapeLLVMCompatible(inShape), elType);

    Type adjustedOutType =
        RankedTensorType::get(makeShapeLLVMCompatible(outShape), elType);

    Type collapsedType =
        RankedTensorType::get(makeShapeLLVMCompatible(collapsedShape), elType);


    Value current = cast(loc, adjustedInType, input);

    if (std::any_of(inAssociations.begin(), inAssociations.end(),
                    [](auto &&x) { return x.size() > 1; })) {
      current = rewriter.create<tensor::CollapseShapeOp>(
          loc, collapsedType, current, inAssociations);
    }

    if (std::any_of(outAssociations.begin(), outAssociations.end(),
                    [](auto &&x) { return x.size() > 1; })) {
      current = rewriter.create<tensor::ExpandShapeOp>(
          loc, adjustedOutType, current, outAssociations);
    }

    auto v = cast(loc, outType, current);
    rewriter.replaceOp(viewOp, v);
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
    Value input = adaptor.getSelf();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    const TypeConverter *typeConverter = getTypeConverter();
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

    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    int64_t j = -1;
    bool elideDynamicBroadcastDimCheck =
        isAssumingStrictSymbolicShapes(rewriter);
    for (auto i : llvm::seq<int64_t>(headOnesCount, inputRank)) {
      if (inputType.isDynamicDim(i)) {
        if (!elideDynamicBroadcastDimCheck) {
          // Make sure that size-1 dynamic dimension does not exist.
          Value dimSize = getDimOp(rewriter, loc, input, i);
          Value dimSizeNotOne = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ne, dimSize, one);
          rewriter.create<cf::AssertOp>(
              loc, dimSizeNotOne,
              rewriter.getStringAttr(
                  "unimplemented: size 1 dynamic dimension is not supported"));
        }
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
    Value input = adaptor.getSelf();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

    // TODO: Handle the case where the dim(th) dimension is dynamic.
    if (inputType.isDynamicDim(dim)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dim(th) dimension is not expected to be dynamic");
    }

    const TypeConverter *typeConverter = getTypeConverter();
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
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    auto inputRank =
        adaptor.getSelf().getType().cast<RankedTensorType>().getRank();
    dim = toPositiveDim(dim, inputRank + 1);
    if (!isValidDim(dim, inputRank + 1))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

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
        op, resultType, adaptor.getSelf(), reassociationMap);
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
    if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0)))
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    int64_t dim1;
    if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");

    auto inVector = adaptor.getSelf();
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
      outputDims.push_back(getDimOp(rewriter, loc, adaptor.getSelf(), i));
    std::swap(outputDims[dim0], outputDims[dim1]);

    Value outVector = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outputDims), elementType);
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
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
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
    if (!matchPattern(op.getDims(), m_TorchListOfConstantInts(dimensions)))
      return rewriter.notifyMatchFailure(op, "all dimensions must be constant");

    Value inVector = adaptor.getSelf();
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

    Value outVector = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outputDims), elementType);
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (unsigned i = 0; i < inputRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (unsigned i = 0; i < inputRank; i++)
      swapExprs.push_back(idExprs[dimensions[i]]);

    AffineMap inputMap =
        AffineMap::get(inputRank, /*symbolCount=*/0, idExprs, op->getContext());
    AffineMap outputMap = AffineMap::get(inputRank, /*symbolCount=*/0,
                                         swapExprs, op->getContext());
    SmallVector<AffineMap> indexingMaps{inputMap, outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputRank, utils::IteratorType::parallel);
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
    const TypeConverter *typeConverter = getTypeConverter();

    auto input = adaptor.getSelf();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();

    SmallVector<Value> resultShape;
    SmallVector<Value> offsets;
    SmallVector<Value> strides;
    if (failed(prepareArgumentsForSlicingOp<AtenSliceTensorOp,
                                            AtenSliceTensorOpAdaptor>(
            op, adaptor, rewriter, resultShape, offsets, strides))) {
      return failure();
    }

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
    const TypeConverter *typeConverter = getTypeConverter();

    // Collect all the tensors to be concatenated.
    auto tensorList = op.getTensors();
    SmallVector<Value> tensorsTorchType;
    if (!getListConstructElements(tensorList, tensorsTorchType))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    auto tensors =
        getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    auto outElemType = newResultType.getElementType();
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i] = torch_to_linalg::convertTensorToElementType(
          rewriter, loc, tensors[i], outElemType);
    }

    int rank = newResultType.getRank();
    Value dimValue = op.getDim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");
    dim = toPositiveDim(dim, rank);
    if (!isValidDim(dim, rank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

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
    for (auto tensor : ArrayRef(tensors).drop_front()) {
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

    Value result = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(sizes), newResultType.getElementType());
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

namespace {
class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value self = adaptor.getSelf();

    SmallVector<Value> inShape;
    if (!getListConstructElements(adaptor.getSize(), inShape)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the size list is not from list construct");
    }
    // For dynamic input dimension we need to use the `broadcastToShape`
    // which in this case is `inShapeConverted` because this shape will yield
    // us the dimension size of the output.
    SmallVector<bool> useBroadcastToShape;
    int64_t inputRank = self.getType().cast<RankedTensorType>().getRank();
    for (size_t i = inShape.size() - inputRank, e = inShape.size(); i < e;
         ++i) {
      int64_t dim;
      if (matchPattern(inShape[i], m_TorchConstantInt(&dim))) {
        if (dim < 0) {
          useBroadcastToShape.push_back(false);
        } else {
          useBroadcastToShape.push_back(true);
        }
      } else {
        // Note: Dynamic -1 (inferred) broadcast shapes are unimplemented.
        useBroadcastToShape.push_back(true);
      }
    }

    SmallVector<Value> inShapeConverted = getTypeConvertedValues(
        rewriter, op.getLoc(), getTypeConverter(), inShape);
    auto newResultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    Value result;
    if (failed(torch_to_linalg::broadcastToGivenShape(
            op, rewriter, self, inShapeConverted, newResultType, result,
            useBroadcastToShape))) {
      return rewriter.notifyMatchFailure(
          op, "unable to perform broadcast operation");
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenContiguousOp : public OpConversionPattern<AtenContiguousOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenContiguousOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getSelf());
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenCopyOp : public OpConversionPattern<AtenCopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value self = adaptor.getSelf();
    Value src = adaptor.getSrc();
    RankedTensorType selfType = self.getType().cast<RankedTensorType>();

    // The non_blocking should be a constant `False`.
    bool nonBlocking;
    if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
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
    if (failed(torch_to_linalg::broadcastToGivenShape(
            op, rewriter, src, selfSizes, selfType, broadcastedSrc))) {
      return rewriter.notifyMatchFailure(
          op, "unable to perform broadcast operation");
    }

    AffineMap id = AffineMap::getMultiDimIdentityMap(selfType.getRank(),
                                                     rewriter.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        selfType.getRank(), utils::IteratorType::parallel);
    Value result = rewriter
                       .create<linalg::GenericOp>(
                           loc,
                           /*resultType=*/selfType,
                           /*inputs=*/broadcastedSrc,
                           /*outputs=*/self,
                           /*indexingMaps=*/llvm::ArrayRef({id, id}),
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

namespace {
class ConvertAtenSliceScatterOp
    : public OpConversionPattern<AtenSliceScatterOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

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
};
} // namespace

namespace {
class ConvertAtenViewAsComplexOp
    : public OpConversionPattern<AtenViewAsComplexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewAsComplexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    MLIRContext *context = rewriter.getContext();

    auto input = adaptor.getSelf();

    RankedTensorType resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    auto elementType = resultType.getElementType();
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      auto currentDimSize = rewriter.create<tensor::DimOp>(loc, input, i);
      resultShape.push_back(currentDimSize);
    }

    Value outTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), elementType);

    SmallVector<AffineExpr> outputExpr;
    for (unsigned i = 0; i < resultType.getRank(); i++) {
      outputExpr.push_back(getAffineDimExpr(i, context));
    }

    Value constantZero =
        getConstant(rewriter, loc, 0, mlir::IndexType::get(context));
    Value constantOne =
        getConstant(rewriter, loc, 1, mlir::IndexType::get(context));

    AffineMap outputMap =
        AffineMap::get(resultType.getRank(), 0, outputExpr, op->getContext());

    SmallVector<AffineMap> indexingMaps{outputMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);
    auto complexVar =
        rewriter
            .create<linalg::GenericOp>(
                loc, outTensor.getType(), ValueRange{}, outTensor, indexingMaps,
                iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> indicesZero;
                  SmallVector<Value> indicesOne;

                  for (int i = 0; i < resultType.getRank(); i++) {
                    indicesZero.push_back(b.create<linalg::IndexOp>(loc, i));
                    indicesOne.push_back(b.create<linalg::IndexOp>(loc, i));
                  }

                  indicesZero.push_back(constantZero);
                  indicesOne.push_back(constantOne);

                  Value realVal =
                      b.create<tensor::ExtractOp>(loc, input, indicesZero);
                  Value imagVal =
                      b.create<tensor::ExtractOp>(loc, input, indicesOne);
                  Value complexVal = b.create<complex::CreateOp>(
                      loc, elementType, realVal, imagVal);
                  b.create<linalg::YieldOp>(loc, complexVal);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, complexVar);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenViewAsRealOp : public OpConversionPattern<AtenViewAsRealOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewAsRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();
    MLIRContext *context = rewriter.getContext();

    auto input = adaptor.getSelf();

    RankedTensorType resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    auto inputElementType = getElementTypeOrSelf(input.getType());
    if (!inputElementType.isa<ComplexType>()) {
      return op.emitError("only ComplexType is allowed as input type");
    }
    Type elementType = resultType.getElementType();

    // returned real tensor has a size increase, where the last dim has size 2
    SmallVector<OpFoldResult> resultShape =
        tensor::getMixedSizes(rewriter, loc, input);
    resultShape.push_back(
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, 2));

    Value outTensor =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elementType);

    SmallVector<AffineExpr> inputExpr;
    for (unsigned i = 0; i < resultType.getRank() - 1; i++) {
      inputExpr.push_back(getAffineDimExpr(i, context));
    }

    AffineMap inputMap =
        AffineMap::get(resultType.getRank(), 0, inputExpr, op->getContext());

    inputExpr.push_back(getAffineDimExpr(resultType.getRank() - 1, context));

    AffineMap outputMap =
        AffineMap::get(resultType.getRank(), 0, inputExpr, op->getContext());

    SmallVector<AffineMap> indexingMaps{inputMap, outputMap};

    SmallVector<utils::IteratorType> iteratorTypes(
        resultType.getRank(), utils::IteratorType::parallel);

    Value constantZero =
        getConstant(rewriter, loc, 0, mlir::IndexType::get(context));
    auto realVar =
        rewriter
            .create<linalg::GenericOp>(
                loc, outTensor.getType(), input, outTensor, indexingMaps,
                iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value realVal =
                      b.create<complex::ReOp>(loc, elementType, args[0]);
                  Value imagVal =
                      b.create<complex::ImOp>(loc, elementType, args[0]);
                  Value lastIndex =
                      b.create<linalg::IndexOp>(loc, inputType.getRank());
                  Value cmpResult = b.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, lastIndex, constantZero);
                  Value yieldValue = b.create<arith::SelectOp>(
                      loc, cmpResult, realVal, imagVal);

                  b.create<linalg::YieldOp>(loc, yieldValue);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, realVar);
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
  target.addIllegalOp<AtenContiguousOp>();
  patterns.add<ConvertAtenContiguousOp>(typeConverter, context);
  target.addIllegalOp<AtenCopyOp>();
  patterns.add<ConvertAtenCopyOp>(typeConverter, context);
  target.addIllegalOp<AtenSliceScatterOp>();
  patterns.add<ConvertAtenSliceScatterOp>(typeConverter, context);
  target.addIllegalOp<AtenViewAsComplexOp>();
  patterns.add<ConvertAtenViewAsComplexOp>(typeConverter, context);
  target.addIllegalOp<AtenViewAsRealOp>();
  patterns.add<ConvertAtenViewAsRealOp>(typeConverter, context);
}
