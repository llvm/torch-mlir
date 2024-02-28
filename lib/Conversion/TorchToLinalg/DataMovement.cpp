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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
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
  Value negone = rewriter.create<arith::ConstantIndexOp>(loc, -1);

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

  Value stepIndex = castIntToIndex(rewriter, loc, adaptor.getStep());
  Value start = toPositiveValidDim(rewriter, loc, torchTypeStart,
                                   builtinTypeStart, zero, dimSize);

  // We cannot use to positive valid dim as for negative strides we need to
  // clamp to `-1` so that the full tensor bounds are available:
  Value end = builtinTypeEnd;
  if (torchTypeEnd.getType().isa<Torch::NoneType>()) {
    end = dimSize;
  } else {
    end = castIntToIndex(rewriter, loc, end);
    Value endcmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, end, zero);
    Value endadd = rewriter.create<arith::AddIOp>(loc, end, dimSize);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, endadd, end);
    endcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, end,
                                            zero);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, negone, end);
    endcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, end,
                                            dimSize);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, dimSize, end);
  }

  // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
  resultShape = getTensorSizes(rewriter, loc, input);
  Value len = rewriter.create<arith::SubIOp>(loc, end, start);

  // We check the difference between start and end to determine the total size:
  Value stepcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 stepIndex, zero);
  Value stepsign = rewriter.create<arith::SelectOp>(loc, stepcmp, one, negone);
  Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
  resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, stepsign);
  resultSize = rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);

  // Clamp the size to [0, ...]:
  Value szcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               resultSize, zero);
  resultSize = rewriter.create<arith::SelectOp>(loc, szcmp, zero, resultSize);
  resultShape[dim] = resultSize;

  strides.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);

  offsets[dim] = start;
  strides[dim] = stepIndex;
  return success();
}

// Example:
// input =  tensor([[[0., 1., 2., 3.],
//                   [4., 5., 6., 7.]]])
// torch.ops.aten.reflection_pad1d(input, (3,1));
//                                 padding_left = 3,
//                                 padding_right = 1
// output = tensor([[[3., 2., 1., 0., 1., 2., 3., 2.],
//                   [7., 6., 5., 4., 5., 6., 7., 6.]]])
// Checks: 1) Each of padding_left and padding_right must be non-negative and
//            less than the size of the last dimension.
// Implementation: a) Construct a result tensor of
// shape of input tensor except for the last dimension.
//                    The last dimension of the result tensor should be last
//                    dimension of input tensor + left padding size + right
//                    padding size. Initialize result tensor to all zeros
//                 b) Setup affine map to take slice from input tensor of size
//                 left padding starting from
//                    second column onwards as first column is reflection
//                    boundary
//                 c) Reflect the affine map to have resultant slice reflected
//                 d) Take the slice and write from begining in result tensor
//                 e) write the original tensor next into result tensor
//                 f) Setup affine map to take slice from input tensor of right
//                 padding size ending
//                    at second last column as last column is reflection
//                    boundary for right padding
//                 g) Reflect the affine map to have resultant slice reflected
//                 h) Take the slice and write from left padding size + orignal
//                 tensor last dim size
//                    into result tensor
// Uses the ideas/code used for AtenReflectionPad2dOp
namespace {
class ConvertAtenReflectionPad1dOp
    : public OpConversionPattern<AtenReflectionPad1dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenReflectionPad1dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only constant int padding range is supported");

    MLIRContext *context = rewriter.getContext();
    Location loc = op.getLoc();

    // Lambda Unitility Functions
    // Create an Integer expression of x + y
    auto createIAdd = [&](Value x, Value y) {
      return rewriter.create<arith::AddIOp>(loc, x, y);
    };

    // Create an integer expression of x - y
    auto createISub = [&](Value x, Value y) {
      return rewriter.create<arith::SubIOp>(loc, x, y);
    };

    enum PadLocation { PAD_LEFT = 0, PAD_RIGHT = 1, PAD_CENTER = 2 };

    Value input = adaptor.getSelf();
    Type indexType = rewriter.getIndexType();
    Value zero = getConstant(rewriter, loc, 0, indexType);
    Value one = getConstant(rewriter, loc, 1, indexType);
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto outputType = llvm::cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    unsigned numDims = inputType.getRank();
    assert(numDims >= 2 && "Not enough input dimensions");
    int64_t lastDim = numDims - 1;
    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
    Value lastDimSize = inputShape[lastDim]; // input [1,2,4], then lastDim = 2,
                                             // inputShape[2] will give 4

    Value tileWidth[3], extractOffset[3], insertOffset[3];

    tileWidth[PAD_LEFT] =
        getConstant(rewriter, loc, padInts[PAD_LEFT], indexType);
    tileWidth[PAD_RIGHT] =
        getConstant(rewriter, loc, padInts[PAD_RIGHT], indexType);
    tileWidth[PAD_CENTER] = lastDimSize;

    extractOffset[PAD_LEFT] = one;
    // The offset for the right hand padding "bar" is:
    //   [right] lastDimSize - (tileWidth[PAD_RIGHT] + one)
    extractOffset[PAD_RIGHT] =
        createISub(lastDimSize, createIAdd(tileWidth[PAD_RIGHT], one));
    extractOffset[PAD_CENTER] = zero;

    insertOffset[PAD_LEFT] = zero;
    insertOffset[PAD_RIGHT] = createIAdd(lastDimSize, tileWidth[PAD_LEFT]);
    insertOffset[PAD_CENTER] = tileWidth[PAD_LEFT];

    SmallVector<Value> resultShape{inputShape};
    // Result's last dimension will have size:
    // lastDimSize + left padding size + right padding size
    resultShape[lastDim] =
        createIAdd(resultShape[lastDim],
                   createIAdd(tileWidth[PAD_LEFT], tileWidth[PAD_RIGHT]));
    Value resultTensor = createZeroInitTensor(rewriter, loc, resultShape,
                                              inputType.getElementType());

    // Helper to reflect/reverse the i-th dimension of an affine map without
    // symbols. This only works if applied on a tensor for which the
    // corresponding dimension has a statically known size
    auto reflectDim = [](AffineMap map, unsigned numDims, int64_t i,
                         int64_t size) {
      AffineExpr d = map.getResult(i);
      return map.replace(d, size - d - 1, numDims,
                         0); // left reflect for (3,1) on input shape (1,2,4).
                             // size = 3, lastDim=2, numDims=3
    };

    SmallVector<utils::IteratorType> iteratorTypes{
        numDims, utils::IteratorType::parallel};
    auto idMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    SmallVector<Value> allOneStrides(numDims, one);

    auto addTileToResult = [&](PadLocation padPosition) {
      // Create the tile by extracting a slice from the input tensor.
      SmallVector<Value> extractShape{inputShape};
      extractShape[lastDim] = tileWidth[padPosition];
      SmallVector<Value> extractOffsets(numDims, zero);
      extractOffsets[lastDim] = extractOffset[padPosition];
      Value tile = rewriter.create<tensor::ExtractSliceOp>(
          loc, input, extractOffsets, extractShape, allOneStrides);

      auto inputMap = AffineMap::getMultiDimIdentityMap(numDims, context);
      // Setup the affine map function to resverse the tile along the horizontal
      // for left and right slices
      if (padPosition < PAD_CENTER) {
        inputMap = reflectDim(inputMap, numDims, lastDim, padInts[padPosition]);
        // Take reflected slice as per inputMap
        tile = rewriter
                   .create<linalg::GenericOp>(
                       loc, llvm::cast<RankedTensorType>(tile.getType()), tile,
                       tile, ArrayRef({inputMap, idMap}), iteratorTypes,
                       [](OpBuilder &b, Location nestedLoc, ValueRange args) {
                         b.create<linalg::YieldOp>(nestedLoc, args[0]);
                       })
                   .getResult(0);
      }
      // Insert the tile in the resultTensor
      SmallVector<Value> insertOffsets(numDims, zero);
      insertOffsets[lastDim] = insertOffset[padPosition];
      resultTensor = rewriter.create<tensor::InsertSliceOp>(
          loc, tile, resultTensor, insertOffsets, extractShape, allOneStrides);
    };

    if (padInts[PAD_LEFT] > 0)
      addTileToResult(PAD_LEFT);
    if (padInts[PAD_RIGHT] > 0)
      addTileToResult(PAD_RIGHT);
    addTileToResult(PAD_CENTER);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType, resultTensor);
    return success();
  }
};
} // namespace

namespace {

// Lower the aten.reflection.pad_2d operator into a sequence of
// tensor.extract_slice, linalg.generic, and tensor_insert_slice
// operations.

// To understand the lowering, consider this pytorch example:
//
// >>> t = torch.tensor([[[1.0,2,3],[4,5,6], [7,8,9]]])
// >>> t
// tensor([[[1., 2., 3.],
//         [4., 5., 6.],
//         [7., 8., 9.]]])
// >>> torch.ops.aten.reflection_pad2d(t, [1,2,1,2])
// tensor([[[5., 4., 5., 6., 5., 4.],
//          [2., 1., 2., 3., 2., 1.],
//          [5., 4., 5., 6., 5., 4.],
//          [8., 7., 8., 9., 8., 7.],
//          [5., 4., 5., 6., 5., 4.],
//          [2., 1., 2., 3., 2., 1.]]])
//
// The result can be subdivided into "tiles" corresponding to either
// the input tensor (in the center) or slices of the input tensor
// whose width and height is determined by the padding sizes and which
// are reflected through the side of the central input tensor that
// they touch.
// In the example above, the tiles are:
// top left: [[5]]
// top center: [[4,5,6]]
// top right: [[5,4]]
// center left [[2,1],[5,4],[8,7]]
// center: copy of the input tensor
// center right: [[2,1],[5,4],[8,7]]
// bottom left: [[5,4],[2,1]]
// center bottom: [[2,3,2]]
// center right: [[2,1]]
//
// The lowering uses a tensor.extract_slice operation to create each tile,
// a linalg.generic for the reflection, and a tensor.insert_slice to
// insert the tile in the resulting tensor.
class ConvertAtenReflectionPad2dOp
    : public OpConversionPattern<AtenReflectionPad2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenReflectionPad2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    SmallVector<int64_t> padInts;
    if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");

    Location loc = op.getLoc();
    // Some generic helper functions for creating arithmetic operations.
    auto createAdd = [&](Value x, Value y) {
      return rewriter.create<arith::AddIOp>(loc, x, y);
    };

    auto createAdds = [&](std::initializer_list<Value> values) {
      assert(values.size() >= 2);
      return std::accumulate(values.begin() + 1, values.end(), data(values)[0],
                             createAdd);
    };

    auto createSub = [&](Value x, Value y) {
      return rewriter.create<arith::SubIOp>(loc, x, y);
    };

    auto createSubs = [&](std::initializer_list<Value> values) {
      assert(values.size() >= 2);
      return std::accumulate(values.begin() + 1, values.end(), data(values)[0],
                             createSub);
    };

    // Enums for specifying the coordinates of a tile.  An "h" prefix
    // is used to stand for "horizontal" and "v" for "vertical"
    // throughout.
    enum PadHLoc { LEFT = 0, RIGHT = 1, HCENTER = 2 };
    enum PadVLoc { TOP = 0, BOTTOM = 1, VCENTER = 2 };

    // Helper functions for obtaining information about the operator's
    // padding arguments.
    auto getHPadArgument = [&](PadHLoc l) {
      assert(l < HCENTER);
      return padInts[l];
    };

    auto getVPadArgument = [&](PadVLoc l) {
      assert(l < VCENTER);
      return padInts[2 + l];
    };

    auto shouldCreateTile = [&](PadVLoc v, PadHLoc h) {
      if (!(h == HCENTER || getHPadArgument(h) > 0))
        return false;
      if (!(v == VCENTER || getVPadArgument(v) > 0))
        return false;

      return true;
    };

    Value input = adaptor.getSelf();
    MLIRContext *context = rewriter.getContext();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto outputType = llvm::cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    unsigned numDims = inputType.getRank();

    assert(numDims >= 2 && "Not enough input dimensions");

    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
    int64_t hDim = numDims - 1;
    int64_t vDim = numDims - 2;
    Value hDimSize = inputShape[hDim];
    Value vDimSize = inputShape[vDim];

    assert(getHPadArgument(LEFT) < inputType.getShape()[hDim] &&
           "Left padding too large");
    assert(getHPadArgument(RIGHT) < inputType.getShape()[hDim] &&
           "Right padding too large");
    assert(getVPadArgument(TOP) < inputType.getShape()[vDim] &&
           "Top padding too large");
    assert(getVPadArgument(BOTTOM) < inputType.getShape()[vDim] &&
           "Bottom padding too large");

    Type indexType = rewriter.getIndexType();
    Value zero = getConstant(rewriter, loc, 0, indexType);
    Value one = getConstant(rewriter, loc, 1, indexType);

    Value tileWidth[3];
    tileWidth[HCENTER] = hDimSize;
    for (auto h : {LEFT, RIGHT})
      tileWidth[h] = getConstant(rewriter, loc, getHPadArgument(h), indexType);

    Value tileHeight[3];
    tileHeight[VCENTER] = vDimSize;
    for (auto v : {TOP, BOTTOM})
      tileHeight[v] = getConstant(rewriter, loc, getVPadArgument(v), indexType);

    // Helper to reflect/reverse the i-th dimension of an affine map
    // without symbols. This only works if applied on a tensor
    // for which the corresponding dimension has a statically
    // known size which is good enough since we only apply
    // it to reflect the padding slices.
    auto reflectDim = [](AffineMap map, unsigned numDims, int64_t i,
                         int64_t size) {
      AffineExpr d = map.getResult(i);
      return map.replace(d, size - d - 1, numDims, 0);
    };

    // Create output shape and tensor
    SmallVector<Value> resultShape{inputShape};
    resultShape[vDim] =
        createAdds({resultShape[vDim], tileHeight[TOP], tileHeight[BOTTOM]});
    resultShape[hDim] =
        createAdds({resultShape[hDim], tileWidth[LEFT], tileWidth[RIGHT]});

    Value resultTensor = createZeroInitTensor(rewriter, loc, resultShape,
                                              inputType.getElementType());

    // Construction of the tiles

    // Example: central left tile
    //
    // Let m the width of the left padding as returned by getHPadargument(LEFT)
    // and n the size of the input tensor's "horizontal" dimension, i.e.
    // hDimSize. Assume that the subtensor of the input tensor in the relevant
    // (i.e. last two) dimensions is:
    //
    //     x_1,1 x_1,2 ... x_1,m
    //     x_2,1 x_2,2 ... x_2,m
    //              .
    //              .
    //              .
    //     x_n,1 x_n,2 ... x_n,m
    //
    // The padding tile consists of the columns 2, ..., m + 1
    // of the input in reverse order. The first column gets
    // skipped because this is the column through which the
    // reflection happens.
    //
    //      x_1,m x_1,m-1 ... x_1,2
    //      x_2,m x_1,m-1 ... x_2,2
    //              .
    //              .
    //              .
    //      x_n,m x_n,m-1 ... x_n,2
    //
    // The tile will be inserted to the left of the copy of the input tensor
    // in the output tensor, i.e. with horizontal offset 0.
    // The top padding determines the vertical offset.

    // Tiles on the diagonal (e.g. (TOP, LEFT)) are reflected through
    // two sides, i.e. their columns and rows must be reversed.

    // Setup information about the tiles

    // Compute the offsets for extracting the slice from the
    // input. We need to skip the row or column through which
    // the tile should be reflected, if any (none for the center tile).
    Value extractHOffset[3];
    extractHOffset[LEFT] = one;
    extractHOffset[HCENTER] = zero;
    extractHOffset[RIGHT] = createSubs({hDimSize, tileWidth[RIGHT], one});

    Value extractVOffset[3];
    extractVOffset[TOP] = one;
    extractVOffset[VCENTER] = zero;
    extractVOffset[BOTTOM] = createSubs({vDimSize, tileHeight[BOTTOM], one});

    // Compute the horizontal and vertical offsets for inserting
    // the tiles in the resultTensor.
    Value insertHOffset[3];
    insertHOffset[LEFT] = zero;
    insertHOffset[HCENTER] = tileWidth[LEFT];
    insertHOffset[RIGHT] = createAdd(hDimSize, tileWidth[LEFT]);

    Value insertVOffset[3];
    insertVOffset[TOP] = zero;
    insertVOffset[VCENTER] = tileHeight[TOP];
    insertVOffset[BOTTOM] = createAdd(vDimSize, tileHeight[TOP]);

    auto shouldHReflect = [](PadHLoc l) { return l == LEFT || l == RIGHT; };
    auto shouldVReflect = [](PadVLoc l) { return l == TOP || l == BOTTOM; };

    SmallVector<utils::IteratorType> iteratorTypes{
        numDims, utils::IteratorType::parallel};
    auto idMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    SmallVector<Value> allOneStrides(numDims, one);

    auto createTile = [&](PadVLoc verticalPos, PadHLoc horizontalPos) {
      // Create the tile by extracting a slice from the input tenor.
      SmallVector<Value> extractShape{inputShape};
      extractShape[hDim] = tileWidth[horizontalPos];
      extractShape[vDim] = tileHeight[verticalPos];

      SmallVector<Value> extractOffsets(numDims, zero);
      extractOffsets[hDim] = extractHOffset[horizontalPos];
      extractOffsets[vDim] = extractVOffset[verticalPos];

      Value tile = rewriter.create<tensor::ExtractSliceOp>(
          loc, input, extractOffsets, extractShape, allOneStrides);

      // Reverse the tile along the horizontal, vertical, or both
      // dimensions.
      auto inputMap = AffineMap::getMultiDimIdentityMap(numDims, context);
      if (shouldHReflect(horizontalPos)) {
        inputMap =
            reflectDim(inputMap, numDims, hDim, getHPadArgument(horizontalPos));
      }
      if (shouldVReflect(verticalPos)) {
        inputMap =
            reflectDim(inputMap, numDims, vDim, getVPadArgument(verticalPos));
      }

      tile = rewriter
                 .create<linalg::GenericOp>(
                     loc, llvm::cast<RankedTensorType>(tile.getType()), tile,
                     tile, ArrayRef({inputMap, idMap}), iteratorTypes,
                     [](OpBuilder &b, Location nestedLoc, ValueRange args) {
                       b.create<linalg::YieldOp>(nestedLoc, args[0]);
                     })
                 .getResult(0);

      // Insert the tile in the resultTensor.
      SmallVector<Value> insertOffsets(numDims, zero);
      insertOffsets[hDim] = insertHOffset[horizontalPos];
      insertOffsets[vDim] = insertVOffset[verticalPos];

      resultTensor = rewriter.create<tensor::InsertSliceOp>(
          loc, tile, resultTensor, insertOffsets, extractShape, allOneStrides);
    };

    for (auto v : {TOP, BOTTOM, VCENTER})
      for (auto h : {LEFT, RIGHT, HCENTER})
        if (shouldCreateTile(v, h))
          createTile(v, h);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType, resultTensor);

    return success();
  }
};
} // namespace

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
/// The `ConvertAtenViewOp` conversion pattern converts `aten.View` op to
/// one `linalg.TensorExpandShape` op for all expanded dimensions and one
/// `linalg.TensorCollapseShape` op for all collapsed dimensions. Cases where
/// there is neither an expand or collapse of dimensions (e.g. [2, 3] -> [3, 2])
/// is not handled. Additionally, certain dynamic dimension cases rely on naive
/// assumptions or aren't supported.
/// TODO: Handle all the other cases of `aten.View` op.
class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  // If one of the two dims arrays has size 1, a mapping is created from the one
  // dimension of the size-1 array to all the dimensions of the other array. For
  // example for inputs: xDims = [6], yDims = [2, 3] the result in the indices
  // arrays will be: xIndices = [0], yIndices = [0, 1].
  //
  // An error is returned if the dimension size of the size-1 array is not equal
  // to the product of all the dimension sizes in the other array, or if neither
  // of the arrays is size-1.
  static LogicalResult mapAllDimsToSingleDim(ArrayRef<int64_t> xDims,
                                             ArrayRef<int64_t> yDims,
                                             SmallVector<int64_t> &xIndices,
                                             SmallVector<int64_t> &yIndices) {
    if (xDims.empty() || yDims.empty())
      return failure();

    auto isValidReduction = [](int64_t expectedReductionProduct,
                               ArrayRef<int64_t> arrayToReduce) -> bool {
      if (llvm::count(arrayToReduce, kUnknownSize) > 0 ||
          expectedReductionProduct == kUnknownSize)
        return true;
      return productReduce(arrayToReduce) == expectedReductionProduct;
    };

    if (xDims.size() == 1) {
      if (!isValidReduction(xDims[0], yDims))
        return failure();
      xIndices.assign({0});
      yIndices.assign(llvm::to_vector(llvm::seq<int64_t>(0, yDims.size())));
      return success();
    } else if (yDims.size() == 1) {
      if (!isValidReduction(yDims[0], xDims))
        return failure();
      yIndices.assign({0});
      xIndices.assign(llvm::to_vector(llvm::seq<int64_t>(0, xDims.size())));
      return success();
    }
    return failure();
  }

  // Starting from the beginning of the dims arrays, this helper finds the
  // smallest set of consecutive dims in each array such that the product of the
  // dim sizes in the two subsets is equal. The indices arrays are populated
  // with the indices of the dims arrays that correspond to the subsets found.
  //
  // An error is returned if two subsets of dims with total number of elements
  // equal to each other is not found.
  static LogicalResult mapStaticallyKnownDims(ArrayRef<int64_t> xDims,
                                              ArrayRef<int64_t> yDims,
                                              SmallVector<int64_t> &xIndices,
                                              SmallVector<int64_t> &yIndices) {
    if (xDims.empty() || yDims.empty())
      return failure();
    int64_t xTotalSize = xDims[0];
    int64_t yTotalSize = yDims[0];
    SmallVector<int64_t> xIndicesResult({0});
    SmallVector<int64_t> yIndicesResult({0});
    size_t nextXIndex = 1;
    size_t nextYIndex = 1;
    while (xTotalSize != yTotalSize) {
      if (xTotalSize < yTotalSize) {
        if (nextXIndex == xDims.size() || xDims[nextXIndex] == kUnknownSize)
          return failure();
        xTotalSize *= xDims[nextXIndex];
        xIndicesResult.push_back(nextXIndex++);
      } else {
        if (nextYIndex == yDims.size() || yDims[nextYIndex] == kUnknownSize)
          return failure();
        yTotalSize *= yDims[nextYIndex];
        yIndicesResult.push_back(nextYIndex++);
      }
    }

    xIndices.assign(std::move(xIndicesResult));
    yIndices.assign(std::move(yIndicesResult));
    return success();
  }

  // Calculates the size of a dynamic dimension if all other dimensions are
  // statically known, and rewrites that dynamic dimension with the static size.
  //
  // Note: this function assumes that all the dimensions in `inputShape` map to
  // all the dimensions in `outputShape`.
  static void calculateSingleDynamicSize(MutableArrayRef<int64_t> inputShape,
                                         MutableArrayRef<int64_t> outputShape) {
    if (inputShape.empty() || outputShape.empty())
      return;
    int64_t inputDynamicDimCount = llvm::count(inputShape, kUnknownSize);
    int64_t outputDynamicDimCount = llvm::count(outputShape, kUnknownSize);
    if (inputDynamicDimCount + outputDynamicDimCount != 1)
      return;

    int64_t inputProduct = productReduce(inputShape);
    int64_t outputProduct = productReduce(outputShape);

    if (inputDynamicDimCount == 1) {
      inputProduct /= kUnknownSize;
      *llvm::find(inputShape, kUnknownSize) = outputProduct / inputProduct;
    } else {
      outputProduct /= kUnknownSize;
      *llvm::find(outputShape, kUnknownSize) = inputProduct / outputProduct;
    }
  }

  // Gets the shapes of the input and output tensors, making a best-effort
  // attempt to extract static shape information given the inputs to
  // `aten.view`.
  static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
  getInputAndOutputShape(Value inputTorchTensor,
                         SmallVector<Value> outputSizeTorchInt) {
    SmallVector<int64_t> inputShape(
        inputTorchTensor.getType().cast<BaseTensorType>().getSizes());
    SmallVector<int64_t> outputShape(outputSizeTorchInt.size(), kUnknownSize);
    for (auto [outputDim, outputDimSize] :
         llvm::enumerate(outputSizeTorchInt)) {
      int64_t inputDim;
      int64_t outputDimSizeInt;
      // Match torch.aten.size.int(inputTensor, inputDim) with constant inputDim
      if (matchPattern(outputDimSize,
                       m_TorchTensorSizeInt(inputTorchTensor, &inputDim))) {
        outputShape[outputDim] = inputShape[inputDim];
      } else if (matchPattern(outputDimSize,
                              m_TorchConstantInt(&outputDimSizeInt))) {
        if (outputDimSizeInt != -1) {
          outputShape[outputDim] = outputDimSizeInt;
        }
      }
    }

    calculateSingleDynamicSize(inputShape, outputShape);
    return std::make_pair(inputShape, outputShape);
  }

  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto inputType = input.getType().cast<RankedTensorType>();
    SmallVector<Value> inputSize = getTensorSizes(rewriter, loc, input);
    int64_t inputRank = inputType.getRank();
    const TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();
    if (resultRank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "result shape of rank 0 is invalid");

    if (inputRank == 0) {
      Value expanded =
          rewriter
              .create<tensor::ExpandShapeOp>(loc, resultType, input,
                                             ArrayRef<ReassociationIndices>())
              .getResult();
      rewriter.replaceOp(op, expanded);
      return success();
    }

    // Extract the desired output size as a list of integers. This list should
    // have been created using the operation `torch.prim.ListConstruct`.
    SmallVector<Value> outputSizeTorchInt;
    if (!getListConstructElements(op.getSize(), outputSizeTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: the target size is "
                                         "not constructed from ListConstruct");
    }
    if (llvm::count_if(outputSizeTorchInt, [](Value size) -> bool {
          int64_t sizeInt;
          if (matchPattern(size, m_TorchConstantInt(&sizeInt)))
            return sizeInt == -1;
          return false;
        }) > 1) {
      return rewriter.notifyMatchFailure(
          op, "at most one element in size list is allowed to be -1");
    }
    SmallVector<Value> outputSizeInt = getTypeConvertedValues(
        rewriter, loc, typeConverter, outputSizeTorchInt);
    if (resultRank != (int64_t)outputSizeInt.size()) {
      return rewriter.notifyMatchFailure(
          op, "desired size list length mismatches with the result type rank");
    }

    auto [inputShape, outputShape] =
        getInputAndOutputShape(op.getSelf(), outputSizeTorchInt);

    // Currently, we only handle the cases where each dimension is either
    // being expanded or collapsed. We do not handle cases where it's neither
    // collapsing nor expanding like view of [2,3] for 3x2 tensor.
    // TODO: For neither collapsing nor expanding, we could find a intermediate
    // shape to collapse and then expanded to the target shape. Like [2,3] =>
    // [6] => [3, 2].

    // Iterate through the view op size list to do the following:
    //   Mark dims in unchangedDims for size list items where the output dim
    // size comes from a `torch.aten.size.int(inputTensor, inputDim)`. We
    // naively assume this means the corresponding dimension is not expanded or
    // collapsed. Note this may technically not always be true.
    // TODO: think of a way better way to at least detect when this assumption
    // is violated for the cases of dynamic dimensions.
    bool inputHasOneDynDim = llvm::count(inputShape, kUnknownSize) == 1;
    bool outputHasOneDynDim = llvm::count(outputShape, kUnknownSize) == 1;
    bool singleDynDimsAreEqual =
        inputHasOneDynDim && outputHasOneDynDim &&
        productReduce(inputShape) == productReduce(outputShape);
    SmallVector<std::pair<int64_t, int64_t>> unchangedDims;
    for (auto [outputDim, outputDimSize] :
         llvm::enumerate(outputSizeTorchInt)) {
      int64_t inputDim;
      // Match torch.aten.size.int(inputTensor, inputDim) with constant inputDim
      if (matchPattern(outputDimSize,
                       m_TorchTensorSizeInt(op.getSelf(), &inputDim))) {
        unchangedDims.push_back(std::make_pair(inputDim, outputDim));
      } else if (singleDynDimsAreEqual &&
                 outputShape[outputDim] == kUnknownSize) {
        // If the input and output have a single dynamic dimension and the
        // product of the other dimensions is the same, then we know that the
        // dynamic dimension is unchanged.
        inputDim = std::distance(inputShape.begin(),
                                 llvm::find(inputShape, kUnknownSize));
        unchangedDims.push_back(std::make_pair(inputDim, outputDim));
      }
    }
    // Mark the end of the input/output shapes
    unchangedDims.push_back(std::make_pair(inputRank, resultRank));

    // Association indices for expand/collapse ops. These two vectors
    // are populated such that two entries at the same index corresponds
    // to an expand or collapse. For example,
    //
    // inputAssociations:  [[0, 1], [2]]
    // outputAssociations: [[0],    [1, 2, 3]]
    //
    // indicates that the first two dims of the input tensor
    // are collapsed into the first dim of the output, and the
    // third dim of the input is expanded into the last three dims
    // of the output.
    SmallVector<ReassociationIndices> inputAssociations;
    SmallVector<ReassociationIndices> outputAssociations;

    // The for loop does the following:
    // 1. Attempt to match the indices from inputDim and outputDim to the next
    // boundary found from `torch.aten.size.int(inputTensor, inputDim)`, or
    // until (inputRank, resultRank) if there is no such op. Look at the first
    // dimension of the input and output and collapse the larger one by finding
    // a minimal set of opposing indices with the same number of elements. If
    // the number of dims to the next boundary is 1, then we assume all
    // remaining opposing dims must collapse into it.
    // 2. For handling of dynamic dimensions, we first assume they are only
    // split if we can easily compute the correct size.
    //      e.g. [2, -1] -> [2, 3, 4]
    // This mainly happens at the edges of boundaries. Otherwise we try to match
    // the dynamic dimension with the one across from it and give up if we can't
    // reason about how the dimensions are associated.
    //      e.g. [-1, -1] -> [2, 3, 4]
    // For more information, see description of helper functions used in the
    // `if-else` cases inside the while loop.
    int64_t inputDim = 0, outputDim = 0;
    for (auto [nextUnchangedInput, nextUnchangedOutput] : unchangedDims) {
      // Used for ensuring that we don't have an ambiguous expansion
      bool assumedDynamicDimNotSplit = false;
      while (inputDim < nextUnchangedInput && outputDim < nextUnchangedOutput) {
        auto inputShapeSlice =
            MutableArrayRef<int64_t>(inputShape)
                .slice(inputDim, nextUnchangedInput - inputDim);
        auto outputShapeSlice =
            MutableArrayRef<int64_t>(outputShape)
                .slice(outputDim, nextUnchangedOutput - outputDim);
        SmallVector<int64_t> inputSliceIndices;
        SmallVector<int64_t> outputSliceIndices;

        // TODO: this can be removed by replacing it with a checkDimEqualHelper
        // that takes into account the product of all the dimensions being
        // reduced
        if (assumedDynamicDimNotSplit && inputShapeSlice.size() == 1 &&
            outputShapeSlice.size() != 1 &&
            inputShapeSlice[0] == kUnknownSize) {
          return rewriter.notifyMatchFailure(
              op, "found ambiguous expand of dynamic input sizes "
                  "(e.g. [-1, -1] -> [-1, -1, -1])");
        }

        if (succeeded(mapAllDimsToSingleDim(inputShapeSlice, outputShapeSlice,
                                            inputSliceIndices,
                                            outputSliceIndices))) {
          calculateSingleDynamicSize(inputShapeSlice, outputShapeSlice);
          // Update shape to pass the tensor.expand_shape and
          // tensor.collapse_shape verifiers. If one of the dimensions of the
          // tensor being flattened is dynamic, the size of the flattened tensor
          // must also be dynamic.
          if (inputShapeSlice.size() == 1 &&
              llvm::count(outputShapeSlice, kUnknownSize) > 0) {
            inputShapeSlice[0] = kUnknownSize;
          } else if (outputShapeSlice.size() == 1 &&
                     llvm::count(inputShapeSlice, kUnknownSize) > 0) {
            outputShapeSlice[0] = kUnknownSize;
          }
        } else if (succeeded(mapStaticallyKnownDims(
                       inputShapeSlice, outputShapeSlice, inputSliceIndices,
                       outputSliceIndices))) {
          /// `mapStaticallyKnownDims` maps the smallest number of
          /// input and output dimensions in the slice statically
          /// known to have the same number of elements.
        } else if (inputShapeSlice[0] == kUnknownSize) {
          // If the input is dynamic, assume it is not split
          checkDimEqualHelper(rewriter, loc, inputSize[inputDim],
                              outputSizeInt[outputDim]);
          // If output dimension is not dynamic, improve static information of
          // input
          inputShape[inputDim] = outputShape[outputDim];
          inputSliceIndices.push_back(0);
          outputSliceIndices.push_back(0);
          assumedDynamicDimNotSplit = true;
        } else {
          return rewriter.notifyMatchFailure(
              op, "unimplemented: found unhandled case of expansion/collapse "
                  "in `aten.view`");
        }

        inputAssociations.emplace_back();
        outputAssociations.emplace_back();
        for (int64_t inputSliceIndex : inputSliceIndices)
          inputAssociations.back().push_back(inputSliceIndex + inputDim);
        for (int64_t outputSliceIndex : outputSliceIndices)
          outputAssociations.back().push_back(outputSliceIndex + outputDim);
        inputDim = inputAssociations.back().back() + 1;
        outputDim = outputAssociations.back().back() + 1;
      }

      // Handle any leading or trailing size-1 dimensions and append the
      // associations for the dims matching `aten.size.int`.
      if (nextUnchangedInput != inputRank) {
        assert(nextUnchangedOutput != resultRank &&
               "`nextUnchangedInput` and `nextUnchangedOutput` should equal "
               "the respective input and output rank at the same time");
        inputAssociations.emplace_back();
        outputAssociations.emplace_back();
      }
      while (inputDim <= nextUnchangedInput && inputDim < inputRank) {
        if (inputDim != nextUnchangedInput && inputShape[inputDim] != 1) {
          return rewriter.notifyMatchFailure(
              op, "unimplemented: only collapsing of static size-1 into "
                  "unchanged dim supported");
        }
        inputAssociations.back().push_back(inputDim++);
      }
      while (outputDim <= nextUnchangedOutput && outputDim < resultRank) {
        if (outputDim != nextUnchangedOutput && outputShape[outputDim] != 1) {
          return rewriter.notifyMatchFailure(
              op, "unimplemented: only expanding of static size-1 out of "
                  "unchanged dim supported");
        }
        outputAssociations.back().push_back(outputDim++);
      }
    }

    auto cast = [&](Location loc, Type t, Value v) -> Value {
      return rewriter.createOrFold<tensor::CastOp>(loc, t, v);
    };

    // Check if the shapes already match up to dynamic sizes. If so, we can just
    // cast as the result type because the previous loop sets up the necessary
    // dim checks in case of dynamic sizes.
    if (llvm::all_of(
            inputAssociations,
            [](ReassociationIndices indices) { return indices.size() == 1; }) &&
        llvm::all_of(outputAssociations, [](ReassociationIndices indices) {
          return indices.size() == 1;
        })) {

      auto castResult = cast(loc, resultType, input);
      rewriter.replaceOp(op, castResult);

      return success();
    }

    // TODO: audit possibility of sparsity on these tensors
    Type adjustedResultType = RankedTensorType::get(
        makeShapeLLVMCompatible(outputShape), resultType.getElementType());
    Type adjustedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(inputShape), resultType.getElementType());
    Value castedInput = cast(loc, adjustedInputType, input);
    std::optional<Value> expandedInput;
    std::optional<Value> collapsedInput;

    if (llvm::any_of(inputAssociations, [](ReassociationIndices indices) {
          return indices.size() > 1;
        })) {

      SmallVector<int64_t> intermediateShape;
      for (auto i : llvm::seq(0, (int)outputAssociations.size())) {
        int sum = 1;

        for (auto j : llvm::seq(0, (int)outputAssociations[i].size())) {
          if (outputShape[outputAssociations[i][j]] < 0) {
            sum = kUnknownSize;
            break;
          }
          sum *= outputShape[outputAssociations[i][j]];
        }

        intermediateShape.push_back(sum);
      }

      // TODO: audit possibility of sparsity on these tensor
      Type intermediateResultType =
          RankedTensorType::get(makeShapeLLVMCompatible(intermediateShape),
                                resultType.getElementType());

      expandedInput =
          rewriter
              .create<tensor::CollapseShapeOp>(loc, intermediateResultType,
                                               castedInput, inputAssociations)
              .getResult();
    }

    if (llvm::any_of(outputAssociations, [](ReassociationIndices indices) {
          return indices.size() > 1;
        })) {

      collapsedInput = rewriter
                           .create<tensor::ExpandShapeOp>(
                               loc, adjustedResultType,
                               expandedInput.has_value() ? expandedInput.value()
                                                         : castedInput,
                               outputAssociations)
                           .getResult();
    }

    Value result = collapsedInput.has_value() ? collapsedInput.value()
                                              : expandedInput.value();

    auto castResult = cast(loc, resultType, result);
    rewriter.replaceOp(op, castResult);

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
    int rank = newResultType.getRank();
    Value dimValue = op.getDim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");
    dim = toPositiveDim(dim, rank);
    if (!isValidDim(dim, rank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

    auto outElemType = newResultType.getElementType();
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto inputType = cast<RankedTensorType>(tensors[i].getType());
      if (inputType.getElementType() != outElemType) {
        tensors[i] = torch_to_linalg::convertTensorToElementType(
            rewriter, loc, tensors[i], outElemType);
      }
    }

    llvm::SmallVector<Value> filteredTensors;
    for (auto tensor : tensors) {
      auto inputType = cast<RankedTensorType>(tensor.getType());
      if (inputType.getDimSize(dim) != 0) {
        filteredTensors.push_back(tensor);
      }
    }

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, newResultType, dim,
                                                  filteredTensors);
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
    // TODO: audit possibility of sparsity on these tensor
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

namespace {
class ConvertAtenDiagonalOp : public OpConversionPattern<AtenDiagonalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDiagonalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    int64_t offset;
    if (!matchPattern(op.getOffset(), m_TorchConstantInt(&offset)))
      return rewriter.notifyMatchFailure(op, "offset must be constant");
    int64_t dim1;
    if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");
    int64_t dim2;
    if (!matchPattern(op.getDim2(), m_TorchConstantInt(&dim2)))
      return rewriter.notifyMatchFailure(op, "dim2 must be constant");

    Value inputMatrix = adaptor.getSelf();
    RankedTensorType inputType = inputMatrix.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();

    if (inputRank < 2)
      return rewriter.notifyMatchFailure(
          op, "input must have at least two dimensions");
    int64_t outputRank = inputRank - 1;

    dim1 = toPositiveDim(dim1, inputRank);
    if (!isValidDim(dim1, inputRank))
      return rewriter.notifyMatchFailure(op, "dim1 out of range");
    dim2 = toPositiveDim(dim2, inputRank);
    if (!isValidDim(dim2, inputRank))
      return rewriter.notifyMatchFailure(op, "dim2 out of range");
    if (dim1 == dim2)
      return rewriter.notifyMatchFailure(
          op, "diagonal dimensions cannot be identical");

    Type elementType = inputType.getElementType();
    RankedTensorType outputType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Location loc = op.getLoc();

    Value dim1Size, dim2Size;
    dim1Size = getDimOp(rewriter, loc, inputMatrix, dim1);
    dim2Size = getDimOp(rewriter, loc, inputMatrix, dim2);

    // compute the length of the diagonal with possible offset
    // if the offset is very large or very small, diagSize=0 and an empty tensor
    // is returned
    Value indexZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value indexMinusOne = rewriter.create<arith::ConstantIndexOp>(loc, -1);
    Value indexOffset = rewriter.create<arith::ConstantIndexOp>(loc, offset);
    Value offsetIsNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, indexOffset, indexZero);
    Value sizeForNegativeOffset = rewriter.create<arith::MaxSIOp>(
        loc,
        rewriter.create<arith::MinSIOp>(
            loc, rewriter.create<arith::AddIOp>(loc, dim1Size, indexOffset),
            dim2Size),
        indexZero);
    Value sizeForPositiveOffset = rewriter.create<arith::MaxSIOp>(
        loc,
        rewriter.create<arith::MinSIOp>(
            loc, rewriter.create<arith::SubIOp>(loc, dim2Size, indexOffset),
            dim1Size),
        indexZero);
    Value diagSize = rewriter.create<arith::SelectOp>(
        loc, offsetIsNegative, sizeForNegativeOffset, sizeForPositiveOffset);

    // depending on its sign, the offset affects only the row or column indices
    // of the diagonal
    Value diagStart1 = rewriter.create<arith::SelectOp>(
        loc, offsetIsNegative,
        rewriter.create<arith::MulIOp>(loc, indexOffset, indexMinusOne),
        indexZero);
    Value diagStart2 = rewriter.create<arith::SelectOp>(loc, offsetIsNegative,
                                                        indexZero, indexOffset);

    SmallVector<Value> outputDims;
    for (auto i = 0; i < inputRank; i++) {
      if (!(i == dim1 || i == dim2))
        outputDims.push_back(getDimOp(rewriter, loc, inputMatrix, i));
    }
    outputDims.push_back(diagSize);

    Value outputMatrix = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(outputDims), elementType);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::getMultiDimIdentityMap(outputRank, rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputRank, utils::IteratorType::parallel);

    auto diagonal =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputMatrix.getType(), ValueRange{}, outputMatrix,
                indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> diagIndices;
                  Value indexOnDiag =
                      b.create<linalg::IndexOp>(loc, outputRank - 1);
                  Value dim1Index =
                      b.create<arith::AddIOp>(loc, indexOnDiag, diagStart1);
                  Value dim2Index =
                      b.create<arith::AddIOp>(loc, indexOnDiag, diagStart2);

                  // specify at which input indices the diagonal values are
                  // extracted
                  for (int indIn = 0, indOut = 0; indIn < inputRank; indIn++) {
                    if (indIn == dim1)
                      diagIndices.push_back(dim1Index);
                    else if (indIn == dim2)
                      diagIndices.push_back(dim2Index);
                    else {
                      diagIndices.push_back(
                          b.create<linalg::IndexOp>(loc, indOut));
                      indOut++;
                    }
                  }
                  Value diagElt = b.create<tensor::ExtractOp>(
                      loc, elementType, inputMatrix, diagIndices);
                  b.create<linalg::YieldOp>(loc, diagElt);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType, diagonal);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenReflectionPad1dOp>();
  patterns.add<ConvertAtenReflectionPad1dOp>(typeConverter, context);
  target.addIllegalOp<AtenReflectionPad2dOp>();
  patterns.add<ConvertAtenReflectionPad2dOp>(typeConverter, context);
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
  target.addIllegalOp<AtenDiagonalOp>();
  patterns.add<ConvertAtenDiagonalOp>(typeConverter, context);
}
