//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/APInt.h"

#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static int64_t productReduce(ArrayRef<int64_t> a) {
  return accumulate(a.begin(), a.end(), /*init=*/static_cast<int64_t>(1),
                    std::multiplies<int64_t>());
}

template <typename OpTy, typename OpAdaptor>
LogicalResult prepareArgumentsForSlicingOp(OpTy op, OpAdaptor adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           int64_t &dim,
                                           SmallVector<Value> &resultShape,
                                           SmallVector<Value> &offsets,
                                           SmallVector<Value> &strides) {
  Location loc = op.getLoc();
  auto input = adaptor.getSelf();
  RankedTensorType inputType = cast<RankedTensorType>(input.getType());

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value negone = arith::ConstantIndexOp::create(rewriter, loc, -1);

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

  Value stepIndex = castIntToIndex(rewriter, loc, adaptor.getStep());
  Value start = toPositiveValidDim(rewriter, loc, torchTypeStart,
                                   builtinTypeStart, zero, dimSize);

  // We cannot use to positive valid dim as for negative strides we need to
  // clamp to `-1` so that the full tensor bounds are available:
  Value end = builtinTypeEnd;
  if (isa<Torch::NoneType>(torchTypeEnd.getType())) {
    end = dimSize;
  } else {
    end = castIntToIndex(rewriter, loc, end);
    Value endcmp = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::slt, end, zero);
    Value endadd = arith::AddIOp::create(rewriter, loc, end, dimSize);
    end = arith::SelectOp::create(rewriter, loc, endcmp, endadd, end);
    endcmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                   end, zero);
    end = arith::SelectOp::create(rewriter, loc, endcmp, negone, end);
    endcmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                   end, dimSize);
    end = arith::SelectOp::create(rewriter, loc, endcmp, dimSize, end);
  }

  // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
  resultShape = getTensorSizes(rewriter, loc, input);
  Value len = arith::SubIOp::create(rewriter, loc, end, start);

  // We check the difference between start and end to determine the total size:
  Value stepcmp = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, stepIndex, zero);
  Value stepsign = arith::SelectOp::create(rewriter, loc, stepcmp, one, negone);
  Value resultSize = arith::AddIOp::create(rewriter, loc, len, stepIndex);
  resultSize = arith::SubIOp::create(rewriter, loc, resultSize, stepsign);
  resultSize =
      arith::FloorDivSIOp::create(rewriter, loc, resultSize, stepIndex);

  // Clamp the size to [0, ...]:
  Value szcmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                      resultSize, zero);
  resultSize = arith::SelectOp::create(rewriter, loc, szcmp, zero, resultSize);
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
      return arith::AddIOp::create(rewriter, loc, x, y);
    };

    // Create an integer expression of x - y
    auto createISub = [&](Value x, Value y) {
      return arith::SubIOp::create(rewriter, loc, x, y);
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
      Value tile = tensor::ExtractSliceOp::create(
          rewriter, loc, input, extractOffsets, extractShape, allOneStrides);

      auto inputMap = AffineMap::getMultiDimIdentityMap(numDims, context);
      // Setup the affine map function to resverse the tile along the horizontal
      // for left and right slices
      if (padPosition < PAD_CENTER) {
        inputMap = reflectDim(inputMap, numDims, lastDim, padInts[padPosition]);
        // Take reflected slice as per inputMap
        tile = linalg::GenericOp::create(
                   rewriter, loc, llvm::cast<RankedTensorType>(tile.getType()),
                   tile, tile, ArrayRef({inputMap, idMap}), iteratorTypes,
                   [](OpBuilder &b, Location nestedLoc, ValueRange args) {
                     linalg::YieldOp::create(b, nestedLoc, args[0]);
                   })
                   .getResult(0);
      }
      // Insert the tile in the resultTensor
      SmallVector<Value> insertOffsets(numDims, zero);
      insertOffsets[lastDim] = insertOffset[padPosition];
      resultTensor = tensor::InsertSliceOp::create(rewriter, loc, tile,
                                                   resultTensor, insertOffsets,
                                                   extractShape, allOneStrides);
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
      return arith::AddIOp::create(rewriter, loc, x, y);
    };

    auto createAdds = [&](std::initializer_list<Value> values) {
      assert(values.size() >= 2);
      return std::accumulate(values.begin() + 1, values.end(), data(values)[0],
                             createAdd);
    };

    auto createSub = [&](Value x, Value y) {
      return arith::SubIOp::create(rewriter, loc, x, y);
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

    auto verifyPadding = [&](int64_t padArgument, int64_t dim,
                             StringRef errorMessage) {
      auto padValue =
          arith::ConstantIndexOp::create(rewriter, loc, padArgument);
      Value index = arith::ConstantIndexOp::create(rewriter, loc, dim);
      Value shapeDim = tensor::DimOp::create(rewriter, loc, input, index);
      Value cmpPred = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sle, padValue, shapeDim);
      cf::AssertOp::create(rewriter, loc, cmpPred,
                           rewriter.getStringAttr(errorMessage));
    };

    verifyPadding(getHPadArgument(LEFT), hDim, "Left padding too large");
    verifyPadding(getHPadArgument(RIGHT), hDim, "Right padding too large");
    verifyPadding(getVPadArgument(TOP), vDim, "Top padding too large");
    verifyPadding(getVPadArgument(BOTTOM), vDim, "Bottom padding too large");

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

      Value tile = tensor::ExtractSliceOp::create(
          rewriter, loc, input, extractOffsets, extractShape, allOneStrides);

      auto inputMap = AffineMap::getMultiDimIdentityMap(numDims, context);

      tile =
          linalg::GenericOp::create(
              rewriter, loc, llvm::cast<RankedTensorType>(tile.getType()), tile,
              tile, ArrayRef({inputMap, idMap}), iteratorTypes,
              [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
                // Use linalg.index to reflect the dims
                SmallVector<Value> extractIndices(numDims);
                for (unsigned i = 0; i < numDims; i++)
                  extractIndices[i] = linalg::IndexOp::create(b, nestedLoc, i);

                auto reflectDim = [&](int64_t padSize, Value dim) {
                  Value reflectDimSize = getConstant(rewriter, loc, padSize - 1,
                                                     rewriter.getIndexType());
                  return arith::SubIOp::create(b, loc, reflectDimSize, dim);
                };

                // Reverse the tile along the horizontal, vertical, or both
                // dimensions.
                if (shouldHReflect(horizontalPos))
                  extractIndices[hDim] = reflectDim(
                      getHPadArgument(horizontalPos), extractIndices[hDim]);

                if (shouldVReflect(verticalPos))
                  extractIndices[vDim] = reflectDim(
                      getVPadArgument(verticalPos), extractIndices[vDim]);

                Value extractValue = tensor::ExtractOp::create(
                    rewriter, nestedLoc, tile, extractIndices);
                linalg::YieldOp::create(b, nestedLoc, extractValue);
              })
              .getResult(0);

      // Insert the tile in the resultTensor.
      SmallVector<Value> insertOffsets(numDims, zero);
      insertOffsets[hDim] = insertHOffset[horizontalPos];
      insertOffsets[vDim] = insertVOffset[verticalPos];

      resultTensor = tensor::InsertSliceOp::create(rewriter, loc, tile,
                                                   resultTensor, insertOffsets,
                                                   extractShape, allOneStrides);
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
    auto type = cast<RankedTensorType>(adaptor.getSelf().getType());
    auto inputRank = type.getRank();
    if (inputRank == 1) {
      // If input rank is equal to 1, then there's no scope for flattening the
      // input tensor.
      rewriter.replaceOp(op, adaptor.getSelf());
      return success();
    }

    auto resultType =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
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
    Value collapsedTensor = tensor::CollapseShapeOp::create(
        rewriter, op->getLoc(), adaptor.getSelf(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                collapsedTensor);
    return success();
  }
};
} // namespace

// Lower aten.unflatten.int into tensor.expand_shape
namespace {
class ConvertAtenUnflattenIntOp
    : public OpConversionPattern<AtenUnflattenIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUnflattenIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    BaseTensorType outputTensorType = cast<BaseTensorType>(op.getType());
    if (!outputTensorType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: output must have known sizes");

    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "unimplemented: unranked tensor");
    auto inputTensorType = cast<Torch::ValueTensorType>(self.getType());
    if (!inputTensorType || !inputTensorType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "Expected input type having sizes");
    }
    int inputRank = inputTensorType.getSizes().size();
    auto outputSizes = outputTensorType.getSizes();
    int outputRank = outputSizes.size();

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires dim to be constants");

    dimInt = toPositiveDim(dimInt, inputRank);
    if (!isValidDim(dimInt, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");

    auto sizesOp = op.getSizes().getDefiningOp<Torch::PrimListConstructOp>();
    int numSizes = sizesOp.getNumOperands();

    int64_t numDynamicReassocDims = 0;
    for (int64_t i = dimInt; i < dimInt + numSizes; i++) {
      if (outputSizes[i] == Torch::kUnknownSize)
        numDynamicReassocDims++;
    }

    SmallVector<Value> reassocSizes;
    if (!getListConstructElements(op.getSizes(), reassocSizes) &&
        numDynamicReassocDims > 1)
      return rewriter.notifyMatchFailure(
          op, "Must be able to either infer expansion dims, or retrieve them "
              "from list construct");

    auto expandTy = getTypeConverter()->convertType(outputTensorType);
    Value expand;
    // When there are less than two dynamic reassociation dims, this will lower
    // to tensor.expand_shape. Otherwise, this lowers to tensor.reshape.
    // TODO: in the numDynamicReassocDims >= 2 case, lower to expand_shape with
    // explicitly provided outputShape once
    // https://github.com/iree-org/iree/issues/17760 is resolved.
    if (numDynamicReassocDims < 2) {
      SmallVector<ReassociationIndices> reassociations(inputRank);
      if (inputRank > 0) {
        for (int i = 0; i < dimInt; ++i)
          reassociations[i].push_back(i);
        for (int i = 0; i < numSizes; ++i)
          reassociations[dimInt].push_back(i + dimInt);
        for (int i = dimInt + numSizes; i < outputRank; ++i)
          reassociations[i - numSizes + 1].push_back(i);
      }
      expand = tensor::ExpandShapeOp::create(rewriter, loc, expandTy,
                                             adaptor.getSelf(), reassociations)
                   .getResult();
    } else {
      reassocSizes = getTypeConvertedValues(rewriter, loc, getTypeConverter(),
                                            reassocSizes);
      SmallVector<Value> inputShape =
          getTensorSizes(rewriter, loc, adaptor.getSelf());
      inputShape = castIndexVectorToInt64Vector(rewriter, loc, inputShape);
      SmallVector<Value> outputShape(inputShape.begin(),
                                     inputShape.begin() + dimInt);
      if (inputRank > 0) {
        for (int i = 0; i < numSizes; ++i)
          outputShape.push_back(reassocSizes[i]);
        for (int i = dimInt + numSizes; i < outputRank; ++i)
          outputShape.push_back(inputShape[i - numSizes + 1]);
      }

      RankedTensorType shapeType = RankedTensorType::get(
          ArrayRef<int64_t>{outputRank}, rewriter.getIntegerType(64));
      Value shapeValue =
          tensor::FromElementsOp::create(rewriter, loc, shapeType, outputShape);
      expand = tensor::ReshapeOp::create(rewriter, loc, expandTy,
                                         adaptor.getSelf(), shapeValue)
                   .getResult();
    }
    rewriter.replaceOp(op, expand);
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
    if (xTotalSize == kUnknownSize || yTotalSize == kUnknownSize)
      return failure();
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

  // Starting from the beginning of the dims arrays, this helper finds the
  // smallest set of consecutive dims in each array that satisfies one of
  // the following conditions.
  // 1. The product of the static dim sizes in the two subsets is equal.
  // 2. The product of the dim size multiplied by the multiplier for the unknown
  // one in both subsets is equal.
  // The indices arrays are populated with the indices of the dims arrays that
  // correspond to the subsets found.
  //
  // An error is returned if two subsets of dims with total number of elements
  // equal to each other is not found.
  static LogicalResult mapParallelUnknownDims(ArrayRef<int64_t> xDims,
                                              ArrayRef<int64_t> yDims,
                                              SmallVector<int64_t> &xIndices,
                                              SmallVector<int64_t> &yIndices,
                                              int64_t xMultiplier,
                                              int64_t yMultiplier) {
    if (xDims.empty() || yDims.empty())
      return failure();
    if (llvm::count(xDims, kUnknownSize) > 1 ||
        llvm::count(yDims, kUnknownSize) > 1)
      return failure();

    int64_t xTotalSize = xDims[0];
    int64_t yTotalSize = yDims[0];
    SmallVector<int64_t> xIndicesResult({0});
    SmallVector<int64_t> yIndicesResult({0});
    size_t nextXIndex = 1;
    size_t nextYIndex = 1;
    bool xHasUnknownSize = false;
    bool yHasUnknownSize = false;
    if (xTotalSize == kUnknownSize) {
      xHasUnknownSize = true;
      xTotalSize = xMultiplier;
    }
    if (yTotalSize == kUnknownSize) {
      yHasUnknownSize = true;
      yTotalSize = yMultiplier;
    }

    while (xTotalSize != yTotalSize || xHasUnknownSize != yHasUnknownSize) {
      if ((!xHasUnknownSize && yHasUnknownSize) || xTotalSize < yTotalSize) {
        if (nextXIndex == xDims.size())
          return failure();
        if (xDims[nextXIndex] == kUnknownSize) {
          // No support for more than one unknown dim.
          if (xHasUnknownSize)
            return failure();
          xTotalSize *= xMultiplier;
          xHasUnknownSize = true;
        } else {
          xTotalSize *= xDims[nextXIndex];
        }
        xIndicesResult.push_back(nextXIndex++);
      } else {
        if (nextYIndex == yDims.size())
          return failure();
        if (yDims[nextYIndex] == kUnknownSize) {
          // No support for more than one unknown dim.
          if (yHasUnknownSize)
            return failure();
          yTotalSize *= yMultiplier;
          yHasUnknownSize = true;
        } else {
          yTotalSize *= yDims[nextYIndex];
        }
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
        cast<BaseTensorType>(inputTorchTensor.getType()).getSizes());
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

  // Gets the ratio between the unknown dimensions in the input shape and the
  // output shape. This ratio is used to match parallel unknown dimensions.
  static std::pair<int64_t, int64_t>
  getMultiplier(SmallVector<int64_t> inputShape,
                SmallVector<int64_t> outputShape) {
    int64_t totalInputElements = std::abs(productReduce(inputShape));
    int64_t totalOutputElements = std::abs(productReduce(outputShape));
    APInt GCD = llvm::APIntOps::GreatestCommonDivisor(
        APInt(64, totalInputElements), APInt(64, totalOutputElements));
    int64_t gcd = *(GCD.getRawData());
    int64_t inputMultiplier = totalOutputElements / gcd;
    int64_t outputMultiplier = totalInputElements / gcd;
    return std::make_pair(inputMultiplier, outputMultiplier);
  }

  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getParentOp()->hasAttr("torch.disable_legacy_view"))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "legacy view lowering diabled");
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    const TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    int64_t resultRank = resultType.getRank();
    if (resultRank == 0) {
      rewriter
          .replaceOpWithNewOp<tensor::CollapseShapeOp>(
              op, resultType, input, ArrayRef<ReassociationIndices>())
          .getResult();
      return success();
    }

    if (inputRank == 0) {
      llvm::SmallVector<int64_t> outshape(resultRank, 1);
      auto expandTy =
          RankedTensorType::get(outshape, resultType.getElementType());
      Value expand =
          tensor::ExpandShapeOp::create(rewriter, op.getLoc(), expandTy, input,
                                        ArrayRef<ReassociationIndices>());
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, expand);
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
    int64_t inputDynDim = llvm::count(inputShape, kUnknownSize);
    int64_t outputDynDim = llvm::count(outputShape, kUnknownSize);
    if (outputDynDim > 1)
      return rewriter.notifyMatchFailure(
          op, "Cannot support more than one output dynamic dimension");

    bool inputHasOneDynDim = inputDynDim == 1;
    bool outputHasOneDynDim = outputDynDim == 1;
    bool singleDynDimsAreEqual =
        inputHasOneDynDim && outputHasOneDynDim &&
        productReduce(inputShape) == productReduce(outputShape);
    SmallVector<std::pair<int64_t, int64_t>> unchangedDims;

    auto [inputMultiplier, outputMultiplier] =
        getMultiplier(inputShape, outputShape);

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
    SmallVector<std::pair<int64_t, int64_t>> checkDimPairs;
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
        } else if (succeeded(mapParallelUnknownDims(
                       inputShapeSlice, outputShapeSlice, inputSliceIndices,
                       outputSliceIndices, inputMultiplier,
                       outputMultiplier))) {
          /// `mapParallelUnknownDims` maps the smallest number of
          /// input and output dimensions in the slice statically known
          /// or parallel unknown to have the same number of elements.
          assumedDynamicDimNotSplit = true;
        } else if (inputShapeSlice[0] == kUnknownSize) {
          // Defer the dynamic shape check to avoid DialectConversion assertion:
          if (outputShapeSlice[0] != kUnknownSize) {
            checkDimPairs.push_back(
                std::pair<int64_t, int64_t>(inputDim, outputDim));
          }

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

    SmallVector<Value> inputSize = getTensorSizes(rewriter, loc, input);

    SmallVector<Value> outputSizeInt = getTypeConvertedValues(
        rewriter, loc, typeConverter, outputSizeTorchInt);
    if (resultRank != (int64_t)outputSizeInt.size()) {
      return rewriter.notifyMatchFailure(
          op, "desired size list length mismatches with the result type rank");
    }

    for (auto [inputDim, outputDim] : checkDimPairs) {
      checkDimEqualHelper(rewriter, loc, inputSize[inputDim],
                          outputSizeInt[outputDim]);
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
          tensor::CollapseShapeOp::create(rewriter, loc, intermediateResultType,
                                          castedInput, inputAssociations)
              .getResult();
    }

    if (llvm::any_of(outputAssociations, [](ReassociationIndices indices) {
          return indices.size() > 1;
        })) {

      collapsedInput =
          tensor::ExpandShapeOp::create(
              rewriter, loc, adjustedResultType,
              expandedInput.has_value() ? expandedInput.value() : castedInput,
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
class ConvertAtenViewOpToReshape : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getParentOp()->hasAttr("torch.disable_legacy_view"))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "legacy view lowering diabled");
    SmallVector<Value> sizes;
    if (!getListConstructElements(op.getSize(), sizes))
      return op.emitError(
          "unimplemented: the tensor size list is not from list construct");

    auto loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto self = adaptor.getSelf();
    const TypeConverter *typeConverter = getTypeConverter();

    // Convert to the `linalg` types, count the number of negative values,
    // and determine the product of non-negative values. This lets us compute
    // the inferred dimensions sizes.
    auto sizeTy =
        cast<IntegerType>(typeConverter->convertType(sizes.front().getType()));
    Value one = arith::ConstantOp::create(b, sizeTy,
                                          rewriter.getIntegerAttr(sizeTy, 1));
    Value zero = arith::ConstantOp::create(b, sizeTy,
                                           rewriter.getIntegerAttr(sizeTy, 0));
    Value count = zero;
    Value knownSize = one;
    for (auto &size : sizes) {
      Value convert = typeConverter->materializeTargetConversion(rewriter, loc,
                                                                 sizeTy, size);

      Value mul = arith::MulIOp::create(b, knownSize, convert);
      Value add = arith::AddIOp::create(b, count, one);
      Value isNeg =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt, convert, zero);

      knownSize = arith::SelectOp::create(b, isNeg, knownSize, mul);
      count = arith::SelectOp::create(b, isNeg, add, count);
      size = convert;
    }

    // Check we are only inferring one dimension if not in strict mode. In
    // strict mode, there will only ever statically be one inferred dim.
    if (!isAssumingStrictSymbolicShapes(rewriter)) {
      Value countPred =
          arith::CmpIOp::create(b, arith::CmpIPredicate::sle, count, one);
      cf::AssertOp::create(
          b, loc, countPred,
          b.getStringAttr(
              "must have at most one inferred (negative) dimension"));
    }

    // Determine the total size of the inferred dimension and update the
    // inferred dimension:
    auto selfTy = cast<RankedTensorType>(self.getType());
    Value totalSize = one;
    for (int i = 0, s = selfTy.getRank(); i < s; ++i) {
      Value index = arith::ConstantIndexOp::create(b, i);
      Value dim = tensor::DimOp::create(b, self, index);
      dim = arith::IndexCastOp::create(b, sizeTy, dim);
      totalSize = arith::MulIOp::create(b, totalSize, dim);
    }

    Value inferredSize = arith::DivSIOp::create(b, totalSize, knownSize);
    for (auto &size : sizes) {
      Value isNeg =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt, size, zero);
      size = arith::SelectOp::create(b, isNeg, inferredSize, size);
    }

    auto ty = RankedTensorType::get(sizes.size(), sizes.front().getType());
    auto outputDims = tensor::FromElementsOp::create(b, ty, sizes);

    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, resultType, self,
                                                   outputDims);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenViewOpStrict : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isAssumingStrictSymbolicShapes(rewriter))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "not strict symbolic shapes");
    SmallVector<Value> sizeValues;
    if (!getListConstructElements(op.getSize(), sizeValues))
      return op.emitError(
          "unimplemented: the tensor size list is not from list construct");

    auto loc = op.getLoc();
    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    auto self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    // Handle collapse to 0D.
    if (sizeValues.empty()) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, adaptor.getSelf(), ArrayRef<ReassociationIndices>{});
      return success();
    }

    // If there is a static inferred dimension (-1), then we emit a
    // flatten/unflatten and let that proceed through its lowering.
    // Otherwise, emit a tensor.reshape. Note that this relies on the fact that
    // Torch does not allow such an op to have a symbolic inferred dim.
    int inferredDim = -1;
    bool staticSizes = true;
    for (int i = 0, e = sizeValues.size(); i < e; ++i) {
      int64_t dim;
      if (!matchPattern(sizeValues[i], m_TorchConstantInt(&dim))) {
        staticSizes = false;
        continue;
      }
      if (dim == -1) {
        inferredDim = i;
        break;
      }
    }

    // While it should be illegal to have a view op with fully known sizes
    // and a dynamic shape, in reality, torch IR is a bit loosey and
    // progressively resolves to this state. There are delicate invariants
    // on the ops we produce that require this, so we enforce.
    if (staticSizes && !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(loc,
                                         "view cannot be converted with static "
                                         "sizes and a dynamic result type");
    }

    // Handle inferred dim case.
    // TODO: Remove the restriction on staticSizes once flatten/unflatten
    // reliably work with multiple dynamic dimensions.
    if (inferredDim >= 0 && staticSizes) {
      if (!staticSizes) {
        return rewriter.notifyMatchFailure(
            loc, "view to flatten/unflatten only supported for static sizes");
      }
      // This is a torch-torch conversion, so only non adapted types are
      // involved.
      auto selfTy = dyn_cast<ValueTensorType>(op.getSelf().getType());
      if (!selfTy || !selfTy.hasSizes())
        return failure();

      // Work out the 1D flattened type.
      int64_t flatDim = 1;
      auto selfSizes = selfTy.getSizes();
      for (int64_t dim : selfSizes) {
        if (dim == kUnknownSize) {
          flatDim = kUnknownSize;
          break;
        }
        flatDim *= dim;
      }
      // Flatten to 1D.
      ValueTensorType flatType = rewriter.getType<ValueTensorType>(
          ArrayRef<int64_t>{flatDim}, selfTy.getOptionalDtype());
      Value dimStart = Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(0));
      Value dimEnd = Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(selfSizes.size() - 1));
      Value flatSelf = Torch::AtenFlattenUsingIntsOp::create(
          rewriter, loc, flatType, op.getSelf(), dimStart, dimEnd);

      // Unflatten to requested size.
      rewriter.replaceOpWithNewOp<AtenUnflattenIntOp>(
          op, op.getResult().getType(), flatSelf, dimStart, op.getSize());
      return success();
    }

    // Generate output dims, either based on whether there is an inferred dim
    // present or all dims are specified.
    auto sizeTy = cast<IntegerType>(
        typeConverter->convertType(sizeValues.front().getType()));
    SmallVector<Value> outputDimValues;
    assert(sizeTy && "Type converter did not handle size");
    if (inferredDim >= 0) {
      // Inferred dim. If the above flatten/unflatten logic ever catches
      // everything, this branch can go away entirely.
      Value one = arith::ConstantOp::create(rewriter, loc, sizeTy,
                                            rewriter.getIntegerAttr(sizeTy, 1));
      Value sizeProduct = one;
      // Multiply the non-inferred target sizes.
      for (int i = 0, e = sizeValues.size(); i < e; ++i) {
        if (i == inferredDim)
          continue;
        Value size = sizeValues[i];
        Value convertedSize = typeConverter->materializeTargetConversion(
            rewriter, loc, sizeTy, size);
        assert(convertedSize && "Type converter did not handle size");
        sizeProduct =
            arith::MulIOp::create(rewriter, loc, sizeProduct, convertedSize);
      }

      // Multiply the self tensor sizes.
      Value selfProduct = one;
      for (int i = 0, e = selfTy.getRank(); i < e; ++i) {
        Value index = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value dim = tensor::DimOp::create(rewriter, loc, self, index);
        dim = arith::IndexCastOp::create(rewriter, loc, sizeTy, dim);
        selfProduct = arith::MulIOp::create(rewriter, loc, selfProduct, dim);
      }

      Value inferredSize =
          arith::DivUIOp::create(rewriter, loc, selfProduct, sizeProduct);
      for (int i = 0, e = sizeValues.size(); i < e; ++i) {
        if (i == inferredDim) {
          outputDimValues.push_back(inferredSize);
        } else {
          outputDimValues.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, sizeTy, sizeValues[i]));
        }
      }
    } else {
      // No inferred dim. So output dims are just pass through.
      for (Value torchSize : sizeValues) {
        outputDimValues.push_back(typeConverter->materializeTargetConversion(
            rewriter, loc, sizeTy, torchSize));
      }
    }

    // Normal lowering to reshape with fully computed sizes.
    auto outputDimsTy = RankedTensorType::get(
        outputDimValues.size(), outputDimValues.front().getType());
    auto outputDims = tensor::FromElementsOp::create(
        rewriter, loc, outputDimsTy, outputDimValues);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getSelf(), outputDims);
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
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    int64_t inputRank = inputType.getRank();

    const TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
    auto resultShape = resultType.getShape();
    int64_t resultRank = resultType.getRank();

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    // No change in rank so we just cast to the output type:
    if (inputRank == resultRank) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
      return success();
    }

    // In case the operand tensor type is statically shaped with all dimensions
    // being unit extent, it will be collapsed to a 0-D tensor.
    if (resultRank == 0) {
      SmallVector<ReassociationIndices> reassociation;
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, input, reassociation);
      return success();
    }

    SmallVector<ReassociationIndices> reassociation(resultRank);
    // First dimensions are guaranteed to match to eachother:
    int64_t i = 0;
    int64_t j = 0;

    for (i = 0; i < inputRank && j < resultRank; i++) {
      reassociation[j].push_back(i);
      j = inputShape[i] == resultShape[j] ? j + 1 : j;
    }

    // Squeeze in the remaining 1s:
    for (; i < inputRank; ++i) {
      if (inputShape[i] != 1)
        return rewriter.notifyMatchFailure(op,
                                           "non-unary dim cannot be squeezed");
      reassociation.back().push_back(i);
    }

    // Make sure that result type rank is compatible with the squeezed size:
    if (j != resultRank)
      return rewriter.notifyMatchFailure(
          op, "expected output size mismatches with the result type rank");

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, resultType, input,
                                                         reassociation);
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
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    auto squeezeTensorInfo =
        squeezeTensor(rewriter, op, adaptor.getSelf(), dim);
    if (failed(squeezeTensorInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");
    }

    rewriter.replaceOp(op, squeezeTensorInfo.value());
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

    auto unsqueezeTensorInfo =
        unsqueezeTensor(rewriter, op, adaptor.getSelf(), dim);
    if (failed(unsqueezeTensorInfo)) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot generate unsqueeze tensor");
    }

    rewriter.replaceOp(op, unsqueezeTensorInfo.value());
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
    auto inType = cast<RankedTensorType>(inVector.getType());
    auto inputRank = inType.getRank();
    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    if (inputRank <= 1 && inType == outType) {
      rewriter.replaceOp(op, {adaptor.getSelf()});
      return success();
    }
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

    Value outVector = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(outputDims), elementType);

    SmallVector<int64_t> permutation(inputRank);
    std::iota(permutation.begin(), permutation.end(), 0);
    permutation[dim0] = dim1;
    permutation[dim1] = dim0;

    auto transpose = linalg::TransposeOp::create(rewriter, loc, inVector,
                                                 outVector, permutation)
                         .getResult();
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
    Value result;
    if (failed(torch_to_linalg::permuteTensor(op, rewriter, op->getLoc(),
                                              dimensions, inVector, result)))
      return rewriter.notifyMatchFailure(
          op, "failed to perform permutation of tensor");

    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, result);
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
    RankedTensorType resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));

    SmallVector<Value> resultShape, offsets, strides;
    int64_t dim;
    if (failed(prepareArgumentsForSlicingOp<AtenSliceTensorOp,
                                            AtenSliceTensorOpAdaptor>(
            op, adaptor, rewriter, dim, resultShape, offsets, strides))) {
      return failure();
    }

    // If stride is negative, then flip the input tensor corresponding to that
    // dim, update the stride for flipped tensor by multiplying it by -1, and
    // update the offset as follows:
    // flipped_offset = input_shape[dim] - (result_shape[dim] * flipped_stride)
    //
    // For example:
    // Input = [0, 1, 2, 3, 4, 5]
    // stride = [-2], result_shape = [2], offset = [3]
    // Result = [3, 1]
    // After flipping:
    // Input = [5, 4, 3, 2, 1, 0]
    // stride = [2], result_shape = [2], offset = [6 - (2 * 2)] = [2]
    // Result = [3, 1]

    Value flippedInput = torch_to_linalg::flipTensor(rewriter, loc, input,
                                                     SmallVector<int64_t>{dim});
    Value cstDim = arith::ConstantIndexOp::create(rewriter, loc, dim);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value isNegativeStride = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, strides[dim], zero);
    strides[dim] = math::AbsIOp::create(rewriter, loc, strides[dim]);
    Value resShapeMulStride =
        arith::MulIOp::create(rewriter, loc, resultShape[dim], strides[dim]);
    Value inputDim = tensor::DimOp::create(rewriter, loc, input, cstDim);
    Value flippedOffset =
        arith::SubIOp::create(rewriter, loc, inputDim, resShapeMulStride);
    offsets[dim] = arith::SelectOp::create(rewriter, loc, isNegativeStride,
                                           flippedOffset, offsets[dim]);

    input = arith::SelectOp::create(rewriter, loc, isNegativeStride,
                                    flippedInput, input);

    SmallVector<int64_t> dynShape(resultType.getRank(), ShapedType::kDynamic);
    auto sliceType = RankedTensorType::get(
        dynShape, resultType.getElementType(), resultType.getEncoding());
    Value result = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, input, offsets, resultShape, strides);

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
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));
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
    int64_t inputRank = cast<RankedTensorType>(self.getType()).getRank();
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
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
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
    RankedTensorType selfType = cast<RankedTensorType>(self.getType());

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
    Value result = linalg::GenericOp::create(
                       rewriter, loc,
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
                         linalg::YieldOp::create(b, loc, result);
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

    RankedTensorType resultType = cast<RankedTensorType>(
        typeConverter->convertType(op->getResult(0).getType()));

    SmallVector<Value> resultShape, offsets, strides;
    int64_t dim;
    if (failed(prepareArgumentsForSlicingOp<AtenSliceScatterOp,
                                            AtenSliceScatterOpAdaptor>(
            op, adaptor, rewriter, dim, resultShape, offsets, strides))) {
      return failure();
    }

    Value src = adaptor.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    int64_t srcRank = srcType.getRank();
    SmallVector<int64_t> srcAbstractSizes(srcRank, kUnknownSize);
    // TODO: audit possibility of sparsity on these tensor
    auto abstractSrcType = RankedTensorType::get(
        makeShapeLLVMCompatible(srcAbstractSizes), srcType.getElementType());
    Value abstractSrc =
        tensor::CastOp::create(rewriter, loc, abstractSrcType, src);

    Value result = tensor::InsertSliceOp::create(
        rewriter, loc, abstractSrc, input, offsets, resultShape, strides);

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
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));

    auto elementType = resultType.getElementType();
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < resultType.getRank(); i++) {
      auto currentDimSize = tensor::DimOp::create(rewriter, loc, input, i);
      resultShape.push_back(currentDimSize);
    }

    Value outTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(resultShape), elementType);

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
        linalg::GenericOp::create(
            rewriter, loc, outTensor.getType(), ValueRange{}, outTensor,
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              SmallVector<Value> indicesZero;
              SmallVector<Value> indicesOne;

              for (int i = 0; i < resultType.getRank(); i++) {
                indicesZero.push_back(linalg::IndexOp::create(b, loc, i));
                indicesOne.push_back(linalg::IndexOp::create(b, loc, i));
              }

              indicesZero.push_back(constantZero);
              indicesOne.push_back(constantOne);

              Value realVal =
                  tensor::ExtractOp::create(b, loc, input, indicesZero);
              Value imagVal =
                  tensor::ExtractOp::create(b, loc, input, indicesOne);
              Value complexVal = complex::CreateOp::create(b, loc, elementType,
                                                           realVal, imagVal);
              linalg::YieldOp::create(b, loc, complexVal);
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
        cast<RankedTensorType>(typeConverter->convertType(op.getType()));

    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    auto inputElementType = getElementTypeOrSelf(input.getType());
    if (!isa<ComplexType>(inputElementType)) {
      return op.emitError("only ComplexType is allowed as input type");
    }
    Type elementType = resultType.getElementType();

    // returned real tensor has a size increase, where the last dim has size 2
    SmallVector<OpFoldResult> resultShape =
        tensor::getMixedSizes(rewriter, loc, input);
    resultShape.push_back(
        rewriter.createOrFold<arith::ConstantIndexOp>(loc, 2));

    Value outTensor =
        tensor::EmptyOp::create(rewriter, loc, resultShape, elementType);

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
        linalg::GenericOp::create(
            rewriter, loc, outTensor.getType(), input, outTensor, indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value realVal =
                  complex::ReOp::create(b, loc, elementType, args[0]);
              Value imagVal =
                  complex::ImOp::create(b, loc, elementType, args[0]);
              Value lastIndex =
                  linalg::IndexOp::create(b, loc, inputType.getRank());
              Value cmpResult = arith::CmpIOp::create(
                  b, loc, arith::CmpIPredicate::eq, lastIndex, constantZero);
              Value yieldValue =
                  arith::SelectOp::create(b, loc, cmpResult, realVal, imagVal);

              linalg::YieldOp::create(b, loc, yieldValue);
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
    RankedTensorType inputType = cast<RankedTensorType>(inputMatrix.getType());
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
    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Location loc = op.getLoc();

    Value dim1Size, dim2Size;
    dim1Size = getDimOp(rewriter, loc, inputMatrix, dim1);
    dim2Size = getDimOp(rewriter, loc, inputMatrix, dim2);

    // compute the length of the diagonal with possible offset
    // if the offset is very large or very small, diagSize=0 and an empty tensor
    // is returned
    Value indexZero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value indexMinusOne = arith::ConstantIndexOp::create(rewriter, loc, -1);
    Value indexOffset = arith::ConstantIndexOp::create(rewriter, loc, offset);
    Value offsetIsNegative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sle, indexOffset, indexZero);
    Value sizeForNegativeOffset = arith::MaxSIOp::create(
        rewriter, loc,
        arith::MinSIOp::create(
            rewriter, loc,
            arith::AddIOp::create(rewriter, loc, dim1Size, indexOffset),
            dim2Size),
        indexZero);
    Value sizeForPositiveOffset = arith::MaxSIOp::create(
        rewriter, loc,
        arith::MinSIOp::create(
            rewriter, loc,
            arith::SubIOp::create(rewriter, loc, dim2Size, indexOffset),
            dim1Size),
        indexZero);
    Value diagSize =
        arith::SelectOp::create(rewriter, loc, offsetIsNegative,
                                sizeForNegativeOffset, sizeForPositiveOffset);

    // depending on its sign, the offset affects only the row or column indices
    // of the diagonal
    Value diagStart1 = arith::SelectOp::create(
        rewriter, loc, offsetIsNegative,
        arith::MulIOp::create(rewriter, loc, indexOffset, indexMinusOne),
        indexZero);
    Value diagStart2 = arith::SelectOp::create(rewriter, loc, offsetIsNegative,
                                               indexZero, indexOffset);

    SmallVector<Value> outputDims;
    for (auto i = 0; i < inputRank; i++) {
      if (!(i == dim1 || i == dim2))
        outputDims.push_back(getDimOp(rewriter, loc, inputMatrix, i));
    }
    outputDims.push_back(diagSize);

    Value outputMatrix = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(outputDims), elementType);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::getMultiDimIdentityMap(outputRank, rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputRank, utils::IteratorType::parallel);

    auto diagonal =
        linalg::GenericOp::create(
            rewriter, loc, outputMatrix.getType(), ValueRange{}, outputMatrix,
            indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              SmallVector<Value> diagIndices;
              Value indexOnDiag =
                  linalg::IndexOp::create(b, loc, outputRank - 1);
              Value dim1Index =
                  arith::AddIOp::create(b, loc, indexOnDiag, diagStart1);
              Value dim2Index =
                  arith::AddIOp::create(b, loc, indexOnDiag, diagStart2);

              // specify at which input indices the diagonal values are
              // extracted
              for (int indIn = 0, indOut = 0; indIn < inputRank; indIn++) {
                if (indIn == dim1)
                  diagIndices.push_back(dim1Index);
                else if (indIn == dim2)
                  diagIndices.push_back(dim2Index);
                else {
                  diagIndices.push_back(
                      linalg::IndexOp::create(b, loc, indOut));
                  indOut++;
                }
              }
              Value diagElt = tensor::ExtractOp::create(
                  b, loc, elementType, inputMatrix, diagIndices);
              linalg::YieldOp::create(b, loc, diagElt);
            })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputType, diagonal);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenDiagEmbedOp : public OpConversionPattern<AtenDiagEmbedOp> {

  static SmallVector<Value>
  getDiagEmbedResultShape(OpBuilder &b, Location loc, Value tensor,
                          int64_t offset, int64_t dim1, int64_t dim2) {
    auto inputType = cast<RankedTensorType>(tensor.getType());
    auto inputRank = inputType.getRank();

    // output tensor always has 1 extra dimension
    auto resultRank = inputRank + 1;

    // regardless of offset sign, output tensor is same
    Value constOffset = arith::ConstantIndexOp::create(b, loc, offset);
    Value absOffset = math::AbsIOp::create(b, loc, constOffset);

    // diagonal size is determined by last input dimension
    auto lastInputDim = getDimOp(b, loc, tensor, inputRank - 1);
    Value diagDim = arith::AddIOp::create(b, loc, lastInputDim, absOffset);

    // output shape has same dimensions as input
    // except for the diagonal dimensions
    int input_dim_idx = 0;
    SmallVector<Value> resultShape;
    for (unsigned int i = 0; i < resultRank; i++) {
      if (i == dim1 || i == dim2)
        resultShape.push_back(diagDim);
      else
        resultShape.push_back(getDimOp(b, loc, tensor, input_dim_idx++));
    }

    return resultShape;
  }

public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDiagEmbedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();

    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();
    auto resultRank = inputRank + 1;

    int64_t offset;
    if (!matchPattern(op.getOffset(), m_TorchConstantInt(&offset)))
      return rewriter.notifyMatchFailure(op, "offset is not constant");

    int64_t dim1;
    if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 is not constant");
    dim1 = toPositiveDim(dim1, resultRank);
    if (!isValidDim(dim1, resultRank))
      return rewriter.notifyMatchFailure(
          op, "dim1 can only be in closed range [" +
                  std::to_string(-resultRank) + "," +
                  std::to_string(resultRank - 1) + "]");

    int64_t dim2;
    if (!matchPattern(op.getDim2(), m_TorchConstantInt(&dim2)))
      return rewriter.notifyMatchFailure(op, "dim2 is not constant");
    dim2 = toPositiveDim(dim2, resultRank);
    if (!isValidDim(dim2, resultRank))
      return rewriter.notifyMatchFailure(
          op, "dim2 can only be in closed range [" +
                  std::to_string(-resultRank) + "," +
                  std::to_string(resultRank - 1) + "]");

    if (dim1 == dim2)
      return rewriter.notifyMatchFailure(op, "dim1 and dim2 can not be equal");

    // add linalg.fill
    Type resultElemType = inputType.getElementType();
    auto resultShape =
        getDiagEmbedResultShape(rewriter, loc, input, offset, dim1, dim2);
    Value zeroTensor =
        createZeroInitTensor(rewriter, loc, resultShape, resultElemType);

    // add linalg.generic with diagonal access pattern affine indexing maps
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(resultRank),
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);
    Value resultTensor =
        linalg::GenericOp::create(
            rewriter, loc, zeroTensor.getType(), ValueRange{}, zeroTensor,
            /*indexingMaps=*/indexingMaps,
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value dim1Index = linalg::IndexOp::create(b, loc, dim1);
              Value dim2Index = linalg::IndexOp::create(b, loc, dim2);

              // to pick right element from input, first add all dimensions
              // except last one, then last will be either dim1 or dim2
              // depending upon lower or upper diagonal defined by offset
              // sign
              SmallVector<Value> inputIndices;
              for (unsigned int i = 0; i < resultRank; i++) {
                if (i != dim1 && i != dim2) {
                  inputIndices.push_back(linalg::IndexOp::create(b, loc, i));
                }
              }

              // adjust output diagonal indices and last input Index based
              // on offset
              Value dim1IdxAdjusted;
              Value dim2IdxAdjusted;
              if (offset < 0) {
                Value absOffset =
                    arith::ConstantIndexOp::create(b, loc, -offset);
                dim1IdxAdjusted = dim1Index;
                dim2IdxAdjusted =
                    arith::AddIOp::create(b, loc, dim2Index, absOffset);
                inputIndices.push_back(linalg::IndexOp::create(b, loc, dim2));
              } else {
                Value constOffset =
                    arith::ConstantIndexOp::create(b, loc, offset);
                dim1IdxAdjusted =
                    arith::AddIOp::create(b, loc, dim1Index, constOffset);
                dim2IdxAdjusted = dim2Index;
                inputIndices.push_back(linalg::IndexOp::create(b, loc, dim1));
              }

              Value isDiagonal =
                  arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq,
                                        dim1IdxAdjusted, dim2IdxAdjusted);

              Value inputElem = tensor::ExtractOp::create(
                  b, loc, resultElemType, input, inputIndices);

              Value result = arith::SelectOp::create(rewriter, loc, isDiagonal,
                                                     inputElem, args[0]);
              linalg::YieldOp::create(b, loc, result);
            })
            .getResult(0);

    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, resultTensor);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenUnfoldOp : public OpConversionPattern<AtenUnfoldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUnfoldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto self = adaptor.getSelf();
    RankedTensorType selfType = cast<RankedTensorType>(self.getType());

    int64_t dimension;
    if (!matchPattern(op.getDimension(), m_TorchConstantInt(&dimension))) {
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dimension");
    }
    int64_t size;
    if (!matchPattern(op.getSize(), m_TorchConstantInt(&size))) {
      return rewriter.notifyMatchFailure(op, "only support constant int size");
    }
    int64_t step;
    if (!matchPattern(op.getStep(), m_TorchConstantInt(&step))) {
      return rewriter.notifyMatchFailure(op, "only support constant int step");
    }

    if (step <= 0) {
      return rewriter.notifyMatchFailure(op, "step must be greater than zero.");
    }

    int64_t selfRank = selfType.getRank();

    // Zero-Rank case
    if (selfRank == 0) {
      // Empty tensor
      if (size == 0) {
        RankedTensorType resultType =
            RankedTensorType::get({0}, selfType.getElementType());
        Value emptyTensor = tensor::EmptyOp::create(
            rewriter, loc, resultType.getShape(), resultType.getElementType());

        rewriter.replaceOp(op, emptyTensor);
        return success();
      }

      Value unsqueezedSelf = tensor::ExpandShapeOp::create(
          rewriter, loc, RankedTensorType::get({1}, selfType.getElementType()),
          self, ArrayRef<ReassociationIndices>{});
      rewriter.replaceOp(op, unsqueezedSelf);
      return success();
    }

    auto shape = selfType.getShape();

    if (dimension < 0) {
      dimension = toPositiveDim(dimension, selfRank);
    }
    if (!isValidDim(dimension, selfRank)) {
      return rewriter.notifyMatchFailure(op, "dimension out of range");
    }

    Value dimSize = tensor::DimOp::create(rewriter, loc, self, dimension);

    Value sizeValue = arith::ConstantIndexOp::create(rewriter, loc, size);
    Value sizeCheck = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ule, sizeValue, dimSize);
    cf::AssertOp::create(
        rewriter, loc, sizeCheck,
        rewriter.getStringAttr("size must be <= target dimension"));

    /* Calculate output shape of unfold op:
     *  https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
     *  outputShape[dimension] is set to numBlocks, with size appended as an
     *  additional dimension
     */
    SmallVector<OpFoldResult> outputShape;
    for (int64_t i = 0; i < selfRank; i++) {
      if (i == dimension) {
        outputShape.push_back(getDynamicOrStaticNumBlocks(
            rewriter, loc, shape[dimension], dimSize, size, step));
      } else if (shape[i] == ShapedType::kDynamic) {
        outputShape.push_back(
            OpFoldResult(tensor::DimOp::create(rewriter, loc, self, i)));
      } else {
        outputShape.push_back(rewriter.getIndexAttr(shape[i]));
      }
    }
    outputShape.push_back(rewriter.getIndexAttr(size));

    // Empty tensor to insert values into
    Value outputTensor = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                                 selfType.getElementType());

    /**
     * Use reindexing to map output indices to input indices
     * i.e. In output of rank 3 case:
     *     (i, j, k) => (i', j') where i' = i * step + k and j' = j
     *       if dimension == 0
     *     (i, j, k) => (i', j') where i' = i and j' = j * step + k
     *       if dimension == 1
     */
    MLIRContext *context = rewriter.getContext();
    SmallVector<AffineExpr> outputExprs;
    for (int dim = 0; dim < selfRank; ++dim) {
      if (dim == dimension) {
        auto idxLast = getAffineDimExpr(selfRank, context);
        auto idxDimension = getAffineDimExpr(dimension, context);

        AffineExpr dimIdx =
            idxLast + idxDimension * rewriter.getAffineConstantExpr(step);
        outputExprs.push_back(dimIdx);
      } else {
        outputExprs.push_back(getAffineDimExpr(dim, context));
      }
    }

    int64_t outputRank = selfRank + 1;
    auto inputAffineMap = AffineMap::get(outputRank, 0, outputExprs, context);
    auto outputAffineMap =
        AffineMap::getMultiDimIdentityMap(outputRank, context);

    SmallVector<utils::IteratorType> iteratorTypes(
        outputRank, utils::IteratorType::parallel);

    Value result =
        linalg::GenericOp::create(
            rewriter, loc, outputTensor.getType(), self, outputTensor,
            ArrayRef({inputAffineMap, outputAffineMap}), iteratorTypes,
            [](OpBuilder &b, Location nestedLoc, ValueRange args) {
              linalg::YieldOp::create(b, nestedLoc, args[0]);
            })
            .getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  OpFoldResult getDynamicOrStaticNumBlocks(OpBuilder &rewriter, Location loc,
                                           int64_t shapeDim, Value dimSize,
                                           int64_t size, int64_t step) const {
    /**
     * numBlocks = (shape[dimension] - size) // step + 1
     */
    if (shapeDim == ShapedType::kDynamic) {
      Value numBlocksSubOp = arith::SubIOp::create(
          rewriter, loc, dimSize,
          arith::ConstantIndexOp::create(rewriter, loc, size));
      Value numBlocksDivOp = arith::DivUIOp::create(
          rewriter, loc, numBlocksSubOp,
          arith::ConstantIndexOp::create(rewriter, loc, step));
      Value numBlocks = arith::AddIOp::create(
          rewriter, loc, arith::ConstantIndexOp::create(rewriter, loc, 1),
          numBlocksDivOp);
      return OpFoldResult(numBlocks);
    }

    int64_t staticNumBlocks = (shapeDim - size) / step + 1;
    return rewriter.getIndexAttr(staticNumBlocks); // Use static value
  }
};
} // namespace

namespace {
class ConvertSparseOperatorOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  static bool isSparsePrimitive(StringRef prim) {
    return llvm::find(legalizedNames, prim) != legalizedNames.end();
  }

  // Rewriting method.
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSparsePrimitive(op.getNameAttr()))
      return failure();
    // Conversion is completed specified by information in the sparse tensor
    // type. Thus, we can rewrite all legalizedNames to the same construct.
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<sparse_tensor::ConvertOp>(
        op, resultType, adaptor.getOperands()[0]);
    return success();
  }

private:
  // The operators that legalize to sparse tensor conversions.
  static SmallVector<StringRef> legalizedNames;
};
// Static initializer.
SmallVector<StringRef> ConvertSparseOperatorOp::legalizedNames = {
    "torch.aten._to_dense", "torch.aten._to_sparse", "torch.aten._to_csr",
    "torch.aten._to_csc",   "torch.aten._to_bsr",    "torch.aten._to_bsc",
    "torch.aten.to_dense",  "torch.aten.to_sparse",  "torch.aten.to_csr",
    "torch.aten.to_csc",    "torch.aten.to_bsr",     "torch.aten.to_bsc",
};
} // namespace

void mlir::torch::torch_to_linalg::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  // Add some legal ops for torch-torch lowering.
  target.addLegalOp<ConstantIntOp>();

  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenReflectionPad1dOp>();
  patterns.add<ConvertAtenReflectionPad1dOp>(typeConverter, context);
  target.addIllegalOp<AtenReflectionPad2dOp>();
  patterns.add<ConvertAtenReflectionPad2dOp>(typeConverter, context);
  target.addIllegalOp<AtenFlattenUsingIntsOp>();
  patterns.add<ConvertAtenFlattenUsingIntsOp>(typeConverter, context);
  patterns.add<ConvertAtenUnflattenIntOp>(typeConverter, context);
  target.addIllegalOp<AtenUnflattenIntOp>();

  // View op sadness: In the future, we only want ConvertAtenViewOpStrict,
  // but this requires work upstream to fully generalize reshape handling.
  // In the meantime, the analysis based ConvertAtenViewOp tries hard to
  // produce expand/collapse shapes, the ConvertAtenViewOpStrict does the
  // right thing but cannot be fully supported for dynamic shapes, and
  // ConvertAtenViewOpToReshape overly pessimizes and generates a lot of IR
  // due to not statically switching between inferred and non-inferred view
  // cases. They are ordered by optimiality of the lowerings they generate
  // when they are able.
  target.addIllegalOp<AtenViewOp>();
  patterns.add<ConvertAtenViewOp>(typeConverter, context, /*benefit=*/300);
  patterns.add<ConvertAtenViewOpStrict>(typeConverter, context,
                                        /*benefit=*/200);
  patterns.add<ConvertAtenViewOpToReshape>(typeConverter, context,
                                           /*benefit=*/100);
  target.addIllegalOp<AtenUnfoldOp>();
  patterns.add<ConvertAtenUnfoldOp>(typeConverter, context);
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
  target.addIllegalOp<AtenDiagEmbedOp>();
  patterns.add<ConvertAtenDiagEmbedOp>(typeConverter, context);
  // Rewrite all special sparse conversions hidden as operators.
  target.addDynamicallyLegalOp<OperatorOp>([&](Torch::OperatorOp op) {
    return !ConvertSparseOperatorOp::isSparsePrimitive(op.getNameAttr());
  });
  patterns.add<ConvertSparseOperatorOp>(typeConverter, context);
}
