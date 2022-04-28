//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static SmallVector<OpFoldResult>
getIndexIntsAsOpFoldResult(OpBuilder &b, SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(
      ints, [&](int64_t val) -> OpFoldResult { return b.getIndexAttr(val); }));
}

// Helper function to get the padding tensor given the padding int values.
Value torch_to_linalg::getPaddedTensor(
    Operation *op, OpBuilder &b, Value &input,
    SmallVectorImpl<int64_t> &lowPaddingInts,
    SmallVectorImpl<int64_t> &highPaddingInts, Value pad) {
  Location loc = op->getLoc();
  Type rankedTensorType =
      tensor::PadOp::inferResultType(input.getType().cast<RankedTensorType>(),
                                     lowPaddingInts, highPaddingInts);
  SmallVector<OpFoldResult> lowPaddings =
      getIndexIntsAsOpFoldResult(b, lowPaddingInts);
  SmallVector<OpFoldResult> highPaddings =
      getIndexIntsAsOpFoldResult(b, highPaddingInts);
  Value paddedInput = tensor::createPadScalarOp(
      rankedTensorType, input, pad, /*low=*/lowPaddings, /*high=*/highPaddings,
      /*packing=*/false, loc, b);
  return paddedInput;
}

// Helper function to get the padding tensor given the padding int values.
// It's assumed that the padding on the low end and high end are the same,
// and that zero padding is required.
Value torch_to_linalg::getZeroPaddedTensor(
    Operation *op, OpBuilder &b, Value &input,
    SmallVectorImpl<int64_t> &paddingInts) {
  assert(input.getType().isa<RankedTensorType>() &&
         "input must be RankedTensorType");
  Location loc = op->getLoc();
  Value c0 = b.create<arith::ConstantOp>(
      loc,
      b.getZeroAttr(input.getType().cast<RankedTensorType>().getElementType()));
  return getPaddedTensor(op, b, input, paddingInts, paddingInts, c0);
}

Value torch_to_linalg::getOutputDimForConvOps(OpBuilder &b, Location loc,
                                              Value in, Value paddingInt,
                                              Value dilationInt,
                                              Value kernelSizeInt,
                                              Value strideInt) {
  Value c1 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value c2 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(2));

  Value doublePadding = b.create<arith::MulIOp>(loc, paddingInt, c2);
  // in + 2 * padding
  Value inAddDoublePadding =
      b.create<arith::AddIOp>(loc, castIndexToInt64(b, loc, in), doublePadding);

  // dilation * (kernelSize - 1)
  Value kernelSizeSub1 = b.create<arith::SubIOp>(loc, kernelSizeInt, c1);
  Value dilationTimesKernelSize =
      b.create<arith::MulIOp>(loc, dilationInt, kernelSizeSub1);

  Value temp =
      b.create<arith::SubIOp>(loc, inAddDoublePadding, dilationTimesKernelSize);
  Value dividend = b.create<arith::SubIOp>(loc, temp, c1);
  Value division = b.create<arith::FloorDivSIOp>(loc, dividend, strideInt);
  Value out = b.create<arith::AddIOp>(loc, division, c1);
  return castIntToIndex(b, loc, out);
}

Value torch_to_linalg::createReductionLinalgGeneric(
    OpBuilder &b, Location loc, Value tensorOperand,
    const DenseSet<int64_t> &dimSet, bool keepDim, Value initElem,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  auto inputType = tensorOperand.getType().cast<RankedTensorType>();

  // Get the result shape by obtaining the size of each
  // dimension in the input tensor that is not getting reduced.
  // If `keepDim` is true, the rank of the output tensor
  // is kept the same as the rank of the input tensor, and the
  // reduced dimensions are set to have size 1.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    auto currentDimSize = b.create<tensor::DimOp>(loc, tensorOperand, i);
    if (!dimSet.contains(i))
      resultShape.push_back(currentDimSize);
    else if (keepDim)
      resultShape.push_back(c1);
  }

  // Create the affine expressions that will be used to
  // iterate over the input and output tensors.
  // Here we also set the type of iterator: parallel or reduction.
  SmallVector<AffineExpr> exprs;
  SmallVector<StringRef> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  for (auto size : llvm::enumerate(inputType.getShape())) {
    exprs.push_back(b.getAffineDimExpr(size.index()));

    if (dimSet.contains(size.index())) {
      iteratorTypes.push_back(getReductionIteratorTypeName());
      // If `keepDim`, create affine map to the first element
      // in the current dimension.
      if (keepDim)
        resultExprs.push_back(b.getAffineConstantExpr(0));
    } else {
      iteratorTypes.push_back(getParallelIteratorTypeName());
      resultExprs.push_back(b.getAffineDimExpr(size.index()));
    }
  }

  auto indexingMaps = AffineMap::inferFromExprList({exprs, resultExprs});
  Value accumulator =
      createInitTensor(b, loc, resultShape, initElem.getType(), initElem);

  return b
      .create<linalg::GenericOp>(
          loc, /*resultTensorTypes=*/accumulator.getType(),
          /*inputs=*/tensorOperand,
          /*outputs=*/accumulator, indexingMaps, iteratorTypes, bodyBuild)
      .getResult(0);
}

Value torch_to_linalg::createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    Type resultElementType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  // The overall error handling strategy here is best viewed by thinking about
  // what happens for a single result dimension. This loop not structured that
  // way because it is hard to create the affine maps for each operand unless
  // we structure the loop to iterate over tensor operands as the outer loop
  // instead of inner loop. This pseudocode gives better intuition:
  // ```
  // for each result dimension:
  //   for each tensor operand:
  //     if it doesn't even have high enough rank relative to the result:
  //       continue
  //     if it is a static size-1 along this result dimension:
  //       continue
  //     if this is the first tensor operand that didn't continue above:
  //       take its dimension size as the size of the non-broadcasted
  //       traversal along this dimension (this may include a dynamic size-1,
  //       **non-broadcasted** traversal!)
  //     emit error check "if the size does not match the non-broadcasted
  //     traversal size along this dimension, error"
  // ```
  SmallVector<int64_t> operandRanks;
  operandRanks.resize(tensorOperands.size());
  llvm::transform(tensorOperands, operandRanks.begin(), [](Value tensor) {
    return tensor.getType().dyn_cast<RankedTensorType>().getRank();
  });

  auto resultRankIt =
      std::max_element(operandRanks.begin(), operandRanks.end());
  assert(resultRankIt != operandRanks.end() && "Unable to get result rank.");
  int64_t resultRank = *resultRankIt;

  // Initialize the resultShape to all 1's, as a fallback in case
  // all sizes along that result dimension are statically 1.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape(resultRank, c1);
  SmallVector<AffineMap> indexingMaps;
  for (Value tensorOperand : tensorOperands) {
    SmallVector<AffineExpr> exprs;
    auto type = tensorOperand.getType().cast<RankedTensorType>();
    for (auto size : llvm::enumerate(type.getShape())) {
      // If the size is statically known to be 1, we don't want any
      // error guards to be spuriously emitted, since we are specifically
      // allowing size-1 broadcasts in this case, as they correspond to a
      // constant-0 indexing map.
      if (size.value() == 1) {
        exprs.push_back(b.getAffineConstantExpr(0));
        continue;
      }

      // The rank of this operand might be smaller than the overall rank of
      // the broadcast. Add an offset to correlate it to the correct
      // dimension of the result.
      auto resultDim = size.index() + (resultRank - type.getRank());

      // The generated linalg op will now be iterating along the full size
      // of this dimension. Record that fact.
      exprs.push_back(b.getAffineDimExpr(resultDim));

      // Now, we need to ensure that such iteration is not going to trigger
      // undefined behavior, by doing appropriate checks against the current
      // dimension size.
      auto currentDimSize = getDimOp(b, loc, tensorOperand, size.index());

      // If the result size of this dimension has so far only hit the
      // statically-known-to-be-1 case above (i.e., we have not yet assigned a
      // new Value to `resultShape[resultDim]`), then we have no other dynamic
      // values to check against, and merely need to record the current
      // dimension size.
      if (resultShape[resultDim] == c1) {
        resultShape[resultDim] = currentDimSize;
        continue;
      }

      // We prohibit the size-1 dynamic broadcasting scenario, so just check
      // for exact equality with the running result size.
      // This is the check which protects against the undefined behavior of
      // the generated linalg op in the case of iterating two operands with
      // dimensions sizes that are expected to match.
      auto equalToRunning =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                  resultShape[resultDim], currentDimSize);
      b.create<cf::AssertOp>(loc, equalToRunning,
                             "mismatched size for broadcast");
    }
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, b.getContext()));
  }

  SmallVector<StringRef> iteratorTypes(resultRank,
                                       getParallelIteratorTypeName());
  // Add the indexing map for the outs init tensor.
  indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

  Value initTensor = b.create<linalg::InitTensorOp>(
      loc, getAsOpFoldResult(resultShape), resultElementType);
  return b
      .create<linalg::GenericOp>(loc,
                                 /*resultTensorTypes=*/initTensor.getType(),
                                 /*inputs=*/tensorOperands,
                                 /*outputs=*/initTensor, indexingMaps,
                                 iteratorTypes, bodyBuild)
      .getResult(0);
}
