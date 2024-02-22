//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
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
  Value paddedInput =
      b.create<tensor::PadOp>(loc, rankedTensorType, input, /*low=*/lowPaddings,
                              /*high=*/highPaddings, pad);
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

// Helper function that adds dynamic padding to a tensor, ignoring unpaddedDims
// dimensions at the beginning. The high and low padding are the same, and the
// padding value is zero.
Value torch_to_linalg::getDynamicZeroPaddedTensor(
    Operation *op, OpBuilder &b, Value &input, SmallVectorImpl<Value> &padding,
    int unpaddedDims, Value pad) {
  assert(input.getType().isa<RankedTensorType>() &&
         "input must be RankedTensorType");
  unsigned int inRank = input.getType().cast<RankedTensorType>().getRank();
  Location loc = op->getLoc();

  SmallVector<Value> inputDims = getTensorSizes(b, loc, input);
  Value c0 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
  SmallVector<Value> paddingIncludingUnchanged(unpaddedDims, c0);
  paddingIncludingUnchanged.append(padding);
  assert(unpaddedDims + padding.size() == inRank &&
         "sum of unpaddedDims and padding.size() must equal to inputRank");
  for (auto pad = paddingIncludingUnchanged.begin();
       pad < paddingIncludingUnchanged.end(); pad++)
    *pad = castIntToIndex(b, loc, *pad);

  Type elementType = input.getType().cast<RankedTensorType>().getElementType();
  // TODO: audit possibility of sparsity on this tensor
  Type inputType =
      RankedTensorType::get(makeShapeLLVMCompatible(llvm::ArrayRef<int64_t>(
                                SmallVector<int64_t>(inRank, kUnknownSize))),
                            elementType);

  SmallVector<OpFoldResult> paddingValues =
      getAsOpFoldResult(paddingIncludingUnchanged);
  return b.create<tensor::PadOp>(loc, inputType, input, /*low=*/paddingValues,
                                 /*high=*/paddingValues, pad);
}

Value torch_to_linalg::getOutputDimForConvOps(OpBuilder &b, Location loc,
                                              Value in, Value paddingInt,
                                              Value dilationInt,
                                              Value kernelSizeInt,
                                              Value strideInt, bool ceilMode) {
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
  Value division;
  if (ceilMode)
    division = b.create<arith::CeilDivSIOp>(loc, dividend, strideInt);
  else
    division = b.create<arith::FloorDivSIOp>(loc, dividend, strideInt);
  Value out = b.create<arith::AddIOp>(loc, division, c1);
  return castIntToIndex(b, loc, out);
}

Value torch_to_linalg::getOutputDimForConvTransposeOps(
    OpBuilder &b, Location loc, Value in, Value paddingInt, Value dilationInt,
    Value kernelSizeInt, Value strideInt, Value outputPaddingInt) {
  Value c1 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value c2 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(2));

  // (in - 1) * stride
  Value inStrided =
      b.create<arith::SubIOp>(loc, castIndexToInt64(b, loc, in), c1);
  inStrided = b.create<arith::MulIOp>(loc, inStrided, strideInt);

  // 2 * padding
  Value doublePadding = b.create<arith::MulIOp>(loc, paddingInt, c2);

  // (kernelSize - 1) * dilation
  Value kernelDilated = b.create<arith::SubIOp>(loc, kernelSizeInt, c1);
  kernelDilated = b.create<arith::MulIOp>(loc, kernelDilated, dilationInt);

  Value out = b.create<arith::SubIOp>(loc, inStrided, doublePadding);
  out = b.create<arith::AddIOp>(loc, out, kernelDilated);
  out = b.create<arith::AddIOp>(loc, out, outputPaddingInt);
  out = b.create<arith::AddIOp>(loc, out, c1);

  return castIntToIndex(b, loc, out);
}

Value torch_to_linalg::createReductionLinalgGeneric(
    OpBuilder &b, Location loc, const ReductionOpInfo &opInfo, Value initElem,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  auto inputType = opInfo.tensorOperand.getType().cast<RankedTensorType>();

  // Get the result shape by obtaining the size of each
  // dimension in the input tensor that is not getting reduced.
  // If `opInfo.keepDim` is true, the rank of the output tensor
  // is kept the same as the rank of the input tensor, and the
  // reduced dimensions are set to have size 1.
  auto c1 = b.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    auto currentDimSize = b.create<tensor::DimOp>(loc, opInfo.tensorOperand, i);
    if (!opInfo.dimSet.contains(i))
      resultShape.push_back(currentDimSize);
    else if (opInfo.keepDim)
      resultShape.push_back(c1);
  }

  // Create the affine expressions that will be used to
  // iterate over the input and output tensors.
  // Here we also set the type of iterator: parallel or reduction.
  SmallVector<AffineExpr> exprs;
  SmallVector<utils::IteratorType> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  for (auto size :
       llvm::enumerate(makeShapeTorchCompatible(inputType.getShape()))) {
    exprs.push_back(b.getAffineDimExpr(size.index()));

    if (opInfo.dimSet.contains(size.index())) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
      // If `opInfo.keepDim`, create affine map to the first element
      // in the current dimension.
      if (opInfo.keepDim)
        resultExprs.push_back(b.getAffineConstantExpr(0));
    } else {
      iteratorTypes.push_back(utils::IteratorType::parallel);
      resultExprs.push_back(b.getAffineDimExpr(size.index()));
    }
  }

  auto indexingMaps =
      AffineMap::inferFromExprList({exprs, resultExprs}, b.getContext());
  Value accumulator =
      createInitTensor(b, loc, resultShape, initElem.getType(), initElem);

  return b
      .create<linalg::GenericOp>(
          loc, /*resultTensorTypes=*/accumulator.getType(),
          /*inputs=*/opInfo.tensorOperand,
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
  //       **non-broadcasted** traversal unless if
  //       isAssumingStrictSymbolicShapes!)
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
  bool elideDynamicBroadcastCheck = isAssumingStrictSymbolicShapes(b);
  for (Value tensorOperand : tensorOperands) {
    SmallVector<AffineExpr> exprs;
    auto type = tensorOperand.getType().cast<RankedTensorType>();
    for (auto size :
         llvm::enumerate(makeShapeTorchCompatible(type.getShape()))) {
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
      if (!elideDynamicBroadcastCheck) {
        auto equalToRunning =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                    resultShape[resultDim], currentDimSize);
        b.create<cf::AssertOp>(loc, equalToRunning,
                               "mismatched size for broadcast");
      }
    }
    indexingMaps.push_back(AffineMap::get(
        /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, b.getContext()));
  }

  SmallVector<utils::IteratorType> iteratorTypes(resultRank,
                                                 utils::IteratorType::parallel);
  // Add the indexing map for the outs init tensor.
  indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

  Value initTensor = b.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(resultShape), resultElementType);
  return b
      .create<linalg::GenericOp>(loc,
                                 /*resultTensorTypes=*/initTensor.getType(),
                                 /*inputs=*/tensorOperands,
                                 /*outputs=*/initTensor, indexingMaps,
                                 iteratorTypes, bodyBuild)
      .getResult(0);
}

// Broadcasts input tensor based on the broadcastToShape.
LogicalResult torch_to_linalg::broadcastToGivenShape(
    Operation *op, PatternRewriter &rewriter, Value input,
    SmallVector<Value> broadcastToShape, RankedTensorType broadcastType,
    Value &result, SmallVector<bool> useBroadcastToShape) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  int64_t inputRank = inputType.getRank();
  int64_t outputRank = broadcastToShape.size();
  ArrayRef<int64_t> outputShape = broadcastType.getShape();
  SmallVector<int64_t> inputShape =
      makeShapeTorchCompatible(inputType.getShape());
  if (outputRank < inputRank) {
    return rewriter.notifyMatchFailure(
        op, "invalid shape: broadcastToShape size must not be smaller than the "
            "size of the input shape");
  }

  Type elementType = inputType.getElementType();
  Location loc = op->getLoc();
  SmallVector<OpFoldResult> outShape;
  bool elideDynamicBroadcastCheck = isAssumingStrictSymbolicShapes(rewriter);

  // Vector indicating broadcasted status when assuming strict symbolic shapes.
  SmallVector<bool> broadcastedStatus;

  // Create affine map and shapes for tensor initialization.
  SmallVector<AffineExpr> outExpr;
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  Value zeroIndex =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIndex =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  size_t diff = outputRank - inputRank;
  bool hasDynamicNumpyBroadcast = false;
  for (size_t i = 0, e = outputRank; i < e; i++) {
    Value shapeValue = broadcastToShape[i];
    size_t j = i - diff;
    bool isDynamic = i >= diff && inputShape[j] == kUnknownSize;

    // Inherit static output shapes if present.
    if (outputShape[i] != ShapedType::kDynamic) {
      outShape.push_back(rewriter.getIndexAttr(outputShape[i]));
      if (i < diff) {
        if (outputShape[i] < 0) {
          return rewriter.notifyMatchFailure(
              op, "invalid shape: negative values not allowed in new broadcast "
                  "dimensions");
        }
        continue;
      }
      if (isDynamic) {
        hasDynamicNumpyBroadcast = true;
      } else if (inputShape[j] != outputShape[i] && inputShape[j] != 1) {
        return rewriter.notifyMatchFailure(
            op, "invalid shape: static mismatch in input and output broadcast "
                "shapes");
      }

      // If strict symbolic shapes are assumed and the input shape is dynamic,
      // we can assume that dim is not broadcasted.
      broadcastedStatus.push_back(inputShape[j] != outputShape[i] &&
                                  !isDynamic);
      continue;
    }

    if (i < diff) {
      if (!elideDynamicBroadcastCheck) {
        Value isValid = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, shapeValue, zero);
        rewriter.create<cf::AssertOp>(
            loc, isValid,
            rewriter.getStringAttr(
                "negative values not allowed in new dimensions"));
      }
      outShape.push_back(castIntToIndex(rewriter, loc, shapeValue));
      continue;
    }
    if (inputShape[j] == 1) {
      // Broadcast singleton dimension
      Value isNegative = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, shapeValue, zero);
      Value select = rewriter.create<arith::SelectOp>(
          loc, isNegative, oneIndex, castIntToIndex(rewriter, loc, shapeValue));
      outShape.push_back(select);
      broadcastedStatus.push_back(true);
      continue;
    }

    // Case of dynamic input dimension wherein the shape to broadcast will
    // yield us the dimension size of the output.
    Value dim;
    if (!useBroadcastToShape.empty() && useBroadcastToShape[j]) {
      dim = castIntToIndex(rewriter, loc, broadcastToShape[i]);
      if (isDynamic) {
        hasDynamicNumpyBroadcast = true;
      }
      if (!elideDynamicBroadcastCheck) {
        Value isValid = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, shapeValue, zero);
        rewriter.create<cf::AssertOp>(
            loc, isValid,
            rewriter.getStringAttr(
                "unimplemented: dynamic negative broadcast sizes"));
      }
    } else {
      dim = getDimOp(rewriter, loc, input, j);
    }
    // We can safely assume this dimension is not broadcasted with strict
    // symbols.
    broadcastedStatus.push_back(false);
    outShape.push_back(dim);
  }

  Value outTensor =
      rewriter.create<tensor::EmptyOp>(loc, outShape, elementType);

  // If we know there are no ? -> ? broadcasted dims, or we are assuming
  // strict symbols, we can safely use standard linalg style broadcasting
  // semantics.
  if (!hasDynamicNumpyBroadcast || elideDynamicBroadcastCheck) {
    // If no dims are broadcasted and the rank doesn't change, we can just fold
    // the op away entirely.
    if (!llvm::any_of(broadcastedStatus, [](bool b) { return b; }) &&
        inputRank == outputRank) {
      result = rewriter.create<tensor::CastOp>(loc, outTensor.getType(), input);
      return success();
    }

    SmallVector<AffineExpr> inputExprs;
    for (int64_t i = 0, e = inputRank; i < e; ++i) {
      if (broadcastedStatus[i]) {
        inputExprs.push_back(rewriter.getAffineConstantExpr(0));
        continue;
      }
      inputExprs.push_back(rewriter.getAffineDimExpr(i + diff));
    }

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(outputRank, 0, inputExprs, rewriter.getContext()),
        rewriter.getMultiDimIdentityMap(outputRank)};
    SmallVector<utils::IteratorType> iteratorTypes(
        outputRank, utils::IteratorType::parallel);
    result = rewriter
                 .create<linalg::GenericOp>(
                     loc, outTensor.getType(), input, outTensor, indexingMaps,
                     iteratorTypes,
                     [&](OpBuilder &b, Location loc, ValueRange args) {
                       b.create<linalg::YieldOp>(loc, args[0]);
                     })
                 .getResult(0);
    return success();
  }

  // Fall back to numpy-style dynamic broadcasting in the form of a single
  // linalg op.
  SmallVector<AffineMap> indexingMaps = {
      rewriter.getMultiDimIdentityMap(outputRank)};
  SmallVector<utils::IteratorType> iteratorTypes(outputRank,
                                                 utils::IteratorType::parallel);
  result = rewriter
               .create<linalg::GenericOp>(
                   loc, outTensor.getType(), ValueRange(), outTensor,
                   indexingMaps, iteratorTypes,
                   [&](OpBuilder &b, Location loc, ValueRange args) {
                     // `loopIndices` contains IV of the linalg loops which
                     // would be used to extract values from the input tensor
                     // later on.
                     SmallVector<Value> loopIndices;
                     for (size_t i = 0, e = outputRank; i < e; ++i) {
                       if (i < diff)
                         continue;
                       loopIndices.push_back(b.create<linalg::IndexOp>(loc, i));
                     }
                     // `inputIndicesToExtract` contains i-th linalg loop IV if
                     // the i-th input dimension is not 1, else it contains a
                     // zero index.
                     SmallVector<Value> inputIndicesToExtract;
                     for (size_t i = 0, n = inputRank; i < n; i++) {
                       if (inputShape[i] == 1) {
                         inputIndicesToExtract.push_back(zeroIndex);
                       } else {
                         Value inputDim = getDimOp(b, loc, input, i);
                         Value isEqual = b.create<arith::CmpIOp>(
                             loc, arith::CmpIPredicate::eq, inputDim, oneIndex);
                         Value select = rewriter.create<arith::SelectOp>(
                             loc, isEqual, zeroIndex, loopIndices[i]);
                         inputIndicesToExtract.push_back(select);
                       }
                     }
                     // Extract and yield the value from input tensor at
                     // `inputIndicesToExtract` indices.
                     Value result = b.create<tensor::ExtractOp>(
                         loc, input, inputIndicesToExtract);
                     b.create<linalg::YieldOp>(loc, result);
                   })
               .getResult(0);

  return success();
}

Value torch_to_linalg::removeSizeInformation(OpBuilder &b, Location loc,
                                             Value tensor) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  auto rank = tensorType.getRank();
  SmallVector<int64_t> unknownSizes(rank, kUnknownSize);
  return b.create<tensor::CastOp>(
      loc, tensorType.clone(makeShapeLLVMCompatible(unknownSizes)), tensor);
}

Value torch_to_linalg::convertTensorToElementType(OpBuilder &b, Location loc,
                                                  Value tensor,
                                                  Type elementType) {
  auto dtypePromoteBody = [&](OpBuilder &builder, Location loc,
                              ValueRange payloadArgs) {
    Value elem =
        convertScalarToDtype(builder, loc, payloadArgs[0], elementType);
    builder.create<linalg::YieldOp>(loc, elem);
  };
  return torch_to_linalg::createElementwiseLinalgGeneric(
      b, loc, {tensor}, elementType, dtypePromoteBody);
}

FailureOr<Type> torch_to_linalg::getBackendTypeForScalarType(
    MLIRContext *context, torch_upstream::ScalarType dtypeInt) {
  FailureOr<Type> maybeType =
      getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
  if (failed(maybeType)) {
    return failure();
  }
  Type type = *maybeType;
  // The linalg-on-tensors backend currently expects integers to be signless.
  if (auto intType = type.dyn_cast<IntegerType>()) {
    type = IntegerType::get(context, intType.getWidth(), IntegerType::Signless);
  }
  return type;
}

bool torch_to_linalg::isUnsignedTorchType(Type type) {
  if (auto tty = dyn_cast<ValueTensorType>(type))
    return isUnsignedTorchType(tty.getDtype());
  if (isa<mlir::FloatType>(type))
    return false;
  if (isa<QInt8Type>(type))
    return false;
  if (isa<QUInt8Type>(type))
    return true;
  if (isa<QInt32Type>(type))
    return false;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.isUnsigned();
  llvm_unreachable("Unknown type checked for signedness");
  return false;
}
