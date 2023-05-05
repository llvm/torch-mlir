//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// The parameter input is expected to be of RankedTensorType.
Value torch_to_tcp::broadcastRankInLeadingDims(
    ConversionPatternRewriter &rewriter, Value input, int64_t rankIncrease) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();

  SmallVector<ReassociationExprs> reassociationMap(inputType.getRank());
  if (inputType.getRank() > 0) {
    for (int64_t axis = 0; axis < rankIncrease; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    for (int64_t inputAxis = 0; inputAxis < inputType.getRank(); ++inputAxis)
      reassociationMap[inputAxis].push_back(
          rewriter.getAffineDimExpr(inputAxis + rankIncrease));
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape(rankIncrease, 1);
  resultShape.insert(resultShape.end(), inputShape.begin(), inputShape.end());
  auto resultType =
      inputType.cloneWith(ArrayRef(resultShape), inputType.getElementType());

  return rewriter.create<tensor::ExpandShapeOp>(
      input.getDefiningOp()->getLoc(), resultType, input, reassociationMap);
}

// The parameters input and target are expected to be of RankedTensorType.
Value torch_to_tcp::broadcastShapeInLeadingDims(
    ConversionPatternRewriter &rewriter, Value input, Value target,
    int64_t numLeadingAxes) {
  Operation *op = input.getDefiningOp();
  SmallVector<int64_t> axes;
  SmallVector<Value> dimSizes;
  for (int64_t axis = 0; axis < numLeadingAxes; ++axis) {
    axes.push_back(axis);
    dimSizes.push_back(
        rewriter.createOrFold<tensor::DimOp>(op->getLoc(), target, axis));
  }

  auto axesAttr = rewriter.getI64ArrayAttr(axes);
  return rewriter.create<tcp::BroadcastOp>(op->getLoc(), target.getType(),
                                           input, dimSizes, axesAttr);
}

// The parameters input and target are expected to be of RankedTensorType.
Value torch_to_tcp::broadcastInLeadingDimsToMatchShape(
    ConversionPatternRewriter &rewriter, Value input, Value target) {
  RankedTensorType targetType = target.getType().cast<RankedTensorType>();
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();

  Value result = input;
  if (inputType.getRank() < targetType.getRank()) {
    int64_t rankIncrease = targetType.getRank() - inputType.getRank();
    result = torch_to_tcp::broadcastRankInLeadingDims(rewriter, result,
                                                      rankIncrease);
    result = torch_to_tcp::broadcastShapeInLeadingDims(rewriter, result, target,
                                                       rankIncrease);
  }

  return result;
}

Value torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
    ConversionPatternRewriter &rewriter, Value input, Value target,
    Type resultType, int64_t axisInOutput) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  RankedTensorType targetType = target.getType().cast<RankedTensorType>();

  auto inputRank = inputType.getRank();
  auto targetRank = targetType.getRank();

  // This utility only accepts 0D and 1D inputs
  assert(inputRank < 2 && "Only 0D and 1D tensors are supported!");

  Value result = input;

  // First: Broadcast Rank
  // Case 1: 0D -> ND
  // [] -> [1, 1, 1, 1]
  // reassociation map = [[]]
  // Case 2: 1D -> ND
  // [C] -> [1, C, 1, 1] if axisInOutput = 1
  // reassociation map = [[0, 1, 2, 3]]
  SmallVector<ReassociationExprs> reassociationMap(inputRank);
  SmallVector<int64_t> resultShape(targetRank, 1);
  if (inputRank == 1) {
    for (int64_t axis = 0; axis < targetRank; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    resultShape[axisInOutput] = inputType.getShape()[0];
  }
  Type expandResultType =
      targetType.cloneWith(ArrayRef(resultShape), resultType);
  result = rewriter.create<tensor::ExpandShapeOp>(
      result.getDefiningOp()->getLoc(), expandResultType, input,
      reassociationMap);

  // Second: Broadcast Shape
  // Case 1: 0D -> ND
  // [1, 1, 1, 1] -> [N, C, H, W]
  // Second: Broadcast Shape
  // Case 2: 1D -> ND
  // [1, C, 1, 1] -> [N, C, H, W]
  SmallVector<int64_t> axes;
  SmallVector<Value> dimSizes;
  for (int64_t axis = 0; axis < targetRank; ++axis) {
    if (inputRank == 0 || axis != axisInOutput) {
      axes.push_back(axis);
      dimSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          result.getDefiningOp()->getLoc(), target, axis));
    }
  }
  auto axesAttr = rewriter.getI64ArrayAttr(axes);

  Type broadcastResultType =
      targetType.cloneWith(targetType.getShape(), resultType);
  result = rewriter.create<tcp::BroadcastOp>(result.getDefiningOp()->getLoc(),
                                             broadcastResultType, result,
                                             dimSizes, axesAttr);

  return result;
}

SmallVector<int64_t>
torch_to_tcp::getShapeFromPrimList(ArrayRef<Value> listVal) {
  SmallVector<int64_t> resultShape;
  for (Value value : listVal) {
    int64_t num;
    if (matchPattern(value, m_TorchConstantInt(&num)))
      resultShape.push_back(num);
    else
      resultShape.push_back(ShapedType::kDynamic);
  }
  return resultShape;
}

Value torch_to_tcp::broadcast0DOr1DFromShape(
    ConversionPatternRewriter &rewriter, Value input, ArrayRef<Value> targetVal,
    SmallVector<int64_t> resultShape, int64_t axisInOutput) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto inputRank = inputType.getRank();
  RankedTensorType targetType = input.getType().cast<RankedTensorType>();

  int64_t targetRank = 0;
  SmallVector<Value> dimSizes;
  for (Value value : targetVal) {
    targetRank++;
    Value newDimSize = rewriter.create<torch::TorchConversion::ToI64Op>(
        input.getDefiningOp()->getLoc(), value);
    dimSizes.push_back(rewriter.create<arith::IndexCastOp>(
        input.getDefiningOp()->getLoc(), rewriter.getIndexType(), newDimSize));
  }

  SmallVector<ReassociationExprs> reassociationMap(inputRank);
  SmallVector<int64_t> expandShape(targetRank, 1);

  if (inputRank == 1) {
    for (int64_t axis = 0; axis < targetRank; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    resultShape[axisInOutput] = inputType.getShape()[0];
    expandShape[axisInOutput] = inputType.getShape()[0];
  }

  Value result = input;
  auto resultType =
      targetType.cloneWith(ArrayRef(expandShape), targetType.getElementType());
  result = rewriter.create<tensor::ExpandShapeOp>(
      result.getDefiningOp()->getLoc(), resultType, input, reassociationMap);

  SmallVector<int64_t> axes;
  for (int64_t axis = 0; axis < targetRank; ++axis) {
    if (inputRank == 0 || axis != axisInOutput) {
      axes.push_back(axis);
    }
  }
  auto axesAttr = rewriter.getI64ArrayAttr(axes);
  resultType =
      targetType.cloneWith(ArrayRef(resultShape), targetType.getElementType());
  result = rewriter.create<tcp::BroadcastOp>(
      result.getDefiningOp()->getLoc(), resultType, result, dimSizes, axesAttr);

  return result;
}

// TODO: Add unit tests for all getConstTensor* functions below
template <typename T>
std::optional<Value>
torch_to_tcp::impl::getConstTensorUtil(PatternRewriter &rewriter, Operation *op,
                                       ArrayRef<T> vec, ArrayRef<int64_t> shape,
                                       RankedTensorType type) {
  uint64_t numTotalElements = 1;
  for (int64_t a : shape) {
    assert(a >= 0 && "getConstTensor(): Only static shapes supported");
    numTotalElements *= a;
  }

  if (vec.size() != numTotalElements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto constAttr = DenseElementsAttr::get(type, vec);

  auto constOp = rewriter.create<tcp::ConstOp>(op->getLoc(), type, constAttr);
  return constOp.getResult();
}

template <typename T>
std::optional<Value>
torch_to_tcp::getConstTensor(PatternRewriter &rewriter, Operation *op,
                             ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  auto constType =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));

  return torch_to_tcp::impl::getConstTensorUtil<T>(rewriter, op, vec, shape,
                                                   constType);
}

template std::optional<Value>
torch_to_tcp::getConstTensor<int8_t>(PatternRewriter &, Operation *,
                                     ArrayRef<int8_t> vec,
                                     ArrayRef<int64_t> shape);

template std::optional<Value>
torch_to_tcp::getConstTensor<int32_t>(PatternRewriter &, Operation *,
                                      ArrayRef<int32_t> vec,
                                      ArrayRef<int64_t> shape);

template std::optional<Value>
torch_to_tcp::getConstTensor<int64_t>(PatternRewriter &, Operation *,
                                      ArrayRef<int64_t> vec,
                                      ArrayRef<int64_t> shape);

template <>
std::optional<Value>
torch_to_tcp::getConstTensor<float>(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<float> vec,
                                    ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(shape, rewriter.getF32Type());

  return torch_to_tcp::impl::getConstTensorUtil<float>(rewriter, op, vec, shape,
                                                       constType);
}

template <>
std::optional<Value>
torch_to_tcp::getConstTensor<double>(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<double> vec,
                                     ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(shape, rewriter.getF64Type());

  return torch_to_tcp::impl::getConstTensorUtil<double>(rewriter, op, vec,
                                                        shape, constType);
}

template <>
std::optional<Value>
torch_to_tcp::getConstTensor<APInt>(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<APInt> vec,
                                    ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));

  return torch_to_tcp::impl::getConstTensorUtil<APInt>(rewriter, op, vec, shape,
                                                       constType);
}

bool torch_to_tcp::getConstTensorWithType(ConversionPatternRewriter &rewriter,
                                          Operation *op, Value &constOp,
                                          Type resultType, int fillVal) {
  if (resultType.isInteger(64)) {
    constOp = *torch_to_tcp::getConstTensor<int64_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int64_t>(fillVal)), {});
  } else if (resultType.isInteger(32)) {
    constOp = *torch_to_tcp::getConstTensor<int32_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int32_t>(fillVal)), {});
  } else if (resultType.isInteger(8)) {
    constOp = *torch_to_tcp::getConstTensor<int8_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int8_t>(fillVal)), {});
  } else if (resultType.isF32()) {
    constOp = *torch_to_tcp::getConstTensor<float>(
        rewriter, op, llvm::ArrayRef(static_cast<float>(fillVal)), {});
  } else if (resultType.isF64()) {
    constOp = *torch_to_tcp::getConstTensor<double>(
        rewriter, op, llvm::ArrayRef(static_cast<double>(fillVal)), {});
  } else {
    return false;
  }
  return true;
}
