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

using namespace mlir;
using namespace mlir::tcp;

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
  auto resultType = inputType.cloneWith(makeArrayRef(resultShape),
                                        inputType.getElementType());

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

template <typename T>
std::optional<Value>
torch_to_tcp::getConstTensor(PatternRewriter &rewriter, Operation *op,
                             ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tcp::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

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
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tcp::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

template <>
std::optional<Value>
torch_to_tcp::getConstTensor<double>(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<double> vec,
                                     ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tcp::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

template <>
std::optional<Value>
torch_to_tcp::getConstTensor<APInt>(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<APInt> vec,
                                    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tcp::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}
