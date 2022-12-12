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

Value torch_to_tcp::broadcastRankInLeadingDims(
    ConversionPatternRewriter &rewriter, Value input, int64_t rankIncrease) {
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();

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
