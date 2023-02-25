//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch_to_tcp {

// Helper function to expand the rank of the input tensor. Works by
// adding 1-dim shape to the leading dims using `tensor::ExpandShapeOp`.
Value broadcastRankInLeadingDims(ConversionPatternRewriter &rewriter,
                                 Value input, int64_t rankIncrease);

// Helper function to broadcast all 1-dim shapes in input to match
// that of target, without altering the rank, using `tcp::BroadcastOp`.
Value broadcastShapeInLeadingDims(ConversionPatternRewriter &rewriter,
                                  Value input, Value target,
                                  int64_t numLeadingAxes);

// Helper function to do both rank and shape leading-dim broadcasting
// of the input to match target.
Value broadcastInLeadingDimsToMatchShape(ConversionPatternRewriter &rewriter,
                                         Value input, Value target);

// Helper function to broadcast a 0D or 1D input tensor to match rank and shape
// of target. For the 1D case, this projects the input vector to the
// `axisInOutput` in the result.
//
// Case 1: 0D->ND
// Example: [] -> [N, C, H, W]
//   First: Broadcast Rank
//      [] -> [1, 1, 1, 1]
//   Second: Broadcast Shape
//      [1, 1, 1, 1] -> [N, C, H, W]
//
// Case 2: 1D->ND
// Example: [C] -> [N, C, H, W] (`axisInOutput = 1`)
//   First: Broadcast Rank
//      [C] -> [1, C, 1, 1]
//   Second: Broadcast Shape
//      [1, C, 1, 1] -> [N, C, H, W]
Value broadcast0DOr1DToNDAndMatchShape(ConversionPatternRewriter &rewriter,
                                       Value input, Value target,
                                       int64_t axisInOutput = 0);

// Utility function to create a tcp.const op with given content and shape.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape);
namespace impl {
template <typename T>
std::optional<Value>
getConstTensorUtil(PatternRewriter &rewriter, Operation *op, ArrayRef<T> vec,
                   ArrayRef<int64_t> shape, RankedTensorType type);
}

} // namespace torch_to_tcp
} // namespace mlir
