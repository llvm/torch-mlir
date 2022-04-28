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
namespace torch {
namespace torch_to_linalg {

// Helper function to get the padding tensor given the padding int values.
Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                      SmallVectorImpl<int64_t> &lowPaddingInts,
                      SmallVectorImpl<int64_t> &highPaddingInts, Value pad);

// Helper function to get the padding tensor given the padding int values.
// It's assumed that the padding on the low end and high end are the same,
// and that zero padding is required.
Value getZeroPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                          SmallVectorImpl<int64_t> &paddingInts);

// Helper function to caculate the output tensor dims for convolution-like ops.
// Along each dim:
// dim_out =
//  floor((dim_in + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1
Value getOutputDimForConvOps(OpBuilder &b, Location loc, Value in,
                             Value paddingInt, Value dilationInt,
                             Value kernelSizeInt, Value strideInt);

// Create a reduction of `tensorOperand`, reducing along the dimensions
// in `dimSet`. If `keepDim` is true, the output tensor is the same
// rank as the `tensorOperand` and reduced dimensions are set to size 1.
// `initElem` is the element used to initialize the output tensor
// where the reduction will be stored.
Value createReductionLinalgGeneric(
    OpBuilder &b, Location loc, Value tensorOperand,
    const DenseSet<int64_t> &dimSet, bool keepDim, Value initElem,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild);

// Create a pointwise operation that uses values in `tensorOperands`, such that
// the element type of the resulting tensor is `resultElementType`.
Value createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    Type resultElementType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild);
} // namespace torch_to_linalg
} // namespace torch
} // namespace mlir
