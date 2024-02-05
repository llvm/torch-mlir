//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"

namespace mlir {
namespace torch {
namespace torch_to_linalg {

struct ReductionOpInfo {
  bool keepDim;
  Value tensorOperand;
  DenseSet<int64_t> dimSet;
};

// Helper function to get the padding tensor given the padding int values.
Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                      SmallVectorImpl<int64_t> &lowPaddingInts,
                      SmallVectorImpl<int64_t> &highPaddingInts, Value pad);

// Helper function to get the padding tensor given the padding int values.
// It's assumed that the padding on the low end and high end are the same,
// and that zero padding is required.
Value getZeroPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                          SmallVectorImpl<int64_t> &paddingInts);

// Helper function that adds dynamic padding to a tensor, ignoring unpaddedDims
// dimensions at the beginning. The high and low padding are the same, and the
// padding value is zero.
Value getDynamicZeroPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                                 SmallVectorImpl<Value> &padding,
                                 int unpaddedDims = 0, Value pad = {});

// Helper function to caculate the output tensor dims for convolution-like ops.
// Along each dim:
// dim_out =
//  floor((dim_in + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1
Value getOutputDimForConvOps(OpBuilder &b, Location loc, Value in,
                             Value paddingInt, Value dilationInt,
                             Value kernelSizeInt, Value strideInt,
                             bool ceilMode = false);

// As above but for transposed convolution ops
// Along each dim:
// dim_out =
//  (dim_in - 1) * stride - 2 * padding + dilation * (kernelSize - 1) +
//  output_padding + 1
Value getOutputDimForConvTransposeOps(OpBuilder &b, Location loc, Value in,
                                      Value paddingInt, Value dilationInt,
                                      Value kernelSizeInt, Value strideInt,
                                      Value outputPaddingInt);

// Create a reduction of `opInfo.tensorOperand`, reducing along the dimensions
// in `opInfo.dimSet`. If `opInfo.keepDim` is true, the output tensor is the
// same rank as the `opInfo.tensorOperand` and reduced dimensions are set to
// size 1. `initElem` is the element used to initialize the output tensor where
// the reduction will be stored.
Value createReductionLinalgGeneric(
    OpBuilder &b, Location loc, const ReductionOpInfo &opInfo, Value initElem,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild);

// Create a pointwise operation that uses values in `tensorOperands`, such that
// the element type of the resulting tensor is `resultElementType`.
Value createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    Type resultElementType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild);

// Broadcasts input tensor based on the broadcastToShape.
LogicalResult broadcastToGivenShape(Operation *op, PatternRewriter &rewriter,
                                    Value input,
                                    SmallVector<Value> broadcastToShape,
                                    RankedTensorType broadcastType,
                                    Value &result,
                                    SmallVector<bool> useBroadcastToShape = {});

// Cast a tensor to a rank-equivalent tensor of unknown size, i.e. <1x2xf32> ->
// <?x?xf32>
Value removeSizeInformation(OpBuilder &b, Location loc, Value tensor);

// Converts a tensor' element type to the specified `elementType`.
Value convertTensorToElementType(OpBuilder &b, Location loc, Value tensor,
                                 Type elementType);

// Convert a scalar type to the corresponding builtin type in the
// linalg-on-tensors backend.
FailureOr<Type>
getBackendTypeForScalarType(MLIRContext *context,
                            torch_upstream::ScalarType dtypeInt);

bool isUnsignedTorchType(Type type);

} // namespace torch_to_linalg
} // namespace torch
} // namespace mlir
