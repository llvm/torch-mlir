//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_STABLEHLOLEGALIZEUTILS_H
#define TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_STABLEHLOLEGALIZEUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hlo {

using mlir::ConversionPatternRewriter;

// Create chlo::ConstantLikeOp
template <typename T>
Value getConstantLike(OpBuilder &rewriter, Location loc, T constant, Value val);

// Create a 32-bit float constant operator from a float
Value getStablehloConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                       float val);

// Create a 64-bit float constant operator from a double
Value getStablehloConstTensorSingleF64(PatternRewriter &rewriter, Operation *op,
                                       double val);

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 StableHLO constant, need to pass in llvm::APInt instead.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape);

template <typename T>
Value getSplatConstTensor(ConversionPatternRewriter &rewriter, Operation *op,
                          T val, Type dtype, llvm::ArrayRef<int64_t> dshape);

Value scalarToStablehloTensor(ConversionPatternRewriter &rewriter,
                              Operation *op, Value scalarValue, Type dtype);

Value promoteType(PatternRewriter &rewriter, Location loc, Value input,
                  TensorType outType);

Value promoteAndBroadcast(ConversionPatternRewriter &rewriter, Value input,
                          TensorType outType);

SmallVector<int64_t> toPositiveDims(ArrayRef<int64_t> dims, int64_t rank);

// Get the dimension sizes of the input tensor, given the dimension axes
FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     ArrayRef<int64_t> inpDims,
                                                     size_t dimSizeIndexBits);

// Get the dimension sizes of the input tensor
FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     size_t dimSizeIndexBits);

// Get the dimension sizes of the input tensor, given the dimension axes
FailureOr<SmallVector<Value, 4>> getDimIndexOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     ArrayRef<int64_t> inpDims);

// Get the dimension sizes of the input tensor
FailureOr<SmallVector<Value, 4>>
getDimIndexOfTensor(PatternRewriter &rewriter, Operation *op, Value value);

// Get a tensor that unsqueezed the specified dimensions of the input tensor
FailureOr<Value> unsqueezeTensor(PatternRewriter &rewriter, Operation *op,
                                 Value tensor, ArrayRef<int64_t> inputUnsqzDims,
                                 size_t dimSizeIndexBits);

// Get a tensor that collapse the specified dimensions of the input tensor
FailureOr<Value> collapseTensor(PatternRewriter &rewriter, Operation *op,
                                Value tensor, int64_t collapseStartDim,
                                int64_t collapseEndDim,
                                size_t dimSizeIndexBits);

// Get a tensor that splits the specified dimensions of the input tensor
FailureOr<Value> splitTensor(PatternRewriter &rewriter, Operation *op,
                             Value tensor, int64_t splitDim,
                             int64_t outerLength, size_t dimSizeIndexBits);

Value getConstantOfShape(PatternRewriter &rewriter, Location loc,
                         const APFloat &constant, Value shape,
                         TensorType outType);
} // namespace hlo
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_STABLEHLOLEGALIZEUTILS_H
