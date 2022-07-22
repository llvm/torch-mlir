//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H
#define TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

using mlir::ConversionPatternRewriter;

// Create a 32-bit float constant operator from a float
Value getMhloConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val);

// Create a 64-bit float constant operator from a double
Value getMhloConstTensorSingleF64(PatternRewriter &rewriter, Operation *op,
                                  double val);

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 MHLO constant, need to pass in llvm::APInt instead.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<T> vec, ArrayRef<int64_t> shape);

template <typename T>
Value getSplatConstTensor(ConversionPatternRewriter &rewriter, Operation *op,
                          T val, Type dtype, llvm::ArrayRef<int64_t> dshape);

LogicalResult torchScalarToMhloTensor(ConversionPatternRewriter &rewriter,
                                      Operation *op, Value torchScalarValue,
                                      Value &mhloTensor, Type dtype,
                                      llvm::ArrayRef<int64_t> dshape,
                                      bool doBroadcast = true);

LogicalResult torchAlphaToMhloTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value alphaScalar,
                                     Value &alphaTensor, Type dtype,
                                     llvm::ArrayRef<int64_t> dshape,
                                     bool checkForUnity);

Value promoteAndBroadcast(ConversionPatternRewriter &rewriter, Value input,
                          TensorType outType);
} // namespace mhlo
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMHLO_MHLOLEGALIZEUTILS_H
