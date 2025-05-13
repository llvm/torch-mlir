//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZEUTILS_H
#define TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"        // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h" // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"      // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"               // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                    // from @llvm-project
#include "mlir/IR/PatternMatch.h"                    // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"    // from @llvm-project
#include "mlir/Support/LLVM.h"                       // from @llvm-project

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from input framework scaling, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, StringRef rounding_mode,
                   bool scale32);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
                          Value input_val, double input_scale,
                          int64_t input_zp);

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter &rewriter, Operation *op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type);

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type);

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val);

// Create an int8_t const tosa.mul shift tensor from an int
Value getTosaMulShiftConstTensor(PatternRewriter &rewriter, Operation *op,
                                 int32_t shift);

// Create a zero constant tensor of the desired type and shape.
std::optional<Value> getZerosLikeTensor(PatternRewriter &rewriter,
                                        Operation *op, Type type);

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
// To create INT48 TOSA constant, need to pass in llvm::APInt instead.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape,
                                    std::optional<Type> dtype = {});

// Default function to create tosa.cast op. This should be called instead of
// directly calling rewriter.create<tosa::CastOp>.
std::optional<Value> tosaCastTensorToType(PatternRewriter &rewriter, Value src,
                                          TensorType destType);

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(ImplicitLocOpBuilder &builder, Type result_ty,
                        Args &&...args) {
  return CreateOpAndInferShape<TosaOp>(builder, result_ty, args...);
}

template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(PatternRewriter &rewriter, Location loc, Type result_ty,
                        Args &&...args) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return CreateOpAndInfer<TosaOp>(builder, result_ty, args...);
}

template <typename TosaOp, typename... Args>
void CreateReplaceOpAndInfer(PatternRewriter &rewriter, Operation *op,
                             Type result_ty, Args &&...args) {
  auto result =
      CreateOpAndInfer<TosaOp>(rewriter, op->getLoc(), result_ty, args...);
  rewriter.replaceOp(op, result->getResults());
}

// Get accumulator type for AvgPool2dOp.
LogicalResult getAvgPool2dAccType(PatternRewriter &rewriter, Value input,
                                  TypeAttr &accType);

// Get accumulator type for TOSA convolution ops
LogicalResult getConvOpsAccType(PatternRewriter &rewriter,
                                RankedTensorType inputTy,
                                RankedTensorType weightTy,
                                RankedTensorType outputTy, TypeAttr &accType);

} // namespace tosa
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZEUTILS_H
