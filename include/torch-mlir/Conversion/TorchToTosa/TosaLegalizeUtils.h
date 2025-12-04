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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"            // from @llvm-project
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
                   int64_t input_zp, int64_t output_zp,
                   tosa::RoundingMode rounding_mode, bool scale32);

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
                          Value input_val, double input_scale,
                          int64_t input_zp);

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

FailureOr<Value> getConvBiasForNoneType(Operation *op,
                                        PatternRewriter &rewriter,
                                        Type inputElemTy, Type outputElemTy,
                                        int64_t numOutputChannels);

// Emit an explicit zero-valued `tosa.pad` around an NHWC tensor so that later
// avg_pool lowering can run with `pad = 0`. `padExtents` is ordered as
// {top, bottom, left, right}. Returns the padded tensor value.
Value emitExplicitZeroPadNHWC(Location loc, PatternRewriter &rewriter,
                              Operation *op, Value inputNHWC,
                              ArrayRef<int64_t> padExtents);

// Get the zero point from a torch.tensor or torch.qtensor value.
// If the value is a quantized tensor, it extracts the zero point as a
// scalar integer value. If the value is a float tensor, it returns a
// constant 0.
FailureOr<Value> getZeroPointValue(PatternRewriter &rewriter, Operation *op,
                                   Value tensor, Type elemType);

} // namespace tosa
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZEUTILS_H
