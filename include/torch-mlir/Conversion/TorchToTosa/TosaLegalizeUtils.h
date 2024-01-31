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

#include "mlir/Dialect/Quant/QuantTypes.h"        // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"                    // from @llvm-project

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from input framework scaling, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
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

LogicalResult tosaCastTensorToType(PatternRewriter &rewriter, Operation *op,
                                   Value src, Type destType, Value &result);

Value promoteType(PatternRewriter &rewriter, Value input, TensorType outType);

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference during the framework to TOSA lowering.
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInfer(PatternRewriter &rewriter, Location loc, Type result_ty,
                        Args &&...args) {
  auto op = rewriter.create<TosaOp>(loc, result_ty, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), op.getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(result_ty);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = result_ty.cast<ShapedType>().getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge = ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  auto new_ty = newKnowledge.getType();
  result.setType(new_ty);
  return op;
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

} // namespace tosa
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZEUTILS_H
