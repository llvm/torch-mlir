//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"       // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h" // from @llvm-project
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from input framework tensor, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getI32ArrayAttr({multiplier}), rewriter.getI32ArrayAttr({shift}),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
                          Value input_val, double input_scale,
                          int64_t input_zp) {
  // Output is always int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
                      input_zp, 0, false, true);
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                     ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
llvm::Optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<APInt> vec,
                                            ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<float> vec,
                                            ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(PatternRewriter &,
                                                       Operation *,
                                                       ArrayRef<int32_t> vec,
                                                       ArrayRef<int64_t> shape);

template llvm::Optional<Value> getConstTensor<int64_t>(PatternRewriter &,
                                                       Operation *,
                                                       ArrayRef<int64_t> vec,
                                                       ArrayRef<int64_t> shape);
} // namespace tosa
} // namespace mlir
