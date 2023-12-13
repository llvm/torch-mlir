//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

// Simple rewrites for the default domain.
// See: https://onnx.ai/onnx/operators/
// For operators that are effectively version invariant, we register with
// sinceVersion==1. We interpret this to include the following spec
// diffs that are irrelevant to this level of lowering:
//   * Supported element types.
//   * Limited broadcasting to full broadcasting support.
//
// There are a lot of spec revisions that basically generalized elementwise
// to be more normal and a direct translation vs a special case. This
// results in a lot of ONNX test cases that all reduce to the exact same
// thing here, so we simplify.
void mlir::torch::onnx_c::populateDefaultDomainQtoZ(
    OnnxCustomOpConversionPattern &patterns) {

  patterns.onOp(
      "Selu", 6, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        float alpha, gamma;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.f32FloatAttr(alpha, "alpha") ||
            binder.f32FloatAttr(gamma, "gamma") ||
            binder.tensorResultType(resultType))
          return failure();

        Value vAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), alpha));

        Value vScale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), gamma));

        Value vInputScale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getFloatAttr(rewriter.getF64Type(), 1.0));

        rewriter.replaceOpWithNewOp<Torch::AtenEluOp>(
            binder.op, resultType, operand, vAlpha, vScale, vInputScale);
        return success();
      });

  patterns.onOp(
      "Transpose", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        auto loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();
        auto operandType = operand.getType().cast<Torch::ValueTensorType>();
        TensorType tensorType = operandType.toBuiltinTensor();
        if (!tensorType || !tensorType.hasRank())
          return failure();

        // Default permutation is to reverse orders:
        int64_t rank = tensorType.getRank();
        llvm::SmallVector<int64_t> reverse(rank);
        for (int64_t i = 0; i < rank; ++i) {
          reverse[i] = rank - i - 1;
        }

        llvm::SmallVector<int64_t> permutations;
        if (failed(binder.s64IntegerArrayAttr(permutations, "perm", reverse)))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to obtain permutations");

        if (static_cast<int64_t>(permutations.size()) != rank)
          return rewriter.notifyMatchFailure(
              binder.op, "Permutation length does not match operand rank");

        llvm::SmallVector<int64_t> shape(tensorType.getShape());
        llvm::SmallVector<int64_t> current(rank);
        for (int64_t i = 0; i < rank; ++i) {
          current[i] = i;
        }

        for (int64_t i = 0; i < rank; ++i) {
          if (current[i] == permutations[i])
            continue;

          int64_t target = i + 1;
          for (; target < rank; ++target) {
            if (current[target] == permutations[i])
              break;
          }

          std::swap(shape[i], shape[target]);
          std::swap(current[i], current[target]);

          Value dim0 = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));

          Value dim1 = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), target));

          operand = rewriter.create<Torch::AtenTransposeIntOp>(
              loc,
              Torch::ValueTensorType::get(tensorType.getContext(), shape,
                                          operandType.getOptionalDtype()),
              operand, dim0, dim1);
        }

        rewriter.replaceOp(binder.op, operand);

        return success();
      });
}
