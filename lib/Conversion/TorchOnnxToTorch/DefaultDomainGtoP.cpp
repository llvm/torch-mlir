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
void mlir::torch::onnx_c::populateDefaultDomainGtoP(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp("MatMul", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Greater", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenGtTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("GreaterOrEqual", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenGeTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Less", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLtTensorOp>(
                      binder.op, resultType, lhs, rhs);
		              return success();
                });
  patterns.onOp("LessOrEqual", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLeTensorOp>(
                      binder.op, resultType, lhs, rhs);
		              return success();
                });
  patterns.onOp(
      "GatherElements", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices;
        int64_t axis;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(axis, "axis", 0))
          return failure();
        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value sparseGrad = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<Torch::AtenGatherOp>(
            binder.op, resultType, data, constAxis, indices, sparseGrad);
        return success();
      });
  patterns.onOp(
      "Gemm", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value a, b, c;
        float alpha, beta;
        int64_t transA, transB;
        if (binder.tensorOperandAtIndex(a, 0) ||
            binder.tensorOperandAtIndex(b, 1) ||
            binder.tensorOperandAtIndex(c, 2) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.f32FloatAttr(alpha, "alpha", 1.0) ||
            binder.f32FloatAttr(beta, "beta", 1.0) ||
            binder.tensorResultType(resultType))
          return failure();

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

        auto transpose = [&](Value m) -> Value {
          auto tty = m.getType().cast<Torch::ValueTensorType>();
          auto shape = tty.getOptionalSizes();
          if (shape.has_value()) {
            llvm::SmallVector<int64_t> newShape(shape.value());
            std::reverse(newShape.begin(), newShape.end());
            shape = std::move(newShape);
          }
          auto oty = Torch::ValueTensorType::get(tty.getContext(), shape,
                                                 tty.getOptionalDtype());
          return rewriter.create<Torch::AtenTransposeIntOp>(binder.getLoc(),
                                                            oty, m, zero, one);
        };

        if (transA) {
          a = transpose(a);
        }

        if (transB) {
          b = transpose(b);
        }

        Value mm =
            rewriter.create<Torch::AtenMmOp>(binder.getLoc(), resultType, a, b);
        if (alpha == 1.0 && beta == 1.0) {
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, mm, c, one);
          return success();
        }

        if (alpha != 1.0 && beta != 1.0) {
          Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
              binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(alpha));
          mm = rewriter.create<Torch::AtenMulScalarOp>(
              binder.getLoc(), resultType, mm, constAlpha);
          alpha = 1.0;
        }

        if (alpha != 1.0) {
          std::swap(alpha, beta);
          std::swap(mm, c);
        }

        Value constBeta = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));
        rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
            binder.op, resultType, mm, c, constBeta);
        return success();
      });
  patterns.onOp("LeakyRelu", 16,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  float alpha;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType) ||
                      binder.f32FloatAttr(alpha, "alpha", 0.01))
                    return failure();
                  Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                      rewriter.getF64FloatAttr(alpha));
                  rewriter.replaceOpWithNewOp<Torch::AtenLeakyReluOp>(
                      binder.op, resultType, operand, constAlpha);
                  return success();
                });
}