//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

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
void mlir::torch::onnx_c::populateDefaultDomainAtoF(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp("Abs", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAbsOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  // TODO: Acos unimplemented in torch-mlir
  // TODO: Acosh unimplemented in torch-mlir
  // Add became forward compatible with Torch in version 7.
  patterns.onOp("Add", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  Value const1 = rewriter.create<Torch::ConstantIntOp>(
                      binder.getLoc(), rewriter.getType<Torch::IntType>(),
                      rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
                  rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
                      binder.op, resultType, lhs, rhs, const1);
                  return success();
                });
  // TODO: AffineGrid
  patterns.onOp("And", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenLogicalAndOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "ArgMax", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        bool keepDims;
        int64_t axis;
        bool selectLastIndex;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.s64BoolAttr(keepDims, "keepdims", true) ||
            binder.s64IntegerAttr(axis, "axis", 0) ||
            binder.s64BoolAttr(selectLastIndex, "select_last_index", false))
          return failure();

        if (selectLastIndex) {
          // TODO: Figure out how to support this case. Need to add a reverse
          // or something.
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: select_last_index=true");
        }

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(operand.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value constKeepDims = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(keepDims));
        rewriter.replaceOpWithNewOp<Torch::AtenArgmaxOp>(
            binder.op, resultType, operand, constAxis, constKeepDims);
        return success();
      });
  patterns.onOp(
      "ArgMin", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        bool keepDims;
        int64_t axis;
        bool selectLastIndex;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.s64BoolAttr(keepDims, "keepdims", true) ||
            binder.s64IntegerAttr(axis, "axis", 0) ||
            binder.s64BoolAttr(selectLastIndex, "select_last_index", false))
          return failure();

        if (selectLastIndex) {
          // TODO: Figure out how to support this case. Need to add a reverse
          // or something.
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: select_last_index=true");
        }

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(operand.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value constKeepDims = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(keepDims));
        rewriter.replaceOpWithNewOp<Torch::AtenArgminOp>(
            binder.op, resultType, operand, constAxis, constKeepDims);
        return success();
      });
  // TODO: Asin unimplemented in torch-mlir
  // TODO: Asinh unimplemented in torch-mlir
  // TODO: Atanh unimplemented in torch-mlir
  patterns.onOp("Atan", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAtanOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "BitShift", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        std::string direction;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(direction, "direction", ""))
          return failure();
        if (direction == "LEFT") {
          rewriter.replaceOpWithNewOp<Torch::AtenBitwiseLeftShiftTensorOp>(
              binder.op, resultType, lhs, rhs);
        } else {
          rewriter.replaceOpWithNewOp<Torch::AtenBitwiseRightShiftTensorOp>(
              binder.op, resultType, lhs, rhs);
        }
        return success();
      });
  patterns.onOp(
      "BitwiseAnd", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        std::string direction;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenBitwiseAndTensorOp>(
            binder.op, resultType, lhs, rhs);
        return success();
      });
  patterns.onOp("BitwiseNot", 18,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenBitwiseNotOp>(
                      binder.op, resultType, operand);
                  return success();
                });
}
