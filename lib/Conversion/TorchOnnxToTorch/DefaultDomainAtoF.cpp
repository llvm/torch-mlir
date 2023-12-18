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
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

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
  patterns.onOp("Acos", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAcosOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "AveragePool", 19,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        SmallVector<int64_t> dilation;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET") {
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");
        }
        if (binder.s64IntegerArrayAttr(dilation, "dilations", {})) {
          return failure();
        }
        if (dilation.size() > 0) {
          return rewriter.notifyMatchFailure(
              binder.op, "dilation is not supported by torch.aten.avgpool op");
        }

        Torch::ValueTensorType resultType;
        Value operand;
        bool ceilMode, countIncludePad;
        if (binder.tensorOperand(operand) ||
            binder.s64BoolAttr(ceilMode, "ceil_mode", false) ||
            binder.s64BoolAttr(countIncludePad, "count_include_pad", false) ||
            binder.tensorResultType(resultType))
          return failure();
        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(operand);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;

        SmallVector<int64_t> kernel, padding, strides;
        if (binder.s64IntegerArrayAttr(kernel, "kernel_shape", {})) {
          return failure();
        }
        if (kernel.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "kernel list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(padding, "pads", {0})) {
          return failure();
        }
        if (padding.size() != 1 && padding.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "padding list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(strides, "strides", {1})) {
          return failure();
        }
        if (strides.size() != 1 && strides.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        }

        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        for (int64_t i : kernel) {
          cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        for (int64_t i : padding) {
          cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        for (int64_t i : strides) {
          cstStrides.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        Value kernelSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceilMode);
        Value cstCountIncludePad = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), countIncludePad);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        if (rank == 3) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool1dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad);
          return success();
        } else if (rank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        } else if (rank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        }
        return failure();
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
  patterns.onOp(
      "BitwiseOr", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        std::string direction;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenBitwiseOrTensorOp>(
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
  patterns.onOp(
      "BitwiseXor", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        std::string direction;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenBitwiseXorTensorOp>(
            binder.op, resultType, lhs, rhs);
        return success();
      });
  patterns.onOp(
      "Cast", 19, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t dtypeIntOnnx, dtypeIntTorch;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "to") ||
            binder.tensorResultType(resultType))
          return failure();

        // TODO: Add complete mapping.
        switch (dtypeIntOnnx) {
        case 1:
          dtypeIntTorch = 6; // float
          break;
        case 10:
          dtypeIntTorch = 5; // half
          break;
        case 11:
          dtypeIntTorch = 7; // double
          break;
        case 16:
          dtypeIntTorch = 15; // bfloat16
          break;
        default:
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    dtypeIntTorch));
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, operand, constDtype,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/none);
        return success();
      });
  patterns.onOp("Ceil", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenCeilOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Clip", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        if (binder.op->getNumOperands() == 1) {
          Value source;
          if (binder.tensorOperand(source) ||
              binder.tensorResultType(resultType))
            return failure();
          Value cstNone =
              rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
          rewriter.replaceOpWithNewOp<Torch::AtenClampOp>(
              binder.op, resultType, source, /*min=*/cstNone, /*max=*/cstNone);
          return success();
        } else if (binder.op->getNumOperands() == 2) {
          Value source, min;
          if (binder.tensorOperands(source, min) ||
              binder.tensorResultType(resultType))
            return failure();
          rewriter.replaceOpWithNewOp<Torch::AtenClampMinTensorOp>(
              binder.op, resultType, source, /*min=*/min);
          return success();
        } else if (binder.op->getNumOperands() == 3) {
          Value source, min, max;
          if (binder.tensorOperandAtIndex(source, 0) ||
              binder.tensorOperandAtIndex(min, 1) ||
              binder.tensorOperandAtIndex(max, 2) ||
              binder.tensorResultType(resultType))
            return failure();
          rewriter.replaceOpWithNewOp<Torch::AtenClampTensorOp>(
              binder.op, resultType, source, min, max);
          return success();
        }
        return failure();
      });
  patterns.onOp("Cos", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenCosOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Div", 14,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Equal", 19,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  std::string direction;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenEqTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("Floor", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenFloorOp>(
                      binder.op, resultType, operand);
                  return success();
                });
}
