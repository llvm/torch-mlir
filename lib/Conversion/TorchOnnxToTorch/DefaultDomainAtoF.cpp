//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectResourceBlobManager.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

class Endian {
private:
  static constexpr uint32_t uint32_ = 0x01020304;
  static constexpr uint8_t magic_ = (const uint8_t &)uint32_;

public:
  static constexpr bool little = magic_ == 0x04;
  static constexpr bool big = magic_ == 0x01;
  static_assert(little || big, "Cannot determine endianness!");

private:
  Endian() = delete;
};

static int64_t onnxDtypeIntToTorchDtypeInt(int64_t dtypeIntOnnx) {
  // TODO: Add complete mapping.
  // Where are the ONNX and PyTorch dtype enums defined?
  // ONNX:
  //  https://github.com/shouxieai/tensorRT_Pro/blob/main/onnx/onnx-ml.proto
  // PyTorch:
  //  https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/Utils/TorchUpstream.h#L88

  int64_t dtypeIntTorch = [dtypeIntOnnx]() {
    switch (dtypeIntOnnx) {
    case 1:
      return 6; // float
    case 7:
      return 5; // int64
    case 9:
      return 11; // bool
    case 10:
      return 5; // half
    case 11:
      return 7; // double
    case 16:
      return 15; // bfloat16
    default:
      return -1; // No dtype
    }
  }();

  return dtypeIntTorch;
}

static LogicalResult createTorchTransposeOp(ConversionPatternRewriter &rewriter,
                                            Location loc, Value input,
                                            int64_t dimA, int64_t dimB,
                                            Value &transposed) {
  Type transposedType;
  if (failed(getTransposedType(input.getType().cast<Torch::BaseTensorType>(),
                               dimA, dimB, transposedType)))
    return failure();
  Value cstDimA = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimA));
  Value cstDimB = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimB));
  transposed = rewriter.create<Torch::AtenTransposeIntOp>(
      loc, transposedType, input, cstDimA, cstDimB);
  return success();
}

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
  patterns.onOp("Asin", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAsinOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Asinh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAsinhOp>(
                      binder.op, resultType, operand);
                  return success();
                });
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
  patterns.onOp("Atanh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAtanhOp>(
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
  patterns.onOp("Acosh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenAcoshOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("BatchNormalization", 15,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value input, weight, bias, runningMean, runningVar;
                  bool training;
                  float momentum, eps;
                  if (binder.s64BoolAttr(training, "training_mode", 0))
                    return failure();
                  if (training) {
                    // TODO: Add support for training = true
                    return rewriter.notifyMatchFailure(
                        binder.op, "unsupported conversion: training = true");
                  }

                  if (binder.tensorOperandAtIndex(input, 0) ||
                      binder.tensorOperandAtIndex(weight, 1) ||
                      binder.tensorOperandAtIndex(bias, 2) ||
                      binder.tensorOperandAtIndex(runningMean, 3) ||
                      binder.tensorOperandAtIndex(runningVar, 4) ||
                      binder.f32FloatAttr(momentum, "momentum", 0.9f) ||
                      binder.f32FloatAttr(eps, "epsilon", 1e-05f) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
                      binder.getLoc(), false);
                  Value cstMomentum = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getF64FloatAttr(momentum));
                  Value cstEps = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getF64FloatAttr(eps));

                  rewriter.replaceOpWithNewOp<Torch::AtenBatchNormOp>(
                      binder.op, resultType, input, weight, bias, runningMean,
                      runningVar, /*training=*/cstFalse, cstMomentum, cstEps,
                      /*cudnn_enabled=*/cstFalse);
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
        SmallVector<int64_t> defaultPadding(2 * (rank - 2), 0);
        if (binder.s64IntegerArrayAttr(padding, "pads", defaultPadding)) {
          return failure();
        }
        if (padding.size() != 2 * (rank - 2)) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "padding list size does not match twice the number of axes");
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
      "Bernoulli", 15,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t dtypeIntOnnx, dtypeIntTorch;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "dtype", -1) ||
            binder.tensorResultType(resultType))
          return failure();

        SmallString<64> name("torch.onnx.");
        name.append("seed");
        auto attr = binder.op->getAttr(name);
        if (attr) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");
        }

        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value bernoulli = rewriter.create<Torch::AtenBernoulliOp>(
            binder.getLoc(), input.getType(), input, /*generator=*/none);

        if (dtypeIntOnnx == -1) {
          // True, if dtype attribute value is not present.
          rewriter.replaceOp(binder.op, bernoulli);
          return success();
        }
        dtypeIntTorch = onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (dtypeIntTorch == -1) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    dtypeIntTorch));
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, bernoulli, constDtype,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/none);
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
  patterns.onOp("BitwiseAnd", 18,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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
  patterns.onOp("BitwiseOr", 18,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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
  patterns.onOp("BitwiseXor", 18,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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
      "Cast", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t dtypeIntOnnx, dtypeIntTorch;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "to") ||
            binder.tensorResultType(resultType))
          return failure();

        dtypeIntTorch = onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (dtypeIntTorch == -1) {
          auto message = llvm::formatv("unimplemented support for the given "
                                       "dtype conversion (onnx 'type' = {0})",
                                       dtypeIntOnnx);
          auto y = rewriter.notifyMatchFailure(binder.op, message);

          return y;
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
  patterns.onOp(
      "CastLike", 15, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input, target;
        if (binder.tensorOperands(input, target) ||
            binder.tensorResultType(resultType))
          return failure();

        // TODO: Add support to handle the `saturate` attribute.
        // Ignoring it right now, since it's only using during the float8
        // conversions which are not supported in Torch-MLIR right now.

        Torch::ValueTensorType targetTy =
            target.getType().cast<Torch::ValueTensorType>();
        if (!targetTy.hasDtype()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "target tensor must have a dtype");
        }
        Type targetDtype = targetTy.getDtype();
        Value constDtype = Torch::getDtypeIntValueForType(
            rewriter, binder.getLoc(), targetDtype);
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, input, constDtype,
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
  patterns.onOp(
      "Concat", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> tensors;
        int64_t dim;
        if (binder.tensorOperands(tensors, binder.op->getNumOperands()) ||
            binder.s64IntegerAttr(dim, "axis", 0) ||
            binder.tensorResultType(resultType))
          return failure();
        Type listElemType =
            tensors[0]
                .getType()
                .cast<Torch::BaseTensorType>()
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = rewriter.create<Torch::PrimListConstructOp>(
            binder.op->getLoc(), listType, tensors);
        Value cstDim = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dim));
        rewriter.replaceOpWithNewOp<Torch::AtenCatOp>(binder.op, resultType,
                                                      tensorList, cstDim);
        return success();
      });
  patterns.onOp(
      "Constant", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType))
          return failure();
        auto dtype = resultType.getDtype();

        float floatValue;
        if (binder.op->hasAttr("torch.onnx.value_float") &&
            !binder.f32FloatAttr(floatValue, "value_float", 0.0)) {
          auto splatAttr =
              SplatElementsAttr::get(resultType.toBuiltinTensor().clone(dtype),
                                     rewriter.getFloatAttr(dtype, floatValue));
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, splatAttr);
          return success();
        }

        int64_t intValue;
        if (binder.op->hasAttr("torch.onnx.value_int") &&
            !binder.s64IntegerAttr(intValue, "value_int", 0)) {
          auto splatAttr =
              SplatElementsAttr::get(resultType.toBuiltinTensor().clone(dtype),
                                     rewriter.getIntegerAttr(dtype, intValue));
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, splatAttr);
          return success();
        }

        if (DenseResourceElementsAttr attr =
                binder.op->getAttr("torch.onnx.value")
                    .dyn_cast_or_null<DenseResourceElementsAttr>()) {
          // Bytes are stored in little endian order. Big endian support will
          // require swizzling.
          if (!Endian::little) {
            binder.op->emitError(
                "unimplemented: importing on big endian systems");
            return failure();
          }

          auto ty = cast<ShapedType>(attr.getType());
          auto ptr = attr.getRawHandle().getBlob()->getData();
          DenseElementsAttr denseAttr =
              DenseElementsAttr::getFromRawBuffer(ty, ptr);
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, denseAttr);
          return success();
        }

        if (ElementsAttr attr = binder.op->getAttr("torch.onnx.value")
                                    .dyn_cast_or_null<ElementsAttr>()) {
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, attr);
          return success();
        }

        llvm::SmallVector<int64_t> intValues;
        if (!binder.s64IntegerArrayAttr(intValues, "value_ints", {}) &&
            !intValues.empty()) {
          llvm::SmallVector<APInt> apValues;
          for (auto intVal : intValues) {
            apValues.push_back(APInt(dtype.getIntOrFloatBitWidth(), intVal));
          }
          auto attr = DenseElementsAttr::get(
              resultType.toBuiltinTensor().clone(dtype), apValues);
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, attr);
          return success();
        }

        return failure();
      });
  patterns.onOp(
      "Conv", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET") {
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");
        }

        Torch::ValueTensorType resultType;
        Value input, weight;
        int64_t group;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(weight, 1) ||
            binder.s64IntegerAttr(group, "group", 1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto weightTensorType = weight.getType().cast<Torch::ValueTensorType>();
        if (!weightTensorType || !weightTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected weight type having sizes");
        }
        ArrayRef<int64_t> weightShape = weightTensorType.getSizes();
        SmallVector<int64_t> kernelShape;
        if (binder.s64IntegerArrayAttr(kernelShape, "kernel_shape", {}))
          return failure();
        if (kernelShape.size()) {
          if (kernelShape.size() != weightShape.size() - 2) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "unsupported conversion: kernel_shape list size should have "
                "number of values equal to weight_rank - 2");
          } else {
            for (unsigned i = 0; i < kernelShape.size(); i++) {
              if (weightShape[i + 2] != kernelShape[i]) {
                return rewriter.notifyMatchFailure(
                    binder.op, "unsupported conversion: kernel_shape value "
                               "should be equal to the weight tensor shape");
              }
            }
          }
        }

        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;

        SmallVector<int64_t> padding, strides, dilations;
        SmallVector<int64_t> defaultPadding, defaultStrides, defaultDilations;
        for (unsigned i = 0; i < rank - 2; i++) {
          defaultPadding.push_back(0);
          defaultStrides.push_back(1);
          defaultDilations.push_back(1);
        }
        // Padding for the beginning and ending along each spatial axis, it can
        // take any value greater than or equal to 0. The value represent the
        // number of pixels added to the beginning and end part of the
        // corresponding axis. pads format should be as follow [x1_begin,
        // x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added
        // at the beginning of axis i and xi_end, the number of pixels added at
        // the end of axis i.
        if (binder.s64IntegerArrayAttr(padding, "pads", defaultPadding)) {
          return failure();
        }
        if (padding.size() != rank - 2 && padding.size() != 2 * (rank - 2)) {
          return rewriter.notifyMatchFailure(
              binder.op, "padding list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(dilations, "dilations",
                                       defaultDilations)) {
          return failure();
        }
        if (dilations.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "dilations list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(strides, "strides", defaultStrides)) {
          return failure();
        }
        if (strides.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        }

        SmallVector<Value> cstPadding, cstStrides, cstDilations,
            cstOutputPadding;
        if (padding.size() != 2 * (rank - 2)) {
          for (int64_t i : padding) {
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          }
        } else {
          for (unsigned i = 0; i < padding.size() / 2; i++) {
            if (padding[i] != padding[i + (padding.size() / 2)]) {
              // TODO: Add support for different padding values for the
              // beginning and ending along each spatial axis
              return rewriter.notifyMatchFailure(
                  binder.op,
                  "unsupported conversion: padding values for the beginning "
                  "and ending along each spatial axis must be equal");
            }
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
          }
        }
        for (int64_t i : dilations) {
          cstDilations.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        for (int64_t i : strides) {
          cstStrides.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        cstOutputPadding = {cstZero, cstZero};

        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value dilationsList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstDilations);
        Value stridesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value outputPaddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstOutputPadding);
        Value transposed =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value bias;
        if (binder.op->getNumOperands() == 3) {
          if (binder.tensorOperandAtIndex(bias, 2)) {
            return failure();
          }
        } else {
          bias = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        }
        Value cstGroup = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(group));

        rewriter.replaceOpWithNewOp<Torch::AtenConvolutionOp>(
            binder.op, resultType, input, weight, bias, stridesList,
            paddingList, dilationsList, transposed, outputPaddingList,
            cstGroup);
        return success();
      });
  patterns.onOp(
      "ConvTranspose", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET") {
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");
        }
        SmallVector<int64_t> outputShape;
        if (binder.s64IntegerArrayAttr(outputShape, "output_shape", {}))
          return failure();
        if (outputShape.size()) {
          // TODO: Add support for non-None output_shape value.
          return rewriter.notifyMatchFailure(
              binder.op,
              "unsupported conversion: output_shape should be absent");
        }
        Torch::ValueTensorType resultType;
        Value input, weight;
        int64_t group;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(weight, 1) ||
            binder.s64IntegerAttr(group, "group", 1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto weightTensorType = weight.getType().cast<Torch::ValueTensorType>();
        if (!weightTensorType || !weightTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected weight type having sizes");
        }
        ArrayRef<int64_t> weightShape = weightTensorType.getSizes();
        SmallVector<int64_t> kernelShape;
        if (binder.s64IntegerArrayAttr(kernelShape, "kernel_shape", {}))
          return failure();
        if (kernelShape.size()) {
          if (kernelShape.size() != weightShape.size() - 2) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "unsupported conversion: kernel_shape list size should have "
                "number of values equal to weight_rank - 2");
          } else {
            for (unsigned i = 0; i < kernelShape.size(); i++) {
              if (weightShape[i + 2] != kernelShape[i]) {
                return rewriter.notifyMatchFailure(
                    binder.op, "unsupported conversion: kernel_shape value "
                               "should be equal to the weight tensor shape");
              }
            }
          }
        }

        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;

        SmallVector<int64_t> padding, strides, dilations, outputPadding;
        SmallVector<int64_t> defaultPadding, defaultStrides, defaultDilations,
            defaultOutputPadding;
        for (unsigned i = 0; i < rank - 2; i++) {
          defaultPadding.push_back(0);
          defaultStrides.push_back(1);
          defaultDilations.push_back(1);
          defaultOutputPadding.push_back(0);
        }
        // Padding for the beginning and ending along each spatial axis, it can
        // take any value greater than or equal to 0. The value represent the
        // number of pixels added to the beginning and end part of the
        // corresponding axis. pads format should be as follow [x1_begin,
        // x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added
        // at the beginning of axis i and xi_end, the number of pixels added at
        // the end of axis i.
        if (binder.s64IntegerArrayAttr(padding, "pads", defaultPadding)) {
          return failure();
        }
        if (padding.size() != rank - 2 && padding.size() != 2 * (rank - 2)) {
          return rewriter.notifyMatchFailure(
              binder.op, "padding list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(dilations, "dilations",
                                       defaultDilations)) {
          return failure();
        }
        if (dilations.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "dilations list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(strides, "strides", defaultStrides)) {
          return failure();
        }
        if (strides.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        }
        if (binder.s64IntegerArrayAttr(outputPadding, "output_padding",
                                       defaultOutputPadding)) {
          return failure();
        }
        if (outputPadding.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "output_padding list size does not match the number of axes");
        }

        SmallVector<Value> cstPadding, cstStrides, cstDilations,
            cstOutputPadding;
        if (padding.size() != 2 * (rank - 2)) {
          for (int64_t i : padding) {
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          }
        } else {
          for (unsigned i = 0; i < padding.size() / 2; i++) {
            if (padding[i] != padding[i + (padding.size() / 2)]) {
              // TODO: Add support for different padding values for the
              // beginning and ending along each spatial axis
              return rewriter.notifyMatchFailure(
                  binder.op,
                  "unsupported conversion: padding values for the beginning "
                  "and ending along each spatial axis must be equal");
            }
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
          }
        }
        for (int64_t i : dilations) {
          cstDilations.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        for (int64_t i : strides) {
          cstStrides.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        for (int64_t i : outputPadding) {
          cstOutputPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }

        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value dilationsList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstDilations);
        Value stridesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value outputPaddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstOutputPadding);
        Value transposed =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value bias;
        if (binder.op->getNumOperands() == 3) {
          if (binder.tensorOperandAtIndex(bias, 2)) {
            return failure();
          }
        } else {
          bias = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        }
        Value cstGroup = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(group));

        rewriter.replaceOpWithNewOp<Torch::AtenConvolutionOp>(
            binder.op, resultType, input, weight, bias, stridesList,
            paddingList, dilationsList, transposed, outputPaddingList,
            cstGroup);
        return success();
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
  patterns.onOp("Cosh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenCoshOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "CumSum", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value operand;
        Value axisTensor;
        if (binder.tensorOperands(operand, axisTensor) ||
            binder.tensorResultType(resultType))
          return failure();

        int64_t exclusive;
        int64_t reverse;
        // if bind succeeds and either is set, fail because not implemented
        if (!binder.s64IntegerAttr(exclusive, "exclusive", 0))
          if (exclusive != 0)
            return rewriter.notifyMatchFailure(
                binder.op, "unsupported onnx.CumSum conversion: exclusive");
        if (!binder.s64IntegerAttr(reverse, "reverse", 0))
          if (reverse != 0)
            return rewriter.notifyMatchFailure(
                binder.op, "unsupported onnx.CumSum conversion: reverse");

        // deal with neg axis: if (axis < 0) axis += rank
        int64_t rank =
            cast<Torch::ValueTensorType>(operand.getType()).getSizes().size();
        Value rankVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), rank));
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));

        Value axisScalar = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), axisTensor);
        Value isNegative = rewriter.create<Torch::AtenLtIntOp>(
            binder.getLoc(), axisScalar, zero);
        isNegative =
            rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(), isNegative);
        Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), isNegative, rankVal);
        Value dim = rewriter.create<Torch::AtenAddIntOp>(
            binder.getLoc(), axisScalar, finalOffset);

        Torch::BaseTensorType resultTensorType =
            resultType.cast<Torch::BaseTensorType>();
        if (!resultTensorType.hasDtype()) {
          return rewriter.notifyMatchFailure(
              binder.op, "expected result type to have a dtype");
        }
        // resultTensorType.print(llvm::outs());
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        rewriter.replaceOpWithNewOp<Torch::AtenCumsumOp>(binder.op, resultType,
                                                         operand, dim, none);
        return success();
      });
  patterns.onOp(
      "DepthToSpace", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t blockSize;
        std::string mode;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(blockSize, "blocksize") ||
            binder.customOpNameStringAttr(mode, "mode", "DCR") ||
            binder.tensorResultType(resultType))
          return failure();
        auto inputTy = input.getType().dyn_cast<Torch::BaseTensorType>();
        if (!inputTy || !inputTy.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        SmallVector<int64_t> inputSizes{inputTy.getSizes()};
        if (inputSizes.size() != 4) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected input rank to be 4");
        }
        Value b = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(0)));
        Value c = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(1)));
        Value h = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(2)));
        Value w = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), input,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(3)));
        Value cstBlockSize = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(blockSize));
        Value cstBlockSizeSquare = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(blockSize * blockSize));
        Value cDivBlockSizeSquare = rewriter.create<Torch::AtenDivIntOp>(
            binder.getLoc(), c, cstBlockSizeSquare);
        cDivBlockSizeSquare = rewriter.create<Torch::AtenIntFloatOp>(
            binder.getLoc(), cDivBlockSizeSquare);
        Value reshapeSizesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(input.getContext())),
            llvm::SmallVector<Value>{b, cstBlockSize, cstBlockSize,
                                     cDivBlockSizeSquare, h, w});
        int64_t cDivBlockSizeSquareInt =
            inputSizes[1] == Torch::kUnknownSize
                ? Torch::kUnknownSize
                : inputSizes[1] / (blockSize * blockSize);
        SmallVector<int64_t, 6> reshapeSizesInt{
            inputSizes[0],          blockSize,     blockSize,
            cDivBlockSizeSquareInt, inputSizes[2], inputSizes[3]};
        Value reshapedInput = rewriter.create<Torch::AtenReshapeOp>(
            binder.getLoc(),
            inputTy.getWithSizesAndDtype(reshapeSizesInt,
                                         inputTy.getOptionalDtype()),
            input, reshapeSizesList);

        Value transposedInput;
        if (mode == "DCR") {
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), reshapedInput,
                  /*dimA=*/1, /*dimB=*/3, transposedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op");
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), transposedInput,
                  /*dimA=*/2, /*dimB=*/4, transposedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op");
        } else {
          // mode == "CRD"
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), reshapedInput,
                  /*dimA=*/2, /*dimB=*/4, transposedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op");
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), transposedInput,
                  /*dimA=*/3, /*dimB=*/4, transposedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op");
        }
        if (failed(createTorchTransposeOp(
                rewriter, binder.getLoc(), transposedInput,
                /*dimA=*/4, /*dimB=*/5, transposedInput)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to create TorchTranspose op");

        Value hMulBlockSize = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), h, cstBlockSize);
        Value wMulBlockSize = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), w, cstBlockSize);
        reshapeSizesList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(input.getContext())),
            llvm::SmallVector<Value>{b, cDivBlockSizeSquare, hMulBlockSize,
                                     wMulBlockSize});
        rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
            binder.op, resultType, transposedInput, reshapeSizesList);
        return success();
      });
  patterns.onOp(
      "DequantizeLinear", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperands(operands, 3) ||
            binder.tensorResultType(resultType))
          return failure();

        Value operand = operands[0];
        Value scale = operands[1];
        Value zeropoint = operands[2];

        auto operandTy = operand.getType().cast<Torch::ValueTensorType>();

        auto scaleTy = scale.getType().dyn_cast<Torch::ValueTensorType>();
        if (!scaleTy || !scaleTy.hasSizes())
          return rewriter.notifyMatchFailure(binder.op, "requires known rank");
        if (!resultType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "requires known resulty dtype");

        if (scaleTy.getSizes().size() == 0) {
          Type qTy = operandTy.getDtype();

          if (qTy.isUnsignedInteger(8)) {
            qTy = rewriter.getType<Torch::QUInt8Type>();
          } else if (qTy.isSignedInteger(8)) {
            qTy = rewriter.getType<Torch::QInt8Type>();
          } else if (qTy.isSignedInteger(32)) {
            qTy = rewriter.getType<Torch::QInt32Type>();
          } else {
            return rewriter.notifyMatchFailure(binder.op,
                                               "unsupported result dtype");
          }

          auto qTensorTy = rewriter.getType<Torch::ValueTensorType>(
              resultType.getOptionalSizes(), qTy);
          scale = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::FloatType>(), scale);
          zeropoint = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), zeropoint);

          auto quantize =
              rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
                  binder.getLoc(), qTensorTy, operand, scale, zeropoint);
          rewriter.replaceOpWithNewOp<Torch::AtenDequantizeSelfOp>(
              binder.op, resultType, quantize);
          return success();
        }

        return failure();
      });
  patterns.onOp("Div", 14,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "Dropout", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        int64_t numOperands = binder.op->getNumOperands();
        SmallVector<Value> operands;
        int64_t seed;
        if (binder.tensorOperands(operands, numOperands) ||
            binder.s64IntegerAttr(seed, "seed", 0) ||
            binder.tensorResultTypeAtIndex(resultType, 0))
          return failure();

        // Global Seed value is 0.
        if (seed != 0) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected seed value to be 0");
        }

        Value ratio, trainingMode;
        if (numOperands == 3) {
          ratio = rewriter.create<Torch::AtenFloatImplicitOp>(loc, operands[1]);
          Value trainVal = operands[2];
          auto trainTensorType =
              trainVal.getType().dyn_cast<Torch::BaseTensorType>();
          if (!trainTensorType)
            return rewriter.notifyMatchFailure(binder.op,
                                               "train tensor must have a type");

          Type inputDtype = trainTensorType.getOptionalDtype();
          if (!inputDtype || !inputDtype.isInteger(1))
            return rewriter.notifyMatchFailure(
                binder.op,
                "train tensor must have an integer dtype of width 1");

          std::optional<unsigned> inputRank = Torch::getTensorRank(trainVal);
          if (!inputRank || *inputRank != 0)
            return rewriter.notifyMatchFailure(binder.op,
                                               "train tensor must have rank 0");

          if (auto valueTensorLiteralOp =
                  trainVal.getDefiningOp<Torch::ValueTensorLiteralOp>()) {
            auto val = valueTensorLiteralOp.getValue()
                           .cast<DenseElementsAttr>()
                           .getSplatValue<bool>();
            trainingMode = rewriter.create<Torch::ConstantBoolOp>(loc, val);
          } else {
            Value trainingModeScalar =
                rewriter.create<Torch::AtenIntImplicitOp>(loc, operands[2]);
            Value cstOne = rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(1));
            trainingMode = rewriter.create<Torch::AtenEqIntOp>(
                loc, trainingModeScalar, cstOne);
          }
        } else if (numOperands == 2) {
          ratio = rewriter.create<Torch::AtenFloatImplicitOp>(loc, operands[1]);
          trainingMode = rewriter.create<Torch::ConstantBoolOp>(loc, false);
        } else {
          ratio = rewriter.create<Torch::ConstantFloatOp>(
              loc, rewriter.getF64FloatAttr(0.5));
          trainingMode = rewriter.create<Torch::ConstantBoolOp>(loc, false);
        }

        Value dropout = rewriter.create<Torch::AtenDropoutOp>(
            loc, resultType, /*input=*/operands[0], ratio, trainingMode);

        if (binder.op->getNumResults() == 1) {
          rewriter.replaceOp(binder.op, dropout);
          return success();
        }
        Torch::ValueTensorType maskType;
        if (binder.tensorResultTypeAtIndex(maskType, 1))
          return failure();
        Value dtype = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(
                     (int64_t)torch_upstream::ScalarType::Bool));
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        Value mask = rewriter.create<Torch::AtenOnesLikeOp>(
            loc, maskType, operands[0], dtype, /*layout=*/none,
            /*device=*/none, /*pin_memory=*/none, /*memory_format=*/none);
        rewriter.replaceOp(binder.op, {dropout, mask});
        return success();
      });
  patterns.onOp("Equal", 1,
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
  patterns.onOp("Elu", 6,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Location loc = binder.getLoc();
                  Torch::ValueTensorType resultType;
                  Value input;
                  float alpha;
                  if (binder.tensorOperand(input) ||
                      binder.f32FloatAttr(alpha, "alpha") ||
                      binder.tensorResultType(resultType))
                    return failure();
                  Value cstAlpha = rewriter.create<Torch::ConstantFloatOp>(
                      loc, rewriter.getF64FloatAttr(alpha));
                  Value cstOne = rewriter.create<Torch::ConstantFloatOp>(
                      loc, rewriter.getF64FloatAttr(1.0));
                  rewriter.replaceOpWithNewOp<Torch::AtenEluOp>(
                      binder.op, resultType, input, cstAlpha, /*scale=*/cstOne,
                      /*input_scale=*/cstOne);
                  return success();
                });
  patterns.onOp("Erf", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  std::string direction;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenErfOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Exp", 6,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenExpOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Expand", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // uses ideas and code from onnx.Reshape
        Torch::ValueTensorType resultType;
        Value data, shape;
        if (binder.tensorOperands(data, shape) ||
            binder.tensorResultType(resultType))
          return failure();
        Torch::BaseTensorType shapeType =
            shape.getType().cast<Torch::BaseTensorType>();
        SmallVector<int64_t> selectSizes;
        Type selectResultType = shapeType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), shapeType.getOptionalDtype());
        // Variable to store 1-D onnx shape tensor, shapeSizes[0] has the
        // dimension size
        auto shapeSizes =
            dyn_cast<Torch::ValueTensorType>(shape.getType()).getSizes();
        // A constant zero value
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        // Variable to store pytorch int list of shape (dimension)
        SmallVector<Value> dimList;

        // Convert the shape tensor from vector of int64_t to torch int list as
        // we are using torch implementation Torch::AtenBroadcastToOp which
        // takes list of int
        for (int i = 0; i < shapeSizes[0]; i++) {
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, shape, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          dimList.push_back(dim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        rewriter.replaceOpWithNewOp<Torch::AtenBroadcastToOp>(
            binder.op, resultType, data, dimValueList);
        return success();
      });
  patterns.onOp(
      "Flatten", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // Flatten means to partition the input tensor's dimensions
        // into a "left range" spanning 0 to axis - 1 and a "right range"
        // spanning axis to rank - 1.  Each range is then collapsed
        // into a single dimension, resulting in a 2-D tensor.
        // If either range is empty, it is replaced with a single
        // dimension of size 1.
        //
        // For example, for a 4-D input tensor of shape (a, b, c, d)
        // and axis==2, flatten produces a 2-D tensor of shape
        // (a*b, c*d).
        //
        // If instead axis==0, the left range is empty, and the result
        // is (1, a*b*c*d).

        Torch::ValueTensorType resultType;
        Value operand;
        int64_t axis;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(axis, "axis", 1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto operandTy = cast<Torch::ValueTensorType>(operand.getType());
        llvm::SmallVector<int64_t> shape(operandTy.getSizes());
        int64_t rank = shape.size();

        // If axis is negative, count from the right instead of left
        if (axis < 0)
          axis = rank + axis;

        // We collapse in the dimensions to the right of the axis.
        for (int i = axis + 1; i < rank; ++i) {
          bool dynamic = shape[axis] == Torch::kUnknownSize ||
                         shape[i] == Torch::kUnknownSize;
          if (dynamic) {
            shape[axis] = Torch::kUnknownSize;
          } else {
            shape[axis] = shape[axis] * shape[i];
          }
        }

        shape.resize(axis + 1, 1);

        auto baseType = rewriter.getType<Torch::ValueTensorType>(
            shape, operandTy.getDtype());
        Value collapsedRight;
        if (axis >= rank) {
          // If the right range is empty, add a dim of size 1 to the
          // right side of the shape:
          // cr = torch.unsqueeze(x, x.ndim)
          Value rankConst = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(rank));
          collapsedRight = rewriter.create<Torch::AtenUnsqueezeOp>(
              binder.getLoc(), baseType, operand, rankConst);
        } else {
          // Otherwise, collapse the right range into a single dimension:
          // cr = torch._prims.collapse(x, axis, x.ndim - 1)
          Value axisConst = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(axis));
          Value rankLess1Const = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(rank - 1));
          collapsedRight = rewriter.create<Torch::PrimsCollapseOp>(
              binder.getLoc(), baseType, operand, axisConst, rankLess1Const);
        }

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));

        if (axis <= 0) {
          // If the left range is empty, add a dim of size 1 to the
          // left side of the shape:
          // torch.unsqueeze(cr, 0)
          rewriter.replaceOpWithNewOp<Torch::AtenUnsqueezeOp>(
              binder.op, resultType, collapsedRight, zero);
          return success();
        }

        // Otherwise, collapse the left range into a single dimension:
        // torch._prims.collapse(cr, 0, axis - 1)
        Value axisLess1Const = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis - 1));
        rewriter.replaceOpWithNewOp<Torch::PrimsCollapseOp>(
            binder.op, resultType, collapsedRight, zero, axisLess1Const);
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
  patterns.onOp(
      "ConstantOfShape", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value shape;
        if (binder.tensorOperand(shape) || binder.tensorResultType(resultType))
          return failure();

        // convert shape tensor to list of ints
        auto shapeSizes =
            dyn_cast<Torch::ValueTensorType>(shape.getType()).getSizes();
        SmallVector<Value> dimList;
        Torch::BaseTensorType shapeType =
            shape.getType().cast<Torch::BaseTensorType>();
        Type selectResultType = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>({}), shapeType.getOptionalDtype());
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        for (int i = 0; i < shapeSizes[0]; i++) {
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, shape, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          dimList.push_back(dim);
        }

        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        // Get fill_value if it is present.
        // Assumption : resultDType and value attr type match.
        auto attr = binder.op->getAttr("torch.onnx.value");
        auto resultDType = resultType.getDtype();

        // Extract the fill value and dtype
        // ONNX requires value attr to be a tensor
        if (!attr) {
          attr = DenseElementsAttr::get(
              resultType.toBuiltinTensor().clone(resultDType),
              rewriter.getFloatAttr(resultDType, 0.0));
        }

        // If its a dense resource attr we need to convert to a dense type:
        if (DenseResourceElementsAttr rattr =
                attr.dyn_cast_or_null<DenseResourceElementsAttr>()) {
          // Bytes are stored in little endian order. Big endian support will
          // require swizzling.
          if (!Endian::little) {
            binder.op->emitError(
                "unimplemented: importing on big endian systems");
            return failure();
          }

          auto ty = cast<ShapedType>(rattr.getType());
          auto ptr = rattr.getRawHandle().getBlob()->getData();
          auto denseAttr = DenseElementsAttr::getFromRawBuffer(ty, ptr);
          attr = dyn_cast_or_null<SplatElementsAttr>(denseAttr);
        }

        Attribute splattr;
        if (isa<SplatElementsAttr>(attr)) {
          auto denseAttr = attr.cast<DenseElementsAttr>();
          splattr = denseAttr.getSplatValue<Attribute>();
        }

        if (!isa<FloatAttr, IntegerAttr>(splattr)) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "`value` attr tensor only supports types int and float for now.");
        }

        Value splatvalue;
        if (auto intattr = dyn_cast<IntegerAttr>(splattr)) {
          IntegerType intty = cast<IntegerType>(intattr.getType());
          int64_t value;
          if (intty.isUnsignedInteger()) {
            value = intattr.getUInt();
          } else if (intty.isSignedInteger()) {
            value = intattr.getSInt();
          } else {
            value = intattr.getInt();
          }
          splatvalue =
              rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), value);
        }

        if (auto fpattr = dyn_cast<FloatAttr>(splattr))
          splatvalue = rewriter.create<Torch::ConstantFloatOp>(
              binder.getLoc(),
              rewriter.getF64FloatAttr(fpattr.getValueAsDouble()));

        rewriter.replaceOpWithNewOp<Torch::AtenFullOp>(
            binder.op, resultType, dimValueList, splatvalue, /*dtype=*/noneVal,
            /*layout=*/noneVal, /*device=*/noneVal, /*pin_memory=*/noneVal);
        return success();
      });
}
