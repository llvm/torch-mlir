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
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/FormatVariadic.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

namespace {
LogicalResult windowFunctionImpl(OpBinder binder,
                                 ConversionPatternRewriter &rewriter,
                                 Value size, Value a0, Value a1, Value a2,
                                 Torch::ValueTensorType resultType,
                                 int64_t output_datatype, int64_t periodic) {

  Location loc = binder.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);

  double isPeriodicFp = static_cast<double>(periodic);

  Value zero = b.create<Torch::ConstantFloatOp>(rewriter.getF64FloatAttr(0.0));
  Value one = b.create<Torch::ConstantFloatOp>(rewriter.getF64FloatAttr(1.0));
  Value two = b.create<Torch::ConstantFloatOp>(rewriter.getF64FloatAttr(2.0));

  constexpr double pi = llvm::numbers::pi;
  Value tau = b.create<Torch::ConstantFloatOp>(
      rewriter.getFloatAttr(rewriter.getF64Type(), 2.0 * pi));

  Value noneVal = b.create<Torch::ConstantNoneOp>();
  Value cstFalse = b.create<Torch::ConstantBoolOp>(false);
  Value float32Type = b.create<Torch::ConstantIntOp>(
      rewriter.getI64IntegerAttr(/*float32Type*/ 6));

  // Create an f32 ValueTensorType with thse same size as size, the
  // operand
  auto shapeOfOperand =
      dyn_cast<Torch::ValueTensorType>(size.getType()).getOptionalSizes();
  auto f32ResultType = rewriter.getType<Torch::ValueTensorType>(
      shapeOfOperand, rewriter.getF32Type());
  Value periodicSizeFloat = b.create<Torch::AtenToDtypeOp>(
      f32ResultType, size, float32Type, cstFalse, cstFalse, noneVal);
  Value symmetricSizeFloat = b.create<Torch::AtenSubScalarOp>(
      periodicSizeFloat.getType(), periodicSizeFloat, one, one);

  Value isPeriodic =
      b.create<Torch::ConstantFloatOp>(rewriter.getF64FloatAttr(isPeriodicFp));
  Value isSymmetricFloat = b.create<Torch::ConstantFloatOp>(
      rewriter.getF64FloatAttr(1.0 - isPeriodicFp));

  Value periodicComponent = b.create<Torch::AtenMulScalarOp>(
      periodicSizeFloat.getType(), periodicSizeFloat, isPeriodic);
  Value symmetricComponent = b.create<Torch::AtenMulScalarOp>(
      symmetricSizeFloat.getType(), symmetricSizeFloat, isSymmetricFloat);
  Value sizeFloat = b.create<Torch::AtenAddTensorOp>(
      symmetricComponent.getType(), symmetricComponent, periodicComponent, one);

  // Here, size can be used in the place of periodicSizeFloat, as the
  // latter is just a float representation of the former.
  Value scalarLimit = getItemOp<Torch::IntType>(binder, rewriter, size);

  Value rangeArr = b.create<Torch::AtenArangeStartStepOp>(
      resultType, zero, scalarLimit, one, noneVal, noneVal, noneVal, noneVal);

  Value rangeTimesTau =
      b.create<Torch::AtenMulScalarOp>(resultType, rangeArr, tau);
  Value rangeAngular =
      b.create<Torch::AtenDivTensorOp>(resultType, rangeTimesTau, sizeFloat);
  Value twoRangeAngular =
      b.create<Torch::AtenMulScalarOp>(resultType, rangeAngular, two);

  Value cosRangeAngular = b.create<Torch::AtenCosOp>(resultType, rangeAngular);
  Value cosTwoRangeAngular =
      b.create<Torch::AtenCosOp>(resultType, twoRangeAngular);

  Value a1Component =
      b.create<Torch::AtenMulScalarOp>(resultType, cosRangeAngular, a1);
  Value a2Component =
      b.create<Torch::AtenMulScalarOp>(resultType, cosTwoRangeAngular, a2);

  // AtenSubScalarOp actually requires a tensor operand as the LHS, that
  // is, operand #1. Therefore, to avoid errors, the onnx implementation
  // has been modified. a1 has been changed to negative half, and the
  // AtenSubScalarOp has been replaced with AtenAddScalarOp, as the add
  // operation is commutative.
  Value subA1Component =
      b.create<Torch::AtenAddScalarOp>(resultType, a1Component, a0, one);
  Value result = b.create<Torch::AtenAddTensorOp>(resultType, subA1Component,
                                                  a2Component, one);

  std::optional<int64_t> dtypeIntTorch =
      onnxDtypeIntToTorchDtypeInt(output_datatype);
  if (!dtypeIntTorch.has_value()) {
    return rewriter.notifyMatchFailure(
        binder.op, "unimplemented support for the given dtype conversion");
  }
  Value outputDtype = b.create<Torch::ConstantIntOp>(
      rewriter.getType<Torch::IntType>(),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                              dtypeIntTorch.value()));

  rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
      binder.op, resultType, result, outputDtype,
      /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
      /*memory_format=*/noneVal);

  return success();
}

} // namespace

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

        // ONNX allows negative axis.
        auto operandSizes =
            cast<Torch::ValueTensorType>(operand.getType()).getSizes();
        if (axis < 0)
          axis += operandSizes.size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value constKeepDims = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(keepDims));

        if (selectLastIndex) {
          Value dims = createConstantIntList(binder, rewriter, {axis});
          auto operandTy = dyn_cast<Torch::ValueTensorType>(operand.getType());
          operand = rewriter.create<Torch::AtenFlipOp>(
              binder.getLoc(), operandTy, operand, dims);
          Value argmax = rewriter.create<Torch::AtenArgmaxOp>(
              binder.getLoc(), resultType, operand, constAxis, constKeepDims);
          Value offset = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(),
              rewriter.getI64IntegerAttr(operandSizes[axis] - 1));
          Value alpha = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Value sub = rewriter.create<Torch::AtenSubScalarOp>(
              binder.getLoc(), resultType, argmax, offset, alpha);
          rewriter.replaceOpWithNewOp<Torch::AtenAbsOp>(binder.op, resultType,
                                                        sub);
          return success();
        }

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

        // ONNX allows negative axis.
        auto operandSizes =
            cast<Torch::ValueTensorType>(operand.getType()).getSizes();
        if (axis < 0)
          axis += operandSizes.size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));
        Value constKeepDims = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(keepDims));

        if (selectLastIndex) {
          Value dims = createConstantIntList(binder, rewriter, {axis});
          auto operandTy = dyn_cast<Torch::ValueTensorType>(operand.getType());
          operand = rewriter.create<Torch::AtenFlipOp>(
              binder.getLoc(), operandTy, operand, dims);
          Value argmin = rewriter.create<Torch::AtenArgminOp>(
              binder.getLoc(), resultType, operand, constAxis, constKeepDims);
          Value offset = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(),
              rewriter.getI64IntegerAttr(operandSizes[axis] - 1));
          Value alpha = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Value sub = rewriter.create<Torch::AtenSubScalarOp>(
              binder.getLoc(), resultType, argmin, offset, alpha);
          rewriter.replaceOpWithNewOp<Torch::AtenAbsOp>(binder.op, resultType,
                                                        sub);
          return success();
        }

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
      "AveragePool", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        SmallVector<int64_t> dilations;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET") {
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");
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
        if (binder.s64IntegerArrayAttr(
                strides, "strides", llvm::SmallVector<int64_t>(rank - 2, 1))) {
          return failure();
        }
        if (strides.size() != 1 && strides.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        }

        SmallVector<Value> cstKernel, cstPadding, cstStridesDilations;
        for (int64_t i : kernel) {
          cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }
        // Onnx pads format: [x1_begin, x2_begin…x1_end, x2_end,…]
        // Pytorch pads format: [x1, x2,...] or [x], assume begin==end for all
        // axes x.
        int64_t paddingSizeHalf = padding.size() / 2;
        for (int64_t i = 0; i < paddingSizeHalf; ++i) {
          // Check if onnx padding attribute is symmetric.
          if (padding[i] != padding[i + paddingSizeHalf])
            return rewriter.notifyMatchFailure(
                binder.op, "onnx padding attribute is not symmetric");
          cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
        }
        for (int64_t i : strides) {
          cstStridesDilations.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
        }

        // No dilations attribute in pytorch avgpool op, so use this trick to
        // encode dilation into strides. Then in the following torchtolinalg
        // lowering, decode strides into strides + dilation.
        // [strideDim1,strideDim2,...,dilationDim1,dilationDim2,...]
        if (binder.s64IntegerArrayAttr(
                dilations, "dilations",
                llvm::SmallVector<int64_t>(rank - 2, 1))) {
          return failure();
        }
        for (auto dilation : dilations) {
          cstStridesDilations.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(dilation)));
        }

        Value kernelSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesDilationsList =
            rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                cstStridesDilations);
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceilMode);
        Value cstCountIncludePad = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), countIncludePad);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        if (rank == 3) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool1dOp>(
              binder.op, resultType, operand, kernelSizeList,
              stridesDilationsList, paddingList, cstCeilMode,
              cstCountIncludePad);
          return success();
        } else if (rank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool2dOp>(
              binder.op, resultType, operand, kernelSizeList,
              stridesDilationsList, paddingList, cstCeilMode,
              cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        } else if (rank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool3dOp>(
              binder.op, resultType, operand, kernelSizeList,
              stridesDilationsList, paddingList, cstCeilMode,
              cstCountIncludePad,
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
        int64_t dtypeIntOnnx;
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
        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));
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
        int64_t dtypeIntOnnx;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "to") ||
            binder.tensorResultType(resultType))
          return failure();

        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value constDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));
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
            cast<Torch::ValueTensorType>(target.getType());
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
      "Celu", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        float alpha;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha", 1.0f))
          return failure();
        // exp(x/alpha)
        Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));
        Value xDivAlpha = rewriter.create<Torch::AtenDivScalarOp>(
            binder.getLoc(), resultType, operand, constAlpha);
        Value expXDivAlpha = rewriter.create<Torch::AtenExpOp>(
            binder.getLoc(), resultType, xDivAlpha);
        // alpha * (exp(x/alpha) - 1)
        Value constantOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value subOne = rewriter.create<Torch::AtenSubScalarOp>(
            binder.getLoc(), resultType, expXDivAlpha, constantOne,
            constantOne);
        Value mulAlpha = rewriter.create<Torch::AtenMulScalarOp>(
            binder.getLoc(), resultType, subOne, constAlpha);
        Value constantZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value zeroTensor = createRank0Tensor(rewriter, binder.getLoc(),
                                             resultType, constantZero);
        // min(0, alpha * (exp(x/alpha) - 1))
        Value minExpression = rewriter.create<Torch::AtenMinimumOp>(
            binder.getLoc(), resultType, zeroTensor, mulAlpha);

        // max(0, x)
        Value maxExpression = rewriter.create<Torch::AtenMaximumOp>(
            binder.getLoc(), resultType, zeroTensor, operand);
        // max(0,x) + min(0, alpha * (exp(x/alpha) - 1))
        rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
            binder.op, resultType, maxExpression, minExpression, constantOne);
        return success();
      });
  patterns.onOp(
      "CenterCropPad", 18,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input, shape;
        if (binder.tensorOperands(input, shape) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        SmallVector<int64_t> inputShape(inputTy.getSizes());
        SmallVector<int64_t> resultShape(resultType.getSizes());
        int64_t rank = inputShape.size();

        SmallVector<int64_t> axes, defaultAxes(rank);
        std::iota(defaultAxes.begin(), defaultAxes.end(), 0);
        if (binder.s64IntegerArrayAttr(axes, "axes", defaultAxes)) {
          return failure();
        }
        int64_t axesSize = axes.size();

        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value cstTwo = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(2));
        auto scalarTensorType = rewriter.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>{1}, rewriter.getIntegerType(64, /*signed*/ 1));

        int64_t lastChangeDim = 0;
        llvm::SmallVector<int64_t> interShape(inputShape);
        for (int i = 0; i < rank; i++) {
          if (inputShape[i] != resultShape[i]) {
            interShape[i] = -1;
            lastChangeDim = i;
          }
          if (interShape[i] == ShapedType::kDynamic)
            interShape[i] = Torch::kUnknownSize;
        }
        auto interType = rewriter.getType<Torch::ValueTensorType>(
            interShape, resultType.getOptionalDtype());

        Value modeVal = rewriter.create<Torch::ConstantStrOp>(
            binder.getLoc(), rewriter.getStringAttr("floor"));
        for (int i = 0; i < axesSize; i++) {
          if (axes[i] < 0)
            axes[i] += rank;
          if (inputShape[axes[i]] == resultShape[axes[i]])
            continue;

          auto opType = axes[i] == lastChangeDim ? resultType : interType;
          Value axis = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(axes[i]));
          Value k = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i));
          Value kTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
              binder.getLoc(), scalarTensorType, k);
          Value sel = rewriter.create<Torch::AtenIndexSelectOp>(
              binder.getLoc(), scalarTensorType, shape, cstZero, kTensor);
          Value outputDimSize = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), sel);
          Value inputDimSize = rewriter.create<Torch::AtenSizeIntOp>(
              binder.getLoc(), input,
              rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(axes[i])));

          if (inputShape[axes[i]] > resultShape[axes[i]]) {
            Value sub = rewriter.create<Torch::AtenSubIntOp>(
                binder.getLoc(), inputDimSize, outputDimSize);
            Value subTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
                binder.getLoc(), scalarTensorType, sub);
            Value div = rewriter.create<Torch::AtenDivScalarModeOp>(
                binder.getLoc(), scalarTensorType, subTensor, cstTwo, modeVal);
            Value start = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), div);
            Value end = rewriter.create<Torch::AtenAddIntOp>(
                binder.getLoc(), start, outputDimSize);
            input = rewriter.create<Torch::AtenSliceTensorOp>(
                binder.getLoc(), opType, input, axis, start, end, cstOne);
          } else {
            Value sub = rewriter.create<Torch::AtenSubIntOp>(
                binder.getLoc(), outputDimSize, inputDimSize);
            Value subTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
                binder.getLoc(), scalarTensorType, sub);
            Value div = rewriter.create<Torch::AtenDivScalarModeOp>(
                binder.getLoc(), scalarTensorType, subTensor, cstTwo, modeVal);
            Value start = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), div);
            Value end = rewriter.create<Torch::AtenAddIntOp>(
                binder.getLoc(), start, inputDimSize);

            SmallVector<Value> zerosShapeValues;
            for (int j = 0; j < rank; j++) {
              if (j == axes[i]) {
                zerosShapeValues.push_back(outputDimSize);
              } else {
                Value dimSize = rewriter.create<Torch::AtenSizeIntOp>(
                    binder.getLoc(), input,
                    rewriter.create<Torch::ConstantIntOp>(
                        binder.getLoc(), rewriter.getI64IntegerAttr(j)));
                zerosShapeValues.push_back(dimSize);
              }
            }
            Value zerosShapeList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                rewriter.getType<Torch::ListType>(
                    rewriter.getType<Torch::IntType>()),
                zerosShapeValues);
            Value zeros = rewriter.create<Torch::AtenZerosOp>(
                binder.getLoc(), opType, zerosShapeList, none, none, none,
                none);
            input = rewriter.create<Torch::AtenSliceScatterOp>(
                binder.getLoc(), opType, zeros, input, axis, start, end,
                cstOne);
          }
        }

        rewriter.replaceOp(binder.op, input);
        return success();
      });
  patterns.onOp(
      "Clip", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // https://onnx.ai/onnx/operators/onnx__Clip.html

        // Inputs and outputs must be tensors.
        Value source;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperandAtIndex(source, 0) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        // Min and max can be args (version 11+) or attributes (version 6-).
        // They default to numeric_limits::lowest() and numeric_limits::max().
        Value min;
        Value max;
        if (binder.op->getNumOperands() >= 2)
          min = binder.op->getOperand(1);
        if (binder.op->getNumOperands() == 3)
          max = binder.op->getOperand(2);

        // Note: attribute versions of the op only support float types.
        auto resultDtype = resultType.getDtype();
        if (!min && binder.op->hasAttr("torch.onnx.min")) {
          float minValue;
          if (binder.f32FloatAttr(minValue, "min",
                                  std::numeric_limits<float>::lowest()))
            return failure();
          auto minSplatAttr = SplatElementsAttr::get(
              resultType.toBuiltinTensor(),
              rewriter.getFloatAttr(resultDtype, minValue));
          min = rewriter.create<Torch::ValueTensorLiteralOp>(
              binder.getLoc(), resultType, minSplatAttr);
        }
        if (!max && binder.op->hasAttr("torch.onnx.max")) {
          float maxValue;
          if (binder.f32FloatAttr(maxValue, "max",
                                  std::numeric_limits<float>::max()))
            return failure();
          auto maxSplatAttr = SplatElementsAttr::get(
              resultType.toBuiltinTensor(),
              rewriter.getFloatAttr(resultDtype, maxValue));
          max = rewriter.create<Torch::ValueTensorLiteralOp>(
              binder.getLoc(), resultType, maxSplatAttr);
        }

        if (!min && !max) {
          // Cliping with no limits is a no-op.
          rewriter.replaceOp(binder.op, source);
          return success();
        }

        if (!max) {
          rewriter.replaceOpWithNewOp<Torch::AtenClampMinTensorOp>(
              binder.op, resultType, source, min);
          return success();
        }

        rewriter.replaceOpWithNewOp<Torch::AtenClampTensorOp>(
            binder.op, resultType, source, min, max);
        return success();
      });
  patterns.onOp(
      "Compress", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand, conditionTensor;
        int64_t axis;
        if (binder.tensorOperands(operand, conditionTensor) ||
            binder.s64IntegerAttr(axis, "axis", INT64_MAX) ||
            binder.tensorResultType(resultType))
          return failure();

        auto shapeSizes =
            dyn_cast<Torch::ValueTensorType>(operand.getType()).getSizes();
        auto resultSizes = resultType.getSizes();

        // flatten input tensor if using default axis
        if (axis == INT64_MAX) {
          SmallVector<int64_t> nonzeroShape = {resultSizes[0]};
          auto dtype =
              dyn_cast<Torch::ValueTensorType>(conditionTensor.getType())
                  .getDtype();
          auto nonzeroType =
              rewriter.getType<Torch::ValueTensorType>(nonzeroShape, dtype);
          Value indexVal = rewriter.create<Torch::AtenNonzeroOp>(
              binder.getLoc(), nonzeroType, conditionTensor);
          Value cstZero = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(0));
          Value cstNegOne = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(-1));
          int64_t numElements = 1;
          for (auto i : shapeSizes) {
            numElements *= i;
          }
          SmallVector<int64_t> flattenShape = {numElements};
          auto flattenType = rewriter.getType<Torch::ValueTensorType>(
              flattenShape, resultType.getDtype());
          Value flattenTensor = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              binder.getLoc(), flattenType, operand, cstZero, cstNegOne);
          rewriter.replaceOpWithNewOp<Torch::AtenIndexSelectOp>(
              binder.op, resultType, flattenTensor, cstZero, indexVal);
          return success();
        }

        // Negative axis value means counting dimensions from the back
        if (axis < 0)
          axis += shapeSizes.size();
        SmallVector<int64_t> nonzeroShape = {resultSizes[axis]};
        auto dtype = dyn_cast<Torch::ValueTensorType>(conditionTensor.getType())
                         .getDtype();
        auto nonzeroType =
            rewriter.getType<Torch::ValueTensorType>(nonzeroShape, dtype);
        Value indexVal = rewriter.create<Torch::AtenNonzeroOp>(
            binder.getLoc(), nonzeroType, conditionTensor);
        Value dimVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        rewriter.replaceOpWithNewOp<Torch::AtenIndexSelectOp>(
            binder.op, resultType, operand, dimVal, indexVal);
        return success();
      });
  patterns.onOp(
      "Concat", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> tensors;
        int64_t dim;
        if (binder.tensorOperands(tensors, binder.op->getNumOperands()) ||
            binder.s64IntegerAttr(dim, "axis", 0) ||
            binder.tensorResultType(resultType))
          return failure();
        Type listElemType =
            cast<Torch::BaseTensorType>(tensors[0].getType())
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
              SplatElementsAttr::get(resultType.toBuiltinTensor(),
                                     rewriter.getFloatAttr(dtype, floatValue));
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, splatAttr);
          return success();
        }

        int64_t intValue;
        if (binder.op->hasAttr("torch.onnx.value_int") &&
            !binder.s64IntegerAttr(intValue, "value_int", 0)) {
          auto splatAttr =
              SplatElementsAttr::get(resultType.toBuiltinTensor(),
                                     rewriter.getIntegerAttr(dtype, intValue));
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, splatAttr);
          return success();
        }

        if (DenseResourceElementsAttr attr =
                dyn_cast_or_null<DenseResourceElementsAttr>(
                    binder.op->getAttr("torch.onnx.value"))) {
          // Bytes are stored in little endian order. Big endian support will
          // require swizzling.
          if (!Endian::little) {
            binder.op->emitError(
                "unimplemented: importing on big endian systems");
            return failure();
          }

          auto ty = cast<ShapedType>(attr.getType());
          ElementsAttr denseAttr;
          auto ptr = attr.getRawHandle().getBlob();
          if (!ptr) {
            denseAttr = DenseResourceElementsAttr::get(
                ty, "__onnx_constant_not_found_possibly_due_to_being_elided__",
                AsmResourceBlob());
            rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
                binder.op, resultType, denseAttr);
            return success();
          }
          auto data = ptr->getData();
          if (cast<ShapedType>(attr.getType()).getElementType().isInteger(1)) {
            llvm::SmallVector<APInt> newContents;
            for (auto val : data) {
              APInt apval(1, val);
              newContents.push_back(apval);
            }
            denseAttr = DenseElementsAttr::get(ty, newContents);
          } else {
            denseAttr = DenseElementsAttr::getFromRawBuffer(ty, data);
          }

          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, denseAttr);
          return success();
        }

        if (ElementsAttr attr = dyn_cast_or_null<ElementsAttr>(
                binder.op->getAttr("torch.onnx.value"))) {
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
          auto attr =
              DenseElementsAttr::get(resultType.toBuiltinTensor(), apValues);
          rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(
              binder.op, resultType, attr);
          return success();
        }

        return failure();
      });
  patterns.onOp(
      "Col2Im", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input, blockShape, imageShape;
        SmallVector<int64_t> dilations, strides, pads;

        // TODO: The length of dilations should be len(imageShape), and the same
        // goes for strides. The length of pads should be 2 * len(imageShape).
        // But, as at the moment we are only supporting 3D or 4D input,
        // len(imageShape) must necessarily be 2, hence the lengths of the
        // default values.
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(imageShape, 1) ||
            binder.tensorOperandAtIndex(blockShape, 2) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerArrayAttr(dilations, "dilations",
                                       SmallVector<int64_t>{1, 1}) ||
            binder.s64IntegerArrayAttr(strides, "strides",
                                       SmallVector<int64_t>{1, 1}) ||
            binder.s64IntegerArrayAttr(pads, "pads",
                                       SmallVector<int64_t>{0, 0, 0, 0}))
          return failure();

        auto imageShapeTy = cast<Torch::ValueTensorType>(imageShape.getType());
        auto imageShapeSizes = imageShapeTy.getSizes();

        auto blockShapeTy = cast<Torch::ValueTensorType>(blockShape.getType());
        auto blockShapeSizes = blockShapeTy.getSizes();

        // Check that neither imageShape nor blockShape have dynamic shapes.
        if (imageShapeSizes[0] == Torch::kUnknownSize ||
            blockShapeSizes[0] == Torch::kUnknownSize) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "Dynamic shapes are not allowed for imageShape and blockShape");
        }

        // TODO: Add support for 5D input tensors.
        if (imageShapeSizes[0] != 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected length of imageShape to be equal to 2");
        }
        if (blockShapeSizes[0] != 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected length of blockShape to be equal to 2");
        }
        if (dilations.size() != 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected length of dilations to be equal to 2");
        }
        if (strides.size() != 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected length of strides to be equal to 2");
        }

        // TODO: Disable this check and add support for different
        // paddings on lower and higher ends of each axis.
        // Because we have already checked that imageShape has 2 elements,
        // we can safely assume that len(padding) will be 4.
        if (pads[0] != pads[2] || pads[1] != pads[3])
          return rewriter.notifyMatchFailure(
              binder.op, "padding on the lower end and the higher end "
                         "on each axis should be the same");

        // Since we know that the padding on the lower end and the higher
        // end on each axis is the same, we can reduce the size of the
        // padding list, and filter out the duplicate elements.
        // (Also, Torch::AtenCol2imOp requires len(padding) to be 2).
        SmallVector<int64_t> padOnEachAxis = {pads[0], pads[1]};
        Value dilationsList =
            createConstantIntList(binder, rewriter, dilations);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value paddingList =
            createConstantIntList(binder, rewriter, padOnEachAxis);

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));

        // Index the imageShape and blockShape tensors, as AtenCol2imOp expects
        // them to be int lists.
        auto select = [&](Value v, Value k,
                          Torch::ValueTensorType ty) -> Value {
          Value kTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
              binder.getLoc(),
              Torch::ValueTensorType::get(
                  binder.op->getContext(), ArrayRef<int64_t>{1},
                  rewriter.getIntegerType(64, /*signed*/ 1)),
              k);

          auto sel = rewriter.create<Torch::AtenIndexSelectOp>(
              binder.getLoc(),
              Torch::ValueTensorType::get(ty.getContext(), ArrayRef<int64_t>{1},
                                          ty.getOptionalDtype()),
              v, zero, kTensor);
          Value item = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), sel);
          return item;
        };

        SmallVector<Value> imageShapeContainer, blockShapeContainer;
        for (int64_t i = 0; i < imageShapeSizes[0]; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i));

          // Passing in the shapeType of each of these tensors avoids
          // repeated casts, as these have already been calculated.
          imageShapeContainer.push_back(select(imageShape, k, imageShapeTy));
          blockShapeContainer.push_back(select(blockShape, k, blockShapeTy));
        }

        Value imageShapeAsList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            imageShapeContainer);
        Value blockShapeAsList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            blockShapeContainer);

        rewriter.replaceOpWithNewOp<Torch::AtenCol2imOp>(
            binder.op, resultType, input, imageShapeAsList, blockShapeAsList,
            dilationsList, paddingList, stridesList);
        return success();
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

        auto weightTensorType = cast<Torch::ValueTensorType>(weight.getType());
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
        Value paddedInput = input;
        Value paddingList;
        if (padding.size() != 2 * (rank - 2)) {
          for (int64_t i : padding) {
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          }
          paddingList = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              cstPadding);
        } else {
          // ONNX offers pads in the format listing all starting dims, then all
          // ending dims, e.g. {t, l, b, r} for conv2d. Torch by default accepts
          // only starting dims, e.g. {t, l}. However, we can support padding at
          // the beginning and end of each dimension by first performing
          // torch.nn.functional.pad on the input. But this requires the pad
          // values to be rearranged since torch pad() takes pads in the order
          // rightmost dim start and end, then next to last, and so on, e.g. {l,
          // r, t, b}.
          bool matchedPads = true;
          for (unsigned i = 0; i < padding.size() / 2; i++) {
            if (padding[i] != padding[i + (padding.size() / 2)]) {
              matchedPads = false;
              break;
            }
          }
          if (matchedPads) {
            for (unsigned i = 0; i < padding.size() / 2; i++) {
              cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
            }
            paddingList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                cstPadding);
          } else {
            SmallVector<Value> padsRearrange;
            SmallVector<Value> inputPaddingList;
            for (uint32_t i = 0; i < padding.size() / 2; i++) {
              padsRearrange.emplace_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
              padsRearrange.emplace_back(rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(
                                       padding[(padding.size() / 2) + i])));
              inputPaddingList.emplace_back(
                  rewriter.create<Torch::ConstantIntOp>(
                      binder.getLoc(), rewriter.getI64IntegerAttr(0)));
            }
            // The conv op itself will have no padding since the actual padding
            // is performed using the torch.pad preceding it.
            paddingList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                inputPaddingList);
            Value padsSizeList =
                rewriter
                    .create<Torch::PrimListConstructOp>(
                        binder.getLoc(),
                        Torch::ListType::get(
                            rewriter.getType<Torch::IntType>()),
                        padsRearrange)
                    .getResult();
            Value modeVal = rewriter.create<Torch::ConstantStrOp>(
                binder.getLoc(), rewriter.getStringAttr("constant"));
            Value constantValue;
            auto inputTensorType =
                cast<Torch::ValueTensorType>(input.getType());
            if (isa<IntegerType>(inputTensorType.getDtype()))
              constantValue = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getI64IntegerAttr(0));
            if (isa<FloatType>(inputTensorType.getDtype()))
              constantValue = rewriter.create<Torch::ConstantFloatOp>(
                  binder.getLoc(), rewriter.getF64FloatAttr(0.0f));
            // Pad output shape must be computed explicitly from the pad values
            SmallVector<int64_t> newInputShape(inputTensorType.getSizes());
            for (uint32_t i = 0; i < padding.size() / 2; i++) {
              newInputShape[2 + i] +=
                  padding[i] + padding[(padding.size() / 2) + i];
            }
            auto padTy = rewriter.getType<Torch::ValueTensorType>(
                newInputShape, inputTensorType.getDtype());
            paddedInput = rewriter.create<Torch::AtenPadOp>(
                binder.getLoc(), padTy, input, padsSizeList, modeVal,
                constantValue);
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
            binder.op, resultType, paddedInput, weight, bias, stridesList,
            paddingList, dilationsList, transposed, outputPaddingList,
            cstGroup);
        return success();
      });
  patterns.onOp(
      "ConvInteger", 10,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET")
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");

        Torch::ValueTensorType resultType;
        Value input, weight, inputZp, weightZp;
        int64_t group;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(weight, 1) ||
            binder.s64IntegerAttr(group, "group", 1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
        auto weightTy = dyn_cast<Torch::ValueTensorType>(weight.getType());
        if (!weightTy || !weightTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "Expected weight type having sizes");
        ArrayRef<int64_t> weightShape = weightTy.getSizes();
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
              if (weightShape[i + 2] != kernelShape[i])
                return rewriter.notifyMatchFailure(
                    binder.op, "unsupported conversion: kernel_shape value "
                               "should be equal to the weight tensor shape");
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
        SmallVector<int64_t> defaultPadding(rank - 2, 0),
            defaultStrides(rank - 2, 1), defaultDilations(rank - 2, 1);
        // Padding for the beginning and ending along each spatial axis, it can
        // take any value greater than or equal to 0. The value represent the
        // number of pixels added to the beginning and end part of the
        // corresponding axis. pads format should be as follow [x1_begin,
        // x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added
        // at the beginning of axis i and xi_end, the number of pixels added at
        // the end of axis i.
        if (binder.s64IntegerArrayAttr(padding, "pads", defaultPadding))
          return failure();
        if (padding.size() != rank - 2 && padding.size() != 2 * (rank - 2))
          return rewriter.notifyMatchFailure(
              binder.op, "padding list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(dilations, "dilations",
                                       defaultDilations))
          return failure();
        if (dilations.size() != rank - 2)
          return rewriter.notifyMatchFailure(
              binder.op,
              "dilations list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(strides, "strides", defaultStrides))
          return failure();
        if (strides.size() != rank - 2)
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");

        Value scale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(1.0));
        if (binder.tensorOperandAtIndex(inputZp, 2)) {
          inputZp = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(0));
        } else {
          inputZp = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), inputZp);
        }
        if (binder.tensorOperandAtIndex(weightZp, 3))
          weightZp = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(0));
        // TODO: support per channel quantization if weightZp is a 1-D tensor
        if (auto zpTy = dyn_cast<Torch::ValueTensorType>(weightZp.getType())) {
          for (auto dim : zpTy.getSizes())
            if (dim != 1)
              return failure();
          weightZp = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), weightZp);
        }

        SmallVector<Value> cstPadding;
        if (padding.size() != 2 * (rank - 2)) {
          for (int64_t i : padding) {
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          }
        } else {
          for (unsigned i = 0; i < padding.size() / 2; i++) {
            if (padding[i] != padding[i + (padding.size() / 2)])
              // TODO: Add support for different padding values for the
              // beginning and ending along each spatial axis
              return rewriter.notifyMatchFailure(
                  binder.op,
                  "unsupported conversion: padding values for the beginning "
                  "and ending along each spatial axis must be equal");
            cstPadding.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(padding[i])));
          }
        }

        Value paddingList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            cstPadding);
        Value dilationsList =
            createConstantIntList(binder, rewriter, dilations);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value outputPaddingList =
            createConstantIntList(binder, rewriter, {0, 0});
        Value transposed =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value bias = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstGroup = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(group));

        Type inputQTy = getQTorchTypeFromTorchIntType(inputTy);
        Type weightQTy = getQTorchTypeFromTorchIntType(weightTy);
        input = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), inputQTy, input, scale, inputZp);
        weight = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), weightQTy, weight, scale, weightZp);

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

        auto weightTensorType = cast<Torch::ValueTensorType>(weight.getType());
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
        Torch::ValueTensorType resultType;
        Value operand, axisTensor;
        int64_t exclusive, reverse;
        if (binder.tensorOperands(operand, axisTensor) ||
            binder.s64IntegerAttr(exclusive, "exclusive", 0) ||
            binder.s64IntegerAttr(reverse, "reverse", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        Torch::BaseTensorType resultTensorType =
            cast<Torch::BaseTensorType>(resultType);
        if (!resultTensorType.hasDtype()) {
          return rewriter.notifyMatchFailure(
              binder.op, "expected result type to have a dtype");
        }

        // deal with neg axis: if (axis < 0) axis += rank
        int64_t rank =
            cast<Torch::ValueTensorType>(operand.getType()).getSizes().size();
        Value rankVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), rank));
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));

        Value axisScalar = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), axisTensor);
        Value isNegative = rewriter.create<Torch::AtenLtIntOp>(
            binder.getLoc(), axisScalar, cstZero);
        isNegative =
            rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(), isNegative);
        Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
            binder.getLoc(), isNegative, rankVal);
        Value axis = rewriter.create<Torch::AtenAddIntOp>(
            binder.getLoc(), axisScalar, finalOffset);
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value res;
        if (reverse) {
          Value dims = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{axis});
          Value flip = rewriter.create<Torch::AtenFlipOp>(
              binder.getLoc(), resultType, operand, dims);
          Value cumsum = rewriter.create<Torch::AtenCumsumOp>(
              binder.getLoc(), resultType, flip, axis, none);
          res = rewriter.create<Torch::AtenFlipOp>(binder.getLoc(), resultType,
                                                   cumsum, dims);
        } else {
          res = rewriter.create<Torch::AtenCumsumOp>(
              binder.getLoc(), resultType, operand, axis, none);
        }

        if (exclusive)
          res = rewriter.create<Torch::AtenSubTensorOp>(
              binder.getLoc(), resultType, res, operand, cstOne);
        rewriter.replaceOp(binder.op, res);
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
        auto inputTy = dyn_cast<Torch::BaseTensorType>(input.getType());
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
      "DeformConv", 19,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        auto loc = binder.getLoc();

        // get operands
        llvm::SmallVector<Value> operands;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();
        if (operands.size() < 3 || operands.size() > 5)
          return failure();
        auto inputType =
            dyn_cast<Torch::ValueTensorType>(operands[0].getType());
        if (!inputType || !inputType.hasSizes() ||
            inputType.getSizes().size() != 4)
          return rewriter.notifyMatchFailure(
              binder.op, "Unsupported: DeformConv with input rank != 4");
        unsigned rank = inputType.getSizes().size();
        auto weightType =
            dyn_cast<Torch::ValueTensorType>(operands[1].getType());
        if (!weightType || !weightType.hasSizes())
          return failure();
        auto offsetType =
            dyn_cast<Torch::ValueTensorType>(operands[2].getType());
        if (!offsetType || !offsetType.hasSizes())
          return failure();

        // get attributes
        SmallVector<int64_t> dilations, kernelShape, pads, strides;
        SmallVector<int64_t> defaultDilations(rank - 2, 0);
        SmallVector<int64_t> defaultPads(2 * (rank - 2), 0);
        SmallVector<int64_t> defaultStrides(rank - 2, 1);
        int64_t group, offsetGroup;
        if (binder.s64IntegerArrayAttr(dilations, "dilations",
                                       defaultDilations) ||
            binder.s64IntegerArrayAttr(kernelShape, "kernel_shape", {}) ||
            binder.s64IntegerArrayAttr(pads, "pads", defaultPads) ||
            binder.s64IntegerArrayAttr(strides, "strides", defaultStrides) ||
            binder.s64IntegerAttr(group, "group", 1) ||
            binder.s64IntegerAttr(offsetGroup, "offset_group", 1))
          return failure();

        for (unsigned i = 0; i < rank - 2; i++) {
          if (pads[i] != pads[rank + i - 2])
            return rewriter.notifyMatchFailure(
                binder.op, "unsupported: asymmetric padding");
        }

        // Identify and assign names to operands
        Value input, weight, offset, bias, mask;
        bool useMask = false;
        input = operands[0];
        weight = operands[1];
        offset = operands[2];
        if (operands.size() == 4) {
          auto unknownOpdRank = Torch::getTensorRank(operands[3]);
          if (!unknownOpdRank)
            return failure();
          if (*unknownOpdRank == 1)
            bias = operands[3];
          else if (*unknownOpdRank == rank) {
            mask = operands[3];
            useMask = true;
          } else
            llvm_unreachable("onnx.DeformConv: optional 4th operand of "
                             "unexpected rank encountered");
        }
        if (operands.size() == 5) {
          bias = operands[3];
          mask = operands[4];
          useMask = true;
        }

        // assign default operand values if necessary
        ArrayRef<int64_t> weightSizes = weightType.getSizes();
        ArrayRef<int64_t> offsetSizes = offsetType.getSizes();
        if (!bias) {
          int64_t outputChannels = weightSizes[0];
          SmallVector<int64_t> biasShape(1, outputChannels);
          Value biasShapeList = mlir::torch::onnx_c::createConstantIntList(
              binder, rewriter, biasShape);
          Value cstZero = Torch::getConstantWithGivenDtypeAndValue(
              rewriter, loc, 0.0f, inputType.getDtype());
          bias =
              Torch::createInitTensor(rewriter, loc,
                                      rewriter.getType<Torch::ValueTensorType>(
                                          biasShape, inputType.getDtype()),
                                      cstZero, biasShapeList);
        }
        if (!mask) {
          int64_t batchSize = inputType.getSizes()[0];
          int64_t kernelHeight = weightSizes[2];
          int64_t kernelWidth = weightSizes[3];
          int64_t outputHeight = offsetSizes[2];
          int64_t outputWidth = offsetSizes[3];
          int64_t maskDimOne = offsetGroup * kernelHeight * kernelWidth;
          SmallVector<int64_t> maskShape(
              {batchSize, maskDimOne, outputHeight, outputWidth});
          Value cstOne = Torch::getConstantWithGivenDtypeAndValue(
              rewriter, loc, 1.0f, inputType.getDtype());
          Value maskShapeList = mlir::torch::onnx_c::createConstantIntList(
              binder, rewriter, maskShape);
          mask =
              Torch::createInitTensor(rewriter, loc,
                                      rewriter.getType<Torch::ValueTensorType>(
                                          maskShape, inputType.getDtype()),
                                      cstOne, maskShapeList);
        }

        // get attributes as constant values
        SmallVector<Value> dilationValues, padValues, strideValues;
        for (auto i : dilations)
          dilationValues.push_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
        for (auto i : pads)
          padValues.push_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
        for (auto i : strides)
          strideValues.push_back(rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i)));
        Value groupValue = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(group));
        Value offsetGroupValue = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(offsetGroup));
        Value useMaskValue = rewriter.create<Torch::ConstantBoolOp>(
            loc, rewriter.getBoolAttr(useMask));
        rewriter.replaceOpWithNewOp<Torch::TorchvisionDeformConv2dOp>(
            binder.op, resultType, input, weight, offset, mask, bias,
            strideValues[0], strideValues[1], padValues[0], padValues[1],
            dilationValues[0], dilationValues[1], groupValue, offsetGroupValue,
            useMaskValue);
        return success();
      });
  patterns.onOp(
      "Det", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        if (binder.tensorOperand(input) || binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenLinalgDetOp>(binder.op,
                                                            resultType, input);
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

        auto operandTy = cast<Torch::ValueTensorType>(operand.getType());

        auto scaleTy = dyn_cast<Torch::ValueTensorType>(scale.getType());
        if (!scaleTy || !scaleTy.hasSizes())
          return rewriter.notifyMatchFailure(binder.op, "requires known rank");
        if (!resultType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "requires known result dtype");
        if (scaleTy.getSizes().size() == 0 ||
            (scaleTy.getSizes().size() == 1 && scaleTy.getSizes()[0] == 1)) {
          auto qTensorTy = getQTorchTypeFromTorchIntType(operandTy);
          if (!qTensorTy) {
            return rewriter.notifyMatchFailure(binder.op,
                                               "unsupported result dtype");
          }

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

        return rewriter.notifyMatchFailure(binder.op,
                                           "unimplemented: non-scalar scale");
      });
  patterns.onOp("Div", 7,
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
              dyn_cast<Torch::BaseTensorType>(trainVal.getType());
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
            auto val = cast<DenseElementsAttr>(valueTensorLiteralOp.getValue())
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
  patterns.onOp(
      "DynamicQuantizeLinear", 11,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Value input;
        Torch::ValueTensorType resultType, scaleType, zeroPointType;
        if (binder.tensorOperand(input) ||
            binder.tensorResultTypeAtIndex(resultType, 0) ||
            binder.tensorResultTypeAtIndex(scaleType, 1) ||
            binder.tensorResultTypeAtIndex(zeroPointType, 2))
          return failure();

        Value scale, zeroPoint;

        // scale = ( max(0, max(input)) - min(0, min(input)) ) / 255
        Value inputMax =
            rewriter.create<Torch::AtenMaxOp>(loc, scaleType, input);
        Value inputMin =
            rewriter.create<Torch::AtenMinOp>(loc, scaleType, input);
        Value constantZero = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0));
        Value constantOne = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        Value zeroTensor =
            createRank0Tensor(rewriter, loc, scaleType, constantZero);
        Value inputMaxW0 = rewriter.create<Torch::AtenMaximumOp>(
            loc, scaleType, inputMax, zeroTensor);
        Value inputMinW0 = rewriter.create<Torch::AtenMinimumOp>(
            loc, scaleType, inputMin, zeroTensor);
        Value scaleTensor = rewriter.create<Torch::AtenSubTensorOp>(
            loc, scaleType, inputMaxW0, inputMinW0, constantOne);
        // Note: the following is hard-coded for ui8
        Value width = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(255));
        Value widthTensor = createRank0Tensor(rewriter, loc, scaleType, width);
        scaleTensor = rewriter.create<Torch::AtenDivTensorOp>(
            loc, scaleType, scaleTensor, widthTensor);
        // compute the preZeroPoint = 0 - (inputMin/scale)
        // compute the zeroPoint = cast ( round (clip or saturate
        // (preZeroPoint)))
        Value preZeroPoint = rewriter.create<Torch::AtenDivTensorOp>(
            loc, scaleType, inputMin, scaleTensor);
        preZeroPoint = rewriter.create<Torch::AtenSubTensorOp>(
            loc, scaleType, zeroTensor, preZeroPoint, constantOne);
        // saturate to interval [0, 255]
        preZeroPoint = rewriter.create<Torch::AtenClampOp>(
            loc, scaleType, preZeroPoint, /*min=*/constantZero, /*max=*/width);
        // round, then cast to uint8
        preZeroPoint =
            rewriter.create<Torch::AtenRoundOp>(loc, scaleType, preZeroPoint);
        Type qTy = rewriter.getType<Torch::QUInt8Type>();
        auto qTensorTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), qTy);
        auto torchqTy = Torch::getScalarTypeForType(qTy);
        Value tyConst = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    static_cast<int64_t>(torchqTy)));
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
        Value zeroPointTensor = rewriter.create<Torch::AtenToDtypeOp>(
            loc, zeroPointType, preZeroPoint, tyConst,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/none);
        // extract scale and zeroPoint scalars to pass to
        // AtenQuantizePerTensorOp
        zeroPoint = rewriter.create<Torch::AtenItemOp>(
            loc, rewriter.getType<Torch::IntType>(), zeroPointTensor);
        scale = rewriter.create<Torch::AtenItemOp>(
            loc, rewriter.getType<Torch::FloatType>(), scaleTensor);
        Value quantizedTensor = rewriter.create<Torch::AtenQuantizePerTensorOp>(
            loc, qTensorTy, input, scale, zeroPoint, tyConst);
        // get uint8 tensor output
        Value output = rewriter.create<Torch::AtenIntReprOp>(loc, resultType,
                                                             quantizedTensor);
        rewriter.replaceOp(binder.op, {output, scaleTensor, zeroPointTensor});
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
        auto loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value data, shape;
        if (binder.tensorOperands(data, shape) ||
            binder.tensorResultType(resultType))
          return failure();

        auto dataType = cast<Torch::BaseTensorType>(data.getType());
        auto shapeType = cast<Torch::BaseTensorType>(shape.getType());
        if (!dataType.hasSizes() || !shapeType.hasSizes())
          return failure();

        auto shapeSizes = shapeType.getSizes();
        int64_t dataRank = dataType.getSizes().size();
        int64_t shapeRank = shapeSizes.size();
        if (shapeRank != 1 || shapeSizes[0] == Torch::kUnknownSize)
          return failure();

        auto rankDifference = dataRank - shapeSizes[0];

        SmallVector<int64_t> selectSizes;
        Type selectResultType = shapeType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), shapeType.getOptionalDtype());
        // Variable to store 1-D onnx shape tensor, shapeSizes[0] has the
        // dimension size
        // A constant zero value
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        // Variable to store pytorch int list of shape (dimension)
        SmallVector<Value> dimList;

        // Convert the shape tensor from vector of int64_t to torch int list as
        // we are using torch implementation Torch::AtenBroadcastToOp which
        // takes list of int
        for (int i = 0; i < shapeSizes[0]; i++) {
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              loc, selectResultType, shape, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              loc, rewriter.getType<Torch::IntType>(), extract);

          if (i + rankDifference >= 0) {
            Value iv =
                rewriter.create<Torch::ConstantIntOp>(loc, i + rankDifference);
            auto sz = rewriter.create<Torch::AtenSizeIntOp>(
                loc, rewriter.getType<Torch::IntType>(), data, iv);
            dim = rewriter.create<Torch::PrimMaxIntOp>(loc, dim, sz);
          }

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
      "EyeLike", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t dtypeIntOnnx, diagonalIndex;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(dtypeIntOnnx, "dtype", 1) ||
            binder.s64IntegerAttr(diagonalIndex, "k", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        auto operandTy = cast<Torch::ValueTensorType>(operand.getType());
        SmallVector<int64_t> shape(operandTy.getSizes());
        for (unsigned i = 0; i < shape.size(); i++) {
          if (shape[i] == ShapedType::kDynamic)
            shape[i] = Torch::kUnknownSize;
        }

        Value cst0 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cst1 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value nVal = rewriter.create<Torch::AtenSizeIntOp>(binder.getLoc(),
                                                           operand, cst0);
        Value mVal = rewriter.create<Torch::AtenSizeIntOp>(binder.getLoc(),
                                                           operand, cst1);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        std::optional<int64_t> dtypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(dtypeIntOnnx);
        if (!dtypeIntTorch.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }
        Value dtypeVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(dtypeIntTorch.value()));

        // diagonalIndex = 0 populates the main diagonal
        // diagonalIndex > 0 populates an upper diagonal
        // diagonalIndex < 0 populates a lower diagonal
        if (diagonalIndex == 0) {
          rewriter.replaceOpWithNewOp<Torch::AtenEyeMOp>(
              binder.op, resultType, nVal, mVal, dtypeVal, noneVal, noneVal,
              noneVal);
          return success();
        }

        Value diagVal = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(),
            rewriter.getI64IntegerAttr(std::abs(diagonalIndex)));
        Value newN, newM, dimVal, startVal;
        // get shapes of main diag eye op and zeros op
        if (diagonalIndex > 0) {
          newN = nVal;
          newM = rewriter.create<Torch::AtenSubIntOp>(binder.getLoc(), mVal,
                                                      diagVal);
          if (shape[1] != Torch::kUnknownSize) {
            shape[1] -= diagonalIndex;
          }
          dimVal = cst1;
          startVal = mVal;
        } else {
          newN = rewriter.create<Torch::AtenSubIntOp>(binder.getLoc(), nVal,
                                                      diagVal);
          newM = mVal;
          if (shape[0] != Torch::kUnknownSize) {
            shape[0] += diagonalIndex;
          }
          dimVal = cst0;
          startVal = nVal;
        }

        // create main diag eye op
        auto eyeResultType = rewriter.getType<Torch::ValueTensorType>(
            shape, resultType.getOptionalDtype());
        Value eyeOp = rewriter.create<Torch::AtenEyeMOp>(
            binder.getLoc(), eyeResultType, newN, newM, dtypeVal, noneVal,
            noneVal, noneVal);
        // create zeros op
        SmallVector<Value> zerosShapeValues = {nVal, mVal};
        Value zerosShapeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            zerosShapeValues);
        Value zerosOp = rewriter.create<Torch::AtenZerosOp>(
            binder.getLoc(), resultType, zerosShapeList, dtypeVal, noneVal,
            noneVal, noneVal);

        // embeds the values of the eye matrix into zeros
        rewriter.replaceOpWithNewOp<Torch::AtenSliceScatterOp>(
            binder.op, resultType, zerosOp, eyeOp, dimVal,
            /*start=*/diagVal, /*end=*/startVal, /*step=*/cst1);
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
            cast<Torch::BaseTensorType>(shape.getType());
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
          attr =
              DenseElementsAttr::get(resultType.toBuiltinTensor(),
                                     rewriter.getFloatAttr(resultDType, 0.0));
        }

        // If its a dense resource attr we need to convert to a dense type:
        if (DenseResourceElementsAttr rattr =
                dyn_cast_or_null<DenseResourceElementsAttr>(attr)) {
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
          auto denseAttr = cast<DenseElementsAttr>(attr);
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
  patterns.onOp(
      "Einsum", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> tensors;
        std::string equation;
        if (binder.tensorOperands(tensors, binder.op->getNumOperands()) ||
            binder.customOpNameStringAttr(equation, "equation") ||
            binder.tensorResultType(resultType))
          return failure();
        Type listElemType =
            cast<Torch::BaseTensorType>(tensors[0].getType())
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = rewriter.create<Torch::PrimListConstructOp>(
            binder.op->getLoc(), listType, tensors);
        Value cstEquation = rewriter.create<Torch::ConstantStrOp>(
            binder.getLoc(), rewriter.getType<Torch::StringType>(),
            rewriter.getStringAttr(equation));
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        rewriter.replaceOpWithNewOp<Torch::AtenEinsumOp>(
            binder.op, resultType, cstEquation, tensorList, /*path=*/cstNone);
        return success();
      });
  patterns.onOp(
      "BlackmanWindow", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value size;
        Torch::ValueTensorType resultType;
        int64_t periodic, output_datatype;
        if (binder.tensorOperand(size) ||
            binder.s64IntegerAttr(output_datatype, "output_datatype", 1) ||
            binder.s64IntegerAttr(periodic, "periodic", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        Location loc = binder.getLoc();
        Value a0 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.42));
        Value a1 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(-0.5));
        Value a2 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.08));

        auto windowFunctionResult =
            windowFunctionImpl(binder, rewriter, size, a0, a1, a2, resultType,
                               output_datatype, periodic);

        if (failed(windowFunctionResult))
          return failure();

        return success();
      });

  patterns.onOp(
      "HannWindow", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value size;
        Torch::ValueTensorType resultType;
        int64_t periodic, output_datatype;
        if (binder.tensorOperand(size) ||
            binder.s64IntegerAttr(output_datatype, "output_datatype", 1) ||
            binder.s64IntegerAttr(periodic, "periodic", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        Location loc = binder.getLoc();
        Value a0 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.5));
        Value a1 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(-0.5));
        Value a2 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.0));

        auto windowFunctionResult =
            windowFunctionImpl(binder, rewriter, size, a0, a1, a2, resultType,
                               output_datatype, periodic);

        if (failed(windowFunctionResult))
          return failure();

        return success();
      });

  patterns.onOp(
      "HammingWindow", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value size;
        Torch::ValueTensorType resultType;
        int64_t periodic, output_datatype;
        if (binder.tensorOperand(size) ||
            binder.s64IntegerAttr(output_datatype, "output_datatype", 1) ||
            binder.s64IntegerAttr(periodic, "periodic", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        Location loc = binder.getLoc();
        Value a0 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.543478));
        Value a1 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(-0.456522));
        Value a2 = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(0.0));

        auto windowFunctionResult =
            windowFunctionImpl(binder, rewriter, size, a0, a1, a2, resultType,
                               output_datatype, periodic);

        if (failed(windowFunctionResult))
          return failure();

        return success();
      });

  patterns.onOp(
      "DFT", 20, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value inTensor, dftLength, axis;
        Torch::ValueTensorType resultType;
        int64_t inverse, onesided;
        if (binder.tensorOperandAtIndex(inTensor, 0) ||
            binder.s64IntegerAttr(inverse, "inverse", 0) ||
            binder.s64IntegerAttr(onesided, "onesided", 0) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op, "Input Tensor / attrs / resultType bind failed");
        if (!binder.tensorOperandAtIndex(dftLength, 1)) {
          // Convert to int and pass as n
          dftLength = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), dftLength);
        } else {
          // Default for torch is None
          dftLength = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        }
        // Default is same for onnx and torch
        if (!binder.tensorOperandAtIndex(axis, 2)) {
          // convert to int and pass to dims
          axis = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), axis);
        } else {
          // Default in torch is -1 and onnx is -2 (since -1 is for real / img)
          axis = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(-2));
        }

        if (onesided == 1)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unsupported option : onesided");
        // norm default string attr
        Value norm = rewriter.create<Torch::ConstantStrOp>(
            binder.getLoc(), rewriter.getStringAttr(Twine("backward")));
        // Convert from [....., 2] complex number repr for fft consumption.
        Torch::ValueTensorType inType =
            binder.toValidTensorType(inTensor.getType());
        int64_t lastIndex = inType.getSizes().back();
        if (lastIndex != 1 && lastIndex != 2)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected input tensor to have dims [..., 1] or [..., 2]");

        // concat with zeros to make it [..., 2]
        Value inForComplexVal = inTensor;
        ArrayRef<int64_t> inForComplexSizes = inType.getSizes().drop_back();
        if (lastIndex == 1) {
          Value constZeroVal = rewriter.create<Torch::ConstantFloatOp>(
              binder.getLoc(), rewriter.getF64FloatAttr(0));
          Value constOne = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Value constZero = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(0));
          Value padSizeList =
              rewriter
                  .create<Torch::PrimListConstructOp>(
                      binder.getLoc(),
                      Torch::ListType::get(rewriter.getType<Torch::IntType>()),
                      SmallVector<Value>({constZero, constOne}))
                  .getResult();
          Value modeVal = rewriter.create<Torch::ConstantStrOp>(
              binder.getLoc(), rewriter.getStringAttr("constant"));
          SmallVector<int64_t> resSize(inForComplexSizes);
          resSize.push_back(2);
          inForComplexVal = rewriter.create<Torch::AtenPadOp>(
              binder.getLoc(),
              inType.getWithSizesAndDtype(resSize, inType.getOptionalDtype()),
              inTensor, padSizeList, modeVal, constZeroVal);
        }
        Type inComplexTensorType = Torch::ValueTensorType::get(
            binder.op->getContext(), inForComplexSizes,
            mlir::ComplexType::get(inType.getDtype()));
        Value inComplexTensor = rewriter.create<Torch::AtenViewAsComplexOp>(
            binder.getLoc(), inComplexTensorType, inForComplexVal);
        Value ftOp;
        if (inverse == 0) {
          ftOp = rewriter.create<Torch::AtenFftFftOp>(
              binder.getLoc(), inComplexTensorType, inComplexTensor,
              /*n = */ dftLength, /*dim = */ axis, /*norm = */ norm);
        } else {
          ftOp = rewriter.create<Torch::AtenFftIfftOp>(
              binder.getLoc(), inComplexTensorType, inComplexTensor,
              /*n = */ dftLength, /*dim = */ axis, /*norm = */ norm);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenViewAsRealOp>(binder.op,
                                                             resultType, ftOp);
        return success();
      });
}
