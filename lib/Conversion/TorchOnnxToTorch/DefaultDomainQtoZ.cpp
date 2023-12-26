//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

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
  patterns.onOp("Reciprocal", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenReciprocalOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Relu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::AtenReluOp>(binder.op, resultType,
                                                       x);
        return success();
      });
  patterns.onOp("Round", 11,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenRoundOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "ScatterElements", 18,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> valList;
        int64_t axis;
        std::string reduction;
        int64_t numOperands = binder.op->getNumOperands();
        if (binder.tensorOperands(valList, numOperands) ||
            binder.s64IntegerAttr(axis, "axis", 0) ||
            binder.customOpNameStringAttr(reduction, "reduction", "none") ||
            binder.tensorResultType(resultType))
          return failure();

        Value data = valList[0];
        Value indices = valList[1];
        Value updates = valList[2];

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(data.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

        if (reduction == "none") {
          rewriter.replaceOpWithNewOp<Torch::AtenScatterSrcOp>(
              binder.op, resultType, data, constAxis, indices, updates);
          return success();
        }

        // TODO: Implement max and min cases
        if (reduction == "mul") {
          reduction = "multiply";
        } else if (reduction == "max" || reduction == "min") {
          return rewriter.notifyMatchFailure(
              binder.op, "max/min reduction unsupported for scatter elements");
        }

        Value cstStrReduction =
            rewriter.create<Torch::ConstantStrOp>(binder.getLoc(), reduction);

        rewriter.replaceOpWithNewOp<Torch::AtenScatterReduceOp>(
            binder.op, resultType, data, constAxis, indices, updates,
            cstStrReduction);
        return success();
      });
  patterns.onOp(
      "Sigmoid", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::AtenSigmoidOp>(binder.op, resultType,
                                                          x);
        return success();
      });
  patterns.onOp("Sin", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenSinOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Tanh", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenTanhOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("Sqrt", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenSqrtOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Sub", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        Value y;
        if (binder.tensorOperands(x, y) || binder.tensorResultType(resultType))
          return failure();
        Value const1 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        rewriter.replaceOpWithNewOp<Torch::AtenSubTensorOp>(
            binder.op, resultType, x, y, const1);
        return success();
      });
  patterns.onOp(
      "Sum", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        if (binder.op->getNumOperands() == 1) {
          Torch::ValueTensorType resultType;
          Value x;
          if (binder.tensorOperand(x) || binder.tensorResultType(resultType))
            return failure();
          rewriter.replaceOp(binder.op, x);
          return success();
        }
        Torch::ValueTensorType resultType;
        SmallVector<Value> valList;
        int64_t numOperands = binder.op->getNumOperands();
        if (binder.tensorOperands(valList, numOperands) ||
            binder.tensorResultType(resultType))
          return failure();
        Value const1 = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        // Short circuit to binary add
        if (numOperands == 2) {
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, valList[0], valList[1], const1);
          return success();
        }
        // When binder.op->getNumOperands() > 2
        auto baseType = Torch::ValueTensorType::getWithLeastStaticInformation(
            binder.op->getContext());
        Value curr = rewriter.create<Torch::AtenAddTensorOp>(
            binder.getLoc(), resultType, valList[0], valList[1], const1);
        for (int i = 2; i < numOperands; i++) {
          if (i == numOperands - 1) {
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), resultType, curr, valList[i], const1);
          } else {
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), baseType, curr, valList[i], const1);
          }
        }
        rewriter.replaceOp(binder.op, curr);
        return success();
      });
  patterns.onOp("Where", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  SmallVector<Value> valList;
                  int64_t numOperands = binder.op->getNumOperands();
                  if (binder.tensorOperands(valList, numOperands) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  Value condition = valList[0];
                  Value x = valList[1];
                  Value y = valList[2];
                  rewriter.replaceOpWithNewOp<Torch::AtenWhereSelfOp>(
                      binder.op, resultType, condition, x, y);
                  return success();
                });
  patterns.onOp(
      "Xor", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value x;
        Value y;
        if (binder.tensorOperands(x, y) || binder.tensorResultType(resultType))
          return failure();
        rewriter.replaceOpWithNewOp<Torch::AtenLogicalXorOp>(binder.op,
                                                             resultType, x, y);
        return success();
      });
  patterns.onOp(
      "Squeeze", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType))
          return failure();
        Torch::BaseTensorType axesType =
            axes.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = axesType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), axesType.getOptionalDtype());
        auto sizes =
            dyn_cast<Torch::ValueTensorType>(axes.getType()).getSizes();
        if (sizes.size() == 0) {
          rewriter.replaceOpWithNewOp<Torch::AtenSqueezeOp>(binder.op,
                                                            resultType, data);
          return success();
        }
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        Value adjustment = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    adjustmentInt));
        for (int i = 0; i < sizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, axes, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          // deal with neg axis: if (axis < 0) axis += rank
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), dim, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, adjustment);
          Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
              binder.getLoc(), dim, finalOffset);
          dimList.push_back(finalDim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        rewriter.replaceOpWithNewOp<Torch::PrimsSqueezeOp>(
            binder.op, resultType, data, dimValueList);
        return success();
      });
  patterns.onOp(
      "Unsqueeze", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // Unlike squeeze where we are able to lower to Torch::PrimsSqueezeOp,
        // pytorch does not support torch.unsqueeze to insert multiple new dims.
        // discussion can be found here:
        // https://github.com/pytorch/pytorch/issues/9410
        // So, for now, we unroll into multiple unsqueezes.
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType))
          return failure();
        Torch::BaseTensorType axesType =
            axes.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = axesType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), axesType.getOptionalDtype());
        auto sizes =
            dyn_cast<Torch::ValueTensorType>(axes.getType()).getSizes();
        if (sizes.size() == 0) {
          rewriter.replaceOp(binder.op, data);
          return success();
        }
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        Value adjustment = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    adjustmentInt));
        for (int i = 0; i < sizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, axes, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          // deal with neg axis: if (axis < 0) axis += rank
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), dim, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, adjustment);
          Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
              binder.getLoc(), dim, finalOffset);
          dimList.push_back(finalDim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value updatedAxes = rewriter.create<Torch::AtenTensorOp>(
            binder.getLoc(),
            axesType.getWithSizesAndDtype(sizes, axesType.getOptionalDtype()),
            dimValueList, /*dtype=*/noneVal, /*device=*/noneVal, cstFalse);
        // Sort the  list of dims, so we don't run into this situation:
        // data.sizes = [2, 3, 4]
        // dims = [4, 0]
        // index 4 will be invalid to add a singleton dimension because
        // data.sizes.size == 3 We have to work with sorted dims to avoid this
        // situation.
        auto sortIndicesType = axesType.getWithSizesAndDtype(
            axesType.getOptionalSizes(),
            IntegerType::get(binder.op->getContext(), 64, IntegerType::Signed));
        auto sortOpResult = rewriter.create<Torch::AtenSortOp>(
            binder.getLoc(), axes.getType(), sortIndicesType, updatedAxes, zero,
            cstFalse);
        Value result;
        auto baseType = Torch::ValueTensorType::getWithLeastStaticInformation(
            binder.op->getContext());
        // Go through the updated, sorted axes. Do unsqueeze for each dim.
        for (int i = 0; i < sizes[0]; i++) {
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, sortOpResult->getResult(0),
              zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          if (sizes[0] == 1) {
            result = rewriter.create<Torch::AtenUnsqueezeOp>(
                binder.getLoc(), resultType, data, dim);
          } else if (i == 0) {
            result = rewriter.create<Torch::AtenUnsqueezeOp>(
                binder.getLoc(), baseType, data, dim);
          } else if (i == sizes[0] - 1) {
            result = rewriter.create<Torch::AtenUnsqueezeOp>(
                binder.getLoc(), resultType, result, dim);
          } else {
            result = rewriter.create<Torch::AtenUnsqueezeOp>(
                binder.getLoc(), baseType, result, dim);
          }
        }
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp(
      "Softmax", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        int64_t axis;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.tensorResultType(resultType))
          return failure();

        // ONNX allows negative axis.
        if (axis < 0)
          axis +=
              cast<Torch::ValueTensorType>(input.getType()).getSizes().size();

        Value constAxis = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        rewriter.replaceOpWithNewOp<Torch::AtenSoftmaxIntOp>(
            binder.op, resultType, input, constAxis, /*dtype=*/noneVal);
        return success();
      });

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
      "ReduceSum", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();
        Torch::BaseTensorType axesType =
            axes.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = axesType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), axesType.getOptionalDtype());
        auto sizes =
            dyn_cast<Torch::ValueTensorType>(axes.getType()).getSizes();
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        // Deal with case when axes is empty
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            Value keepDimsConstInt = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), keepDims));
            Value keepDimsBool = rewriter.create<Torch::AtenBoolIntOp>(
                binder.getLoc(), keepDimsConstInt);
            rewriter.replaceOpWithNewOp<Torch::AtenSumDimIntListOp>(
                binder.op, resultType, data, /*dim=*/noneVal,
                /*keepdim=*/keepDimsBool, /*dtype=*/noneVal);
          } else {
            rewriter.replaceOp(binder.op, data);
          }
          return success();
        }
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        Value adjustment = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    adjustmentInt));
        // convert axes (tensor) into torch int list while dealing with neg axis
        for (int i = 0; i < sizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, axes, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          // deal with neg axis: if (axis < 0) axis += rank
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), dim, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, adjustment);
          Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
              binder.getLoc(), dim, finalOffset);
          dimList.push_back(finalDim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        Value keepDimBool;
        if (keepDims == 1) {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        } else {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenSumDimIntListOp>(
            binder.op, resultType, data, dimValueList, keepDimBool,
            /*dtype=*/noneVal);
        return success();
      });
  patterns.onOp(
      "ReduceMean", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();
        Torch::BaseTensorType axesType =
            axes.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = axesType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), axesType.getOptionalDtype());
        auto sizes =
            dyn_cast<Torch::ValueTensorType>(axes.getType()).getSizes();
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        // deal with case when axes is empty
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            Value keepDimsConstInt = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), keepDims));
            Value keepDimsBool = rewriter.create<Torch::AtenBoolIntOp>(
                binder.getLoc(), keepDimsConstInt);
            rewriter.replaceOpWithNewOp<Torch::AtenMeanDimOp>(
                binder.op, resultType, data, /*dim=*/noneVal, keepDimsBool,
                /*dtype=*/noneVal);
          } else {
            rewriter.replaceOp(binder.op, data);
          }
          return success();
        }
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        Value adjustment = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    adjustmentInt));
        // convert axes (tensor) into torch int list while dealing with neg axis
        for (int i = 0; i < sizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, axes, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          // deal with neg axis: if (axis < 0) axis += rank
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), dim, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, adjustment);
          Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
              binder.getLoc(), dim, finalOffset);
          dimList.push_back(finalDim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        Value keepDimBool;
        if (keepDims == 1) {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        } else {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenMeanDimOp>(
            binder.op, resultType, data, dimValueList, keepDimBool,
            /*dtype=*/noneVal);
        return success();
      });
  patterns.onOp(
      "ReduceMin", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // AtenAminOp allows us to pass a list of dims
        Torch::ValueTensorType resultType;
        Value data;
        Value axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        // Deal with case when no axes arg is passed
        if (binder.op->getNumOperands() == 1) {
          if (binder.tensorOperand(data) ||
              binder.tensorResultType(resultType) ||
              binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
              binder.s64IntegerAttr(noop_with_empty_axes,
                                    "noop_with_empty_axes", 0))
            return failure();
          if (noop_with_empty_axes == 0) {
            Value keepDimsConstInt = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), keepDims));
            Value keepDimsBool = rewriter.create<Torch::AtenBoolIntOp>(
                binder.getLoc(), keepDimsConstInt);
            int64_t numDims = dyn_cast<Torch::ValueTensorType>(data.getType())
                                  .getSizes()
                                  .size();
            SmallVector<Value> axesList;
            for (int i = 0; i < numDims; i++) {
              Value curr = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(),
                  rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
              axesList.push_back(curr);
            }
            Value axesValueList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                axesList);
            rewriter.replaceOpWithNewOp<Torch::AtenAminOp>(
                binder.op, resultType, data, axesValueList, keepDimsBool);
          } else {
            rewriter.replaceOp(binder.op, data);
          }
          return success();
        }
        if (binder.tensorOperands(data, axes) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();
        Torch::BaseTensorType axesType =
            axes.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = axesType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), axesType.getOptionalDtype());
        auto sizes =
            dyn_cast<Torch::ValueTensorType>(axes.getType()).getSizes();
        // deal with case when axes is empty
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            // create dims list with all dims [0, data.getSizes().size())
            Value keepDimsConstInt = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), keepDims));
            Value keepDimsBool = rewriter.create<Torch::AtenBoolIntOp>(
                binder.getLoc(), keepDimsConstInt);
            int64_t numDims = dyn_cast<Torch::ValueTensorType>(data.getType())
                                  .getSizes()
                                  .size();
            for (int i = 0; i < numDims; i++) {
              Value curr = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(),
                  rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
              dimList.push_back(curr);
            }
            Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
                binder.getLoc(),
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                dimList);
            rewriter.replaceOpWithNewOp<Torch::AtenAminOp>(
                binder.op, resultType, data, dimValueList, keepDimsBool);
          } else {
            rewriter.replaceOp(binder.op, data);
          }
          return success();
        }
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        Value adjustment = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    adjustmentInt));
        // convert axes (tensor) into torch int list while dealing with neg axis
        for (int i = 0; i < sizes[0]; i++) {
          // Go through the axes list and get each dim in the list
          Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value extract = rewriter.create<Torch::AtenSelectIntOp>(
              binder.getLoc(), selectResultType, axes, zero, selectIndex);
          Value dim = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
          // deal with neg axis: if (axis < 0) axis += rank
          Value isNegative =
              rewriter.create<Torch::AtenLtIntOp>(binder.getLoc(), dim, zero);
          isNegative = rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(),
                                                             isNegative);
          Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
              binder.getLoc(), isNegative, adjustment);
          Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
              binder.getLoc(), dim, finalOffset);
          dimList.push_back(finalDim);
        }
        Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            dimList);
        Value keepDimBool;
        if (keepDims == 1) {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        } else {
          keepDimBool =
              rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenAminOp>(
            binder.op, resultType, data, dimValueList, keepDimBool);
        return success();
      });

  patterns.onOp("Shape", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::Aten_ShapeAsTensorOp>(
                      binder.op, resultType, operand);
                  return success();
                });

  patterns.onOp("Sinh", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  rewriter.replaceOpWithNewOp<Torch::AtenSinhOp>(
                      binder.op, resultType, operand);
                  return success();
                });

  patterns.onOp("Tan", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType))
                    return failure();
                  rewriter.replaceOpWithNewOp<Torch::AtenTanOp>(
                      binder.op, resultType, operand);
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

  patterns.onOp(
      "Reshape", 5, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        Value shape;
        int64_t allowzero;
        if (binder.tensorOperands(data, shape) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(allowzero, "allowzero", 0))
          return failure();
        Torch::BaseTensorType shapeType =
            shape.getType().cast<Torch::BaseTensorType>();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Type selectResultType = shapeType.getWithSizesAndDtype(
            llvm::ArrayRef(selectSizes), shapeType.getOptionalDtype());
        auto shapeSizes =
            dyn_cast<Torch::ValueTensorType>(shape.getType()).getSizes();
        auto dataSizes =
            dyn_cast<Torch::ValueTensorType>(data.getType()).getSizes();
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        if (allowzero == 0) {
          // convert shape (tensor) into torch int list while dealing with zero
          // vals
          for (int i = 0; i < shapeSizes[0]; i++) {
            // Go through the shape list and get each dim in the list
            Value selectIndex = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
            Value extract = rewriter.create<Torch::AtenSelectIntOp>(
                binder.getLoc(), selectResultType, shape, zero, selectIndex);
            Value dim = rewriter.create<Torch::AtenItemOp>(
                binder.getLoc(), rewriter.getType<Torch::IntType>(), extract);
            // deal with zero axis values: replace with original dim value in
            // input
            Value isZero =
                rewriter.create<Torch::AtenEqIntOp>(binder.getLoc(), dim, zero);
            isZero =
                rewriter.create<Torch::AtenIntBoolOp>(binder.getLoc(), isZero);
            Value adjustment;
            int64_t inputDimsSize = dataSizes.size();
            if (i < inputDimsSize) {
              adjustment = rewriter.create<Torch::ConstantIntOp>(
                  binder.getLoc(), rewriter.getType<Torch::IntType>(),
                  rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                          dataSizes[i]));
            }
            // Will never have a 0 in the shape tensor input at an index out of
            // bounds of original input dims Therefore, no need to adjust
            else {
              adjustment = zero;
            }
            Value finalOffset = rewriter.create<Torch::AtenMulIntOp>(
                binder.getLoc(), isZero, adjustment);
            Value finalDim = rewriter.create<Torch::AtenAddIntOp>(
                binder.getLoc(), dim, finalOffset);
            dimList.push_back(finalDim);
          }
          Value dimValueList = rewriter.create<Torch::PrimListConstructOp>(
              binder.getLoc(),
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              dimList);
          rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(
              binder.op, resultType, data, dimValueList);
          return success();
        }
        // convert axes (tensor) into torch int list
        for (int i = 0; i < shapeSizes[0]; i++) {
          // Go through the axes list and get each dim in the list
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
        rewriter.replaceOpWithNewOp<Torch::AtenReshapeOp>(binder.op, resultType,
                                                          data, dimValueList);
        return success();
      });
}
