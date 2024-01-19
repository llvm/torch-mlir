//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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

// utilities
//  Templatized function to get an item op of a type
namespace {
template <typename T>
Value getItemOp(OpBinder binder, ConversionPatternRewriter &rewriter,
                Value &ofItem) {
  return rewriter.create<Torch::AtenItemOp>(binder.getLoc(),
                                            rewriter.getType<T>(), ofItem);
}
} // namespace

void mlir::torch::onnx_c::populateDefaultDomainQtoZ(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp("QuantizeLinear", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  llvm::SmallVector<Value> operands;
                  if (binder.tensorOperands(operands, 3) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  Value operand = operands[0];
                  Value scale = operands[1];
                  Value zeropoint = operands[2];

                  auto scaleTy = scale.getType().dyn_cast<Torch::ValueTensorType>();
                  if (!scaleTy || !scaleTy.hasSizes()) return rewriter.notifyMatchFailure(binder.op, "requires known rank");
                  if (!resultType.hasDtype())
                    return rewriter.notifyMatchFailure(
                        binder.op, "requires known result dtype");

                  if (scaleTy.getSizes().size() == 0) {
                    Type qTy = resultType.getDtype();

                    if (qTy.isUnsignedInteger(8)) {
                      qTy = rewriter.getType<Torch::QUInt8Type>();
                    } else if (qTy.isSignedInteger(8)) {
                      qTy = rewriter.getType<Torch::QInt8Type>();
                    } else if (qTy.isSignedInteger(32)) {
                      qTy = rewriter.getType<Torch::QInt32Type>();
                    } else {
                      return rewriter.notifyMatchFailure(binder.op, "unsupported result dtype");
                    }

                    auto qTensorTy = rewriter.getType<Torch::ValueTensorType>(resultType.getOptionalSizes(), qTy);
                    auto torchqTy = Torch::getScalarTypeForType(qTy);

                    Value tyConst = rewriter.create<Torch::ConstantIntOp>(
                        binder.getLoc(), rewriter.getType<Torch::IntType>(),
                        rewriter.getIntegerAttr(rewriter.getIntegerType(64), static_cast<int64_t>(torchqTy)));

                    scale = rewriter.create<Torch::AtenItemOp>(binder.getLoc(), rewriter.getType<Torch::FloatType>(), scale);
                    zeropoint = rewriter.create<Torch::AtenItemOp>(binder.getLoc(), rewriter.getType<Torch::IntType>(), zeropoint);

                    auto quantize = rewriter.create<Torch::AtenQuantizePerTensorOp>(binder.getLoc(), qTensorTy, operand, scale, zeropoint, tyConst);
                    rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType, quantize);
                    return success();
                  }

                  return failure();

                }
  );
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
  // onnx.ReduceMean with axes provided as argument introduced in opset 18
  patterns.onOp(
      "ReduceMean", 18,
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

  // onnx.ReduceMean with axes provided as attribute
  patterns.onOp(
      "ReduceMean", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data;
        llvm::SmallVector<int64_t> axes;
        int64_t keepDims;
        int64_t noop_with_empty_axes;
        if (binder.tensorOperand(data) || binder.tensorResultType(resultType) ||
            binder.s64IntegerArrayAttr(axes, "axes", 0) ||
            binder.s64IntegerAttr(keepDims, "keepdims", 1) ||
            binder.s64IntegerAttr(noop_with_empty_axes, "noop_with_empty_axes",
                                  0))
          return failure();
        SmallVector<Value> dimList;
        SmallVector<int64_t> selectSizes;
        selectSizes.push_back(1);
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        // deal with case when axes is empty
        if (axes.size() == 0) {
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
        int64_t adjustmentInt =
            cast<Torch::ValueTensorType>(data.getType()).getSizes().size();
        // convert axes (tensor) into torch int list while dealing with neg axis
        for (uint64_t i = 0; i < axes.size(); i++) {
          // Go through the axes list and get each dim in the list
          int64_t dim = axes[i];
          if (dim < 0) {
            dim += adjustmentInt;
          }
          // deal with neg axis: if (axis < 0) axis += rank
          Value finalDim = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), dim));
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

  // split with fixed-size parts
  // Arguments:
  // - input: the tensor to split
  // Attributes:
  // - axis: the axis along which to split the input
  // - num_outputs: the number of outputs to produce
  // Outputs:
  // - outputs: the produced outputs. Variadic with num_outputs elements.
  // Note: torch.aten gives a list of tensors, but ONNX gives a variadic list of
  // tensors
  //       so we need to unpack the list
  patterns.onOp(
      "Split", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value self;
        int64_t axis;
        int64_t num_outputs;
        if (binder.tensorOperand(self))
          return rewriter.notifyMatchFailure(
              binder.op, "Not converting to AtenSplitTensorOp due to input "
                         "tensor mismatch");
        if (binder.s64IntegerAttr(axis, "axis", 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get axis attribute");
        if (binder.s64IntegerAttr(num_outputs, "num_outputs", 0))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get num_outputs attribute");

        auto result0Ty =
            binder.op->getResult(0).getType().cast<Torch::ValueTensorType>();
        auto selfTy = self.getType().cast<Torch::ValueTensorType>();

        int64_t dim = axis;
        if (dim < 0)
          dim += selfTy.getSizes().size();

        // set intermediate shape to the shape of the first result
        // if the results are of different shapes
        // set the splitted axis to variable shape
        llvm::SmallVector<int64_t> intermediateShape(result0Ty.getSizes());
        for (auto result : binder.op->getResultTypes()) {
          int64_t d = result.cast<Torch::ValueTensorType>().getSizes()[dim];
          intermediateShape[dim] = d == intermediateShape[dim] ? d : -1;
        }

        Value dimValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), dim));

        Value splitSize = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), num_outputs));

        // TODO: Attempting to use the shape expected by the ONNX mlir as ground
        // truth. For now just use dynamic shapes.
        auto resultOuterType =
            Torch::ListType::get(rewriter.getType<Torch::ValueTensorType>(
                /*std::optional<llvm::ArrayRef<int64_t>>=*/intermediateShape,
                result0Ty.getOptionalDtype()));
        Torch::AtenSplitTensorOp new_op =
            rewriter.create<Torch::AtenSplitTensorOp>(
                binder.getLoc(), resultOuterType, self, splitSize, dimValue);

        // the onnx op is variadic with multiple results, but AtenSplitWithSizes
        // outputs a list so we need to unpack the list
        rewriter.replaceOpWithNewOp<Torch::PrimListUnpackOp>(
            binder.op, binder.op->getResults().getType(), new_op.getResult());

        return success();
      });

  // split with variable parts
  // Arguments:
  // - input: the tensor to split
  // - split: the sizes of the splits to be produced
  // Attributes:
  // - axis: the axis along which to split the input
  // - num_outputs: the number of outputs to produce
  // Outputs:
  // - outputs: the produced outputs. Variadic with num_outputs elements.
  // Note: torch.aten gives a list of tensors, but ONNX gives a variadic list of
  // tensors
  //       so we need to unpack the list
  patterns.onOp(
      "Split", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value self;
        Value split;
        int64_t axis;
        int64_t num_outputs;
        if (binder.tensorOperandAtIndex(self, 0) ||
            binder.tensorOperandAtIndex(split, 1))
          return rewriter.notifyMatchFailure(
              binder.op, "Not converting to AtenSplitWithSizesOp due to input "
                         "tensor mismatch");
        if (binder.s64IntegerAttr(axis, "axis", 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get axis attribute");
        if (binder.s64IntegerAttr(num_outputs, "num_outputs", 0))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to get num_outputs attribute");

        auto result0Ty =
            binder.op->getResult(0).getType().cast<Torch::ValueTensorType>();
        auto selfTy =
            cast<Torch::ValueTensorType>(binder.op->getOperand(0).getType());

        int64_t dim = axis;
        if (dim < 0)
          dim += selfTy.getSizes().size();

        llvm::SmallVector<int64_t> intermediateShape(result0Ty.getSizes());
        for (auto result : binder.op->getResultTypes()) {
          int64_t d = result.cast<Torch::ValueTensorType>().getSizes()[dim];
          intermediateShape[dim] = d == intermediateShape[dim] ? d : -1;
        }

        Torch::PrimTolistOp splitToList = rewriter.create<Torch::PrimTolistOp>(
            binder.getLoc(),
            Torch::ListType::get(rewriter.getType<Torch::IntType>()), split);

        Value dimValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), dim));

        // TODO: Attempting to use the shape expected by the ONNX mlir as ground
        // truth. For now just use dynamic shapes.
        auto resultOuterType =
            Torch::ListType::get(rewriter.getType<Torch::ValueTensorType>(
                /*std::optional<llvm::ArrayRef<int64_t>>=*/intermediateShape,
                result0Ty.getOptionalDtype()));
        Torch::AtenSplitWithSizesOp new_op =
            rewriter.create<Torch::AtenSplitWithSizesOp>(
                binder.getLoc(), resultOuterType, self,
                splitToList.getResult(0), dimValue);

        // the onnx op is variadic with multiple results, but AtenSplitWithSizes
        // outputs a list so we need to unpack the list
        rewriter.replaceOpWithNewOp<Torch::PrimListUnpackOp>(
            binder.op, binder.op->getResults().getType(), new_op.getResult());

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
      "Slice", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultTorchType;
        Value operand, starts, ends;
        // Handle if axes are not provided

        if (binder.tensorOperandAtIndex(operand, 0) ||
            binder.tensorOperandAtIndex(starts, 1) ||
            binder.tensorOperandAtIndex(ends, 2) ||
            binder.tensorResultType(resultTorchType)) {
          return failure();
        }

        auto context = rewriter.getContext();
        auto operandTorchTy = operand.getType().cast<Torch::ValueTensorType>();
        auto operandTy =
            operandTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();

        if (!operandTy)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected tensor operator argument to be a ranked tensor type");

        auto startsTorchTy = starts.getType().cast<Torch::ValueTensorType>();
        auto startsTy =
            startsTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
        int startSize = startsTy.getDimSize(0);

        auto endsTorchTy = ends.getType().cast<Torch::ValueTensorType>();
        auto endsTy =
            endsTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
        int endSize = endsTy.getDimSize(0);
        auto resultTy =
            resultTorchType.toBuiltinTensor().dyn_cast<RankedTensorType>();
        if (!resultTy)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type to be a ranked tensor type");

        Location loc = binder.getLoc();

        // Binding `axes` from its arguments or through a default value
        Value axes;
        if (binder.getNumOperands() >= 4) {
          if (binder.tensorOperandAtIndex(axes, 3)) {
            return failure();
          }
        } else {
          // The default axes value is the range from 0 to the number of
          // dimensions
          Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
          auto defaultAxesType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{operandTy.getRank()},
              rewriter.getIntegerType(64, /*signed*/ 1));
          Value arangeLength = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                      operandTy.getRank()));
          axes = rewriter.create<Torch::AtenArangeOp>(
              loc, defaultAxesType, arangeLength, none, none, none, none);
        }

        // Binding `steps` from its arguments or through a default value
        Value steps;
        if (binder.getNumOperands() >= 5) {
          if (binder.tensorOperandAtIndex(steps, 4)) {
            return failure();
          }
        } else {
          // The default `steps` value is a 1d tensor filled with ones with a
          // size of the dimension of the operand
          Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
          auto defaultStepsType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{operandTy.getRank()},
              rewriter.getIntegerType(64, /*signed*/ 1));
          Value sizeStepInput = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                      operandTy.getRank()));
          Value sizeStepsInput = rewriter.create<Torch::PrimListConstructOp>(
              loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              sizeStepInput);
          steps = rewriter.create<Torch::AtenOnesOp>(
              loc, defaultStepsType, sizeStepsInput, none, none, none, none);
        }

        if (!(endsTy.getRank() == 1 && startsTy.getRank() == 1 &&
              startSize == endSize))
          return rewriter.notifyMatchFailure(
              binder.op, "Expected the rank of starts and ends tensors to be 1 "
                         "and their dimensions to match");

        auto axesTorchTy = axes.getType().cast<Torch::ValueTensorType>();
        auto axesTy =
            axesTorchTy.toBuiltinTensor().dyn_cast<RankedTensorType>();
        int64_t numAxes = axesTy.getDimSize(0);

        if (!(axesTy && numAxes == endSize))
          return rewriter.notifyMatchFailure(
              binder.op, "Axes should be the same size of starts and ends");

        auto stepsTy = steps.getType()
                           .cast<Torch::ValueTensorType>()
                           .toBuiltinTensor()
                           .dyn_cast<RankedTensorType>();

        if (!(stepsTy && stepsTy.getDimSize(0) == endsTy.getDimSize(0)))
          return rewriter.notifyMatchFailure(
              binder.op, "Steps should be the same size of starts and ends");

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        auto select = [&](Value v, Value k) -> Value {
          auto ty = v.getType().cast<Torch::ValueTensorType>();
          auto sel = rewriter.create<Torch::AtenIndexSelectOp>(
              loc,
              Torch::ValueTensorType::get(ty.getContext(), ArrayRef<int64_t>{1},
                                          ty.getOptionalDtype()),
              v, zero, k);
          Value item = rewriter.create<Torch::AtenItemOp>(
              loc, rewriter.getType<Torch::IntType>(), sel);
          return item;
        };

        llvm::SmallVector<int64_t> intermediateShape(operandTy.getShape());
        for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
          if (operandTy.getDimSize(i) != resultTy.getDimSize(i)) {
            intermediateShape[i] = -1;
          }
        }
        auto intermediateType = Torch::ValueTensorType::get(
            context, intermediateShape, resultTorchType.getOptionalDtype());
        for (int i = 0; i < numAxes; ++i) {

          Value k = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), i));
          Value kTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
              loc,
              Torch::ValueTensorType::get(
                  context, ArrayRef<int64_t>{1},
                  rewriter.getIntegerType(64, /*signed*/ 1)),
              k);

          Value start = select(starts, kTensor);
          Value end = select(ends, kTensor);
          Value axis = select(axes, kTensor);
          Value step = select(steps, kTensor);

          auto sliceType = intermediateType;
          if (i == numAxes - 1)
            sliceType = resultTorchType;
          operand = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, sliceType, operand, axis, start, end, step);
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
  patterns.onOp(
      "Range", 11, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // ONNX.Range(start, limit, delta) -- limit is exclusive

        Torch::ValueTensorType resultType;
        Value start, limit, delta;
        auto loc = binder.getLoc();
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        if (binder.tensorOperandAtIndex(start, 0) ||
            binder.tensorOperandAtIndex(limit, 1) ||
            binder.tensorOperandAtIndex(delta, 2) ||
            binder.tensorResultType(resultType))
          return failure();

        // Convert a 0-dimensional/Scalar Tensor ([]) to Scalar Torch Numeric
        // Value torch.tensor(1.1) equivalent in ONNX to 1.1 as an example
        // type of start, limit, delta can be one of: double, float, int16,
        // int32, int64 Assuming start, limit and delta to be same type (could
        // they be different?)
        Torch::BaseTensorType startTensorType =
            start.getType().cast<Torch::BaseTensorType>();
        bool isFloatDType = startTensorType.getDtype().isF64() ||
                            startTensorType.getDtype().isF32();
        bool isIntDType = startTensorType.getDtype().isInteger(16) ||
                          startTensorType.getDtype().isInteger(32) ||
                          startTensorType.getDtype().isInteger(64);
        if (!isFloatDType && !isIntDType) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected the start, limit, delta to be one of "
                         "double, float, int16, int32, int64");
        }
        Value scalarStart, scalarLimit, scalarDelta;
        if (isFloatDType) {
          scalarStart = getItemOp<Torch::FloatType>(binder, rewriter, start);
          scalarLimit = getItemOp<Torch::FloatType>(binder, rewriter, limit);
          scalarDelta = getItemOp<Torch::FloatType>(binder, rewriter, delta);
        } else {
          scalarStart = getItemOp<Torch::IntType>(binder, rewriter, start);
          scalarLimit = getItemOp<Torch::IntType>(binder, rewriter, limit);
          scalarDelta = getItemOp<Torch::IntType>(binder, rewriter, delta);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenArangeStartStepOp>(
            binder.op, resultType, scalarStart, scalarLimit, scalarDelta, none,
            none, none, none);
        return success();
      });
}
