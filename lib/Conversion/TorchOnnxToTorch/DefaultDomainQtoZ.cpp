//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ImplicitLocOpBuilder.h"
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
void mlir::torch::onnx_c::populateDefaultDomainQtoZ(
    OnnxCustomOpConversionPattern &patterns) {

  patterns.onOp(
      "QuantizeLinear", 19,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType outputTy;
        int64_t axis, saturate;
        Value input, scale, zp;
        ImplicitLocOpBuilder b(binder.getLoc(), rewriter);
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.s64IntegerAttr(axis, "axis", 1) ||
            binder.s64IntegerAttr(saturate, "saturate", 1) ||
            binder.tensorResultType(outputTy))
          return failure();

        auto inputTy = input.getType().cast<Torch::ValueTensorType>();
        auto scaleTy = scale.getType().cast<Torch::ValueTensorType>();
        if (!inputTy.hasSizes() || !scaleTy.hasSizes() ||
            !outputTy.hasSizes() || !outputTy.hasDtype())
          return failure();

        IntegerType outputETy = outputTy.getDtype().dyn_cast<IntegerType>();
        if (!outputETy)
          return failure();

        auto scaleShape = scaleTy.getOptionalSizes().value();
        auto outputShape = outputTy.getOptionalSizes().value();

        Value expandList;
        llvm::SmallVector<int64_t> expandShape(outputShape.size() - axis, 1);
        Value cstFalse = b.create<Torch::ConstantBoolOp>(false);
        if (!scaleShape.empty()) {
          llvm::SmallVector<Value> expandDims;
          expandShape[0] = scaleShape.front();
          for (int i = axis + 1, s = outputShape.size(); i < s; ++i) {
            expandDims.push_back(b.create<Torch::ConstantIntOp>(
                b.getType<Torch::IntType>(),
                b.getIntegerAttr(b.getIntegerType(64), i - axis)));
          }

          auto dimListTy =
              b.getType<Torch::ListType>(b.getType<Torch::IntType>());
          expandList =
              b.create<Torch::PrimListConstructOp>(dimListTy, expandDims);
          scale = b.create<Torch::AtenExpandOp>(
              b.getType<Torch::ValueTensorType>(expandShape,
                                                scaleTy.getOptionalDtype()),
              scale, expandList, cstFalse);
        }

        input = b.create<Torch::AtenDivTensorOp>(inputTy, input, scale);

        // Broadcast `zero` to the input shape size and apply if it exists:
        if (!binder.tensorOperandAtIndex(zp, 2)) {
          if (!scaleShape.empty()) {
            auto zpTy = zp.getType().cast<Torch::ValueTensorType>();
            zpTy = b.getType<Torch::ValueTensorType>(expandShape,
                                                     zpTy.getOptionalDtype());
            zp = b.create<Torch::AtenExpandOp>(zpTy, zp, expandList, cstFalse);
          }
          Value const1 = b.create<Torch::ConstantIntOp>(
              b.getType<Torch::IntType>(),
              b.getIntegerAttr(b.getIntegerType(64), 1));
          input = b.create<Torch::AtenAddTensorOp>(inputTy, input, zp, const1);
        }

        // Apply saturation:
        if (saturate) {
          int64_t uMin = 0;
          int64_t uMax =
              APInt::getMaxValue(outputETy.getWidth()).getSExtValue();
          int64_t sMin =
              APInt::getSignedMinValue(outputETy.getWidth()).getSExtValue();
          int64_t sMax =
              APInt::getSignedMaxValue(outputETy.getWidth()).getSExtValue();
          int64_t clampMin = outputETy.isSignless() ? uMin : sMin;
          int64_t clampMax = outputETy.isSignless() ? uMax : sMax;
          Value minVal = b.create<Torch::ConstantIntOp>(
              b.getType<Torch::IntType>(),
              b.getIntegerAttr(b.getIntegerType(64), clampMin));
          Value maxVal = b.create<Torch::ConstantIntOp>(
              b.getType<Torch::IntType>(),
              b.getIntegerAttr(b.getIntegerType(64), clampMax));
          input = b.create<Torch::AtenClampOp>(input.getType(), input, minVal,
                                               maxVal);
        }

        Value none = b.create<Torch::ConstantNoneOp>();
        Value constDtype = b.create<Torch::ConstantIntOp>(
            b.getType<Torch::IntType>(),
            b.getIntegerAttr(b.getIntegerType(64),
                             static_cast<int64_t>(Torch::getScalarTypeForType(
                                 outputTy.getDtype()))));
        input = b.create<Torch::AtenToDtypeOp>(outputTy, input, constDtype,
                                               /*non_blocking=*/cstFalse,
                                               /*copy=*/cstFalse,
                                               /*memory_format=*/none);

        rewriter.replaceOp(binder.op, input);
        return success();
      });

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
        Value result;
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
