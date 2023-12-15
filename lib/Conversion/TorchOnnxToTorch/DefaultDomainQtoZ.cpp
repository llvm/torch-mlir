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
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            Value cstTrue =
                rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
            rewriter.replaceOpWithNewOp<Torch::AtenSumDimIntListOp>(
                binder.op, resultType, data, /*dim=*/noneVal,
                /*keepdim=*/cstTrue, /*dtype=*/noneVal);
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
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            Value cstTrue =
                rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
            rewriter.replaceOpWithNewOp<Torch::AtenMeanDimOp>(
                binder.op, resultType, data, /*dim=*/noneVal,
                /*keepdim=*/cstTrue, /*dtype=*/noneVal);
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
              binder.s64IntegerAttr(keepDims, "keepdims", 1))
            return failure();
          if (noop_with_empty_axes == 0) {
            Value cstTrue =
                rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
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
                binder.op, resultType, data, axesValueList,
                /*keepdim=*/cstTrue);
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
        if (sizes.size() == 1 && sizes[0] == 0) {
          if (noop_with_empty_axes == 0) {
            // create dims list with all dims [0, data.getSizes().size())
            Value cstTrue =
                rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
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
                binder.op, resultType, data, dimValueList, /*keepdim=*/cstTrue);
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
}
