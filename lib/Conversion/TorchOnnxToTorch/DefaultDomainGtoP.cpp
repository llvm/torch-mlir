//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
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
void mlir::torch::onnx_c::populateDefaultDomainGtoP(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "HardSigmoid", 6,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value tensorOperand;
        float alpha, beta;
        if (binder.tensorOperand(tensorOperand) ||
            binder.f32FloatAttr(alpha, "alpha", 0.2f) ||
            binder.f32FloatAttr(beta, "beta", 0.5f) ||
            binder.tensorResultType(resultType))
          return failure();

        // HardSigmoid computes the following expression:
        //   max(0, min(1, alpha * x + beta))
        Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));

        Value constBeta = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));

        // Expression: alpha * x + beta
        Value alpha_x_plus_beta = rewriter.create<Torch::AtenAddScalarOp>(
            binder.getLoc(), resultType, tensorOperand, constBeta,
            /*alpha=*/constAlpha);

        // Expression: min(1, alpha * x + beta)
        Value constantOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value oneTensor = createRank0Tensor(rewriter, binder.getLoc(),
                                            resultType, constantOne);
        Value minExpression = rewriter.create<Torch::AtenMinimumOp>(
            binder.getLoc(), resultType, oneTensor, alpha_x_plus_beta);

        // Expression: max(0, min(1, alpha * x + beta))
        Value constantZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value zeroTensor = createRank0Tensor(rewriter, binder.getLoc(),
                                             resultType, constantZero);
        rewriter.replaceOpWithNewOp<Torch::AtenMaximumOp>(
            binder.op, resultType, zeroTensor, minExpression);
        return success();
      });
  patterns.onOp(
      "Gelu", 20, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value operand;
        Torch::ValueTensorType resultType;
        std::string approximate;

        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(approximate, "approximate", "none"))
          return failure();

        Value vApproximate = rewriter.create<Torch::ConstantStrOp>(
            binder.getLoc(), rewriter.getType<Torch::StringType>(),
            rewriter.getStringAttr(approximate));

        rewriter.replaceOpWithNewOp<Torch::AtenGeluOp>(binder.op, resultType,
                                                       operand, vApproximate);
        return success();
      });
  patterns.onOp(
      "GridSample", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        Value grid;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(grid, 1) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op, "operand grid_sampler bind failure");

        auto inputTensorType = cast<Torch::ValueTensorType>(input.getType());
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        uint32_t inputRank = inputShape.size();
        auto gridTensorType = cast<Torch::ValueTensorType>(grid.getType());
        ArrayRef<int64_t> gridShape = gridTensorType.getSizes();
        uint32_t gridRank = gridShape.size();

        if (inputRank != 4)
          return rewriter.notifyMatchFailure(binder.op,
                                             "only input rank 4 supported");
        if (gridRank != 4)
          return rewriter.notifyMatchFailure(binder.op,
                                             "only grid rank 4 supported");
        if (inputShape[0] != gridShape[0])
          return rewriter.notifyMatchFailure(
              binder.op, "N must be same for input and grid");
        if (gridShape[3] != 2)
          return rewriter.notifyMatchFailure(binder.op,
                                             "gridShape[3] expected to be 2");
        std::string iModeString;
        int64_t iModeInt;
        if (binder.customOpNameStringAttr(iModeString, "mode", "linear"))
          return rewriter.notifyMatchFailure(binder.op, "mode bind failure");

        if (iModeString == "linear" || iModeString == "bilinear") {
          iModeInt = 0;
        } else if (iModeString == "nearest") {
          iModeInt = 1;
        } else {
          return rewriter.notifyMatchFailure(
              binder.op, "currently only mode : linear and nearest supported");
        }

        std::string padding;
        if (binder.customOpNameStringAttr(padding, "padding_mode", "zeros"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "padding_mode bind failure");
        if (padding != "zeros")
          return rewriter.notifyMatchFailure(
              binder.op, "currently only padding_mode : zeros supported");
        int64_t align;
        if (binder.s64IntegerAttr(align, "align_corners", 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "align_corners bind failure");

        Value interpolationMode = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), iModeInt));

        Value paddingMode = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        bool alignMode = align;
        Value alignCorners = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(alignMode));

        rewriter.replaceOpWithNewOp<Torch::AtenGridSamplerOp>(
            binder.op, resultType, input, grid, interpolationMode, paddingMode,
            alignCorners);
        return success();
      });
  patterns.onOp(
      "If", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value conditionTensor;
        if (binder.tensorOperand(conditionTensor)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "condition bind failure");
        }

        auto conditionType =
            conditionTensor.getType().cast<Torch::ValueTensorType>();
        if (!conditionType || conditionType.getSizes().size() != 1)
          return rewriter.notifyMatchFailure(
              binder.op, "condition must have one single element per "
                         "https://onnx.ai/onnx/operators/onnx__If.html");
        auto conditionInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            conditionTensor);
        auto conditionBool = rewriter.create<Torch::AtenBoolIntOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(), conditionInt);

        llvm::SmallVector<mlir::Type> resultTypes;
        if (binder.tensorResultTypes(resultTypes)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }

        Region *thenRegion, *elseRegion;
        if (binder.getRegionAtIndex(elseRegion, 0) ||
            binder.getRegionAtIndex(thenRegion, 1)) {
          return rewriter.notifyMatchFailure(binder.op, "region bind failure");
        }

        auto primIfOp = rewriter.create<Torch::PrimIfOp>(
            binder.getLoc(), TypeRange(resultTypes), conditionBool);

        auto inlineIfCase = [&](Region &srcRegion, Region &dstRegion) {
          rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
        };
        inlineIfCase(*thenRegion, primIfOp.getThenRegion());
        inlineIfCase(*elseRegion, primIfOp.getElseRegion());

        auto replaceTerminator = [&](Region &region) {
          PatternRewriter::InsertionGuard guard(rewriter);
          Operation *terminator = region.front().getTerminator();
          rewriter.setInsertionPoint(terminator);
          rewriter.replaceOpWithNewOp<Torch::PrimIfYieldOp>(
              terminator, terminator->getOperands());
        };
        replaceTerminator(primIfOp.getThenRegion());
        replaceTerminator(primIfOp.getElseRegion());

        rewriter.replaceOp(binder.op, primIfOp.getResults());
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
  patterns.onOp("LessOrEqual", 1,
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
  patterns.onOp("Log", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenLogOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp("LSTM", 1, onnx_c::OnnxLstmExpander);
  patterns.onOp(
      "LogSoftmax", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value input;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperand(input) || binder.tensorResultType(resultType))
          return failure();
        int64_t axis;
        if (binder.s64IntegerAttr(axis, "axis", -1))
          return rewriter.notifyMatchFailure(binder.op, "axis bind failure");
        Value axisConst = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        rewriter.replaceOpWithNewOp<Torch::AtenLogSoftmaxIntOp>(
            binder.op, resultType, input, axisConst, none);
        return success();
      });
  patterns.onOp(
      "LogSoftmax", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value input;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperand(input) || binder.tensorResultType(resultType))
          return failure();

        int64_t axis;
        if (binder.s64IntegerAttr(axis, "axis", 1))
          return rewriter.notifyMatchFailure(binder.op, "axis bind failure");
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unsupported: unranked tensor");
        int64_t rank = *maybeRank;
        // if negative axis is provided, then flip it to a positive axis
        if (axis < 0) {
          axis = rank + axis;
        }
        // need input type and sizes to flatten/unflatten later.
        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        if (!inputTy || !inputTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "failed to get input type or sizes");

        Value axisConst = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value cstEnd = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(rank - 1));

        // The old version of LogSoftmax flattens post-axis dims, performs
        // LogSoftmax on the flattened dim, then unflattens back to the original
        // shape.

        // this section gets some size information necessary for
        // flattening/unflattening
        if (!inputTy || !inputTy.hasSizes())
          return failure();
        llvm::ArrayRef<int64_t> allDims(inputTy.getSizes());
        llvm::ArrayRef<int64_t> rightDims(allDims.begin() + axis,
                                          allDims.end());
        llvm::SmallVector<int64_t> leftDims(allDims.begin(),
                                            allDims.begin() + axis);
        int64_t prodRightSizes = 1;
        llvm::SmallVector<Value> rightDimConsts;
        for (int64_t n : rightDims) {
          rightDimConsts.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(n)));
          if (n == Torch::kUnknownSize) {
            prodRightSizes = -1;
            break;
          }
          prodRightSizes *= n;
        }
        leftDims.push_back(prodRightSizes);
        // the following list will be used to unflatten the right side
        Value rightDimsPrimList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            rightDimConsts);
        auto flatRightTy = rewriter.getType<Torch::ValueTensorType>(
            leftDims, inputTy.getOptionalDtype());
        // flatten input
        Value inputFlatRight = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
            binder.getLoc(), flatRightTy, input, axisConst, cstEnd);
        // compute lsm over flattened index
        Value outputFlatRight = rewriter.create<Torch::AtenLogSoftmaxIntOp>(
            binder.getLoc(), flatRightTy, inputFlatRight, axisConst, none);
        // unflatten
        rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
            binder.op, resultType, outputFlatRight, axisConst,
            rightDimsPrimList);
        return success();
      });
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
  patterns.onOp(
      "MatMulInteger", 10,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs, lhsZp, rhsZp;
        if (binder.tensorOperandAtIndex(lhs, 0) ||
            binder.tensorOperandAtIndex(rhs, 1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto lhsTy = dyn_cast<Torch::ValueTensorType>(lhs.getType());
        auto rhsTy = dyn_cast<Torch::ValueTensorType>(rhs.getType());

        if (binder.tensorOperandAtIndex(lhsZp, 2)) {
          lhsZp = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        }

        if (binder.tensorOperandAtIndex(rhsZp, 3)) {
          rhsZp = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        }

        if (auto zpTy = dyn_cast<Torch::ValueTensorType>(lhsZp.getType())) {
          for (auto dim : zpTy.getSizes())
            if (dim != 1)
              return failure();
          lhsZp = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), lhsZp);
        }

        if (auto zpTy = dyn_cast<Torch::ValueTensorType>(rhsZp.getType())) {
          for (auto dim : zpTy.getSizes())
            if (dim != 1)
              return failure();
          rhsZp = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(), rhsZp);
        }

        Value scale = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(1.0));

        auto q = [&](Type qty) -> Type {
          if (qty.isSignedInteger(8))
            return rewriter.getType<Torch::QInt8Type>();
          if (qty.isUnsignedInteger(8))
            return rewriter.getType<Torch::QUInt8Type>();
          if (qty.isSignedInteger(32))
            return rewriter.getType<Torch::QInt32Type>();
          return {};
        };

        Type lhsQTy = rewriter.getType<Torch::ValueTensorType>(
            lhsTy.getOptionalSizes(), q(lhsTy.getDtype()));
        Type rhsQTy = rewriter.getType<Torch::ValueTensorType>(
            rhsTy.getOptionalSizes(), q(rhsTy.getDtype()));

        lhs = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), lhsQTy, lhs, scale, lhsZp);
        rhs = rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
            binder.getLoc(), rhsQTy, rhs, scale, rhsZp);

        rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(binder.op, resultType,
                                                         lhs, rhs);
        return success();
      });
  patterns.onOp("Mul", 7,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenMulTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp("NonZero", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNonzeroOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "MaxPool", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "auto_pad bind failure");
        if (autoPad != "NOTSET")
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");

        Torch::ValueTensorType resultType;
        Value operand;
        bool ceilMode;
        int64_t storageOrder;
        // TODO: Add support for indices output and storage_order
        if (binder.tensorOperand(operand) ||
            binder.s64BoolAttr(ceilMode, "ceil_mode", false) ||
            binder.s64IntegerAttr(storageOrder, "storage_order", 0) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op,
              "operand/ceil_mode/storage_order/resultType bind failure");
        if (storageOrder != 0)
          return rewriter.notifyMatchFailure(
              binder.op, "storage_order setting is not supported.");
        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(operand);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        int64_t rank = *maybeRank;
        int64_t spatial = rank - 2;

        SmallVector<int64_t> kernel, padding, strides, dilations;
        if (binder.s64IntegerArrayAttr(kernel, "kernel_shape", {}))
          return rewriter.notifyMatchFailure(binder.op,
                                             "kernel_shape bind failure");
        if (kernel.size() != static_cast<size_t>(spatial))
          return rewriter.notifyMatchFailure(
              binder.op, "kernel list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(padding, "pads", {}))
          return rewriter.notifyMatchFailure(binder.op, "pads bind failure");
        if (!padding.empty() &&
            padding.size() != static_cast<size_t>(2 * spatial))
          return rewriter.notifyMatchFailure(
              binder.op, "padding list must contain (begin,end) pair for each "
                         "spatial axis");
        if (binder.s64IntegerArrayAttr(strides, "strides", {}))
          return rewriter.notifyMatchFailure(binder.op, "strides bind failure");
        if (!strides.empty() && strides.size() != static_cast<size_t>(spatial))
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        if (binder.s64IntegerArrayAttr(dilations, "dilations", {}))
          return rewriter.notifyMatchFailure(binder.op,
                                             "dilations bind failure");

        if (padding.empty())
          padding.resize(spatial, 0);
        if (strides.empty())
          strides.resize(spatial, 1);
        if (dilations.empty())
          dilations.resize(spatial, 1);

        // If the padding is symmetric we can push the padding operation to the
        // torch operator.
        if (padding.size() == static_cast<size_t>(2 * spatial)) {
          bool equal = true;
          for (int i = 0; i < spatial; ++i) {
            equal = equal && (padding[i] == padding[i + spatial]);
          }
          if (equal)
            padding.resize(spatial);
        }

        // Torch pool operators require equal padding on each size of each
        // dimension so we materialize the padding behavior explicitly and set
        // the padding to 0.
        if (padding.size() == static_cast<size_t>(2 * spatial)) {
          auto operandTy = cast<Torch::ValueTensorType>(operand.getType());
          llvm::SmallVector<int64_t> shuffledPadding(spatial * 2);
          llvm::SmallVector<int64_t> paddedShape(operandTy.getSizes());
          shuffledPadding.resize(2 * rank);
          for (int i = 0; i < spatial; ++i) {
            paddedShape[i + 2] += padding[i] + padding[i + spatial];
            shuffledPadding[2 * i] = padding[i];
            shuffledPadding[2 * i + 1] = padding[i + spatial];
          }

          Value shuffledPaddingList =
              createConstantIntList(binder, rewriter, padding);
          Value zero;
          if (resultType.getDtype().isa<FloatType>()) {
            zero = rewriter.create<Torch::ConstantFloatOp>(
                binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(
                    std::numeric_limits<double>::lowest()));
          } else if (resultType.getDtype().isa<IntegerType>()) {
            zero = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(
                                     std::numeric_limits<int64_t>::lowest()));
          }

          auto paddedInputTy = rewriter.getType<Torch::ValueTensorType>(
              paddedShape, operandTy.getDtype());
          operand = rewriter.create<Torch::AtenConstantPadNdOp>(
              binder.getLoc(), paddedInputTy, operand, shuffledPaddingList,
              zero);
          padding.clear();
          padding.resize(spatial, 0);
        }

        Value kernelSizeList = createConstantIntList(binder, rewriter, kernel);
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value dilationsList =
            createConstantIntList(binder, rewriter, dilations);
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceilMode);

        if (rank == 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: AtenMaxPool1dOp");
        if (rank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        }
        if (rank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        }
        return rewriter.notifyMatchFailure(binder.op, "No rank is matched.");
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
  patterns.onOp(
      "InstanceNormalization", 6,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        float eps;

        if (binder.tensorOperands(operands, 3) ||
            binder.tensorResultType(resultType) || operands.size() != 3 ||
            binder.f32FloatAttr(eps, "epsilon", 1e-05f)) {
          return failure();
        }
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value boolTrue =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value boolFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        auto epsValue = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getF64FloatAttr(eps));

        auto momentum = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getF64FloatAttr(0.0f));
        rewriter.replaceOpWithNewOp<Torch::AtenInstanceNormOp>(
            binder.op, resultType, /* input */ operands[0],
            /* weight */ operands[1],
            /* bias */ operands[2], /* running mean */ none,
            /* running var */ none,
            /* use input stats */ boolTrue, momentum, epsValue,
            /* cudnn enabled */ boolFalse);
        return success();
      });
  patterns.onOp(
      "Max", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) || operands.size() == 0) {
          return failure();
        }
        Value result = operands[0];
        for (uint64_t i = 1; i < operands.size(); i++) {
          result = rewriter.create<Torch::AtenMaximumOp>(
              binder.getLoc(), resultType, result, operands[i]);
        }
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp(
      "Min", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) || operands.size() == 0) {
          return failure();
        }
        Value result = operands[0];
        for (uint64_t i = 1; i < operands.size(); i++) {
          result = rewriter.create<Torch::AtenMinimumOp>(
              binder.getLoc(), resultType, result, operands[i]);
        }
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp("Neg", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenNegOp>(
                      binder.op, resultType, operand);
                  return success();
                });
  patterns.onOp(
      "Not", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        auto loc = binder.getLoc();
        auto operandTy = cast<Torch::ValueTensorType>(operand.getType());
        auto eTy = operandTy.getDtype();

        if (!eTy.isInteger(1)) {
          auto i1ty = rewriter.getI1Type();
          auto ty = rewriter.getType<Torch::ValueTensorType>(
              operandTy.getSizes(), i1ty);
          auto torchqTy = Torch::getScalarTypeForType(i1ty);
          Value tyConst = rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                      static_cast<int64_t>(torchqTy)));
          Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
          Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
          operand = rewriter.create<Torch::AtenToDtypeOp>(
              loc, ty, operand, tyConst,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/none);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenBitwiseNotOp>(
            binder.op, resultType, operand);
        return success();
      });
  patterns.onOp("Or", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenBitwiseOrTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "GatherND", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices;
        int64_t batchDimCount;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(batchDimCount, "batch_dims", 0))
          return failure();

        Location loc = binder.getLoc();
        auto dataTy = cast<Torch::ValueTensorType>(data.getType());
        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        if (!dataTy || !dataTy.hasSizes())
          return failure();
        if (!indicesTy || !indicesTy.hasSizes())
          return failure();

        // step 1. Get shapes and ranks of data and indices. The last dimension
        // of indices is expected to be static.
        ArrayRef<int64_t> dataShape = dataTy.getSizes();
        int64_t dataRank = dataShape.size();
        ArrayRef<int64_t> indicesShape = indicesTy.getSizes();
        int64_t indicesRank = indicesShape.size();
        int64_t indicesLastDim = indicesShape.back();
        // Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and
        // batch_dims integer b, onnx.gather_nd gathers slices of data into an
        // output tensor of rank q + r - indices_shape[-1] - 1 - b.
        // indices_shape[-1] must be static to have deterministic output rank.
        if (dataRank < 1 || indicesRank < 1)
          return rewriter.notifyMatchFailure(
              binder.op, "expected data and indices rank to be >= 1");
        if (batchDimCount >= std::min(dataRank, indicesRank))
          return rewriter.notifyMatchFailure(
              binder.op, "batch_dims should be strictly less than "
                         "min(rank(data), rank(indices))");
        if (indicesLastDim == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op, "expected last dimension of indices to be static");

        // step 2. Get dimension list of data.
        SmallVector<int64_t> batchShape;
        SmallVector<Value> batchDims;
        SmallVector<Value> dataDims;
        for (int64_t i = 0; i < dataRank; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), i);
          Value dataDim = rewriter.create<Torch::AtenSizeIntOp>(loc, data, k);
          dataDims.push_back(dataDim);
          if (i < batchDimCount) {
            batchShape.push_back(dataShape[i]);
            batchDims.push_back(dataDim);
          }
        }

        // step 3. Get dimension list of indices.
        Value constZero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        SmallVector<Value> indicesDimsMinusOne;
        SmallVector<Value> unflattenIndicesDims;
        Value indicesFlattenDim = constOne;
        for (int64_t i = 0; i < indicesRank - 1; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), i);
          Value indicesDim =
              rewriter.create<Torch::AtenSizeIntOp>(loc, indices, k);
          indicesDimsMinusOne.push_back(indicesDim);
          if (i >= batchDimCount) {
            unflattenIndicesDims.push_back(indicesDim);
            indicesFlattenDim = rewriter.create<Torch::AtenMulIntOp>(
                loc, indicesFlattenDim, indicesDim);
          }
        }
        ArrayRef<int64_t> indicesShapeMinusOne = indicesShape.drop_back();

        // Algorithm: We can not directly perform torch.gather as it requires
        // the ranks of data(`r`) and indices(`q`) to be same. So we will
        // perform collapse and reshape operations to match the ranks of data
        // and indices(making sure the semantics of the onnx.gather_nd are
        // preserved), perform torch.gather operation, later unflatten the
        // gather result to match onnx.gather_nd output. For example, assuming
        // indices is of shape (4, 5, 3, 2), data is (4, 10, 11, 7, 4) and
        // batch_dims(`b`)=1. Firstly, modify indices to 1-D indexing as the
        // torch.gather op supports only single dimensional indexing. (this
        // algorithm would have been simpler if we can get a torch op that
        // supports indexing at multiple dimensions simultaneously). 1-D indexed
        // indices will be of shape (4, 5, 3, 1), now materialize it to
        // `r-b-indices_shape[-1]` dimension of data i.e. reshaping it to the
        // shape (4, 5, 3, 1, 1). Next step is to flatten+expand the indices and
        // flatten the data to (4, 15, 7, 4) and (4, 110, 7, 4) shapes
        // respectively and then perform the torch.gather operation. Post the
        // gather operation, unflatten the indices dimensions of result to (4,
        // 5, 3, 7, 4) which is our required result.

        // step 4. Convert indices_shape[-1] dimensional indexing to 1D
        // indexing.
        Value sliceDim = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(indicesRank - 1));
        SmallVector<int64_t> indicesSliceShape(indicesShapeMinusOne);
        indicesSliceShape.push_back(1);
        auto indicesSliceTy = rewriter.getType<Torch::ValueTensorType>(
            indicesSliceShape, indicesTy.getOptionalDtype());

        Value start = constZero;
        Value updatedIndices;
        for (int64_t i = 0; i < indicesLastDim; ++i) {
          Value end = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i + 1));
          Value indicesSlice = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, indicesSliceTy, indices, sliceDim, start, end,
              /*step=*/constOne);
          start = end;
          // Apply bounds checking on the indices slice.
          auto boolTy = rewriter.getType<Torch::ValueTensorType>(
              indicesSliceShape, rewriter.getI1Type());
          Value lt = rewriter.create<Torch::AtenLtScalarOp>(
              loc, boolTy, indicesSlice, constZero);
          Value add = rewriter.create<Torch::AtenAddScalarOp>(
              loc, indicesSliceTy, indicesSlice, dataDims[batchDimCount + i],
              /*alpha=*/constOne);
          indicesSlice = rewriter.create<Torch::AtenWhereSelfOp>(
              loc, indicesSliceTy, lt, add, indicesSlice);
          if (i == 0) {
            updatedIndices = indicesSlice;
            continue;
          }
          updatedIndices = rewriter.create<Torch::AtenAddTensorOp>(
              loc, indicesSliceTy, indicesSlice, updatedIndices,
              dataDims[batchDimCount + i]);
        }

        // step 5. Compute all the required result types here.
        SmallVector<int64_t> reshapeIndicesShape(indicesShapeMinusOne);
        SmallVector<Value> reshapeIndicesDims(indicesDimsMinusOne);
        // Determine the collapsed dim size of indices(index_shape[-1] is not
        // part of collapsing as we already removed it by 1-D indexing).
        SmallVector<int64_t> flattenIndicesShape(batchShape);
        auto indicesCt = 1;
        for (int64_t i = batchDimCount; i < indicesRank - 1; ++i) {
          if (indicesShape[i] == Torch::kUnknownSize) {
            indicesCt = Torch::kUnknownSize;
            break;
          }
          indicesCt *= indicesShape[i];
        }
        flattenIndicesShape.push_back(indicesCt);
        // Determine the collapsed dim size of data.
        SmallVector<int64_t> flattenDataShape(batchShape);
        auto dataCt = 1;
        for (int64_t i = 0; i < indicesLastDim; ++i) {
          int64_t sz = dataShape[i + batchDimCount];
          if (sz == Torch::kUnknownSize) {
            dataCt = Torch::kUnknownSize;
            break;
          }
          dataCt *= sz;
        }
        flattenDataShape.push_back(dataCt);
        // Compute the shape of expand op.
        SmallVector<Value> expandIndicesDims(batchDims);
        expandIndicesDims.push_back(indicesFlattenDim);
        SmallVector<int64_t> expandIndicesShape(batchShape);
        expandIndicesShape.push_back(indicesCt);
        // Append `r-b-indices_shape[-1]` unit or data dims appropriately to all
        // result types.
        for (int64_t i = batchDimCount + indicesLastDim; i < dataRank; ++i) {
          reshapeIndicesShape.push_back(1);
          flattenIndicesShape.push_back(1);
          flattenDataShape.push_back(dataShape[i]);
          expandIndicesShape.push_back(dataShape[i]);
          reshapeIndicesDims.push_back(constOne);
          expandIndicesDims.push_back(dataDims[i]);
        }

        // step 6. Reshape 1-D indexed indices to match the rank of flattened
        // data by inserting unit dimensions.
        auto intListTy = rewriter.getType<Torch::ListType>(
            rewriter.getType<Torch::IntType>());
        Value reshapeIndicesSizeList =
            rewriter.create<Torch::PrimListConstructOp>(loc, intListTy,
                                                        reshapeIndicesDims);
        auto reshapeIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            reshapeIndicesShape, indicesTy.getOptionalDtype());
        Value reshapedIndices = rewriter.create<Torch::AtenViewOp>(
            loc, reshapeIndicesTy, updatedIndices, reshapeIndicesSizeList);

        // step 7. Flatten `q-b-1` dimensions of the indices.
        auto flattenIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            flattenIndicesShape, indicesTy.getOptionalDtype());
        Value batchDimCountVal = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(batchDimCount));
        Value flattenedIndices = reshapedIndices;
        if (indicesRank == 1) {
          flattenedIndices = rewriter.create<Torch::AtenUnsqueezeOp>(
              loc, flattenIndicesTy, reshapedIndices, constZero);
        } else if (indicesRank > 1) {
          Value endDim = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(indicesRank - 2));
          flattenedIndices = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              loc, flattenIndicesTy, reshapedIndices, batchDimCountVal, endDim);
        }

        // step 8. Expand `r-b-indices_shape[-1]` dims of flattened indices.
        auto expandIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            expandIndicesShape, indicesTy.getOptionalDtype());
        Value expandIndicesSizeList =
            rewriter.create<Torch::PrimListConstructOp>(loc, intListTy,
                                                        expandIndicesDims);
        Value constFalse = rewriter.create<Torch::ConstantBoolOp>(
            loc, rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        Value expandedIndices = rewriter.create<Torch::AtenExpandOp>(
            loc, expandIndicesTy, flattenedIndices, expandIndicesSizeList,
            /*implicit=*/constFalse);

        // step 9. Flatten indices_shape[-1] dimensions of data.
        auto flattenDataTy = rewriter.getType<Torch::ValueTensorType>(
            flattenDataShape, dataTy.getOptionalDtype());
        Value endDim = rewriter.create<Torch::ConstantIntOp>(
            loc,
            rewriter.getI64IntegerAttr(batchDimCount + indicesLastDim - 1));
        Value flattenedData = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
            loc, flattenDataTy, data, batchDimCountVal, endDim);

        // step 10. Now we have flattenedData and expandedIndices of same rank
        // to perform gather operation.
        auto gatherTy = rewriter.getType<Torch::ValueTensorType>(
            expandIndicesShape, dataTy.getOptionalDtype());
        Value gather = rewriter.create<Torch::AtenGatherOp>(
            loc, gatherTy, flattenedData, batchDimCountVal, expandedIndices,
            /*sparseGrad=*/constFalse);

        // step 11. Unflatten the collapsed indices dims of gather result.
        if (indicesRank == 1) {
          rewriter.replaceOpWithNewOp<Torch::AtenSqueezeDimOp>(
              binder.op, resultType, gather, /*dim=*/constZero);
          return success();
        }
        Value unflattenSizeList = rewriter.create<Torch::PrimListConstructOp>(
            loc, intListTy, unflattenIndicesDims);
        rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
            binder.op, resultType, gather, batchDimCountVal, unflattenSizeList);
        return success();
      });
  patterns.onOp(
      "Gather", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, indices;
        int64_t axis;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(axis, "axis", 0))
          return failure();
        Location loc = binder.getLoc();
        auto ctx = binder.op->getContext();
        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        auto dataTy = cast<Torch::ValueTensorType>(data.getType());
        if (!dataTy || !dataTy.hasSizes() || !indicesTy.hasSizes())
          return failure();

        int64_t dataRank = dataTy.getSizes().size();
        int64_t indicesRank = indicesTy.getSizes().size();
        axis = axis < 0 ? axis + dataRank : axis;

        Value index = rewriter.create<Torch::ConstantIntOp>(
            loc, Torch::IntType::get(ctx), rewriter.getI64IntegerAttr(axis));

        // Apply bounds checking on the input:
        auto intTy = rewriter.getType<Torch::IntType>();
        auto boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, intTy, rewriter.getI64IntegerAttr(0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            loc, intTy, rewriter.getI64IntegerAttr(1));
        Value lt =
            rewriter.create<Torch::AtenLtScalarOp>(loc, boolTy, indices, zero);
        Value dim =
            rewriter.create<Torch::AtenSizeIntOp>(loc, intTy, data, index);
        Value add = rewriter.create<Torch::AtenAddScalarOp>(loc, indicesTy,
                                                            indices, dim, one);
        indices = rewriter.create<Torch::AtenWhereSelfOp>(loc, indicesTy, lt,
                                                          add, indices);

        auto intListTy = rewriter.getType<Torch::ListType>(
            rewriter.getType<Torch::IntType>());

        llvm::SmallVector<Value> indicesDims;
        for (int i = 0, s = indicesTy.getSizes().size(); i < s; ++i) {
          Value k = rewriter.create<Torch::ConstantIntOp>(binder.getLoc(), i);
          indicesDims.push_back(rewriter.create<Torch::AtenSizeIntOp>(
              binder.getLoc(), indices, k));
        }

        Value indicesSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(), intListTy, indicesDims);

        // Determine the collapsed dim size:
        auto indicesCt = 1;
        for (auto sz : indicesTy.getSizes()) {
          if (sz == Torch::kUnknownSize) {
            indicesCt = Torch::kUnknownSize;
            break;
          }

          indicesCt *= sz;
        }

        auto flattenTy = rewriter.getType<Torch::ValueTensorType>(
            SmallVector<int64_t>{indicesCt}, indicesTy.getOptionalDtype());

        if (indicesRank == 0) {
          indices = rewriter.create<Torch::AtenUnsqueezeOp>(
              binder.getLoc(), flattenTy, indices, zero);
        } else if (indicesRank > 1) {
          Value rank = rewriter.create<Torch::AtenDimOp>(loc, intTy, indices);
          Value end = rewriter.create<Torch::AtenSubIntOp>(loc, rank, one);
          indices = rewriter.create<Torch::AtenFlattenUsingIntsOp>(
              loc, flattenTy, indices, zero, end);
        }

        llvm::SmallVector<int64_t> gatherShape(dataTy.getSizes());
        gatherShape[axis] = indicesCt;
        auto gatherTy = rewriter.getType<Torch::ValueTensorType>(
            gatherShape, dataTy.getOptionalDtype());
        Value gather = rewriter.create<Torch::AtenIndexSelectOp>(
            loc, gatherTy, data, index, indices);

        if (indicesRank == 1) {
          rewriter.replaceOp(binder.op, gather);
          return success();
        }

        if (indicesRank > 1) {
          gather = rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
              binder.op, resultType, gather, index, indicesSizeList);
          return success();
        }

        rewriter.replaceOpWithNewOp<Torch::AtenSqueezeOp>(binder.op, resultType,
                                                          gather);
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
      "Gemm", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value a, b, c;
        float alpha, beta;
        int64_t transA, transB;
        if (binder.tensorOperandAtIndex(a, 0) ||
            binder.tensorOperandAtIndex(b, 1) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.f32FloatAttr(alpha, "alpha", 1.0f) ||
            binder.f32FloatAttr(beta, "beta", 1.0f) ||
            binder.tensorResultType(resultType))
          return failure();

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));

        auto transpose = [&](Value m) -> Value {
          auto tty = cast<Torch::ValueTensorType>(m.getType());
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

        if (binder.getNumOperands() == 2) {
          rewriter.replaceOpWithNewOp<Torch::AtenMmOp>(binder.op, resultType, a,
                                                       b);
          return success();
        }

        if (binder.tensorOperandAtIndex(c, 2))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected either 2 or 3 inputs");

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
  patterns.onOp(
      "GlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTensorType = cast<Torch::ValueTensorType>(operand.getType());
        if (!inputTensorType || !inputTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        unsigned inputRank = inputShape.size();
        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value inputDimSize = rewriter.create<Torch::AtenSizeIntOp>(
                binder.getLoc(), operand, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
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
        Value cstFalse =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        if (inputRank == 3) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool1dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad);
          return success();
        } else if (inputRank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        } else if (inputRank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenAvgPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
          return success();
        }
        return failure();
      });
  patterns.onOp(
      "GlobalMaxPool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTensorType = operand.getType().cast<Torch::ValueTensorType>();
        if (!inputTensorType || !inputTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        unsigned inputRank = inputShape.size();
        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        SmallVector<Value> cstKernel, cstPadding, cstStrides, cstDilations;
        Value cstZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value inputDimSize = rewriter.create<Torch::AtenSizeIntOp>(
                binder.getLoc(), operand, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(inputShape[i])));
          }
          cstPadding.push_back(cstZero);
          cstDilations.push_back(cstOne);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
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
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);

        if (inputRank == 3) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool1dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        } else if (inputRank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        } else if (inputRank == 5) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dOp>(
              binder.op, resultType, operand, kernelSizeList, stridesList,
              paddingList, dilationsList, cstCeilMode);
          return success();
        }
        return failure();
      });
  patterns.onOp(
      "LayerNormalization", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType yType, meanType, invStdDevType;
        Value x, scale, b;
        int64_t axis, stashType;
        float epsilon;
        if (binder.tensorOperandAtIndex(x, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.tensorOperandAtIndex(b, 2) ||
            binder.tensorResultTypeAtIndex(yType, 0) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.f32FloatAttr(epsilon, "epsilon", 0.00001f) ||
            binder.s64IntegerAttr(stashType, "stash_type", 1))
          return failure();
        Value constEpsilon = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(epsilon));
        unsigned rank = 1;
        if (std::optional<unsigned> maybeRank = Torch::getTensorRank(x))
          rank = *maybeRank;
        SmallVector<Value> normalized;
        axis = Torch::toPositiveDim(axis, rank);
        auto xType = cast<Torch::ValueTensorType>(x.getType());
        if (!xType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input (X) to have sizes");
        }
        ArrayRef<int64_t> xShape = xType.getSizes();
        for (int64_t n = axis; n < rank; n++) {
          normalized.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(xShape[n])));
        }
        Value normalized_shape = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            normalized);

        int64_t numResults = binder.op->getNumResults();
        if (numResults == 1) {
          SmallVector<int64_t> reducedShape(rank, 1);
          for (int64_t i = 0; i < axis; i++)
            reducedShape[i] = xShape[i];
          auto reducedType = xType.getWithSizesAndDtype(
              reducedShape, xType.getOptionalDtype());
          Value y = rewriter
                        .create<Torch::AtenNativeLayerNormOp>(
                            binder.getLoc(), yType, /*meanType=*/reducedType,
                            /*invStdDevType=*/reducedType, x, normalized_shape,
                            scale, b, constEpsilon)
                        .getResult0();
          rewriter.replaceOp(binder.op, y);
          return success();
        }
        if (numResults == 3) {
          if (binder.tensorResultTypeAtIndex(meanType, 1) ||
              binder.tensorResultTypeAtIndex(invStdDevType, 2))
            return failure();
          rewriter.replaceOpWithNewOp<Torch::AtenNativeLayerNormOp>(
              binder.op, yType, meanType, invStdDevType, x, normalized_shape,
              scale, b, constEpsilon);
          return success();
        }
        return rewriter.notifyMatchFailure(
            binder.op, "Unimplemented: expected either 1 or 3 results");
      });
  patterns.onOp("LeakyRelu", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value operand;
                  float alpha;
                  if (binder.tensorOperand(operand) ||
                      binder.tensorResultType(resultType) ||
                      binder.f32FloatAttr(alpha, "alpha", 0.01f))
                    return failure();
                  Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
                      binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                      rewriter.getF64FloatAttr(alpha));
                  rewriter.replaceOpWithNewOp<Torch::AtenLeakyReluOp>(
                      binder.op, resultType, operand, constAlpha);
                  return success();
                });
  patterns.onOp(
      "Pad", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, pads, axes;
        std::string mode;

        // TODO: The `axes` parameter is not supported yet.
        if (!binder.tensorOperandAtIndex(axes, 3)) {
          return rewriter.notifyMatchFailure(
              binder.op, "The axes parameter is not supported yet");
        }
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(pads, 1) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(mode, "mode", "constant"))
          return failure();
        Location loc = binder.getLoc();

        // Get pads shape and rank. The pads tensor is expected to be 1-D
        // tensor.
        auto padsTensorType = cast<Torch::ValueTensorType>(pads.getType());
        if (!padsTensorType || !padsTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expect non empty pad tensor");
        }
        ArrayRef<int64_t> padsShape = padsTensorType.getSizes();
        int64_t padsRank = padsShape.size();
        if (padsRank != 1)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expect 1-d pad tensor");

        int64_t padsSize = padsShape[0];
        if (padsSize == Torch::kUnknownSize) {
          // As per onnx.Pad documentation, padSize = 2*num_data_axes
          // (if axes param not passed). Need to be updated when adding
          // support for `axes` param.
          auto dataOpTy = cast<Torch::ValueTensorType>(data.getType());
          TensorType dataTensor = dataOpTy.toBuiltinTensor();
          if (!dataTensor || !dataTensor.hasRank())
            return rewriter.notifyMatchFailure(
                binder.op, "pad length unknown and data operand unranked");
          int64_t dataRank = dataTensor.getRank();
          padsSize = 2 * dataRank;
        }

        Value constantValue;
        if (binder.getNumOperands() >= 3) {
          if (!binder.tensorOperandAtIndex(constantValue, 2)) {
            auto constTy =
                dyn_cast<Torch::BaseTensorType>(constantValue.getType());
            if (!constTy || !constTy.hasDtype())
              return rewriter.notifyMatchFailure(
                  binder.op, "constant ty is unsupport type");

            Type scalarTy = rewriter.getType<Torch::IntType>();
            if (isa<FloatType>(constTy.getDtype()))
              scalarTy = rewriter.getType<Torch::FloatType>();
            constantValue = rewriter.create<Torch::AtenItemOp>(loc, scalarTy,
                                                               constantValue);
          }
        }

        if (!constantValue) {
          auto dataTensorType = cast<Torch::ValueTensorType>(data.getType());
          if (dataTensorType.getDtype().isa<IntegerType>())
            constantValue = rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(0));
          if (dataTensorType.getDtype().isa<FloatType>())
            constantValue = rewriter.create<Torch::ConstantFloatOp>(
                loc, rewriter.getF64FloatAttr(0.0f));

          if (!constantValue)
            return rewriter.notifyMatchFailure(
                binder.op, "expected integer or float data tensor");
        }

        // Extract all the values of 1-D pad tensor and create a list of all
        // these values as torch.pad op expects pad list.
        Value constZero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        SmallVector<Value> padsTensorValue;
        SmallVector<int64_t> emptyShape;
        Type padsElemType =
            Torch::ValueTensorType::get(padsTensorType.getContext(), emptyShape,
                                        padsTensorType.getOptionalDtype());
        for (uint32_t i = 0; i < padsSize; ++i) {
          Value index = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i));
          auto select = rewriter.create<Torch::AtenSelectIntOp>(
              loc, padsElemType, pads, constZero, index);
          Value selectInt = rewriter.create<Torch::AtenItemOp>(
              loc, rewriter.getType<Torch::IntType>(), select);
          padsTensorValue.push_back(selectInt);
        }

        // The torch.pad op expects a different arrangement of padding pairs for
        // each dimension as compared to the onnx.pad op. So, rearranging pad
        // tensor to satisfy torch.pad op semantics.
        SmallVector<Value> padsRearrange;
        for (uint32_t i = 0; i < padsSize / 2; i++) {
          padsRearrange.emplace_back(padsTensorValue[i]);
          padsRearrange.emplace_back(padsTensorValue[(padsSize / 2) + i]);
        }

        Value padsSizeList =
            rewriter
                .create<Torch::PrimListConstructOp>(
                    loc,
                    Torch::ListType::get(rewriter.getType<Torch::IntType>()),
                    padsRearrange)
                .getResult();
        Value modeVal = rewriter.create<Torch::ConstantStrOp>(
            loc, rewriter.getStringAttr(mode));

        rewriter.replaceOpWithNewOp<Torch::AtenPadOp>(
            binder.op, resultType, data, padsSizeList, modeVal, constantValue);
        return success();
      });
  patterns.onOp("Pow", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value lhs, rhs;
                  if (binder.tensorOperands(lhs, rhs) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenPowTensorTensorOp>(
                      binder.op, resultType, lhs, rhs);
                  return success();
                });
  patterns.onOp(
      "Identity", 14, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value tensor;
        if (binder.tensorOperand(tensor) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        Value noneVal = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        rewriter.replaceOpWithNewOp<Torch::AtenCloneOp>(
            binder.op, resultType, tensor, /*memory_format=*/noneVal);
        return success();
      });
  patterns.onOp(
      "Mean", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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
        Value numOperandsConstant = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), numOperands));
        if (binder.tensorOperands(valList, numOperands) ||
            binder.tensorResultType(resultType))
          return failure();
        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        // Short circuit to binary add
        Value curr = rewriter.create<Torch::AtenAddTensorOp>(
            binder.getLoc(), resultType, valList[0], valList[1], constOne);
        if (numOperands == 2) {
          rewriter.replaceOpWithNewOp<Torch::AtenDivScalarOp>(
              binder.op, resultType, curr, numOperandsConstant);
          return success();
        }
        // When binder.op->getNumOperands() > 2
        auto baseType = Torch::ValueTensorType::getWithLeastStaticInformation(
            binder.op->getContext());
        for (int i = 2; i < numOperands; i++) {
          if (i == numOperands - 1) {
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), resultType, curr, valList[i], constOne);
          } else {
            curr = rewriter.create<Torch::AtenAddTensorOp>(
                binder.getLoc(), baseType, curr, valList[i], constOne);
          }
        }
        rewriter.replaceOpWithNewOp<Torch::AtenDivScalarOp>(
            binder.op, resultType, curr, numOperandsConstant);
        return success();
      });
  patterns.onOp(
      "IsInf", 10, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value tensor;
        int64_t neg;
        int64_t pos;
        if (binder.tensorOperand(tensor) ||
            binder.s64IntegerAttr(neg, "detect_negative", 1) ||
            binder.s64IntegerAttr(pos, "detect_positive", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        if (neg == 0) {
          // replace all negative infs with 0
          tensor = rewriter.create<Torch::AtenReluOp>(
              binder.getLoc(),
              dyn_cast<Torch::ValueTensorType>(tensor.getType()), tensor);
        }
        if (pos == 0) {
          // first use neg op to flip positive inf to negative inf. Then relu to
          // replace all positive infs with 0.
          Value flip = rewriter.create<Torch::AtenNegOp>(
              binder.getLoc(),
              dyn_cast<Torch::ValueTensorType>(tensor.getType()), tensor);
          tensor = rewriter.create<Torch::AtenReluOp>(
              binder.getLoc(), dyn_cast<Torch::ValueTensorType>(flip.getType()),
              flip);
        }
        rewriter.replaceOpWithNewOp<Torch::AtenIsinfOp>(binder.op, resultType,
                                                        tensor);
        return success();
      });
  patterns.onOp("IsNaN", 9,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value tensor;
                  if (binder.tensorOperand(tensor) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenIsnanOp>(
                      binder.op, resultType, tensor);
                  return success();
                });
  patterns.onOp("PRelu", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value tensor;
                  Value slope;
                  if (binder.tensorOperands(tensor, slope) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenPreluOp>(
                      binder.op, resultType, tensor, slope);
                  return success();
                });
  patterns.onOp("Mod", 13,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value self, other;
                  int64_t fmod;
                  if (binder.tensorOperands(self, other) ||
                      binder.tensorResultType(resultType) ||
                      binder.s64IntegerAttr(fmod, "fmod", 0)) {
                    return failure();
                  }

                  if (fmod) {
                    rewriter.replaceOpWithNewOp<Torch::AtenFmodTensorOp>(
                        binder.op, resultType, self, other);
                    return success();
                  }

                  rewriter.replaceOpWithNewOp<Torch::AtenRemainderTensorOp>(
                      binder.op, resultType, self, other);
                  return success();
                });
  patterns.onOp("Mish", 18,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value input;
                  if (binder.tensorOperand(input) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenMishOp>(
                      binder.op, resultType, input);
                  return success();
                });
  patterns.onOp(
      "OneHot", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        llvm::SmallVector<Value> inputs;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperandsList(inputs) ||
            binder.tensorResultType(resultType))
          return failure();

        if (inputs.size() != 3)
          return rewriter.notifyMatchFailure(binder.op, "expected 3 operands");

        int64_t axis;
        if (binder.s64IntegerAttr(axis, "axis", -1))
          return rewriter.notifyMatchFailure(binder.op,
                                             "`axis` attr not found");

        auto loc = binder.getLoc();
        Value indices = inputs[0];
        Value depth = inputs[1];
        Value values = inputs[2];

        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        auto valuesTy = cast<Torch::ValueTensorType>(values.getType());
        auto depthTy = cast<Torch::ValueTensorType>(depth.getType());

        axis = axis < 0 ? axis + indicesTy.getSizes().size() + 1 : axis;

        bool depthIsInt = isa<IntegerType>(depthTy.getDtype());
        Type intTy = rewriter.getType<Torch::IntType>();
        Type floatTy = rewriter.getType<Torch::FloatType>();
        Type depthETy = depthIsInt ? intTy : floatTy;
        depth = rewriter.create<Torch::AtenItemOp>(loc, depthETy, depth);

        if (!depthIsInt)
          depth = rewriter.create<Torch::AtenIntScalarOp>(
              loc, rewriter.getType<Torch::IntType>(), depth);

        auto selectTy = rewriter.getType<Torch::ValueTensorType>(
            llvm::SmallVector<int64_t>{1}, valuesTy.getDtype());

        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));

        Value off = rewriter.create<Torch::AtenSelectIntOp>(loc, selectTy,
                                                            values, zero, zero);
        off = rewriter.create<Torch::AtenItemOp>(
            loc, rewriter.getType<Torch::IntType>(), off);

        Value on = rewriter.create<Torch::AtenSelectIntOp>(loc, selectTy,
                                                           values, zero, one);
        on = rewriter.create<Torch::AtenItemOp>(
            loc, rewriter.getType<Torch::IntType>(), on);

        auto i32Ty = rewriter.getIntegerType(32, true);
        llvm::SmallVector<int64_t> onehotShape(indicesTy.getSizes());
        onehotShape.push_back(Torch::kUnknownSize);
        auto onehotTy =
            rewriter.getType<Torch::ValueTensorType>(onehotShape, i32Ty);

        Value onehot = rewriter.create<Torch::AtenOneHotOp>(
            binder.getLoc(), onehotTy, indices, depth);

        for (int i = valuesTy.getSizes().size(); i > axis; ++i) {
          std::swap(onehotShape[i - 1], onehotShape[i]);
          Value iv0 = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i));
          Value iv1 = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i - 1));

          onehotTy =
              rewriter.getType<Torch::ValueTensorType>(onehotShape, i32Ty);
          onehot = rewriter.create<Torch::AtenTransposeIntOp>(loc, onehotTy,
                                                              onehot, iv1, iv0);
        }

        // Change one hot to an array of booleans to select value:
        auto i1Ty = rewriter.getI1Type();
        auto torchqTy = Torch::getScalarTypeForType(i1Ty);
        Value tyConst = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    static_cast<int64_t>(torchqTy)));

        onehotTy = rewriter.getType<Torch::ValueTensorType>(onehotShape, i1Ty);
        Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
        onehot = rewriter.create<Torch::AtenToDtypeOp>(
            loc, onehotTy, onehot, tyConst,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/none);

        onehotTy = rewriter.getType<Torch::ValueTensorType>(
            onehotShape, resultType.getDtype());
        onehot = rewriter.create<Torch::AtenWhereScalarOp>(loc, onehotTy,
                                                           onehot, on, off);

        rewriter.replaceOp(binder.op, onehot);
        return success();
      });
  patterns.onOp("HardSwish", 14,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value input;
                  if (binder.tensorOperand(input) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  rewriter.replaceOpWithNewOp<Torch::AtenHardswishOp>(
                      binder.op, resultType, input);
                  return success();
                });
}
