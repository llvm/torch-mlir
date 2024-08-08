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
        Value alphaMulX = rewriter.create<Torch::AtenMulScalarOp>(
            binder.getLoc(), resultType, tensorOperand, constAlpha);
        Value constOne = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(1.0));
        Value alphaMulXPlusBeta = rewriter.create<Torch::AtenAddScalarOp>(
            binder.getLoc(), resultType, alphaMulX, constBeta,
            /*alpha=*/constOne);

        // Expression: min(1, alpha * x + beta)
        Value oneTensor =
            createRank0Tensor(rewriter, binder.getLoc(), resultType, constOne);
        Value minExpression = rewriter.create<Torch::AtenMinimumOp>(
            binder.getLoc(), resultType, oneTensor, alphaMulXPlusBeta);

        // Expression: max(0, min(1, alpha * x + beta))
        Value constZero = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getF64FloatAttr(0.0));
        Value zeroTensor =
            createRank0Tensor(rewriter, binder.getLoc(), resultType, constZero);
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
  patterns.onOp("GRU", 1, onnx_c::OnnxGruExpander);
  patterns.onOp(
      "If", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value conditionTensor;
        if (binder.tensorOperand(conditionTensor)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "condition bind failure");
        }

        auto conditionType =
            cast<Torch::ValueTensorType>(conditionTensor.getType());
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
  patterns.onOp(
      "Loop", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // Get all operands (maxTripCount, cond, ....inits....)
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) || operands.size() == 0 ||
            binder.getNumOperands() < 2) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required operands");
        }

        llvm::SmallVector<mlir::Type> operandTypeVec;
        if (binder.tensorOperandTypes(operandTypeVec) ||
            operandTypeVec.size() == 0) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get operandTypes");
        }

        Region *loopBodyIn;
        if (binder.getRegionAtIndex(loopBodyIn, 0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed getting LoopBody Region");
        }

        // MaxTripCount - tensor int64 scalar (or empty)
        Value maxTripCountTensor = operands[0];
        auto maxTripCountInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            maxTripCountTensor);

        // Condition - tensor bool scalar (or empty)
        Value conditionTensor = operands[1];
        auto conditionInt = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            conditionTensor);
        auto conditionBool = rewriter.create<Torch::AtenBoolIntOp>(
            binder.getLoc(), rewriter.getType<Torch::BoolType>(), conditionInt);
        // To be used for "for like" loop case
        auto constBoolTrue = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(true));

        // Others (if present) - variadic (can be tensors and scalar values)
        if (binder.getNumOperands() > 2) {
          operandTypeVec.erase(operandTypeVec.begin(),
                               operandTypeVec.begin() + 2);
          operands.erase(operands.begin(), operands.begin() + 2);
        }

        auto getOpName = [](Operation *op) -> std::string {
          std::string name = op->getName().getStringRef().str();
          if (name != "torch.operator")
            return name;
          // for unconverted onnx ops
          return mlir::dyn_cast<StringAttr>(op->getAttr("name"))
              .getValue()
              .str();
        };

        // PrimLoop Op expectes inputCondition to be boolConstantTrue
        // to decide if the loopOp is `forlike`. Use loopIsForLike to
        // ensure appropriate inputCondition is set
        // Case 1 : loopCondInp -> identity -> terminator(loopCondOut)
        bool loopIsForLike = false;
        auto case1ForLike = [&getOpName](Region *loopBody) -> bool {
          Value onnxLoopBodyCondIn = loopBody->front().getArgument(1);
          if (!onnxLoopBodyCondIn.hasOneUse())
            return false;
          Operation *inpCondUser = *onnxLoopBodyCondIn.getUsers().begin();
          if (getOpName(inpCondUser) != "onnx.Identity") {
            return false;
          }
          if (!inpCondUser->hasOneUse() ||
              getOpName(*(inpCondUser->getUsers().begin())) !=
                  "torch.operator_terminator")
            return false;
          return true;
        };
        loopIsForLike = case1ForLike(loopBodyIn);

        Value loopInitCondition =
            loopIsForLike ? constBoolTrue : conditionBool.getResult();
        auto loc = binder.getLoc();
        mlir::ImplicitLocOpBuilder b(loc, rewriter);
        auto loop = b.create<Torch::PrimLoopOp>(
            TypeRange(operandTypeVec), maxTripCountInt, loopInitCondition,
            ValueRange(operands));

        rewriter.cloneRegionBefore(*loopBodyIn, loop.getRegion(),
                                   loop.getRegion().begin());

        // primLoopOp loopBody expects torch.int as first arg
        // insert torch.int arg in loop body, convert to tensor,
        // replace all uses of old arg, delete old arg.
        auto loopVarArg = loop.getRegion().front().getArgument(0);
        // insert new Arg
        loop.getRegion().front().insertArgument(
            0U, rewriter.getType<Torch::IntType>(), binder.getLoc());
        auto newLoopVarArg = loop.getRegion().front().getArgument(0);

        // convert int arg to tensor of original Type
        rewriter.setInsertionPointToStart(&loop.getRegion().front());
        Value loopVarVal = BlockArgument::Value(loopVarArg);
        auto newTensor = rewriter.create<Torch::PrimNumToTensorScalarOp>(
            loop.getRegion().op_begin()->getLoc(), loopVarVal.getType(),
            newLoopVarArg);

        loopVarArg.replaceAllUsesWith(newTensor);
        loop.getRegion().eraseArgument(1);

        // primLoopOp loopBody has no condition arg
        auto condArg = loop.getRegion().front().getArgument(1);
        if (!condArg.use_empty())
          condArg.replaceAllUsesWith(conditionTensor);

        // replace terminator
        PatternRewriter::InsertionGuard guard(rewriter);
        Operation *terminator = loop.getRegion().front().getTerminator();
        rewriter.setInsertionPoint(terminator);

        // results - n loop carried dependencies and k scan outputs
        // Fail when there are scanOutputs in onnxLoop (K>0);
        // unsupported for now
        if (terminator->getNumOperands() !=
            loop.getRegion().getNumArguments() - 1) {
          return rewriter.notifyMatchFailure(
              binder.op, "scanOutputs in loop body unsupported");
        }

        // Get remaining operands from onnxLoopBody's terminator Op
        // these are all the loop carried dependencies in the loop body
        auto terminatorOperands = terminator->getOperands();
        llvm::SmallVector<Value> remTerminatorOperands(
            terminatorOperands.begin() + 1, terminatorOperands.end());
        Value terminatorCond;
        if (loopIsForLike) {
          terminatorCond = constBoolTrue;
        } else {
          // Only use when loop is not forlike
          Value terminatorCondTensor = terminatorOperands[0];
          auto terminatorCondInt = rewriter.create<Torch::AtenItemOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              terminatorCondTensor);
          auto terminatorCondBool = rewriter.create<Torch::AtenBoolIntOp>(
              binder.getLoc(), rewriter.getType<Torch::BoolType>(),
              terminatorCondInt);
          terminatorCond = terminatorCondBool.getResult();
        }
        rewriter.replaceOpWithNewOp<Torch::PrimLoopConditionOp>(
            terminator, terminatorCond, remTerminatorOperands);

        loop.getRegion().eraseArgument(1);
        rewriter.replaceOp(binder.op, loop);
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
  patterns.onOp("MatMul", 1,
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

        auto lhsQTy = getQTorchTypeFromTorchIntType(lhsTy);
        auto rhsQTy = getQTorchTypeFromTorchIntType(rhsTy);

        if (!lhsQTy || !rhsQTy)
          return rewriter.notifyMatchFailure(binder.op, "failed to get qtype");

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
  patterns.onOp(
      "Multinomial", 7,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value self;
        int64_t onnxDtype, sampleSize;

        if (binder.tensorOperand(self) ||
            binder.s64IntegerAttr(onnxDtype, "dtype", 6) ||
            binder.s64IntegerAttr(sampleSize, "sample_size", 1) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        if (binder.op->hasAttr("torch.onnx.seed")) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented: support not present for seed attribute");
        }

        if (sampleSize <= 0) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "unsupported: sample_size <= 0");
        }

        std::optional<int64_t> torchDtype =
            onnxDtypeIntToTorchDtypeInt(onnxDtype);
        if (!torchDtype.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "unimplemented support for the given dtype conversion");
        }

        Value torchDtypeIntValue = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(torchDtype.value()));
        Value numSamples = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(sampleSize));

        // PRG is seeded globally by default
        Value none = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        // Sample with replacement by default (no onnx equivalent in arguments)
        Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(true));

        // Torch Multinomial always produces a LongTensor
        Torch::ValueTensorType selfType =
            cast<Torch::ValueTensorType>(self.getType());
        Type int64Dtype =
            IntegerType::get(selfType.getContext(), 64, IntegerType::Signed);
        int64_t batchSize = selfType.getSizes()[0];
        SmallVector<int64_t> outShapes({batchSize, sampleSize});
        Torch::ValueTensorType multinomialOutputType =
            Torch::ValueTensorType::get(selfType.getContext(), outShapes,
                                        int64Dtype);
        Value multinomialTensor = rewriter.create<Torch::AtenMultinomialOp>(
            binder.getLoc(), multinomialOutputType, self, numSamples, cstTrue,
            none);

        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, multinomialTensor, torchDtypeIntValue,
            cstFalse, cstFalse, none);

        return success();
      });
  patterns.onOp(
      "NegativeLogLikelihoodLoss", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value self, target, weight, reduction, ignore_index;
        int64_t ignore_index_int;
        std::string reduction_str;

        if (binder.tensorOperandAtIndex(self, 0) ||
            binder.tensorOperandAtIndex(target, 1) ||
            binder.s64IntegerAttr(ignore_index_int, "ignore_index", -100) ||
            binder.customOpNameStringAttr(reduction_str, "reduction", "mean") ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        // optional third tensor argument
        if (binder.tensorOperandAtIndex(weight, 2)) {
          weight = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        }

        ignore_index = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(ignore_index_int));

        // convert string reduction attr to standardized integer enum value
        int reduction_value =
            torch_upstream::get_loss_reduction_enum(reduction_str);
        reduction = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(reduction_value));

        Value nllLoss = rewriter
                            .create<Torch::AtenNllLossForwardOp>(
                                binder.getLoc(), resultType, resultType, self,
                                target, weight, reduction, ignore_index)
                            ->getResult(0);

        rewriter.replaceOp(binder.op, nllLoss);
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

        Torch::ValueTensorType resultTypeOut;
        Value operand;
        int64_t ceilMode, storageOrder;
        // TODO: Add support for indices output and storage_order
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(ceilMode, "ceil_mode", 0) ||
            binder.s64IntegerAttr(storageOrder, "storage_order", 0) ||
            binder.tensorResultTypeAtIndex(resultTypeOut, 0))
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
          for (int i = 0; i < spatial; ++i) {
            paddedShape[i + 2] += padding[i] + padding[i + spatial];
            shuffledPadding[2 * i] = padding[spatial - i - 1];
            shuffledPadding[2 * i + 1] = padding[2 * spatial - i - 1];
          }

          Value shuffledPaddingList =
              createConstantIntList(binder, rewriter, shuffledPadding);
          Value zero;
          if (isa<FloatType>(resultTypeOut.getDtype())) {
            zero = rewriter.create<Torch::ConstantFloatOp>(
                binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(
                    std::numeric_limits<double>::lowest()));
          } else if (isa<IntegerType>(resultTypeOut.getDtype())) {
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

        if (binder.op->getNumResults() == 2) {
          Torch::ValueTensorType resultTypeIndices;
          if (binder.tensorResultTypeAtIndex(resultTypeIndices, 1))
            return failure();

          if (rank == 3)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: AtenMaxPool1dWithIndicesOp");

          if (rank == 4) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dWithIndicesOp>(
                binder.op, resultTypeOut, resultTypeIndices, operand,
                kernelSizeList, stridesList, paddingList, dilationsList,
                cstCeilMode);
            return success();
          }
          if (rank == 5) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dWithIndicesOp>(
                binder.op, resultTypeOut, resultTypeIndices, operand,
                kernelSizeList, stridesList, paddingList, dilationsList,
                cstCeilMode);
            return success();
          }
        } else {
          if (rank == 3) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool1dOp>(
                binder.op, resultTypeOut, operand, kernelSizeList, stridesList,
                paddingList, dilationsList, cstCeilMode);
            return success();
          }
          if (rank == 4) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool2dOp>(
                binder.op, resultTypeOut, operand, kernelSizeList, stridesList,
                paddingList, dilationsList, cstCeilMode);
            return success();
          }
          if (rank == 5) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool3dOp>(
                binder.op, resultTypeOut, operand, kernelSizeList, stridesList,
                paddingList, dilationsList, cstCeilMode);
            return success();
          }
        }
        return rewriter.notifyMatchFailure(binder.op, "No rank is matched.");
      });
  patterns.onOp(
      "MaxRoiPool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<int64_t> pooledShape;
        float spatialScale;
        if (binder.s64IntegerArrayAttr(pooledShape, "pooled_shape", {}) ||
            binder.f32FloatAttr(spatialScale, "spatial_scale", 1.0f)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Attribute bind failure");
        }
        Torch::ValueTensorType resultTy;
        Value input, rois;
        if (binder.tensorOperands(input, rois) ||
            binder.tensorResultType(resultTy)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Operand or result type mismatch");
        }

        Value outputShapeList =
            createConstantIntList(binder, rewriter, pooledShape);
        Location loc = binder.getLoc();

        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        auto roisTy = cast<Torch::ValueTensorType>(rois.getType());
        if (!inputTy || !inputTy.hasSizes())
          return failure();
        if (!roisTy || !roisTy.hasSizes())
          return failure();

        auto intTy = rewriter.getIntegerType(64, true);
        auto floatTy = roisTy.getDtype();
        auto torchIntTy = rewriter.getType<Torch::IntType>();

        Value spatialScaleValue = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getF64FloatAttr(spatialScale));

        Value boolTrue = rewriter.create<Torch::ConstantBoolOp>(
            loc, rewriter.getBoolAttr(true));

        ArrayRef<int64_t> inputShape = inputTy.getSizes();
        int64_t inputRank = inputShape.size();
        if (inputRank < 4) {
          return rewriter.notifyMatchFailure(
              binder.op, "Rank of input tensor must be >= 4");
        }

        ArrayRef<int64_t> roisShape = roisTy.getSizes();
        if (!roisTy.areAllSizesKnown() || roisShape.size() != 2 ||
            roisShape[1] != 5) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected ROIs to be statically sized tensor of shape "
                         "(num_rois, 5)");
        }
        int64_t numRois = roisShape[0];

        /* The implementation is based on the following algorithm:
          MaxRoiPool <pooled_shape, spatial_scale>(
              input : tensor<float>, rois : tensor<?x5xfloat>) => (output)
          {
            * Step 1: Extract ROI specification
              - Each ROI is represented as [batch_id, x1, y1, x2, y2], where
                range is inclusive of x1, y1, x2, and y2
              - The range values are scaled by spatial_scale

            BatchIdxsFloat = Select(rois, dim=1, index=0)
            BatchIdxs = CastLong(BatchIdxsFloat)
            RoiBBsFloat = Slice(rois, dim=1, start=1, end=5, stride=1)
            RoiBBsScaledFloat = MulScalar(RoiBBsFloat, spatial_scale)
            RoiBBsScaled = CastLong(RoiBBsScaledFloat)

            * Step 2: Iteratively pool ROIs
            pooledROIs = []
            for (roiIdx = 0; roiIdx < len(rois); roiIdx++) {
              * Step 2a: For each ROI, we extract batch_id, x1, y1, x2, & y2
              RoiSpec = Select(RoiBBsScaled, 0, roiIdx) : tensor<4xint>
              roiValues = []
              for (specIdx = 0; specIdx < 5; specIdx++) {
                if (specIdx == 0)
                  SpecTensor = Select(BatchIdxs, 1, roiIdx) : tensor<int>
                else
                  SpecTensor = Select(RoiSpec, 0, specIdx-1) : tensor<int>
                SpecValue = Item(specTensor) : torch.int
                roiValues.push(SpecValue)
              }
              BatchIdx, X1, Y1, X2, Y2 = roiValues

              * Step 2b: extract image from input and extract region
                - X2 and Y2 are incremented by 1 to make range inclusive
                - width and height dimension are calculated once outside of loop
                  but intuition is expressed more clearly below

              image = Select(input, 0, BatchIdx)
              widthDim = rank(image) - 1
              heightDim = rank(image) - 2

              imageExtractedY = Slice(image, heightDim, Y1, Y2 + 1, 1)
              region = Slice(image, widthDim, X1, X2 + 1, 1)

              * Step 2c: apply adaptive max pooling to pool region of interest
                         into final pooled size
              pooledROI = AdaptiveMaxPool2d(region, pooled_shape)
              pooledROIs.push(pooledROI)
            }

            * Step 3: Stack pooled regions and return final output
            return output = Stack(pooledRois, dim=0)
          }
        */

        SmallVector<Value> constInts(6);
        for (int i = 0; i <= 5; i++) {
          constInts[i] = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i));
        }

        int64_t widthDim = inputRank - 2;
        Value widthDimValue = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(widthDim));

        int64_t heightDim = inputRank - 3;
        Value heightDimValue = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(heightDim));

        // extract indices of images within batch
        auto batchIdxsShape = SmallVector<int64_t>{Torch::kUnknownSize};
        auto batchIdxsFloatTy =
            rewriter.getType<Torch::ValueTensorType>(batchIdxsShape, floatTy);
        Value batchIdxsFloat = rewriter.create<Torch::AtenSelectIntOp>(
            loc, batchIdxsFloatTy, rois, constInts[1], constInts[0]);
        auto batchIdxsIntTy =
            rewriter.getType<Torch::ValueTensorType>(batchIdxsShape, intTy);
        Value batchIdxs = rewriter.create<Torch::Aten_CastLongOp>(
            loc, batchIdxsIntTy, batchIdxsFloat, boolTrue);

        // extract scaled ranges for regions of interest
        auto roiBBsShape = SmallVector<int64_t>{Torch::kUnknownSize, 4};
        auto roiBBsFloatTy =
            rewriter.getType<Torch::ValueTensorType>(roiBBsShape, floatTy);
        Value roiBBs = rewriter.create<Torch::AtenSliceTensorOp>(
            loc, roiBBsFloatTy, rois, constInts[1], constInts[1], constInts[5],
            constInts[1]);
        Value roiBBsScaledFloat = rewriter.create<Torch::AtenMulScalarOp>(
            loc, roiBBsFloatTy, roiBBs, spatialScaleValue);
        auto roiBBsTy =
            rewriter.getType<Torch::ValueTensorType>(roiBBsShape, intTy);
        Value roiBBsScaled = rewriter.create<Torch::Aten_CastLongOp>(
            loc, roiBBsTy, roiBBsScaledFloat, boolTrue);

        SmallVector<Value> pooledRois;

        for (int64_t i = 0; i < numRois; i++) {
          Value roiIdx = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(i));

          auto roiSpecTy = rewriter.getType<Torch::ValueTensorType>(
              roiBBsTy.getSizes().slice(1), intTy);
          Value roiSpec = rewriter.create<Torch::AtenSelectIntOp>(
              loc, roiSpecTy, roiBBsScaled, constInts[0], roiIdx);

          // Load individual ROI specification values
          SmallVector<Value> roiValues(5);
          for (int specIdx = 0; specIdx < 5; specIdx++) {
            auto intEmptyTensorTy = rewriter.getType<Torch::ValueTensorType>(
                SmallVector<int64_t>{}, intTy);
            Value specTensor;
            if (specIdx == 0) { // batch index
              specTensor = rewriter.create<Torch::AtenSelectIntOp>(
                  loc, intEmptyTensorTy, batchIdxs, constInts[0], roiIdx);
            } else { // roi dimension
              specTensor = rewriter.create<Torch::AtenSelectIntOp>(
                  loc, intEmptyTensorTy, roiSpec, constInts[0],
                  constInts[specIdx - 1]);
            }
            Value specValue =
                rewriter.create<Torch::AtenItemOp>(loc, torchIntTy, specTensor);
            roiValues[specIdx] = specValue;
          }
          Value batchIdx = roiValues[0], roiX1 = roiValues[1],
                roiY1 = roiValues[2], roiX2 = roiValues[3],
                roiY2 = roiValues[4];

          // add 1 to make range ends inclusive as per ONNX implementation
          roiX2 = rewriter.create<Torch::AtenAddOp>(loc, torchIntTy, roiX2,
                                                    constInts[1]);
          roiY2 = rewriter.create<Torch::AtenAddOp>(loc, torchIntTy, roiY2,
                                                    constInts[1]);

          auto imageTy = rewriter.getType<Torch::ValueTensorType>(
              inputShape.slice(1), inputTy.getDtype());
          Value image = rewriter.create<Torch::AtenSelectIntOp>(
              loc, imageTy, input, constInts[0], batchIdx); // (NC x H x W)

          SmallVector<int64_t> imageUnknownShape(imageTy.getSizes());
          imageUnknownShape[heightDim] = Torch::kUnknownSize;
          imageUnknownShape[widthDim] = Torch::kUnknownSize;
          auto imageUnknownTy = rewriter.getType<Torch::ValueTensorType>(
              imageUnknownShape, imageTy.getDtype());

          // extract ROI from image
          Value imageExtractedY = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, imageUnknownTy, image, heightDimValue, roiY1, roiY2,
              constInts[1]);
          Value region = rewriter.create<Torch::AtenSliceTensorOp>(
              loc, imageUnknownTy, imageExtractedY, widthDimValue, roiX1, roiX2,
              constInts[1]);

          SmallVector<int64_t> pooledRegionShape(imageTy.getSizes());
          pooledRegionShape[heightDim] = pooledShape[0];
          pooledRegionShape[widthDim] = pooledShape[1];
          auto pooledRegionTy = rewriter.getType<Torch::ValueTensorType>(
              pooledRegionShape, imageTy.getDtype());
          auto pooledRegionIndicesTy = rewriter.getType<Torch::ValueTensorType>(
              pooledRegionShape, intTy);

          // apply pooling on ROI
          Value pooledRegion =
              rewriter
                  .create<Torch::AtenAdaptiveMaxPool2dOp>(
                      loc, pooledRegionTy, pooledRegionIndicesTy, region,
                      outputShapeList)
                  .getResult0();
          pooledRois.push_back(pooledRegion);
        }

        Value pooledRoisList = rewriter.create<Torch::PrimListConstructOp>(
            loc, Torch::ListType::get(pooledRois[0].getType()), pooledRois);
        rewriter.replaceOpWithNewOp<Torch::AtenStackOp>(
            binder.op, resultTy, pooledRoisList, constInts[0]);

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

        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        Value constZero = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value constOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value axisSize = rewriter.create<Torch::AtenSizeIntOp>(binder.getLoc(),
                                                               data, constAxis);
        Value indicesAdd = rewriter.create<Torch::AtenAddScalarOp>(
            binder.getLoc(), indicesTy, indices, axisSize, constOne);

        auto boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value lt = rewriter.create<Torch::AtenLtScalarOp>(
            binder.getLoc(), boolTy, indices, constZero);
        indices = rewriter.create<Torch::AtenWhereSelfOp>(
            binder.getLoc(), indicesTy, lt, indicesAdd, indices);

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
          std::optional<ArrayRef<int64_t>> shape = tty.getOptionalSizes();
          llvm::SmallVector<int64_t> newShape;
          if (shape.has_value()) {
            newShape.append(shape.value().begin(), shape.value().end());
            std::reverse(newShape.begin(), newShape.end());
            shape = newShape;
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
      "GlobalLpPool", 2,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t p;
        if (binder.tensorOperand(operand) || binder.s64IntegerAttr(p, "p", 2) ||
            binder.tensorResultType(resultType))
          return failure();

        auto inputTensorType = cast<Torch::ValueTensorType>(operand.getType());
        if (!inputTensorType || !inputTensorType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input type having sizes");
        }
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        unsigned inputRank = inputShape.size();
        // only handle 2D, 3D and 5D pooling cases
        if (inputRank > 5 or inputRank < 3) {
          return failure();
        }
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
        Value numElements = cstOne;
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
          numElements = rewriter.create<Torch::AtenMulOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              cstKernel.back(), numElements);
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
        Value abs = rewriter.create<Torch::AtenAbsOp>(binder.getLoc(),
                                                      inputTensorType, operand);
        Value pv = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), p));
        Value pow = rewriter.create<Torch::AtenPowTensorScalarOp>(
            binder.getLoc(), inputTensorType, abs, pv);
        Value avgPool;
        if (inputRank == 3) {
          avgPool = rewriter.create<Torch::AtenAvgPool1dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad);
          avgPool = rewriter.create<Torch::AtenMulScalarOp>(
              binder.getLoc(), resultType, avgPool, numElements);
        } else if (inputRank == 4) {
          avgPool = rewriter.create<Torch::AtenAvgPool2dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        } else { // inputRank == 5
          avgPool = rewriter.create<Torch::AtenAvgPool3dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        }
        Value invP = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(double{1.0 / p}));
        rewriter.replaceOpWithNewOp<Torch::AtenPowTensorScalarOp>(
            binder.op, resultType, avgPool, invP);
        return success();
      });

  patterns.onOp(
      "LpPool", 18, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return failure();
        if (autoPad != "NOTSET") {
          // TODO: Add support for `auto_pad` != "NOTSET"
          return rewriter.notifyMatchFailure(
              binder.op, "unsupported conversion: auto_pad != NOTSET");
        }

        Torch::ValueTensorType resultType;
        Value operand;
        int64_t ceilMode, p;
        if (binder.tensorOperand(operand) ||
            binder.s64IntegerAttr(ceilMode, "ceil_mode", 0) ||
            binder.s64IntegerAttr(p, "p", 2) ||
            binder.tensorResultType(resultType))
          return failure();
        // Determine the rank of input tensor.
        std::optional<unsigned> maybeRank = Torch::getTensorRank(operand);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        unsigned rank = *maybeRank;
        // only 1D, 2D and 3D LpPool is supported.
        if (rank > 5 or rank < 3) {
          return failure();
        }

        SmallVector<int64_t> kernel, padding, strides, dilations;
        SmallVector<int64_t> defaultPadding(2 * (rank - 2), 0);
        if (binder.s64IntegerArrayAttr(kernel, "kernel_shape", {}) ||
            binder.s64IntegerArrayAttr(padding, "pads", defaultPadding) ||
            binder.s64IntegerArrayAttr(
                strides, "strides", llvm::SmallVector<int64_t>(rank - 2, 1)) ||
            binder.s64IntegerArrayAttr(dilations, "dilations", {})) {
          return failure();
        }
        if (kernel.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "kernel list size does not match the number of axes");
        }
        if (padding.size() != 2 * (rank - 2)) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "padding list size does not match twice the number of axes");
        }
        if (strides.size() != rank - 2) {
          return rewriter.notifyMatchFailure(
              binder.op, "strides list size does not match the number of axes");
        }
        if (dilations.size() > 0) {
          return rewriter.notifyMatchFailure(
              binder.op, "dilation is not supported by torch.aten.avgpool op "
                         "and therefore it is not supported for LpPool.");
        }

        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstOne = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value numElements = cstOne;
        for (int64_t i : kernel) {
          cstKernel.push_back(rewriter.create<Torch::ConstantIntOp>(
              binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          numElements = rewriter.create<Torch::AtenMulOp>(
              binder.getLoc(), rewriter.getType<Torch::IntType>(),
              cstKernel.back(), numElements);
        }
        Value kernelSizeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), ceilMode);
        // onnx lp pool doesn't have countIncludePad attribute but set it to
        // true so that in 1D case numElements is correctly undoes divison. For
        // 2D/3D case, division is avoided by divison_override.
        Value cstCountIncludePad =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value pv = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), p));
        auto inputTensorType = cast<Torch::ValueTensorType>(operand.getType());
        Value abs = rewriter.create<Torch::AtenAbsOp>(binder.getLoc(),
                                                      inputTensorType, operand);
        Value pow = rewriter.create<Torch::AtenPowTensorScalarOp>(
            binder.getLoc(), inputTensorType, abs, pv);
        Value avgPool;
        if (rank == 3) {
          avgPool = rewriter.create<Torch::AtenAvgPool1dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad);
          avgPool = rewriter.create<Torch::AtenMulScalarOp>(
              binder.getLoc(), resultType, avgPool, numElements);
        } else if (rank == 4) {
          avgPool = rewriter.create<Torch::AtenAvgPool2dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        } else { // rank == 5
          avgPool = rewriter.create<Torch::AtenAvgPool3dOp>(
              binder.getLoc(), resultType, pow, kernelSizeList, stridesList,
              paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        }
        Value invP = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(double{1.0 / p}));
        rewriter.replaceOpWithNewOp<Torch::AtenPowTensorScalarOp>(
            binder.op, resultType, avgPool, invP);
        return success();
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

        // Since the support for `stash_type` arg does not exist in
        // the torch op so we just check for the stash_type to be same
        // as the input dtype since that won't require us to do any
        // input type conversion and hence can be supported.
        auto xType = cast<Torch::ValueTensorType>(x.getType());
        std::optional<int64_t> stashTypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(stashType);
        if (!stashTypeIntTorch.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for the given stash_type");

        FailureOr<Type> stashDtype = Torch::getTypeForScalarType(
            binder.op->getContext(),
            (torch_upstream::ScalarType)stashTypeIntTorch.value());
        if (failed(stashDtype))
          return failure();
        if (*stashDtype != xType.getOptionalDtype())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: stash_type should be same "
                         "as the input dtype");

        Value constEpsilon = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(epsilon));
        unsigned rank = 1;
        if (std::optional<unsigned> maybeRank = Torch::getTensorRank(x))
          rank = *maybeRank;
        SmallVector<Value> normalized;
        axis = Torch::toPositiveDim(axis, rank);
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
      "LRN", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        int64_t size;
        float alpha, beta, bias;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(size, "size", 2) ||
            binder.f32FloatAttr(alpha, "alpha", 0.0001f) ||
            binder.f32FloatAttr(beta, "beta", 0.75f) ||
            binder.f32FloatAttr(bias, "bias", 1.0f))
          return failure();
        Type dtype = resultType.getOptionalDtype();
        Value constAlpha = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));
        Value constBeta = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));
        Value constBias = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(bias));
        // Please refer to the operator description
        // for more info on the lowering
        // https://onnx.ai/onnx/operators/onnx__LRN.html

        // squared = operand^2
        Location loc = binder.getLoc();
        Torch::ValueTensorType inTy =
            cast<Torch::ValueTensorType>(operand.getType());
        Value sqOperand = rewriter.create<Torch::AtenMulTensorOp>(
            loc, inTy, operand, operand);
        // view it as n x 1 x c x d0 x d..
        if (!inTy.hasSizes()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected input to have sizes");
        }
        ArrayRef<int64_t> inTyShape = inTy.getSizes();
        if (inTyShape.size() < 3) {
          return rewriter.notifyMatchFailure(
              binder.op, "Unsupported: the input dimensions should be >= 3");
        }
        if (inTyShape[1] == Torch::kUnknownSize) {
          return rewriter.notifyMatchFailure(
              binder.op, "Unsupported: the second dimension size must be "
                         "statically known");
        }
        SmallVector<int64_t, 5> viewShapeInt{inTyShape[0], 1, inTyShape[1],
                                             inTyShape[2], Torch::kUnknownSize};
        Torch::ValueTensorType reshapeType =
            rewriter.getType<Torch::ValueTensorType>(viewShapeInt, dtype);
        Value viewShapeListVal =
            createConstantIntList(binder, rewriter, viewShapeInt);
        auto view = rewriter.create<Torch::AtenViewOp>(
            loc, reshapeType, sqOperand, viewShapeListVal);
        // padding
        int64_t highPad = (size - 1) / 2;
        int64_t lowPad = (size - 1) - highPad;
        SmallVector<int64_t> paddingInt{0, 0, 0, 0, lowPad, highPad};
        auto constPadVal = rewriter.create<Torch::ConstantFloatOp>(
            loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value paddingListVal =
            createConstantIntList(binder, rewriter, paddingInt);
        SmallVector<int64_t, 5> paddedShapeInt = viewShapeInt;
        paddedShapeInt[2] += size - 1;
        Torch::ValueTensorType paddedType =
            rewriter.getType<Torch::ValueTensorType>(paddedShapeInt, dtype);
        auto padded = rewriter.create<Torch::AtenConstantPadNdOp>(
            loc, paddedType, view, paddingListVal, constPadVal);
        // avg_pool3d
        SmallVector<int64_t, 3> kernelSize{size, 1, 1};
        Value kernelSizeList =
            createConstantIntList(binder, rewriter, kernelSize);
        SmallVector<int64_t, 3> strides{1, 1, 1};
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        SmallVector<int64_t, 3> padding{0, 0, 0};
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        auto cstCeilMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), false);
        auto cstCountIncludeMode =
            rewriter.create<Torch::ConstantBoolOp>(binder.getLoc(), true);
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        // Output of pooling is same reshape(view) type because
        // of the padding done on the dimensions being pooled.
        auto pool = rewriter.create<Torch::AtenAvgPool3dOp>(
            loc, reshapeType, padded, kernelSizeList, stridesList, paddingList,
            cstCeilMode, cstCountIncludeMode, /*divisor_override=*/cstNone);
        // squeeze
        auto one = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        SmallVector<int64_t, 5> squeezeShapeInt{
            viewShapeInt[0], viewShapeInt[2], viewShapeInt[3], viewShapeInt[4]};
        Torch::ValueTensorType squeezeType =
            rewriter.getType<Torch::ValueTensorType>(squeezeShapeInt, dtype);
        auto squeeze = rewriter.create<Torch::AtenSqueezeDimOp>(
            loc, squeezeType, pool, one);
        // view as input Type
        Value intTyShapeList =
            createConstantIntList(binder, rewriter, inTyShape);
        auto viewAsInput = rewriter.create<Torch::AtenViewOp>(
            loc, inTy, squeeze, intTyShapeList);
        // mul + add + pow + div
        auto mul = rewriter.create<Torch::AtenMulScalarOp>(
            loc, resultType, viewAsInput, constAlpha);
        auto add = rewriter.create<Torch::AtenAddScalarOp>(loc, resultType, mul,
                                                           constBias, one);
        auto pow = rewriter.create<Torch::AtenPowTensorScalarOp>(
            loc, resultType, add, constBeta);

        rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
            binder.op, resultType, operand, pow);
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
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(mode, "mode", "constant"))
          return failure();
        bool cstMode = (mode == "constant");

        // get input rank
        auto dataOpTy = cast<Torch::ValueTensorType>(data.getType());
        TensorType dataTensor = dataOpTy.toBuiltinTensor();
        if (!dataTensor || !dataTensor.hasRank())
          return rewriter.notifyMatchFailure(
              binder.op, "pad length unknown and data operand unranked");
        int64_t dataRank = dataTensor.getRank();
        int64_t padsSize = 2 * dataRank;

        Location loc = binder.getLoc();

        // get pads (earlier versions use an attribute, newer versions use a
        // tensor input)
        SmallVector<Value> padsTensorValue;
        if (binder.tensorOperandAtIndex(pads, 1)) {
          SmallVector<int64_t> defaultPads(2 * dataRank, 0);
          SmallVector<int64_t> padInts;
          if (binder.s64IntegerArrayAttr(padInts, "pads", defaultPads))
            return rewriter.notifyMatchFailure(binder.op,
                                               "pads binder failure");
          // opset_version 1 uses the attribute name "paddings"
          if (padInts == defaultPads) {
            SmallVector<int64_t> paddingsInts;
            if (binder.s64IntegerArrayAttr(paddingsInts, "paddings",
                                           defaultPads))
              return rewriter.notifyMatchFailure(binder.op,
                                                 "paddings binder failure");
            padInts = paddingsInts;
          }
          for (auto p : padInts)
            padsTensorValue.push_back(rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(p)));
        } else {
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
          if (padsShape[0] != Torch::kUnknownSize) {
            // As per onnx.Pad documentation, padSize = 2*num_data_axes
            // (if axes param not passed). Need to be updated when adding
            // support for `axes` param.
            padsSize = padsShape[0];
          }

          // Extract all the values of 1-D pad tensor and create a list of all
          // these values as torch.pad op expects pad list.
          Value constZero = rewriter.create<Torch::ConstantIntOp>(
              loc, rewriter.getI64IntegerAttr(0));
          SmallVector<int64_t> emptyShape;
          Type padsElemType = Torch::ValueTensorType::get(
              padsTensorType.getContext(), emptyShape,
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
        }

        Value constantValue;
        if (binder.getNumOperands() >= 3 && cstMode) {
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

        if (!constantValue && cstMode) {
          auto dataTensorType = cast<Torch::ValueTensorType>(data.getType());
          if (isa<IntegerType>(dataTensorType.getDtype()))
            constantValue = rewriter.create<Torch::ConstantIntOp>(
                loc, rewriter.getI64IntegerAttr(0));
          // Earlier versions used a FLOAT attribute to store the constant
          // value. The following will pick up on any non-default value attr if
          // provided.
          float constantFloat;
          if (isa<FloatType>(dataTensorType.getDtype()) &&
              !binder.f32FloatAttr(constantFloat, "value", 0.0f))
            constantValue = rewriter.create<Torch::ConstantFloatOp>(
                loc, rewriter.getF64FloatAttr(constantFloat));

          if (!constantValue)
            return rewriter.notifyMatchFailure(
                binder.op, "expected integer or float data tensor");
        }

        // for modes other than "constant" a value is not required
        if (!cstMode)
          constantValue = rewriter.create<Torch::ConstantNoneOp>(loc);

        // The torch.pad op expects a different arrangement of padding pairs for
        // each dimension as compared to the onnx.pad op. Rearrange the pad
        // tensor as shown below:
        //
        // [x1_begin, x2_begin, ..., x1_end, x2_end,...] ->
        // [xn_begin, xn_end, ...., x2_begin, x2_end, x1_begin, x1_end]
        SmallVector<Value> padsRearrange;
        for (uint32_t i = padsSize - 1; i >= padsSize / 2; i--) {
          padsRearrange.emplace_back(padsTensorValue[i - padsSize / 2]);
          padsRearrange.emplace_back(padsTensorValue[i]);
        }

        Value padsSizeList =
            rewriter
                .create<Torch::PrimListConstructOp>(
                    loc,
                    Torch::ListType::get(rewriter.getType<Torch::IntType>()),
                    padsRearrange)
                .getResult();

        // lowering to AtenConstantPadNdOp directly allows passing any torch
        // scalar type for the value, whereas AtenPadOp takes an optional float
        // type.
        if (cstMode && !isa<Torch::NoneType>(constantValue.getType())) {
          rewriter.replaceOpWithNewOp<Torch::AtenConstantPadNdOp>(
              binder.op, resultType, data, padsSizeList, constantValue);
          return success();
        }

        // translate a few mismatching mode names ONNX -> Torch
        mode = (mode == "edge") ? "replicate" : mode;
        mode = (mode == "wrap") ? "circular" : mode;

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
      "Identity", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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

        Type boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value zero = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(0));
        Value one = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        Value lt =
            rewriter.create<Torch::AtenLtScalarOp>(loc, boolTy, indices, zero);
        Value add = rewriter.create<Torch::AtenAddScalarOp>(
            loc, indicesTy, indices, depth, one);
        indices = rewriter.create<Torch::AtenWhereSelfOp>(loc, indicesTy, lt,
                                                          add, indices);

        auto selectTy = rewriter.getType<Torch::ValueTensorType>(
            llvm::SmallVector<int64_t>{1}, valuesTy.getDtype());

        bool valuesAreInt = isa<IntegerType>(valuesTy.getDtype());
        Type valuesETy = valuesAreInt ? intTy : floatTy;

        Value off = rewriter.create<Torch::AtenSelectIntOp>(loc, selectTy,
                                                            values, zero, zero);
        off = rewriter.create<Torch::AtenItemOp>(loc, valuesETy, off);

        Value on = rewriter.create<Torch::AtenSelectIntOp>(loc, selectTy,
                                                           values, zero, one);
        on = rewriter.create<Torch::AtenItemOp>(loc, valuesETy, on);

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
          onehot = rewriter.create<Torch::AtenTransposeIntOp>(loc, resultType,
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

        onehot = rewriter.create<Torch::AtenWhereScalarOp>(loc, resultType,
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

  patterns.onOp(
      "Hardmax", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // onnx.Hardmax can be expanded into the following python code:
        //
        // import torch.nn.functional as F
        // def hardmax(tensor, dim=-1):
        //   maximums = torch.argmax(tensor, dim=dim, keepdim=False)
        //   return F.one_hot(maximums)
        //
        // Given an example input:
        // tensor([[1, 2, 3],
        //         [4, 6, 5],
        //         [9, 8, 7]])
        // Above code yields the following:
        // tensor([[0, 0, 1],
        //         [0, 1, 0],
        //         [1, 0, 0]])

        Torch::ValueTensorType resultType;
        int64_t axisValue;
        Value input, axis;
        if (binder.tensorOperand(input) ||
            binder.s64IntegerAttr(axisValue, "axis") ||
            binder.tensorResultType(resultType))
          return failure();

        auto loc = binder.getLoc();

        std::optional<int64_t> axisIntTorch =
            onnxDtypeIntToTorchDtypeInt(axisValue);
        if (!axisIntTorch.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for the given axis conversion");
        axis = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(axisIntTorch.value()));

        // torch.argmax
        Value constKeepDims = rewriter.create<Torch::ConstantBoolOp>(
            loc, rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        Value argmax = rewriter.create<Torch::AtenArgmaxOp>(
            loc, resultType, input, axis, constKeepDims);

        // one_hot
        Value oneInt = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(1));
        rewriter.replaceOpWithNewOp<Torch::AtenOneHotOp>(binder.op, resultType,
                                                         argmax, oneInt);

        return success();
      });
  patterns.onOp("LpNormalization", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  int64_t axis, p;
                  Value input;
                  if (binder.tensorOperand(input) ||
                      binder.s64IntegerAttr(axis, "axis", -1) ||
                      binder.s64IntegerAttr(p, "p", 2) ||
                      binder.tensorResultType(resultType))
                    return failure();

                  auto loc = binder.getLoc();
                  Value cstAxis = rewriter.create<Torch::ConstantIntOp>(
                      loc, rewriter.getI64IntegerAttr(axis));
                  Value cstP = rewriter.create<Torch::ConstantIntOp>(
                      loc, rewriter.getI64IntegerAttr(p));
                  Value cstKeepDim = rewriter.create<Torch::ConstantBoolOp>(
                      loc, rewriter.getBoolAttr(true));
                  Value axisPrimList =
                      rewriter.create<Torch::PrimListConstructOp>(
                          binder.getLoc(),
                          rewriter.getType<Torch::ListType>(
                              rewriter.getType<Torch::IntType>()),
                          llvm::ArrayRef<Value>{cstAxis});

                  SmallVector<int64_t> normSizes(resultType.getSizes());
                  int64_t rank = normSizes.size();
                  axis = axis % rank;
                  axis = (axis < 0) ? axis + rank : axis;
                  normSizes[axis] = 1;
                  auto normType = rewriter.getType<Torch::ValueTensorType>(
                      normSizes, resultType.getDtype());
                  Value norm = rewriter.create<Torch::AtenNormScalarOptDimOp>(
                      loc, normType, input, cstP, axisPrimList, cstKeepDim);

                  rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
                      binder.op, resultType, input, norm);
                  return success();
                });
  patterns.onOp(
      "MaxUnpool", 9, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // TODO: Add support for `output_shape` arg.
        if (binder.op->getNumOperands() == 3)
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: output_shape arg is not supported");

        Torch::ValueTensorType resultType;
        Value data, indices;
        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorOperandAtIndex(indices, 1) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op, "data/indices/resultType bind failure");
        std::optional<unsigned> maybeRank = Torch::getTensorRank(data);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: unranked tensor");
        int64_t rank = *maybeRank;
        int64_t spatial = rank - 2;

        if (rank <= 3 || rank > 5)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Unimplemented: MaxUnpool support "
                                             "only present for rank 4/5 input");

        if (!(resultType.hasSizes() && resultType.areAllSizesKnown()))
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: expected result to have all shapes "
                         "statically known");

        SmallVector<int64_t> resultShape(resultType.getSizes());
        Value resultShapeList =
            createConstantIntList(binder, rewriter, resultShape);
        if (rank == 4) {
          rewriter.replaceOpWithNewOp<Torch::AtenMaxUnpool2dOp>(
              binder.op, resultType, data, indices, resultShapeList);
          return success();
        }

        SmallVector<int64_t> padding, strides;
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

        if (padding.empty())
          padding.resize(spatial, 0);
        if (strides.empty())
          strides.resize(spatial, 1);

        // If the padding is symmetric we can push the padding
        // operation to the torch operator.
        if (padding.size() == static_cast<size_t>(2 * spatial)) {
          bool equal = true;
          for (int i = 0; i < spatial; ++i) {
            equal = equal && (padding[i] == padding[i + spatial]);
          }
          if (equal)
            padding.resize(spatial);
        }

        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);

        rewriter.replaceOpWithNewOp<Torch::AtenMaxUnpool3dOp>(
            binder.op, resultType, data, indices, resultShapeList, stridesList,
            paddingList);
        return success();
      });
  patterns.onOp(
      "GroupNormalization", 18,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input, scale, bias;
        int64_t numGroups, stashType;
        float epsilon;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.tensorOperandAtIndex(bias, 2) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(numGroups, "num_groups") ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5) ||
            binder.s64IntegerAttr(stashType, "stash_type", 1))
          return failure();

        // Since the support for `stash_type` arg does not exist in
        // the torch op so we just check for the stash_type to be same
        // as the input dtype since that won't require us to do any
        // input type conversion and hence can be supported.
        std::optional<int64_t> stashTypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(stashType);
        if (!stashTypeIntTorch.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for the given stash_type");

        FailureOr<Type> stashDtype = Torch::getTypeForScalarType(
            binder.op->getContext(),
            (torch_upstream::ScalarType)stashTypeIntTorch.value());
        if (failed(stashDtype))
          return failure();
        auto inputDtype =
            cast<Torch::ValueTensorType>(input.getType()).getOptionalDtype();
        if (*stashDtype != inputDtype)
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: stash_type != input dtype");

        Value cstEpsilon = rewriter.create<Torch::ConstantFloatOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)epsilon));
        Value cstNumGroups = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(numGroups));
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<Torch::AtenGroupNormOp>(
            binder.op, resultType, input, cstNumGroups, scale, bias, cstEpsilon,
            /*cudnn_enabled=*/cstFalse);
        return success();
      });
  patterns.onOp(
      "Optional", 15, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::OptionalType resultType;
        Value input;

        if (binder.getNumOperands() == 0)
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for missing input element");

        if (binder.tensorListOperand(input))
          if (binder.tensorOperand(input))
            return failure();

        if (binder.optionalResultType(resultType))
          return failure();

        rewriter.replaceOpWithNewOp<Torch::DerefineOp>(binder.op, resultType,
                                                       input);
        return success();
      });
  patterns.onOp("OptionalGetElement", 15,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ListType tensorListResultType;
                  Torch::ValueTensorType tensorResultType;
                  Value input;

                  if (binder.tensorListResultType(tensorListResultType)) {
                    if (binder.tensorResultType(tensorResultType))
                      return failure();

                    if (binder.optionalTensorOperand(input)) {
                      if (binder.tensorOperand(input))
                        return failure();

                      // It means the input is a tensor.
                      rewriter.replaceOp(binder.op, input);
                      return success();
                    }

                    // It means the input is an optional tensor.
                    rewriter.replaceOpWithNewOp<Torch::PrimUncheckedCastOp>(
                        binder.op, tensorResultType, input);
                    return success();
                  }

                  if (binder.optionalTensorListOperand(input)) {
                    if (binder.tensorListOperand(input))
                      return failure();

                    // It means the input is a tensor list.
                    rewriter.replaceOp(binder.op, input);
                    return success();
                  }

                  // It means the input is an optional tensor list.
                  rewriter.replaceOpWithNewOp<Torch::PrimUncheckedCastOp>(
                      binder.op, tensorListResultType, input);
                  return success();
                });
  patterns.onOp(
      "OptionalHasElement", 15,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failed");

        Value input;
        bool output;
        if (!binder.tensorListOperand(input) || !binder.tensorOperand(input) ||
            !binder.optionalTensorListOperand(input) ||
            !binder.optionalTensorOperand(input))
          output = true;
        else
          output = false;

        Value cstOutput = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr((int64_t)output));
        Value cstDtype = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(),
            rewriter.getI64IntegerAttr((int)torch_upstream::ScalarType::Bool));
        Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
            binder.getLoc(), rewriter.getBoolAttr(false));
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());

        Value dataList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            SmallVector<Value>{cstOutput});

        rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
            binder.op, resultType, dataList, /*dtype=*/cstDtype,
            /*layout=*/cstNone, /*requires_grad=*/cstFalse);
        return success();
      });
  patterns.onOp(
      "NonMaxSuppression", 10,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t centerPointBox;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(centerPointBox, "center_point_box", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        // TODO: Add support for non-zero center_point_box value.
        if (centerPointBox != 0)
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: expected center_point_box "
                         "attribute value to be 0");

        // TODO: Add support for optional arguments to be absent.
        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented: expected all 5 args to be present");

        // Squeeze the boxes and scores tensor.
        // In Onnx, the shape of boxes is [BxNx4] while the
        // torchvision expects it to be of shape [Nx4]. Similarly, for
        // the scores tensor shape in Onnx is [BxCxN] while the
        // torchvision expects it to be of shape [N].
        Value boxes = operands[0], scores = operands[1];
        FailureOr<Value> squeezedBoxes = Torch::squeezeTensor(
            rewriter, binder.op, binder.getLoc(), 0, boxes);
        if (failed(squeezedBoxes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze boxes tensor");

        FailureOr<Value> squeezedScores = Torch::squeezeTensor(
            rewriter, binder.op, binder.getLoc(), 0, scores);
        if (failed(squeezedScores))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze scores tensor");
        squeezedScores = Torch::squeezeTensor(
            rewriter, binder.op, binder.getLoc(), 0, squeezedScores.value());
        if (failed(squeezedScores))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze scores tensor");

        boxes = squeezedBoxes.value();
        scores = squeezedScores.value();

        // TODO: Add support for handling score_threshold arg.
        // If score_threshold > min(scores) then the op can't be lowered since
        // the torchvision::nms op doesn't have support for handling the
        // score_threshold arg.
        Value scoreThreshold = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(), operands[4]);
        Value minScores = rewriter.create<Torch::AtenMinOp>(
            binder.getLoc(),
            Torch::ValueTensorType::get(binder.op->getContext(), {},
                                        rewriter.getF32Type()),
            scores);
        minScores = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(), minScores);

        Value scoresCond = rewriter.create<Torch::AtenGeFloatOp>(
            binder.getLoc(), minScores, scoreThreshold);
        rewriter.create<Torch::RuntimeAssertOp>(
            binder.getLoc(), scoresCond,
            rewriter.getStringAttr(
                "unimplemented: score_threshold should be <= min(scores)"));

        Value iouThreshold = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::FloatType>(), operands[3]);
        Value result = rewriter.create<Torch::TorchvisionNmsOp>(
            binder.getLoc(), resultType, boxes, scores, iouThreshold);

        // The result generated by torchvision.nms op is of shape [n], while the
        // onnx expects it to be of shape [n, 3]. Hence, we unsqueeze the tensor
        // and make it of shape [n, 1] and then concatenate it with a zero
        // tensor of shape [n, 2] to make it of shape [n, 3].
        Value dim = rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(1));
        FailureOr<Value> unsqueezedResult =
            Torch::unsqueezeTensor(rewriter, binder.op, result, dim);
        if (failed(unsqueezedResult))
          return rewriter.notifyMatchFailure(
              binder.op, "failed to  unsqueeze result tensor");
        result = unsqueezedResult.value();

        Value numOutputBoxes = rewriter.create<Torch::AtenSizeIntOp>(
            binder.getLoc(), result,
            rewriter.create<Torch::ConstantIntOp>(
                binder.getLoc(), rewriter.getI64IntegerAttr(0)));
        SmallVector<Value> zerosShapeValues{numOutputBoxes};
        zerosShapeValues.push_back(rewriter.create<Torch::ConstantIntOp>(
            binder.getLoc(), rewriter.getI64IntegerAttr(2)));
        Value zerosShapeList = rewriter.create<Torch::PrimListConstructOp>(
            binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            zerosShapeValues);

        std::optional<ArrayRef<int64_t>> resultShape =
            cast<Torch::ValueTensorType>(result.getType()).getOptionalSizes();
        if (!resultShape.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "expected result tensor to have shape");
        llvm::SmallVector<int64_t> zerosShape = {resultShape->front(), 2};
        auto zerosTy = Torch::ValueTensorType::get(
            resultType.getContext(), zerosShape, resultType.getOptionalDtype());
        Value cstNone = rewriter.create<Torch::ConstantNoneOp>(binder.getLoc());
        Value zeros = rewriter.create<Torch::AtenZerosOp>(
            binder.getLoc(), zerosTy, zerosShapeList, cstNone, cstNone, cstNone,
            cstNone);

        Type listElemType =
            cast<Torch::BaseTensorType>(resultType)
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = rewriter.create<Torch::PrimListConstructOp>(
            binder.op->getLoc(), listType, SmallVector<Value>{result, zeros});

        // TODO: Add support for handling max_output_boxes_per_class arg.
        // If numOutputBoxes (N) > max_output_boxes_per_class then the op can't
        // be lowered since the torchvision::nms op doesn't have support for
        // handling the max_output_boxes_per_class arg. Also, we have already
        // constrained the number of classes to be 1 above, so the number of
        // output boxes inferred from the result is num_output_boxes_per_class.
        Value maxOutputBoxesPerClass = rewriter.create<Torch::AtenItemOp>(
            binder.getLoc(), rewriter.getType<Torch::IntType>(), operands[2]);
        Value boxesCond = rewriter.create<Torch::AtenLeIntOp>(
            binder.getLoc(), numOutputBoxes, maxOutputBoxesPerClass);
        rewriter.create<Torch::RuntimeAssertOp>(
            binder.getLoc(), boxesCond,
            rewriter.getStringAttr(
                "unimplemented: number of output boxes per class should be "
                "<= max_output_boxes_per_class"));

        rewriter.replaceOpWithNewOp<Torch::AtenCatOp>(binder.op, resultType,
                                                      tensorList, dim);
        return success();
      });
}
