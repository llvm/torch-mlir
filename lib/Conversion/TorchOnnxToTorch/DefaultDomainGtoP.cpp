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
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));
        Value constBeta = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));

        // Expression: alpha * x + beta
        Value alphaMulX = Torch::AtenMulScalarOp::create(
            rewriter, binder.getLoc(), resultType, tensorOperand, constAlpha);
        Value constOne = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(1.0));
        Value alphaMulXPlusBeta = Torch::AtenAddScalarOp::create(
            rewriter, binder.getLoc(), resultType, alphaMulX, constBeta,
            /*alpha=*/constOne);

        // Expression: min(1, alpha * x + beta)
        Value oneTensor =
            createRank0Tensor(rewriter, binder.getLoc(), resultType, constOne);
        Value minExpression =
            Torch::AtenMinimumOp::create(rewriter, binder.getLoc(), resultType,
                                         oneTensor, alphaMulXPlusBeta);

        // Expression: max(0, min(1, alpha * x + beta))
        Value constZero = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getF64FloatAttr(0.0));
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

        Value vApproximate = Torch::ConstantStrOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::StringType>(),
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

        Value interpolationMode = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), iModeInt));

        Value paddingMode = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

        bool alignMode = align;
        Value alignCorners = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::BoolType>(),
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
        if (!conditionType || conditionType.getSizes().size() > 1)
          return rewriter.notifyMatchFailure(
              binder.op, "condition must have one single element per "
                         "https://onnx.ai/onnx/operators/onnx__If.html");
        auto conditionInt = Torch::AtenItemOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            conditionTensor);
        auto conditionBool = Torch::AtenBoolIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            conditionInt);

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

        auto primIfOp = Torch::PrimIfOp::create(
            rewriter, binder.getLoc(), TypeRange(resultTypes), conditionBool);

        auto inlineIfCase = [&](Region &srcRegion, Region &dstRegion) {
          rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
        };
        inlineIfCase(*thenRegion, primIfOp.getThenRegion());
        inlineIfCase(*elseRegion, primIfOp.getElseRegion());

        auto replaceTerminator = [&](Region &region) -> LogicalResult {
          PatternRewriter::InsertionGuard guard(rewriter);
          Operation *terminator = region.front().getTerminator();
          rewriter.setInsertionPoint(terminator);

          // cast result shape if there is static/dynamic difference
          llvm::SmallVector<Value> terOperands = terminator->getOperands();
          if (terOperands.size() != resultTypes.size())
            return failure();
          for (size_t i = 0; i < terOperands.size(); i++) {
            mlir::Type terType = terOperands[i].getType();
            int64_t terOpRank =
                dyn_cast<Torch::ValueTensorType>(terType).getSizes().size();
            int64_t resRank = dyn_cast<Torch::ValueTensorType>(resultTypes[i])
                                  .getSizes()
                                  .size();
            if (terOpRank != resRank)
              return failure();
            if (terType != resultTypes[i]) {
              Value cast = Torch::TensorStaticInfoCastOp::create(
                  rewriter, binder.getLoc(), resultTypes[i], terOperands[i]);
              terOperands[i] = cast;
            }
          }

          rewriter.replaceOpWithNewOp<Torch::PrimIfYieldOp>(terminator,
                                                            terOperands);
          return success();
        };
        if (failed(replaceTerminator(primIfOp.getThenRegion())) ||
            failed(replaceTerminator(primIfOp.getElseRegion())))
          return rewriter.notifyMatchFailure(binder.op,
                                             "terminator replace failure");

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
        auto maxTripCountInt = Torch::AtenItemOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            maxTripCountTensor);

        // Condition - tensor bool scalar (or empty)
        Value conditionTensor = operands[1];
        auto conditionInt = Torch::AtenItemOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            conditionTensor);
        auto conditionBool = Torch::AtenBoolIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::BoolType>(),
            conditionInt);
        // To be used for "for like" loop case
        auto constBoolTrue = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getBoolAttr(true));

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
        auto loop = Torch::PrimLoopOp::create(
            b, TypeRange(operandTypeVec), maxTripCountInt, loopInitCondition,
            ValueRange(operands));

        rewriter.cloneRegionBefore(*loopBodyIn, loop.getRegion(),
                                   loop.getRegion().begin());

        // primLoopOp loopBody expects torch.int as first arg
        // insert torch.int arg in loop body, convert to tensor,
        // replace all uses of old arg, delete old arg.
        auto loopVar = loop.getRegion().front().getArgument(0);
        // insert new Arg
        loop.getRegion().front().insertArgument(
            0U, rewriter.getType<Torch::IntType>(), binder.getLoc());
        auto newLoopVarArg = loop.getRegion().front().getArgument(0);

        // convert int arg to tensor of original Type
        rewriter.setInsertionPointToStart(&loop.getRegion().front());
        auto loopVarType = dyn_cast<Torch::BaseTensorType>(loopVar.getType());
        if (!loopVarType || !loopVarType.areAllSizesKnown())
          return rewriter.notifyMatchFailure(
              loopVar.getLoc(),
              "loop iteration value must be a tensor with known sizes");
        Value sizes = Torch::toIntListConstruct(rewriter, loopVar.getLoc(),
                                                loopVarType.getSizes());
        auto newTensor = torch::Torch::createInitTensor(
            rewriter, loopVar.getLoc(), loopVarType, newLoopVarArg, sizes);

        loopVar.replaceAllUsesWith(newTensor);
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
          auto terminatorCondInt = Torch::AtenItemOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              terminatorCondTensor);
          auto terminatorCondBool = Torch::AtenBoolIntOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::BoolType>(),
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
        Value axisConst = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        Value none = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
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

        Value axisConst = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(axis));
        Value none = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        Value cstEnd = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(rank - 1));

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
          rightDimConsts.push_back(Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(n)));
          if (n == Torch::kUnknownSize) {
            prodRightSizes = -1;
            break;
          }
          prodRightSizes *= n;
        }
        leftDims.push_back(prodRightSizes);
        // the following list will be used to unflatten the right side
        Value rightDimsPrimList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            rewriter.getType<Torch::ListType>(
                rewriter.getType<Torch::IntType>()),
            rightDimConsts);
        auto flatRightTy = rewriter.getType<Torch::ValueTensorType>(
            leftDims, inputTy.getOptionalDtype());
        // flatten input
        Value inputFlatRight = Torch::AtenFlattenUsingIntsOp::create(
            rewriter, binder.getLoc(), flatRightTy, input, axisConst, cstEnd);
        // compute lsm over flattened index
        Value outputFlatRight = Torch::AtenLogSoftmaxIntOp::create(
            rewriter, binder.getLoc(), flatRightTy, inputFlatRight, axisConst,
            none);
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
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        Value lhs, rhs, lhsZp, rhsZp;
        if (binder.tensorOperandAtIndex(lhs, 0) ||
            binder.tensorOperandAtIndex(rhs, 1) ||
            binder.tensorResultType(resultType))
          return failure();

        if (binder.tensorOperandAtIndex(lhsZp, 2)) {
          lhsZp = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        }

        if (binder.tensorOperandAtIndex(rhsZp, 3)) {
          rhsZp = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        }

        // This op is lowered as follows:
        // lhs = lhs.to(dtype=torch.int32)
        // rhs = rhs.to(dtype=torch.int32)
        // lhs = lhs - lhsZp
        // rhs = rhs - rhsZp
        // res = torch.mm(lhs, rhs)

        // Converting lhs and rhs tensor to `si32` type.
        lhs = Torch::convertTensorToDtype(
            rewriter, loc, lhs,
            mlir::IntegerType::get(binder.op->getContext(), 32,
                                   mlir::IntegerType::Signed));
        rhs = Torch::convertTensorToDtype(
            rewriter, loc, rhs,
            mlir::IntegerType::get(binder.op->getContext(), 32,
                                   mlir::IntegerType::Signed));

        // Subtracting the zero_point values from lhs and rhs.
        Value alpha = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        if (auto lhsZpTy = dyn_cast<Torch::ValueTensorType>(lhsZp.getType()))
          lhs = Torch::AtenSubTensorOp::create(rewriter, loc, lhs.getType(),
                                               lhs, lhsZp, alpha);
        else
          lhs = Torch::AtenSubScalarOp::create(rewriter, loc, lhs.getType(),
                                               lhs, lhsZp, alpha);

        if (auto rhsZpTy = dyn_cast<Torch::ValueTensorType>(rhsZp.getType()))
          rhs = Torch::AtenSubTensorOp::create(rewriter, loc, rhs.getType(),
                                               rhs, rhsZp, alpha);
        else
          rhs = Torch::AtenSubScalarOp::create(rewriter, loc, rhs.getType(),
                                               rhs, rhsZp, alpha);

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
      "MelWeightMatrix", 17,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        llvm::SmallVector<Value> operands;
        Torch::ValueTensorType resultType;
        int64_t output_dtype_attr;
        if (binder.tensorOperands(operands, 5) ||
            binder.tensorResultType(resultType) || operands.size() != 5 ||
            binder.s64IntegerAttr(output_dtype_attr, "output_datatype", 1)) {
          return failure();
        }
        // operands sequence :
        // num_mel_bins, dft_length, sample_rate -> int32/64 tensors
        // lower_edge_hertz, upper_edge_hertz -> f16/32/64

        // Need to backtrack the values of num_mel_bins and dft_length//2+1 from
        // result shape since the inputs are tensors and we cannot know their
        // values at compile time. if the result type does not contain static
        // shapes, then the implementation will be unsupported.
        if (!resultType.areAllSizesKnown())
          return rewriter.notifyMatchFailure(
              binder.op, "Unknown result sizes, not supported.");

        ArrayRef<int64_t> resShape = resultType.getSizes();
        if (resShape.size() != 2)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected result rank to be 2, not supported for other ranks.");

        std::optional<int64_t> torchDTypeInt =
            onnxDtypeIntToTorchDtypeInt(output_dtype_attr);
        if (!torchDTypeInt.has_value()) {
          return rewriter.notifyMatchFailure(
              binder.op, "conversion to given output dtype unsupported");
        }

        // Here Onwards all shapes will be computed using these sizes
        int64_t numSpectrogramBinsInt = resShape[0];
        int64_t numMelBinsInt = resShape[1];
        Torch::ValueTensorType inputIntType = binder.toValidTensorType(
            operands[0].getType()); // Since operands[0 / 1 / 2] will have the
                                    // same int type.
        Torch::ValueTensorType inputFloatType = binder.toValidTensorType(
            operands[3].getType()); // Since operands[3 / 4] will have the same
                                    // float type

        Value numMelBinsItem =
            getItemOp<Torch::IntType>(binder, rewriter, operands[0]);
        Value sampleRateItem =
            getItemOp<Torch::IntType>(binder, rewriter, operands[2]);
        Value lowerEdgeHzItem =
            getItemOp<Torch::FloatType>(binder, rewriter, operands[3]);
        Value upperEdgeHzItem =
            getItemOp<Torch::FloatType>(binder, rewriter, operands[4]);

        // Helpers
        ImplicitLocOpBuilder b(binder.getLoc(), rewriter);
        auto ctx = binder.op->getContext();

        // Recurring shapes
        SmallVector<int64_t> unranked({});
        SmallVector<int64_t> shapeNMB({numMelBinsInt});
        SmallVector<int64_t> shape1xNMB({1, numMelBinsInt});
        SmallVector<int64_t> shapeNSB({numSpectrogramBinsInt});
        SmallVector<int64_t> shapeNSBx1({numSpectrogramBinsInt, 1});
        SmallVector<int64_t> shapeNSBxNMB(
            {numSpectrogramBinsInt, numMelBinsInt});

        // Recurring DTypes
        Type inpFpDType = inputFloatType.getDtype();
        Type inpIntDType = inputIntType.getDtype();
        Type si32Ty = rewriter.getIntegerType(32, true);
        Type f32Ty = rewriter.getF32Type();
        Type i1Ty = rewriter.getI1Type();

        // Value constants
        Value noneConst = Torch::ConstantNoneOp::create(b);
        Value zeroConst =
            Torch::ConstantIntOp::create(b, rewriter.getI64IntegerAttr(0));
        Value oneConst =
            Torch::ConstantIntOp::create(b, rewriter.getI64IntegerAttr(1));
        Value twoConst =
            Torch::ConstantIntOp::create(b, rewriter.getI64IntegerAttr(2));
        Value int32DTypeConst =
            Torch::ConstantIntOp::create(b, rewriter.getI64IntegerAttr(3));
        Value float32DTypeConst =
            Torch::ConstantIntOp::create(b, rewriter.getI64IntegerAttr(6));

        Torch::ValueTensorType dftLenType =
            Torch::ValueTensorType::get(ctx, unranked, inpIntDType);
        Type freqBinsIntType =
            Torch::ValueTensorType::get(ctx, shapeNMB, si32Ty);
        Type freqBinsFltType =
            Torch::ValueTensorType::get(ctx, shapeNMB, f32Ty);

        Value dftLengthDivTwoTensor = Torch::AtenFloorDivideScalarOp::create(
            b, dftLenType, operands[1], twoConst);
        Value numSpectrogramBinsTensor =
            Torch::AtenAddScalarOp::create(b, dftLenType, dftLengthDivTwoTensor,
                                           oneConst, /*alpha =*/oneConst);
        Value numSpectrogramBinsItem = getItemOp<Torch::IntType>(
            binder, rewriter, numSpectrogramBinsTensor);

        // From Ref Impl of Onnx.MelWeightMatrix:
        // https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_mel_weight_matrix.py#L25-L32
        // convert input Freq Hz to Mel
        Value twoFiveNineFiveConst =
            Torch::ConstantFloatOp::create(b, rewriter.getF64FloatAttr(2595));
        Value sevenHConst =
            Torch::ConstantFloatOp::create(b, rewriter.getF64FloatAttr(700));
        Value tenConst =
            Torch::ConstantFloatOp::create(b, rewriter.getF64FloatAttr(10));
        Value oneFltConst =
            Torch::ConstantFloatOp::create(b, rewriter.getF64FloatAttr(1));
        Value LnToLog10Const = Torch::ConstantFloatOp::create(
            b, rewriter.getF64FloatAttr(M_LOG10E));

        Value lfDiv7Hfloat =
            Torch::AtenDivFloatOp::create(b, lowerEdgeHzItem, sevenHConst);
        Type freqType = Torch::ValueTensorType::get(ctx, unranked, inpFpDType);
        Value lfDiv7H =
            Torch::PrimNumToTensorScalarOp::create(b, freqType, lfDiv7Hfloat);
        Value lfDiv7HAdd1 = Torch::AtenAddScalarOp::create(
            b, freqType, lfDiv7H, oneConst, /*alpha =*/oneConst);
        Value lfDiv7HAdd1Ln =
            Torch::AtenLogOp::create(b, freqType, lfDiv7HAdd1);
        Value lfDiv7HAdd1Log10 = Torch::AtenMulScalarOp::create(
            b, freqType, lfDiv7HAdd1Ln, LnToLog10Const);

        Value lfMel = Torch::AtenMulScalarOp::create(
            b, freqType, lfDiv7HAdd1Log10, twoFiveNineFiveConst);

        Value hfDiv7Hfloat =
            Torch::AtenDivFloatOp::create(b, upperEdgeHzItem, sevenHConst);
        Value hfDiv7H =
            Torch::PrimNumToTensorScalarOp::create(b, freqType, hfDiv7Hfloat);
        Value hfDiv7HAdd1 = Torch::AtenAddScalarOp::create(
            b, freqType, hfDiv7H, oneConst, /*alpha =*/oneConst);
        Value hfDiv7HAdd1Ln =
            Torch::AtenLogOp::create(b, freqType, hfDiv7HAdd1);
        Value hfDiv7HAdd1Log10 = Torch::AtenMulScalarOp::create(
            b, freqType, hfDiv7HAdd1Ln, LnToLog10Const);

        Value hfMel = Torch::AtenMulScalarOp::create(
            b, freqType, hfDiv7HAdd1Log10, twoFiveNineFiveConst);

        Value hfSubLf = Torch::AtenSubTensorOp::create(
            b, hfMel.getType(), hfMel, lfMel, /*alpha=*/oneConst);
        Value numMelBinsPlus2 =
            Torch::AtenAddIntOp::create(b, numMelBinsItem, twoConst);
        Value melStep = Torch::AtenDivScalarOp::create(
            b, hfSubLf.getType(), hfSubLf, numMelBinsPlus2);

        Value lowBinsInit = Torch::AtenArangeOp::create(
            b, freqBinsIntType, numMelBinsItem, /*dtype=*/int32DTypeConst,
            /*layout=*/noneConst, /*device=*/noneConst,
            /*pin_memory=*/noneConst);

        Value centerBinsInit = Torch::AtenArangeOp::create(
            b, freqBinsIntType, numMelBinsItem, /*dtype=*/int32DTypeConst,
            /*layout=*/noneConst, /*device=*/noneConst,
            /*pin_memory=*/noneConst);

        Value highBinsInit = Torch::AtenArangeOp::create(
            b, freqBinsIntType, numMelBinsItem, /*dtype=*/int32DTypeConst,
            /*layout=*/noneConst, /*device=*/noneConst,
            /*pin_memory=*/noneConst);

        // Common values used in conversion
        Value dftLenPlusOne = Torch::AtenAddScalarOp::create(
            b, dftLenType, operands[1], oneConst, /*alpha=*/oneConst);
        Value dftLenPlusOneItem =
            getItemOp<Torch::IntType>(binder, rewriter, dftLenPlusOne);
        Value falseConst = Torch::ConstantBoolOp::create(b, false);
        Torch::ValueTensorType unsqueezeBinsResType =
            Torch::ValueTensorType::get(ctx, shape1xNMB, si32Ty);

        // Low bins Mel to hz
        Value lowBinsMulMelStep = Torch::AtenMulTensorOp::create(
            b, freqBinsFltType, lowBinsInit, melStep);
        Value lowBinsScaled = Torch::AtenAddTensorOp::create(
            b, freqBinsFltType, lowBinsMulMelStep, lfMel, /*alpha=*/oneConst);
        Value lbDiv = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, lowBinsScaled, twoFiveNineFiveConst);
        Value lbClone = Torch::AtenCloneOp::create(
            b, freqBinsFltType, lowBinsScaled, /*memory_format=*/noneConst);
        Value lbTenTensor = Torch::AtenFillScalarOp::create(b, freqBinsFltType,
                                                            lbClone, tenConst);
        Value lbPow = Torch::AtenPowTensorTensorOp::create(b, freqBinsFltType,
                                                           lbTenTensor, lbDiv);
        Value lbPowSubOne = Torch::AtenSubScalarOp::create(
            b, freqBinsFltType, lbPow, oneConst, /*alpha=*/oneConst);
        Value lowBinsHz = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, lbPowSubOne, sevenHConst);
        // Normalize freqBinsHz
        Value lbMulDft = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, lowBinsHz, dftLenPlusOneItem);
        Value lowBinsNormalized = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, lbMulDft, sampleRateItem);
        // cast to int32
        Value lowBinsInt = Torch::AtenToDtypeOp::create(
            b, freqBinsIntType, lowBinsNormalized, /*dtype=*/int32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);
        Value lowBins = Torch::AtenUnsqueezeOp::create(
            b, unsqueezeBinsResType, lowBinsInt, /*dim=*/zeroConst);

        // Center bins mel to hz
        Value centerBinsInitInc = Torch::AtenAddScalarOp::create(
            b, freqBinsIntType, centerBinsInit, oneConst, /*alpha=*/oneConst);
        Value centerBinsMulMelStep = Torch::AtenMulTensorOp::create(
            b, freqBinsFltType, centerBinsInitInc, melStep);
        Value centerBinsScaled = Torch::AtenAddTensorOp::create(
            b, freqBinsFltType, centerBinsMulMelStep, lfMel,
            /*alpha=*/oneConst);
        Value cbDiv = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, centerBinsScaled, twoFiveNineFiveConst);
        Value cbClone = Torch::AtenCloneOp::create(
            b, freqBinsFltType, centerBinsScaled, /*memory_format=*/noneConst);
        Value cbTenTensor = Torch::AtenFillScalarOp::create(b, freqBinsFltType,
                                                            cbClone, tenConst);
        Value cbPow = Torch::AtenPowTensorTensorOp::create(b, freqBinsFltType,
                                                           cbTenTensor, cbDiv);
        Value cbPowSubOne = Torch::AtenSubScalarOp::create(
            b, freqBinsFltType, cbPow, oneConst, /*alpha=*/oneConst);
        Value centerBinsHz = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, cbPowSubOne, sevenHConst);
        // Normalize freqBinsHz
        Value cbMulDft = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, centerBinsHz, dftLenPlusOneItem);
        Value centerBinsNormalized = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, cbMulDft, sampleRateItem);
        // cast to int32
        Value centerBinsInt = Torch::AtenToDtypeOp::create(
            b, freqBinsIntType, centerBinsNormalized, /*dtype=*/int32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);
        Value centerBins = Torch::AtenUnsqueezeOp::create(
            b, unsqueezeBinsResType, centerBinsInt, /*dim=*/zeroConst);

        // High bins mel to hz
        Value highBinsInitInc = Torch::AtenAddScalarOp::create(
            b, freqBinsIntType, highBinsInit, twoConst, /*alpha=*/oneConst);
        Value highBinsMulMelStep = Torch::AtenMulTensorOp::create(
            b, freqBinsFltType, highBinsInitInc, melStep);
        Value highBinsScaled = Torch::AtenAddTensorOp::create(
            b, freqBinsFltType, highBinsMulMelStep, lfMel, /*alpha=*/oneConst);
        Value hbDiv = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, highBinsScaled, twoFiveNineFiveConst);
        Value hbClone = Torch::AtenCloneOp::create(
            b, freqBinsFltType, highBinsScaled, /*memory_format=*/noneConst);
        Value hbTenTensor = Torch::AtenFillScalarOp::create(b, freqBinsFltType,
                                                            hbClone, tenConst);
        Value hbPow = Torch::AtenPowTensorTensorOp::create(b, freqBinsFltType,
                                                           hbTenTensor, hbDiv);
        Value hbPowSubOne = Torch::AtenSubScalarOp::create(
            b, freqBinsFltType, hbPow, oneConst, /*alpha=*/oneConst);
        Value highBinsHz = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, hbPowSubOne, sevenHConst);
        // Normalize freqBinsHz
        Value hbMulDft = Torch::AtenMulScalarOp::create(
            b, freqBinsFltType, highBinsHz, dftLenPlusOneItem);
        Value highBinsNormalized = Torch::AtenDivScalarOp::create(
            b, freqBinsFltType, hbMulDft, sampleRateItem);
        // cast to int32
        Value highBinsInt = Torch::AtenToDtypeOp::create(
            b, freqBinsIntType, highBinsNormalized, /*dtype=*/int32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);
        Value highBins = Torch::AtenUnsqueezeOp::create(
            b, unsqueezeBinsResType, highBinsInt, /*dim=*/zeroConst);

        Type iotaInitType = inputIntType.getWithSizesAndDtype(shapeNSB, si32Ty);
        Value iotaInit = Torch::AtenArangeOp::create(
            b, iotaInitType, numSpectrogramBinsItem,
            /*dtype=*/int32DTypeConst,
            /*layout=*/noneConst, /*device=*/noneConst,
            /*pin_memory=*/noneConst);

        Torch::ValueTensorType unsqueezeIotaResType =
            Torch::ValueTensorType::get(ctx, shapeNSBx1, si32Ty);
        Value iota = Torch::AtenUnsqueezeOp::create(b, unsqueezeIotaResType,
                                                    iotaInit, /*dim=*/oneConst);

        Value lowToCenter = Torch::AtenSubTensorOp::create(
            b, unsqueezeBinsResType, centerBins, lowBins, /*alpha=*/oneConst);
        Value centerToHigh = Torch::AtenSubTensorOp::create(
            b, unsqueezeBinsResType, highBins, centerBins, /*alpha=*/oneConst);

        Value oneConstTensor = Torch::createRank0Tensor(
            rewriter, binder.getLoc(),
            Torch::ValueTensorType::get(ctx, std::nullopt, f32Ty), oneConst);

        Type scaledType = inputIntType.getWithSizesAndDtype(shape1xNMB, f32Ty);
        Value upscaleInit = Torch::AtenMaximumOp::create(
            b, unsqueezeBinsResType, oneConstTensor, lowToCenter);
        Value upscale = Torch::AtenToDtypeOp::create(
            b, scaledType, upscaleInit, /*dtype=*/float32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);

        Value downscaleInit = Torch::AtenMaximumOp::create(
            b, unsqueezeBinsResType, oneConstTensor, centerToHigh);
        Value downscale = Torch::AtenToDtypeOp::create(
            b, scaledType, downscaleInit, /*dtype=*/float32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);

        Torch::ValueTensorType binsDiffType =
            Torch::ValueTensorType::get(ctx, shapeNSBxNMB, si32Ty);
        Torch::ValueTensorType diffFloatType =
            Torch::ValueTensorType::get(ctx, shapeNSBxNMB, f32Ty);

        Value iotaSubLBInt = Torch::AtenSubTensorOp::create(
            b, binsDiffType, iota, lowBins, /*alpha=*/oneConst);
        Value iotaSubLB = Torch::AtenToDtypeOp::create(
            b, diffFloatType, iotaSubLBInt, /*dtype=*/float32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);
        Value rampUp = Torch::AtenDivTensorOp::create(b, diffFloatType,
                                                      iotaSubLB, upscale);

        Value hbSubIotaInt = Torch::AtenSubTensorOp::create(
            b, binsDiffType, highBins, iota, /*alpha=*/oneConst);
        Value hbSubIota = Torch::AtenToDtypeOp::create(
            b, diffFloatType, hbSubIotaInt, /*dtype=*/float32DTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);
        Value rampDown = Torch::AtenDivTensorOp::create(b, diffFloatType,
                                                        hbSubIota, downscale);

        // ramp values
        Type iotaCmpBinsType =
            inputIntType.getWithSizesAndDtype(shapeNSBxNMB, i1Ty);

        // Iota Cmp Bins
        Value iotaGtEqCBins =
            Torch::AtenGeTensorOp::create(b, iotaCmpBinsType, iota, centerBins);
        Value iotaEqCBins =
            Torch::AtenEqTensorOp::create(b, iotaCmpBinsType, iota, centerBins);
        Value iotaLtLBins =
            Torch::AtenLtTensorOp::create(b, iotaCmpBinsType, iota, lowBins);
        Value iotaGtLBins =
            Torch::AtenGtTensorOp::create(b, iotaCmpBinsType, iota, highBins);

        // Create output freq ramps Low-Center-High
        Type rampInitType =
            inputIntType.getWithSizesAndDtype(shapeNSBxNMB, f32Ty);
        Value rampInit = Torch::AtenWhereSelfOp::create(
            b, rampInitType, iotaGtEqCBins, rampDown, rampUp);
        Value rampInitLt = Torch::AtenWhereScalarSelfOp::create(
            b, rampInitType, iotaLtLBins, zeroConst, rampInit);
        Value rampInitLtGt = Torch::AtenWhereScalarSelfOp::create(
            b, rampInitType, iotaGtLBins, zeroConst, rampInitLt);

        Type C2HCmpBinsType =
            inputIntType.getWithSizesAndDtype(shape1xNMB, i1Ty);
        Value C2HEqZero = Torch::AtenEqScalarOp::create(
            b, C2HCmpBinsType, centerToHigh, zeroConst);
        Value cornerCases = Torch::AtenLogicalAndOp::create(
            b, iotaCmpBinsType, iotaEqCBins, C2HEqZero);
        Value rampOutput = Torch::AtenWhereScalarSelfOp::create(
            b, rampInitType, cornerCases, oneFltConst, rampInitLtGt);

        Value outputDTypeConst = Torch::ConstantIntOp::create(
            b, rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(torchDTypeInt.value()));
        Value finalOutput = Torch::AtenToDtypeOp::create(
            b, resultType, rampOutput, /*dtype=*/outputDTypeConst,
            /*non_blocking=*/falseConst, /*copy=*/falseConst,
            /*memory_format=*/noneConst);

        rewriter.replaceOp(binder.op, finalOutput);
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

        Value torchDtypeIntValue = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr(torchDtype.value()));
        Value numSamples = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(sampleSize));

        // PRG is seeded globally by default
        Value none = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        // Sample with replacement by default (no onnx equivalent in arguments)
        Value cstTrue = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getBoolAttr(true));

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
        Value multinomialTensor = Torch::AtenMultinomialOp::create(
            rewriter, binder.getLoc(), multinomialOutputType, self, numSamples,
            cstTrue, none);

        Value cstFalse = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getBoolAttr(false));
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
          weight = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        }

        ignore_index = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr(ignore_index_int));

        // convert string reduction attr to standardized integer enum value
        int reduction_value =
            torch_upstream::get_loss_reduction_enum(reduction_str);
        reduction = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr(reduction_value));

        Value nllLoss = Torch::AtenNllLossForwardOp::create(
                            rewriter, binder.getLoc(), resultType, resultType,
                            self, target, weight, reduction, ignore_index)
                            ->getResult(0);

        rewriter.replaceOp(binder.op, nllLoss);
        return success();
      });
  patterns.onOp(
      "NonZero", 13, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        Value zero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        Value one = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        auto rawSize = resultType.getSizes();
        SmallVector<int64_t> torchResultSize(rawSize.rbegin(), rawSize.rend());
        auto torchResultType = rewriter.getType<Torch::ValueTensorType>(
            torchResultSize, resultType.getDtype());
        auto nonZero = Torch::AtenNonzeroOp::create(rewriter, binder.getLoc(),
                                                    torchResultType, operand);
        // The output tensor has a shape of ((n, z)), where (n) is the
        // number of dimensions in the input tensor and (z) is the
        // number of non-zero elements2. This is different from
        // PyTorch's default behavior, where the dimensions are
        // reversed.
        rewriter.replaceOpWithNewOp<Torch::AtenTransposeIntOp>(
            binder.op, resultType, nonZero, zero, one);
        return success();
      });
  patterns.onOp(
      "MaxPool", 12, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        std::string autoPad;
        if (binder.customOpNameStringAttr(autoPad, "auto_pad", "NOTSET"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "auto_pad bind failure");

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

        // set default padding
        if (padding.empty())
          padding.resize(spatial, 0);
        if (strides.empty())
          strides.resize(spatial, 1);
        if (dilations.empty())
          dilations.resize(spatial, 1);

        auto inputTensorType = cast<Torch::ValueTensorType>(operand.getType());

        // Padding for the beginning and ending along each spatial axis, it can
        // take any value greater than or equal to 0. The value represent the
        // number of pixels added to the beginning and end part of the
        // corresponding axis. pads format should be as follow [x1_begin,
        // x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added
        // at the beginning of axis i and xi_end, the number of pixels added at
        // the end of axis i.
        if (autoPad != "NOTSET" && autoPad != "VALID") {
          const bool isSameLower = autoPad == "SAME_LOWER";
          ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
          padding.resize_for_overwrite(2 * spatial);
          for (unsigned dimIdx = 0; dimIdx < spatial; dimIdx++) {
            const int64_t dilatedKernelSize =
                dilations[dimIdx] * (kernel[dimIdx] - 1) + 1;
            int64_t totalPad = ((inputShape[dimIdx + 2] + strides[dimIdx] - 1) /
                                    strides[dimIdx] -
                                1) *
                                   strides[dimIdx] +
                               dilatedKernelSize - inputShape[dimIdx + 2];
            totalPad = totalPad >= 0 ? totalPad : 0;
            padding[dimIdx] =
                isSameLower ? ((totalPad + 1) / 2) : (totalPad / 2);
            padding[spatial + dimIdx] = totalPad - padding[dimIdx];
          }
        }

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
            zero = Torch::ConstantFloatOp::create(
                rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(
                    std::numeric_limits<double>::lowest()));
          } else if (isa<IntegerType>(resultTypeOut.getDtype())) {
            zero = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(
                    std::numeric_limits<int64_t>::lowest()));
          }

          auto paddedInputTy = rewriter.getType<Torch::ValueTensorType>(
              paddedShape, operandTy.getDtype());
          operand = Torch::AtenConstantPadNdOp::create(
              rewriter, binder.getLoc(), paddedInputTy, operand,
              shuffledPaddingList, zero);
          padding.clear();
          padding.resize(spatial, 0);
        }

        Value kernelSizeList = createConstantIntList(binder, rewriter, kernel);
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value dilationsList =
            createConstantIntList(binder, rewriter, dilations);
        Value cstCeilMode =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), ceilMode);

        if (binder.op->getNumResults() == 2) {
          Torch::ValueTensorType resultTypeIndices;
          if (binder.tensorResultTypeAtIndex(resultTypeIndices, 1))
            return failure();

          if (rank == 3) {
            rewriter.replaceOpWithNewOp<Torch::AtenMaxPool1dWithIndicesOp>(
                binder.op, resultTypeOut, resultTypeIndices, operand,
                kernelSizeList, stridesList, paddingList, dilationsList,
                cstCeilMode);
            return success();
          }

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

        Value spatialScaleValue = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(spatialScale));

        Value boolTrue = Torch::ConstantBoolOp::create(
            rewriter, loc, rewriter.getBoolAttr(true));

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
          constInts[i] = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i));
        }

        int64_t widthDim = inputRank - 2;
        Value widthDimValue = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(widthDim));

        int64_t heightDim = inputRank - 3;
        Value heightDimValue = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(heightDim));

        // extract indices of images within batch
        auto batchIdxsShape = SmallVector<int64_t>{Torch::kUnknownSize};
        auto batchIdxsFloatTy =
            rewriter.getType<Torch::ValueTensorType>(batchIdxsShape, floatTy);
        Value batchIdxsFloat = Torch::AtenSelectIntOp::create(
            rewriter, loc, batchIdxsFloatTy, rois, constInts[1], constInts[0]);
        auto batchIdxsIntTy =
            rewriter.getType<Torch::ValueTensorType>(batchIdxsShape, intTy);
        Value batchIdxs = Torch::Aten_CastLongOp::create(
            rewriter, loc, batchIdxsIntTy, batchIdxsFloat, boolTrue);

        // extract scaled ranges for regions of interest
        auto roiBBsShape = SmallVector<int64_t>{Torch::kUnknownSize, 4};
        auto roiBBsFloatTy =
            rewriter.getType<Torch::ValueTensorType>(roiBBsShape, floatTy);
        Value roiBBs = Torch::AtenSliceTensorOp::create(
            rewriter, loc, roiBBsFloatTy, rois, constInts[1], constInts[1],
            constInts[5], constInts[1]);
        Value roiBBsScaledFloat = Torch::AtenMulScalarOp::create(
            rewriter, loc, roiBBsFloatTy, roiBBs, spatialScaleValue);
        auto roiBBsTy =
            rewriter.getType<Torch::ValueTensorType>(roiBBsShape, intTy);
        Value roiBBsScaled = Torch::Aten_CastLongOp::create(
            rewriter, loc, roiBBsTy, roiBBsScaledFloat, boolTrue);

        SmallVector<Value> pooledRois;

        for (int64_t i = 0; i < numRois; i++) {
          Value roiIdx = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i));

          auto roiSpecTy = rewriter.getType<Torch::ValueTensorType>(
              roiBBsTy.getSizes().slice(1), intTy);
          Value roiSpec = Torch::AtenSelectIntOp::create(
              rewriter, loc, roiSpecTy, roiBBsScaled, constInts[0], roiIdx);

          // Load individual ROI specification values
          SmallVector<Value> roiValues(5);
          for (int specIdx = 0; specIdx < 5; specIdx++) {
            auto intEmptyTensorTy = rewriter.getType<Torch::ValueTensorType>(
                SmallVector<int64_t>{}, intTy);
            Value specTensor;
            if (specIdx == 0) { // batch index
              specTensor = Torch::AtenSelectIntOp::create(
                  rewriter, loc, intEmptyTensorTy, batchIdxs, constInts[0],
                  roiIdx);
            } else { // roi dimension
              specTensor = Torch::AtenSelectIntOp::create(
                  rewriter, loc, intEmptyTensorTy, roiSpec, constInts[0],
                  constInts[specIdx - 1]);
            }
            Value specValue = Torch::AtenItemOp::create(rewriter, loc,
                                                        torchIntTy, specTensor);
            roiValues[specIdx] = specValue;
          }
          Value batchIdx = roiValues[0], roiX1 = roiValues[1],
                roiY1 = roiValues[2], roiX2 = roiValues[3],
                roiY2 = roiValues[4];

          // add 1 to make range ends inclusive as per ONNX implementation
          roiX2 = Torch::AtenAddOp::create(rewriter, loc, torchIntTy, roiX2,
                                           constInts[1]);
          roiY2 = Torch::AtenAddOp::create(rewriter, loc, torchIntTy, roiY2,
                                           constInts[1]);

          auto imageTy = rewriter.getType<Torch::ValueTensorType>(
              inputShape.slice(1), inputTy.getDtype());
          Value image = Torch::AtenSelectIntOp::create(
              rewriter, loc, imageTy, input, constInts[0],
              batchIdx); // (NC x H x W)

          SmallVector<int64_t> imageUnknownShape(imageTy.getSizes());
          imageUnknownShape[heightDim] = Torch::kUnknownSize;
          imageUnknownShape[widthDim] = Torch::kUnknownSize;
          auto imageUnknownTy = rewriter.getType<Torch::ValueTensorType>(
              imageUnknownShape, imageTy.getDtype());

          // extract ROI from image
          Value imageExtractedY = Torch::AtenSliceTensorOp::create(
              rewriter, loc, imageUnknownTy, image, heightDimValue, roiY1,
              roiY2, constInts[1]);
          Value region = Torch::AtenSliceTensorOp::create(
              rewriter, loc, imageUnknownTy, imageExtractedY, widthDimValue,
              roiX1, roiX2, constInts[1]);

          SmallVector<int64_t> pooledRegionShape(imageTy.getSizes());
          pooledRegionShape[heightDim] = pooledShape[0];
          pooledRegionShape[widthDim] = pooledShape[1];
          auto pooledRegionTy = rewriter.getType<Torch::ValueTensorType>(
              pooledRegionShape, imageTy.getDtype());
          auto pooledRegionIndicesTy = rewriter.getType<Torch::ValueTensorType>(
              pooledRegionShape, intTy);

          // apply pooling on ROI
          Value pooledRegion =
              Torch::AtenAdaptiveMaxPool2dOp::create(
                  rewriter, loc, pooledRegionTy, pooledRegionIndicesTy, region,
                  outputShapeList)
                  .getResult0();
          pooledRois.push_back(pooledRegion);
        }

        Value pooledRoisList = Torch::PrimListConstructOp::create(
            rewriter, loc, Torch::ListType::get(pooledRois[0].getType()),
            pooledRois);
        rewriter.replaceOpWithNewOp<Torch::AtenStackOp>(
            binder.op, resultTy, pooledRoisList, constInts[0]);

        return success();
      });
  patterns.onOp("Greater", 7,
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
        Value none = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        Value boolTrue =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), true);
        Value boolFalse =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);
        auto epsValue = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getF64FloatAttr(eps));

        auto momentum = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getF64FloatAttr(0.0f));
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
      "MeanVarianceNormalization", 13,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value input;
        SmallVector<int64_t> axes;

        if (binder.tensorOperand(input) ||
            binder.s64IntegerArrayAttr(axes, "axes",
                                       llvm::SmallVector<int64_t>({0, 2, 3})) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }
        if (!resultType.hasSizes() || !resultType.hasDtype()) {
          return failure();
        }
        auto inputTy = cast<Torch::ValueTensorType>(input.getType());
        if (!inputTy || !inputTy.hasSizes()) {
          return failure();
        }
        int64_t inputRank = inputTy.getSizes().size();

        Location loc = binder.getLoc();
        Value keepDim = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value unBiased = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value none = Torch::ConstantNoneOp::create(rewriter, loc);

        ArrayRef<int64_t> output_shape = resultType.getSizes();
        SmallVector<int64_t> reduced_shape(output_shape);

        for (int64_t i : axes) {
          int64_t dim = Torch::toPositiveDim(i, inputRank);
          if (!Torch::isValidDim(dim, inputRank)) {
            return failure();
          }
          reduced_shape[dim] = 1;
        }
        Torch::ValueTensorType reducedOutTy = Torch::ValueTensorType::get(
            resultType.getContext(), reduced_shape, resultType.getDtype());
        SmallVector<Value> cstAxes;
        for (int64_t i : axes) {
          cstAxes.push_back(Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i)));
        }
        Value axes_list = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstAxes);
        Value mean = Torch::AtenMeanDimOp::create(
            rewriter, loc, reducedOutTy, input, axes_list, keepDim, none);
        Value variance = Torch::AtenVarDimOp::create(
            rewriter, loc, reducedOutTy, input, axes_list, unBiased, keepDim);
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstEps = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1e-9));
        variance = Torch::AtenAddScalarOp::create(rewriter, loc, reducedOutTy,
                                                  variance, cstEps, cstOne);
        Value sqrtVar =
            Torch::AtenSqrtOp::create(rewriter, loc, reducedOutTy, variance);
        Value inputMinusMean = Torch::AtenSubTensorOp::create(
            rewriter, loc, resultType, input, mean, cstOne);
        Value meanVarNorm = Torch::AtenDivTensorOp::create(
            rewriter, loc, resultType, inputMinusMean, sqrtVar);

        rewriter.replaceOp(binder.op, meanVarNorm);
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
          result = Torch::AtenMaximumOp::create(
              rewriter, binder.getLoc(), resultType, result, operands[i]);
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
          result = Torch::AtenMinimumOp::create(
              rewriter, binder.getLoc(), resultType, result, operands[i]);
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
          Value tyConst = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                      static_cast<int64_t>(torchqTy)));
          Value none = Torch::ConstantNoneOp::create(rewriter, loc);
          Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
          operand = Torch::AtenToDtypeOp::create(
              rewriter, loc, ty, operand, tyConst,
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
          Value k = Torch::ConstantIntOp::create(rewriter, binder.getLoc(), i);
          Value dataDim = Torch::AtenSizeIntOp::create(rewriter, loc, data, k);
          dataDims.push_back(dataDim);
          if (i < batchDimCount) {
            batchShape.push_back(dataShape[i]);
            batchDims.push_back(dataDim);
          }
        }

        // step 3. Get dimension list of indices.
        Value constZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value constOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        SmallVector<Value> indicesDimsMinusOne;
        SmallVector<Value> unflattenIndicesDims;
        Value indicesFlattenDim = constOne;
        for (int64_t i = 0; i < indicesRank - 1; ++i) {
          Value k = Torch::ConstantIntOp::create(rewriter, binder.getLoc(), i);
          Value indicesDim =
              Torch::AtenSizeIntOp::create(rewriter, loc, indices, k);
          indicesDimsMinusOne.push_back(indicesDim);
          if (i >= batchDimCount) {
            unflattenIndicesDims.push_back(indicesDim);
            indicesFlattenDim = Torch::AtenMulIntOp::create(
                rewriter, loc, indicesFlattenDim, indicesDim);
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
        Value sliceDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(indicesRank - 1));
        SmallVector<int64_t> indicesSliceShape(indicesShapeMinusOne);
        indicesSliceShape.push_back(1);
        auto indicesSliceTy = rewriter.getType<Torch::ValueTensorType>(
            indicesSliceShape, indicesTy.getOptionalDtype());

        Value start = constZero;
        Value updatedIndices;
        for (int64_t i = 0; i < indicesLastDim; ++i) {
          Value end = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i + 1));
          Value indicesSlice = Torch::AtenSliceTensorOp::create(
              rewriter, loc, indicesSliceTy, indices, sliceDim, start, end,
              /*step=*/constOne);
          start = end;
          // Apply bounds checking on the indices slice.
          auto boolTy = rewriter.getType<Torch::ValueTensorType>(
              indicesSliceShape, rewriter.getI1Type());
          Value lt = Torch::AtenLtScalarOp::create(rewriter, loc, boolTy,
                                                   indicesSlice, constZero);
          Value add = Torch::AtenAddScalarOp::create(
              rewriter, loc, indicesSliceTy, indicesSlice,
              dataDims[batchDimCount + i],
              /*alpha=*/constOne);
          indicesSlice = Torch::AtenWhereSelfOp::create(
              rewriter, loc, indicesSliceTy, lt, add, indicesSlice);
          if (i == 0) {
            updatedIndices = indicesSlice;
            continue;
          }
          updatedIndices = Torch::AtenAddTensorOp::create(
              rewriter, loc, indicesSliceTy, indicesSlice, updatedIndices,
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
        Value reshapeIndicesSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListTy, reshapeIndicesDims);
        auto reshapeIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            reshapeIndicesShape, indicesTy.getOptionalDtype());
        Value reshapedIndices =
            Torch::AtenViewOp::create(rewriter, loc, reshapeIndicesTy,
                                      updatedIndices, reshapeIndicesSizeList);

        // step 7. Flatten `q-b-1` dimensions of the indices.
        auto flattenIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            flattenIndicesShape, indicesTy.getOptionalDtype());
        Value batchDimCountVal = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(batchDimCount));
        Value flattenedIndices = reshapedIndices;
        if (indicesRank == 1) {
          flattenedIndices = Torch::AtenUnsqueezeOp::create(
              rewriter, loc, flattenIndicesTy, reshapedIndices, constZero);
        } else if (indicesRank > 1) {
          if (batchDimCount > indicesRank - 2) {
            flattenedIndices = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, flattenIndicesTy, reshapedIndices,
                batchDimCountVal);
          } else {
            Value endDim = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(indicesRank - 2));
            flattenedIndices = Torch::AtenFlattenUsingIntsOp::create(
                rewriter, loc, flattenIndicesTy, reshapedIndices,
                batchDimCountVal, endDim);
          }
        }

        // step 8. Expand `r-b-indices_shape[-1]` dims of flattened indices.
        auto expandIndicesTy = rewriter.getType<Torch::ValueTensorType>(
            expandIndicesShape, indicesTy.getOptionalDtype());
        Value expandIndicesSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListTy, expandIndicesDims);
        Value constFalse = Torch::ConstantBoolOp::create(
            rewriter, loc, rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));
        Value expandedIndices =
            Torch::AtenExpandOp::create(rewriter, loc, expandIndicesTy,
                                        flattenedIndices, expandIndicesSizeList,
                                        /*implicit=*/constFalse);

        // step 9. Flatten indices_shape[-1] dimensions of data.
        auto flattenDataTy = rewriter.getType<Torch::ValueTensorType>(
            flattenDataShape, dataTy.getOptionalDtype());
        Value endDim = Torch::ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(batchDimCount + indicesLastDim - 1));
        Value flattenedData = data;

        if (indicesLastDim != 1) {
          flattenedData = Torch::AtenFlattenUsingIntsOp::create(
              rewriter, loc, flattenDataTy, data, batchDimCountVal, endDim);
        }

        // step 10. Now we have flattenedData and expandedIndices of same rank
        // to perform gather operation.
        auto gatherTy = rewriter.getType<Torch::ValueTensorType>(
            expandIndicesShape, dataTy.getOptionalDtype());
        Value gather =
            Torch::AtenGatherOp::create(rewriter, loc, gatherTy, flattenedData,
                                        batchDimCountVal, expandedIndices,
                                        /*sparseGrad=*/constFalse);

        // step 11. Unflatten the collapsed indices dims of gather result.
        if (indicesRank == 1) {
          rewriter.replaceOpWithNewOp<Torch::AtenSqueezeDimOp>(
              binder.op, resultType, gather, /*dim=*/constZero);
          return success();
        }

        if (unflattenIndicesDims.empty()) {
          rewriter.replaceOpWithNewOp<Torch::AtenSqueezeDimOp>(
              binder.op, resultType, gather, /*dim=*/batchDimCountVal);
          return success();
        }

        Value unflattenSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListTy, unflattenIndicesDims);
        rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
            binder.op, resultType, gather, batchDimCountVal, unflattenSizeList);
        return success();
      });
  patterns.onOp(
      "Gather", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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

        Value index = Torch::ConstantIntOp::create(
            rewriter, loc, Torch::IntType::get(ctx),
            rewriter.getI64IntegerAttr(axis));

        // Apply bounds checking on the input:
        auto intTy = rewriter.getType<Torch::IntType>();
        auto boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value zero = Torch::ConstantIntOp::create(
            rewriter, loc, intTy, rewriter.getI64IntegerAttr(0));
        Value one = Torch::ConstantIntOp::create(rewriter, loc, intTy,
                                                 rewriter.getI64IntegerAttr(1));
        Value lt =
            Torch::AtenLtScalarOp::create(rewriter, loc, boolTy, indices, zero);
        Value dim =
            Torch::AtenSizeIntOp::create(rewriter, loc, intTy, data, index);
        Value add = Torch::AtenAddScalarOp::create(rewriter, loc, indicesTy,
                                                   indices, dim, one);
        indices = Torch::AtenWhereSelfOp::create(rewriter, loc, indicesTy, lt,
                                                 add, indices);

        auto intListTy = rewriter.getType<Torch::ListType>(
            rewriter.getType<Torch::IntType>());

        llvm::SmallVector<Value> indicesDims;
        for (int i = 0, s = indicesTy.getSizes().size(); i < s; ++i) {
          Value k = Torch::ConstantIntOp::create(rewriter, binder.getLoc(), i);
          indicesDims.push_back(Torch::AtenSizeIntOp::create(
              rewriter, binder.getLoc(), indices, k));
        }

        Value indicesSizeList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(), intListTy, indicesDims);

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
          indices = Torch::AtenUnsqueezeOp::create(rewriter, binder.getLoc(),
                                                   flattenTy, indices, zero);
        } else if (indicesRank > 1) {
          Value rank = Torch::AtenDimOp::create(rewriter, loc, intTy, indices);
          Value end = Torch::AtenSubIntOp::create(rewriter, loc, rank, one);
          indices = Torch::AtenFlattenUsingIntsOp::create(
              rewriter, loc, flattenTy, indices, zero, end);
        }

        llvm::SmallVector<int64_t> gatherShape(dataTy.getSizes());
        gatherShape[axis] = indicesCt;
        auto gatherTy = rewriter.getType<Torch::ValueTensorType>(
            gatherShape, dataTy.getOptionalDtype());
        Value gather = Torch::AtenIndexSelectOp::create(rewriter, loc, gatherTy,
                                                        data, index, indices);

        if (indicesRank == 1) {
          rewriter.replaceOp(binder.op, gather);
          return success();
        }

        if (indicesRank > 1) {
          gather = rewriter.replaceOpWithNewOp<Torch::AtenUnflattenIntOp>(
              binder.op, resultType, gather, index, indicesSizeList);
          return success();
        }

        // indicesRank = 0 will select 1 from the axis dim and squeeze it
        // Use AtenSqueezeDimOp for the case of result with dynamic shape
        rewriter.replaceOpWithNewOp<Torch::AtenSqueezeDimOp>(
            binder.op, resultType, gather, index);
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
        Value constAxis = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), axis));

        auto indicesTy = cast<Torch::ValueTensorType>(indices.getType());
        Value constZero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value constOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value axisSize = Torch::AtenSizeIntOp::create(rewriter, binder.getLoc(),
                                                      data, constAxis);
        Value indicesAdd = Torch::AtenAddScalarOp::create(
            rewriter, binder.getLoc(), indicesTy, indices, axisSize, constOne);

        auto boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value lt = Torch::AtenLtScalarOp::create(rewriter, binder.getLoc(),
                                                 boolTy, indices, constZero);
        indices = Torch::AtenWhereSelfOp::create(
            rewriter, binder.getLoc(), indicesTy, lt, indicesAdd, indices);

        Value sparseGrad = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::BoolType>(),
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

        Value zero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));
        Value one = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
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
          return Torch::AtenTransposeIntOp::create(rewriter, binder.getLoc(),
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

        Value mm = Torch::AtenMmOp::create(rewriter, binder.getLoc(),
                                           resultType, a, b);
        if (alpha == 1.0 && beta == 1.0) {
          rewriter.replaceOpWithNewOp<Torch::AtenAddTensorOp>(
              binder.op, resultType, mm, c, one);
          return success();
        }

        if (alpha != 1.0 && beta != 1.0) {
          Value constAlpha = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(alpha));
          mm = Torch::AtenMulScalarOp::create(rewriter, binder.getLoc(),
                                              resultType, mm, constAlpha);
          alpha = 1.0;
        }

        if (alpha != 1.0) {
          std::swap(alpha, beta);
          std::swap(mm, c);
        }

        Value constBeta = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
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
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value inputDimSize = Torch::AtenSizeIntOp::create(
                rewriter, binder.getLoc(), operand, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone =
            Torch::ConstantNoneOp::create(rewriter, binder.getLoc());

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
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value inputDimSize = Torch::AtenSizeIntOp::create(
                rewriter, binder.getLoc(), operand, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(inputShape[i])));
          }
          cstPadding.push_back(cstZero);
          cstDilations.push_back(cstOne);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value dilationsList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstDilations);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstCeilMode =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);

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
        if (inputRank > 5 || inputRank < 3) {
          return failure();
        }
        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value numElements = cstOne;
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(i));
            Value inputDimSize = Torch::AtenSizeIntOp::create(
                rewriter, binder.getLoc(), operand, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, binder.getLoc(),
                rewriter.getI64IntegerAttr(kernelSize)));
          }
          numElements = Torch::AtenMulOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              cstKernel.back(), numElements);
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value abs = Torch::AtenAbsOp::create(rewriter, binder.getLoc(),
                                             inputTensorType, operand);
        Value pv = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), p));
        Value pow = Torch::AtenPowTensorScalarOp::create(
            rewriter, binder.getLoc(), inputTensorType, abs, pv);
        Value avgPool;
        if (inputRank == 3) {
          avgPool = Torch::AtenAvgPool1dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad);
          avgPool = Torch::AtenMulScalarOp::create(
              rewriter, binder.getLoc(), resultType, avgPool, numElements);
        } else if (inputRank == 4) {
          avgPool = Torch::AtenAvgPool2dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        } else { // inputRank == 5
          avgPool = Torch::AtenAvgPool3dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        }
        Value invP = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
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
        if (rank > 5 || rank < 3) {
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
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
        Value numElements = cstOne;
        for (int64_t i : kernel) {
          cstKernel.push_back(Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(i)));
          numElements = Torch::AtenMulOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
              cstKernel.back(), numElements);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        Value cstCeilMode =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), ceilMode);
        // onnx lp pool doesn't have countIncludePad attribute but set it to
        // true so that in 1D case numElements is correctly undoes divison. For
        // 2D/3D case, division is avoided by divison_override.
        Value cstCountIncludePad =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), true);
        Value pv = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), p));
        auto inputTensorType = cast<Torch::ValueTensorType>(operand.getType());
        Value abs = Torch::AtenAbsOp::create(rewriter, binder.getLoc(),
                                             inputTensorType, operand);
        Value pow = Torch::AtenPowTensorScalarOp::create(
            rewriter, binder.getLoc(), inputTensorType, abs, pv);
        Value avgPool;
        if (rank == 3) {
          avgPool = Torch::AtenAvgPool1dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad);
          avgPool = Torch::AtenMulScalarOp::create(
              rewriter, binder.getLoc(), resultType, avgPool, numElements);
        } else if (rank == 4) {
          avgPool = Torch::AtenAvgPool2dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        } else { // rank == 5
          avgPool = Torch::AtenAvgPool3dOp::create(
              rewriter, binder.getLoc(), resultType, pow, kernelSizeList,
              stridesList, paddingList, cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstOne);
        }
        Value invP = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
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

        // Convert dtype if stash_type is different from input dtype
        auto xType = cast<Torch::ValueTensorType>(x.getType());
        Value cstFalse =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);
        Value none = Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        if (*stashDtype != xType.getOptionalDtype()) {
          auto newXType =
              xType.getWithSizesAndDtype(xType.getOptionalSizes(), *stashDtype);
          Value dtypeValue = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(stashTypeIntTorch.value()));
          x = Torch::AtenToDtypeOp::create(
              rewriter, binder.getLoc(), newXType, x, /*dtype=*/dtypeValue,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/none);
        }

        Value constEpsilon = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
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
          normalized.push_back(Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(xShape[n])));
        }
        Value normalized_shape = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            normalized);

        SmallVector<int64_t> reducedShape(rank, 1);
        for (int64_t i = 0; i < axis; i++)
          reducedShape[i] = xShape[i];
        auto reducedType =
            xType.getWithSizesAndDtype(reducedShape, *stashDtype);
        auto y = Torch::AtenNativeLayerNormOp::create(
            rewriter, binder.getLoc(), yType, /*meanType=*/reducedType,
            /*invStdDevType=*/reducedType, x, normalized_shape, scale, b,
            constEpsilon);

        int64_t numResults = binder.op->getNumResults();
        if (numResults == 1) {
          rewriter.replaceOp(binder.op, y.getResult0());
          return success();
        }

        Value meanOutput = y.getResult1();
        Value varOutput = y.getResult2();
        // Convert meanType and varType back if stash_dtype is different
        if (binder.tensorResultTypeAtIndex(meanType, 1) ||
            binder.tensorResultTypeAtIndex(invStdDevType, 2))
          return failure();
        if (*stashDtype != meanType.getOptionalDtype()) {
          Value constDtype = Torch::getDtypeIntValueForType(
              rewriter, binder.getLoc(), meanType.getDtype());
          meanOutput = Torch::AtenToDtypeOp::create(
              rewriter, binder.getLoc(), meanType, meanOutput,
              /*dtype=*/constDtype,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/none);
          varOutput = Torch::AtenToDtypeOp::create(
              rewriter, binder.getLoc(), invStdDevType, varOutput,
              /*dtype=*/constDtype,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/none);
        }
        rewriter.replaceOp(binder.op, {y.getResult0(), meanOutput, varOutput});

        return success();
      });
  patterns.onOp(
      "LeakyRelu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value operand;
        float alpha;
        if (binder.tensorOperand(operand) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha", 0.01f))
          return failure();
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
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
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(alpha));
        Value constBeta = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(beta));
        Value constBias = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(bias));
        // Please refer to the operator description
        // for more info on the lowering
        // https://onnx.ai/onnx/operators/onnx__LRN.html

        // squared = operand^2
        Location loc = binder.getLoc();
        Torch::ValueTensorType inTy =
            cast<Torch::ValueTensorType>(operand.getType());
        Value sqOperand = Torch::AtenMulTensorOp::create(rewriter, loc, inTy,
                                                         operand, operand);
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
        auto view = Torch::AtenViewOp::create(rewriter, loc, reshapeType,
                                              sqOperand, viewShapeListVal);
        // padding
        int64_t highPad = (size - 1) / 2;
        int64_t lowPad = (size - 1) - highPad;
        SmallVector<int64_t> paddingInt{0, 0, 0, 0, lowPad, highPad};
        auto constPadVal = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value paddingListVal =
            createConstantIntList(binder, rewriter, paddingInt);
        SmallVector<int64_t, 5> paddedShapeInt = viewShapeInt;
        paddedShapeInt[2] += size - 1;
        Torch::ValueTensorType paddedType =
            rewriter.getType<Torch::ValueTensorType>(paddedShapeInt, dtype);
        auto padded = Torch::AtenConstantPadNdOp::create(
            rewriter, loc, paddedType, view, paddingListVal, constPadVal);
        // avg_pool3d
        SmallVector<int64_t, 3> kernelSize{size, 1, 1};
        Value kernelSizeList =
            createConstantIntList(binder, rewriter, kernelSize);
        SmallVector<int64_t, 3> strides{1, 1, 1};
        Value stridesList = createConstantIntList(binder, rewriter, strides);
        SmallVector<int64_t, 3> padding{0, 0, 0};
        Value paddingList = createConstantIntList(binder, rewriter, padding);
        auto cstCeilMode =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), false);
        auto cstCountIncludeMode =
            Torch::ConstantBoolOp::create(rewriter, binder.getLoc(), true);
        Value cstNone =
            Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
        // Output of pooling is same reshape(view) type because
        // of the padding done on the dimensions being pooled.
        auto pool = Torch::AtenAvgPool3dOp::create(
            rewriter, loc, reshapeType, padded, kernelSizeList, stridesList,
            paddingList, cstCeilMode, cstCountIncludeMode,
            /*divisor_override=*/cstNone);
        // squeeze
        auto one = Torch::ConstantIntOp::create(rewriter, loc,
                                                rewriter.getI64IntegerAttr(1));
        SmallVector<int64_t, 5> squeezeShapeInt{
            viewShapeInt[0], viewShapeInt[2], viewShapeInt[3], viewShapeInt[4]};
        Torch::ValueTensorType squeezeType =
            rewriter.getType<Torch::ValueTensorType>(squeezeShapeInt, dtype);
        auto squeeze = Torch::AtenSqueezeDimOp::create(rewriter, loc,
                                                       squeezeType, pool, one);
        // view as input Type
        Value intTyShapeList =
            createConstantIntList(binder, rewriter, inTyShape);
        auto viewAsInput = Torch::AtenViewOp::create(rewriter, loc, inTy,
                                                     squeeze, intTyShapeList);
        // mul + add + pow + div
        auto mul = Torch::AtenMulScalarOp::create(rewriter, loc, resultType,
                                                  viewAsInput, constAlpha);
        auto add = Torch::AtenAddScalarOp::create(rewriter, loc, resultType,
                                                  mul, constBias, one);
        auto pow = Torch::AtenPowTensorScalarOp::create(
            rewriter, loc, resultType, add, constBeta);

        rewriter.replaceOpWithNewOp<Torch::AtenDivTensorOp>(
            binder.op, resultType, operand, pow);
        return success();
      });
  patterns.onOp(
      "Pad", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value data, pads, axes;
        std::string mode;

        if (binder.tensorOperandAtIndex(data, 0) ||
            binder.tensorResultType(resultType) ||
            binder.customOpNameStringAttr(mode, "mode", "constant"))
          return failure();

        (void)binder.tensorOperandAtIndex(axes, 3);

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
            padsTensorValue.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(p)));
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
          Value constZero = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));
          SmallVector<int64_t> emptyShape;
          Type padsElemType = Torch::ValueTensorType::get(
              padsTensorType.getContext(), emptyShape,
              padsTensorType.getOptionalDtype());
          for (uint32_t i = 0; i < padsSize; ++i) {
            Value index = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            auto select = Torch::AtenSelectIntOp::create(
                rewriter, loc, padsElemType, pads, constZero, index);
            Value selectInt = Torch::AtenItemOp::create(
                rewriter, loc, rewriter.getType<Torch::IntType>(), select);
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
            constantValue = Torch::AtenItemOp::create(rewriter, loc, scalarTy,
                                                      constantValue);
          }
        }

        if (!constantValue && cstMode) {
          auto dataTensorType = cast<Torch::ValueTensorType>(data.getType());
          if (isa<IntegerType>(dataTensorType.getDtype()))
            constantValue = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(0));
          // Earlier versions used a FLOAT attribute to store the constant
          // value. The following will pick up on any non-default value attr if
          // provided.
          float constantFloat;
          if (isa<FloatType>(dataTensorType.getDtype()) &&
              !binder.f32FloatAttr(constantFloat, "value", 0.0f))
            constantValue = Torch::ConstantFloatOp::create(
                rewriter, loc, rewriter.getF64FloatAttr(constantFloat));

          if (!constantValue)
            return rewriter.notifyMatchFailure(
                binder.op, "expected integer or float data tensor");
        }

        // for modes other than "constant" a value is not required
        if (!cstMode)
          constantValue = Torch::ConstantNoneOp::create(rewriter, loc);

        llvm::SmallVector<Value> begins;
        llvm::SmallVector<Value> ends;
        for (uint32_t i = 0; i < padsSize / 2; ++i)
          begins.push_back(padsTensorValue[i]);
        for (uint32_t i = padsSize / 2; i < padsSize; ++i)
          ends.push_back(padsTensorValue[i]);

        // If we have the axes we need to compute the appropriate pads:
        if (axes) {
          auto axesTy = cast<Torch::ValueTensorType>(axes.getType());
          assert(axesTy.getSizes().size() == 1);
          assert(axesTy.getSizes()[0] != Torch::kUnknownSize);

          auto dataTensorType = cast<Torch::ValueTensorType>(data.getType());
          int64_t rank = dataTensorType.getSizes().size();
          auto boolTy = rewriter.getType<Torch::BoolType>();
          auto intTy = rewriter.getType<Torch::IntType>();
          Value constZero = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));

          // Extract the values:
          int64_t numAxes = axesTy.getSizes()[0];
          Type axesElemType = Torch::ValueTensorType::get(
              axesTy.getContext(), ArrayRef<int64_t>{},
              axesTy.getOptionalDtype());
          llvm::SmallVector<Value> axesExtracted;
          Value rankV = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(rank));
          for (uint32_t i = 0; i < numAxes; ++i) {
            Value index = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            auto select = Torch::AtenSelectIntOp::create(
                rewriter, loc, axesElemType, axes, constZero, index);
            Value selectInt = Torch::AtenItemOp::create(
                rewriter, loc, rewriter.getType<Torch::IntType>(), select);

            Value negAxis = Torch::AtenLtIntOp::create(rewriter, loc, boolTy,
                                                       selectInt, constZero);
            negAxis =
                Torch::AtenIntBoolOp::create(rewriter, loc, intTy, negAxis);
            Value axis = Torch::AtenMulIntOp::create(rewriter, loc, intTy,
                                                     negAxis, rankV);
            axis = Torch::AtenAddIntOp::create(rewriter, loc, intTy, axis,
                                               selectInt);
            axesExtracted.push_back(axis);
          }

          llvm::SmallVector<Value> newBegins;
          llvm::SmallVector<Value> newEnds;

          for (int j = 0; j < rank; ++j) {
            Value newBegin = constZero;
            Value newEnd = constZero;
            Value iv = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(j));

            for (size_t i = 0; i < axesExtracted.size(); ++i) {
              Value begin = begins[i];
              Value end = ends[i];

              Value sameAxis = Torch::AtenEqIntOp::create(rewriter, loc, boolTy,
                                                          axesExtracted[i], iv);
              sameAxis =
                  Torch::AtenIntBoolOp::create(rewriter, loc, intTy, sameAxis);

              begin = Torch::AtenMulIntOp::create(rewriter, loc, intTy,
                                                  sameAxis, begin);
              end = Torch::AtenMulIntOp::create(rewriter, loc, intTy, sameAxis,
                                                end);

              newBegin = Torch::AtenAddIntOp::create(rewriter, loc, intTy,
                                                     newBegin, begin);
              newEnd = Torch::AtenAddIntOp::create(rewriter, loc, intTy, newEnd,
                                                   end);
            }

            newBegins.push_back(newBegin);
            newEnds.push_back(newEnd);
          }

          begins = std::move(newBegins);
          ends = std::move(newEnds);
        }

        // The torch.pad op expects a different arrangement of padding pairs for
        // each dimension as compared to the onnx.pad op. Rearrange the pad
        // tensor as shown below:
        //
        // [x1_begin, x2_begin, ..., x1_end, x2_end,...] ->
        // [xn_begin, xn_end, ...., x2_begin, x2_end, x1_begin, x1_end]
        SmallVector<Value> padsRearrange;
        for (int32_t i = begins.size() - 1; i >= 0; i--) {
          padsRearrange.emplace_back(begins[i]);
          padsRearrange.emplace_back(ends[i]);
        }

        Value padsSizeList =
            Torch::PrimListConstructOp::create(
                rewriter, loc,
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

        Value modeVal = Torch::ConstantStrOp::create(
            rewriter, loc, rewriter.getStringAttr(mode));

        rewriter.replaceOpWithNewOp<Torch::AtenPadOp>(
            binder.op, resultType, data, padsSizeList, modeVal, constantValue);
        return success();
      });
  patterns.onOp(
      "Pow", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // ONNX specifies that the result types matches the type of lhs.
        // In torch, the result type is integer when both operands are integer,
        // and otherwise operand types are promoted to f64.
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.tensorResultType(resultType)) {
          return failure();
        }

        auto loc = binder.getLoc();
        Value cstFalse = Torch::ConstantBoolOp::create(
            rewriter, loc, rewriter.getBoolAttr(false));
        Value none = Torch::ConstantNoneOp::create(rewriter, loc);

        auto powType = resultType;
        if (isa<IntegerType>(resultType.getDtype())) {
          powType = rewriter.getType<Torch::ValueTensorType>(
              resultType.getSizes(), rewriter.getF64Type());
        }

        Value pow = Torch::AtenPowTensorTensorOp::create(rewriter, loc, powType,
                                                         lhs, rhs);

        if (!isa<IntegerType>(resultType.getDtype())) {
          rewriter.replaceOp(binder.op, pow);
          return success();
        }

        auto outDtype = Torch::getScalarTypeForType(resultType.getDtype());
        auto outTyConst = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    static_cast<int64_t>(outDtype)));

        pow = Torch::AtenRoundOp::create(rewriter, loc, powType, pow);
        rewriter.replaceOpWithNewOp<Torch::AtenToDtypeOp>(
            binder.op, resultType, pow, outTyConst, cstFalse, cstFalse, none);

        return success();
      });
  patterns.onOp("Identity", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  Torch::ValueTensorType resultType;
                  Value tensor;
                  if (binder.tensorOperand(tensor) ||
                      binder.tensorResultType(resultType)) {
                    return failure();
                  }
                  Value noneVal =
                      Torch::ConstantNoneOp::create(rewriter, binder.getLoc());
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
        Value numOperandsConstant = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), numOperands));
        if (binder.tensorOperands(valList, numOperands) ||
            binder.tensorResultType(resultType))
          return failure();
        Value constOne = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
        // Short circuit to binary add
        Value curr = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(),
                                                    resultType, valList[0],
                                                    valList[1], constOne);
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
            curr = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(),
                                                  resultType, curr, valList[i],
                                                  constOne);
          } else {
            curr = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(),
                                                  baseType, curr, valList[i],
                                                  constOne);
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
          tensor = Torch::AtenReluOp::create(
              rewriter, binder.getLoc(),
              dyn_cast<Torch::ValueTensorType>(tensor.getType()), tensor);
        }
        if (pos == 0) {
          // first use neg op to flip positive inf to negative inf. Then relu to
          // replace all positive infs with 0.
          Value flip = Torch::AtenNegOp::create(
              rewriter, binder.getLoc(),
              dyn_cast<Torch::ValueTensorType>(tensor.getType()), tensor);
          tensor = Torch::AtenReluOp::create(
              rewriter, binder.getLoc(),
              dyn_cast<Torch::ValueTensorType>(flip.getType()), flip);
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
        depth = Torch::AtenItemOp::create(rewriter, loc, depthETy, depth);

        if (!depthIsInt)
          depth = Torch::AtenIntScalarOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(), depth);

        Type boolTy = rewriter.getType<Torch::ValueTensorType>(
            indicesTy.getSizes(), rewriter.getI1Type());
        Value zero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value one = Torch::ConstantIntOp::create(rewriter, loc,
                                                 rewriter.getI64IntegerAttr(1));
        Value lt =
            Torch::AtenLtScalarOp::create(rewriter, loc, boolTy, indices, zero);
        Value add = Torch::AtenAddScalarOp::create(rewriter, loc, indicesTy,
                                                   indices, depth, one);
        indices = Torch::AtenWhereSelfOp::create(rewriter, loc, indicesTy, lt,
                                                 add, indices);

        auto selectTy = rewriter.getType<Torch::ValueTensorType>(
            llvm::SmallVector<int64_t>{1}, valuesTy.getDtype());

        bool valuesAreInt = isa<IntegerType>(valuesTy.getDtype());
        Type valuesETy = valuesAreInt ? intTy : floatTy;

        Value off = Torch::AtenSelectIntOp::create(rewriter, loc, selectTy,
                                                   values, zero, zero);
        off = Torch::AtenItemOp::create(rewriter, loc, valuesETy, off);

        Value on = Torch::AtenSelectIntOp::create(rewriter, loc, selectTy,
                                                  values, zero, one);
        on = Torch::AtenItemOp::create(rewriter, loc, valuesETy, on);

        auto i32Ty = rewriter.getIntegerType(32, true);
        llvm::SmallVector<int64_t> onehotShape(indicesTy.getSizes());
        onehotShape.push_back(Torch::kUnknownSize);
        auto onehotTy =
            rewriter.getType<Torch::ValueTensorType>(onehotShape, i32Ty);

        Value onehot = Torch::AtenOneHotOp::create(rewriter, binder.getLoc(),
                                                   onehotTy, indices, depth);

        for (int i = indicesTy.getSizes().size(); i > axis; --i) {
          std::swap(onehotShape[i - 1], onehotShape[i]);
          Value iv0 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i));
          Value iv1 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(i - 1));

          onehotTy =
              rewriter.getType<Torch::ValueTensorType>(onehotShape, i32Ty);
          onehot = Torch::AtenTransposeIntOp::create(rewriter, loc, onehotTy,
                                                     onehot, iv1, iv0);
        }

        // Change one hot to an array of booleans to select value:
        auto i1Ty = rewriter.getI1Type();
        auto torchqTy = Torch::getScalarTypeForType(i1Ty);
        Value tyConst = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                    static_cast<int64_t>(torchqTy)));

        onehotTy = rewriter.getType<Torch::ValueTensorType>(onehotShape, i1Ty);
        Value none = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        onehot = Torch::AtenToDtypeOp::create(
            rewriter, loc, onehotTy, onehot, tyConst,
            /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
            /*memory_format=*/none);

        onehot = Torch::AtenWhereScalarOp::create(rewriter, loc, resultType,
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
      "Hardmax", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
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
            binder.s64IntegerAttr(axisValue, "axis", -1) ||
            binder.tensorResultType(resultType))
          return failure();

        auto loc = binder.getLoc();
        auto inputTy = cast<Torch::ValueTensorType>(input.getType());

        if (axisValue < 0)
          axisValue += inputTy.getSizes().size();

        axis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axisValue));

        // torch.argmax
        Value constKeepDims = Torch::ConstantBoolOp::create(
            rewriter, loc, rewriter.getType<Torch::BoolType>(),
            rewriter.getBoolAttr(false));

        SmallVector<int64_t> argmaxShape;
        for (int i = 0, s = inputTy.getSizes().size(); i < s; ++i) {
          if (i == axisValue)
            continue;
          argmaxShape.push_back(inputTy.getSizes()[i]);
        }

        auto argmaxTy = rewriter.getType<Torch::ValueTensorType>(
            argmaxShape, rewriter.getIntegerType(32, IntegerType::Signed));
        Value argmax = Torch::AtenArgmaxOp::create(rewriter, loc, argmaxTy,
                                                   input, axis, constKeepDims);

        // one_hot
        SmallVector<int64_t> onehotShape(argmaxShape);
        onehotShape.push_back(inputTy.getSizes()[axisValue]);
        auto onehotTy = rewriter.getType<Torch::ValueTensorType>(
            onehotShape, resultType.getDtype());
        Value numClasses = Torch::AtenSizeIntOp::create(
            rewriter, binder.getLoc(), input, axis);
        Value onehot = Torch::AtenOneHotOp::create(
            rewriter, binder.getLoc(), onehotTy, argmax, numClasses);

        SmallVector<int64_t> permutation;
        for (int i = 0; i < axisValue; ++i)
          permutation.push_back(i);
        permutation.push_back(onehotShape.size() - 1);
        for (int i = axisValue, s = onehotShape.size(); i < s - 1; ++i)
          permutation.push_back(i);

        SmallVector<Value> permValues;
        for (auto d : permutation) {
          permValues.push_back(Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(d)));
        }

        Value permuteDims = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(rewriter.getType<Torch::IntType>()),
            permValues);
        rewriter.replaceOpWithNewOp<Torch::AtenPermuteOp>(binder.op, resultType,
                                                          onehot, permuteDims);
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
                  Value cstAxis = Torch::ConstantIntOp::create(
                      rewriter, loc, rewriter.getI64IntegerAttr(axis));
                  Value cstP = Torch::ConstantIntOp::create(
                      rewriter, loc, rewriter.getI64IntegerAttr(p));
                  Value cstKeepDim = Torch::ConstantBoolOp::create(
                      rewriter, loc, rewriter.getBoolAttr(true));
                  Value axisPrimList = Torch::PrimListConstructOp::create(
                      rewriter, binder.getLoc(),
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
                  Value norm = Torch::AtenNormScalarOptDimOp::create(
                      rewriter, loc, normType, input, cstP, axisPrimList,
                      cstKeepDim);

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

        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)epsilon));
        Value cstNumGroups = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(numGroups));
        Value cstFalse = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getBoolAttr(false));
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

        Value cstOutput = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr((int64_t)output));
        Value cstDtype = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr((int)torch_upstream::ScalarType::Bool));
        Value cstFalse = Torch::ConstantBoolOp::create(
            rewriter, binder.getLoc(), rewriter.getBoolAttr(false));
        Value cstNone =
            Torch::ConstantNoneOp::create(rewriter, binder.getLoc());

        Value dataList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
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
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t centerPointBox;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(centerPointBox, "center_point_box", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        if (centerPointBox != 0 && centerPointBox != 1)
          return rewriter.notifyMatchFailure(
              binder.op, "expected center_point_box attribute to be 0 or 1");

        // TODO: Support multiple batches and classes
        // Squeeze the boxes and scores tensor.
        // In Onnx, the shape of boxes is [BxNx4] while the
        // torchvision expects it to be of shape [Nx4]. Similarly, for
        // the scores tensor shape in Onnx is [BxCxN] while the
        // torchvision expects it to be of shape [N].
        Value boxes = operands[0], scores = operands[1];
        FailureOr<Value> squeezedBoxes =
            Torch::squeezeTensor(rewriter, binder.op, loc, 0, boxes);
        if (failed(squeezedBoxes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze boxes tensor");
        FailureOr<Value> squeezedScores =
            Torch::squeezeTensor(rewriter, binder.op, loc, 0, scores);
        if (failed(squeezedScores))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze scores tensor");
        squeezedScores = Torch::squeezeTensor(rewriter, binder.op, loc, 0,
                                              squeezedScores.value());
        if (failed(squeezedScores))
          return rewriter.notifyMatchFailure(binder.op,
                                             "failed to squeeze scores tensor");
        boxes = squeezedBoxes.value();
        scores = squeezedScores.value();
        if (centerPointBox == 1) {
          // When center_point_box is 1, the box data is supplied as
          // [[x_center, y_center, width, height], ...]. Slice it to
          // [[x_center, y_center], ...] and [[width, height], ...],
          // calculate the [[x1, y1], ...] and [[x2, y2], ...], and concatnate
          // to [[x1, y1, x2, y2], ...]
          auto boxesTensorType =
              dyn_cast<Torch::ValueTensorType>(boxes.getType());
          Value const0 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));
          Value const1 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(1));
          Value const2 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(2));
          Value const4 = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(4));
          Value const2F = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(2.0));

          // extract scaled ranges for regions of interest
          auto sliceShape = SmallVector<int64_t>{Torch::kUnknownSize, 2};
          auto sliceTensorType = rewriter.getType<Torch::ValueTensorType>(
              sliceShape, boxesTensorType.getDtype());
          Value centers = Torch::AtenSliceTensorOp::create(
              rewriter, loc, sliceTensorType, boxes, const1, const0, const2,
              const1);
          Value sizes = Torch::AtenSliceTensorOp::create(
              rewriter, loc, sliceTensorType, boxes, const1, const2, const4,
              const1);
          Value halfSizes = Torch::AtenDivScalarOp::create(
              rewriter, loc, sizes.getType(), sizes, const2F);
          Value x1y1s = Torch::AtenSubTensorOp::create(
              rewriter, loc, centers.getType(), centers, halfSizes, const1);
          Value x2y2s = Torch::AtenAddTensorOp::create(
              rewriter, loc, centers.getType(), centers, halfSizes, const1);

          Type listElemType = boxesTensorType.getWithSizesAndDtype(
              /*optionalSizes=*/std::nullopt,
              /*optionalDtype=*/nullptr);
          Type listType = Torch::ListType::get(listElemType);
          Value tensorList = Torch::PrimListConstructOp::create(
              rewriter, loc, listType, SmallVector<Value>{x1y1s, x2y2s});
          boxes = Torch::AtenCatOp::create(rewriter, loc, boxesTensorType,
                                           tensorList, const1);
        }

        // TODO: Support score_threshold input
        // Filter out the boxes if the score < score_threshold
        if (operands.size() == 5) {
          Value scoreThreshold = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(), operands[4]);
          Value minScores = Torch::AtenMinOp::create(
              rewriter, loc,
              Torch::ValueTensorType::get(binder.op->getContext(),
                                          SmallVector<int64_t>{},
                                          rewriter.getF32Type()),
              scores);
          minScores = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(), minScores);

          Value scoresCond = Torch::AtenGeFloatOp::create(
              rewriter, loc, minScores, scoreThreshold);
          Torch::RuntimeAssertOp::create(
              rewriter, loc, scoresCond,
              rewriter.getStringAttr(
                  "unimplemented: score_threshold should be <= min(scores)"));
        }

        // Get max_output_boxes_per_class and iou_threshold
        Value cst0 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cst1 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value maxOutputBoxesPerClass = cst0;
        Value iouThreshold = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(0.0));
        if (operands.size() > 3 &&
            !isa<Torch::NoneType>(operands[3].getType())) {
          iouThreshold = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(), operands[3]);
        }
        if (operands.size() > 2 &&
            !isa<Torch::NoneType>(operands[2].getType())) {
          maxOutputBoxesPerClass = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(), operands[2]);
        }

        auto nmsTy = Torch::ValueTensorType::get(
            binder.op->getContext(), SmallVector<int64_t>{-1},
            rewriter.getIntegerType(64, /*signed=*/true));
        Value result = Torch::TorchvisionNmsOp::create(
            rewriter, loc, nmsTy, boxes, scores, iouThreshold);

        // Slice the result if numOutputBoxes (N) > max_output_boxes_per_class
        Value numOutputBoxes =
            Torch::AtenSizeIntOp::create(rewriter, loc, result, cst0);
        Value boxesCond = Torch::AtenGtIntOp::create(
            rewriter, loc, numOutputBoxes, maxOutputBoxesPerClass);

        auto nmsResultTy = Torch::ValueTensorType::get(
            binder.op->getContext(),
            SmallVector<int64_t>{resultType.getSizes()[0]},
            rewriter.getIntegerType(64, /*signed=*/true));
        auto ifSlice = Torch::PrimIfOp::create(
            rewriter, loc, TypeRange({nmsResultTy}), boxesCond);
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.createBlock(&ifSlice.getThenRegion(),
                               ifSlice.getThenRegion().begin());

          Value curResult = Torch::AtenSliceTensorOp::create(
              rewriter, loc, nmsResultTy, result, /*dim=*/cst0, /*start=*/cst0,
              /*end=*/maxOutputBoxesPerClass, /*step=*/cst1);
          Torch::PrimIfYieldOp::create(rewriter, loc, curResult);
        }
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.createBlock(&ifSlice.getElseRegion(),
                               ifSlice.getElseRegion().begin());

          Value curResult = Torch::TensorStaticInfoCastOp::create(
              rewriter, loc, nmsResultTy, result);
          Torch::PrimIfYieldOp::create(rewriter, loc, curResult);
        }
        result = ifSlice.getResult(0);

        // The result generated by torchvision.nms op is of shape [n], while the
        // onnx expects it to be of shape [n, 3]. Hence, we unsqueeze the tensor
        // and make it of shape [n, 1] and then concatenate it with a zero
        // tensor of shape [n, 2] to make it of shape [n, 3].
        FailureOr<Value> unsqueezedResult =
            Torch::unsqueezeTensor(rewriter, binder.op, result, cst1);
        if (failed(unsqueezedResult))
          return rewriter.notifyMatchFailure(
              binder.op, "failed to  unsqueeze result tensor");
        result = unsqueezedResult.value();

        numOutputBoxes =
            Torch::AtenSizeIntOp::create(rewriter, loc, result, cst0);
        SmallVector<Value> zerosShapeValues{numOutputBoxes};
        zerosShapeValues.push_back(Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(2)));
        Value zerosShapeList = Torch::PrimListConstructOp::create(
            rewriter, loc,
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
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value zeros =
            Torch::AtenZerosOp::create(rewriter, loc, zerosTy, zerosShapeList,
                                       cstNone, cstNone, cstNone, cstNone);

        Type listElemType =
            cast<Torch::BaseTensorType>(resultType)
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = Torch::PrimListConstructOp::create(
            rewriter, loc, listType, SmallVector<Value>{zeros, result});
        rewriter.replaceOpWithNewOp<Torch::AtenCatOp>(binder.op, resultType,
                                                      tensorList, cst1);
        return success();
      });
}
