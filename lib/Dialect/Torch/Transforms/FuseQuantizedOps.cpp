//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <stack>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

template <typename SrcOp> struct QuantInfo {
  static constexpr unsigned operandsToQuantize[2] = {0, 1};
};

template <> struct QuantInfo<AtenReluOp> {
  static constexpr unsigned operandsToQuantize[1] = {0};
};

// A QCommutingOp is an Op satisfying:
// 1. Has at most one tensor operand at index 0
// 2. Has a single output, which is a tensor
// 3. Satisfies the commutation relation:
//      [MPTQT -> Dequant -> Op(float)] = [Op(int) -> MPTQT -> Dequant]
// where MPTQT = "Aten_MakePerTensorQuantizedTensorOp"
// and Dequant = "AtenDequantizeSelfOp" or "AtenDequantizeTensorOp"
bool isQCommutingOp(mlir::Operation *op) {
  // if adding a new commuting op here, be sure to add a
  // RemoveUnused pattern for that op to clean up afterwards
  return llvm::isa<AtenTransposeIntOp, AtenReshapeOp, AtenSliceTensorOp,
                   PrimsCollapseOp, AtenViewOp, AtenPadOp, AtenConstantPadNdOp>(
      op);
}

struct QuantizedChain {
  std::stack<mlir::Operation *> commutingOpStack;
  Value dequantOpd, MPTQTOpd, scale, zeroPoint;
};

// The following conversion takes patterns of the form [op0 -> MPTQT -> dequant
// -> Op1 -> Op2 -> ... Opk -> SrcOp] to [op0 -> Int(Op1) -> Int(Op2) -> ... ->
// Int(Opk) -> MPTQT -> SrcOp] for any sequence of q commuting ops
// {Op1,Op2,...,Opk} with k <= depth.
// With depth = 0, this conversion will simply fuse any immediately quantizable
// operands: [MPTQT -> Dequant -> SrcOp (float operands)] to [MPTQT -> SrcOp(int
// operands)]
template <typename SrcOp, unsigned depth>
class QuantizeOperandsPastCommutingOps : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    llvm::SmallVector<Value> operands(op->getOperands());

    // Prevent fusion for 1d convolution ops and just do it as an f32 conv since
    // there isn't a linalg named op for quantized 1-d convolution yet.
    // TODO: Remove this and add support for 1-d quantized convolution.
    int64_t inputRank =
        cast<ValueTensorType>(operands[0].getType()).getSizes().size();
    if (isa<Torch::AtenConvolutionOp>(op) && inputRank < 4)
      return rewriter.notifyMatchFailure(
          op, "1-d quantized convolution is not supported");

    SmallVector<QuantizedChain, 2> operandChains;
    for (unsigned i : QuantInfo<SrcOp>::operandsToQuantize) {
      Value operand = operands[i];
      QuantizedChain chain;
      for (unsigned k = 0; k < depth + 1; k++) {
        auto currOp = operand.getDefiningOp();
        // Case 0 : currOp is a nullptr (e.g., operand is a block argument)
        if (!currOp)
          break;
        // Case 1 : currOp is a q commuting op (continue loop)
        if (isQCommutingOp(currOp)) {
          chain.commutingOpStack.push(currOp);
          // set operand to currOp for next k-iteration
          operand = currOp->getOperand(0);
          continue;
        }
        // Case 2 : currOp is a dequant op (end loop)
        if (llvm::isa<AtenDequantizeSelfOp, AtenDequantizeTensorOp>(currOp)) {
          chain.dequantOpd = currOp->getOperand(0);
          // Bail out if any operand is per-channel quantized, which would
          // require more complex fusion logic.
          if (llvm::isa<Aten_MakePerChannelQuantizedTensorOp>(
                  chain.dequantOpd.getDefiningOp()))
            break;

          auto MPTQTOp =
              chain.dequantOpd
                  .getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>();
          chain.MPTQTOpd = MPTQTOp.getOperand(0);
          chain.scale = MPTQTOp.getOperand(1);
          chain.zeroPoint = MPTQTOp.getOperand(2);
        }
        // either a dequant was found or chain broken, so break loop
        break;
      }

      // if tracing this operand was successful, add it to operandChains.
      if (chain.MPTQTOpd)
        operandChains.push_back(std::move(chain));
    }

    // Continuing the rewriting with only some of the operandsToQuantize traced
    // successfully is possible but leads to "half-quantized" ops which are
    // expected to cause problems in later lowering steps. We opt out of
    // treating these cases for now.
    if (operandChains.size() !=
        std::size(QuantInfo<SrcOp>::operandsToQuantize)) {
      if (!operandChains.empty())
        op.emitWarning("Partially traced quantized operands. This op will "
                       "remain in QDQ form.");
      return rewriter.notifyMatchFailure(
          op, "did not find a complete quantized chain for all operands");
    }

    for (auto &&[i, chain] : llvm::enumerate(operandChains)) {
      // rewrite stack
      Value oldOpd = chain.MPTQTOpd;
      Type intDType =
          cast<ValueTensorType>(chain.MPTQTOpd.getType()).getOptionalDtype();
      while (!chain.commutingOpStack.empty()) {
        // get front of the commuting op stack and replace its first operand
        // with oldOpd
        auto currOp = chain.commutingOpStack.top();
        chain.commutingOpStack.pop();
        llvm::SmallVector<Value> currOperands(currOp->getOperands());
        currOperands[0] = oldOpd;
        // pad ops aren't quite commuting, so we include some extra logic to
        // quantize the padding value
        if (isa<Torch::AtenPadOp, Torch::AtenConstantPadNdOp>(currOp)) {
          Value floatPadValue = currOperands.back();
          Value quantPadValue;
          if (isa<Torch::NoneType>(floatPadValue.getType()))
            quantPadValue =
                AtenFloatScalarOp::create(rewriter, loc, chain.zeroPoint);
          else {
            floatPadValue =
                AtenFloatScalarOp::create(rewriter, loc, floatPadValue);
            quantPadValue = Torch::AtenDivFloatOp::create(
                rewriter, loc, floatPadValue, chain.scale);
            quantPadValue = Torch::AtenAddFloatIntOp::create(
                rewriter, loc, quantPadValue, chain.zeroPoint);
          }
          // clamp pad value to qint range
          if (auto intType = dyn_cast<mlir::IntegerType>(intDType)) {
            bool isSigned = intType.isSignedInteger();
            int64_t width = intType.getWidth();
            assert(width < 64 &&
                   "quantized int bitwidth should be less than 64");
            int64_t minInt = isSigned ? -(1 << (width - 1)) : 0;
            int64_t maxInt = isSigned ? -minInt - 1 : ((1 << width) - 1);
            Value minQValueFloat = ConstantFloatOp::create(
                rewriter, loc, rewriter.getF64FloatAttr(minInt));
            Value maxQValueFloat = ConstantFloatOp::create(
                rewriter, loc, rewriter.getF64FloatAttr(maxInt));
            SmallVector<int64_t> emptyShape;
            auto floatTensorType = rewriter.getType<Torch::ValueTensorType>(
                emptyShape, rewriter.getF64Type());
            Value quantPadValueTensor = createRank0Tensor(
                rewriter, loc, floatTensorType, quantPadValue);
            Value clampedTensor = Torch::AtenClampOp::create(
                rewriter, loc, floatTensorType, quantPadValueTensor,
                minQValueFloat, maxQValueFloat);
            quantPadValue = Torch::AtenItemOp::create(
                rewriter, loc, rewriter.getType<Torch::FloatType>(),
                clampedTensor);
          }
          // quantPadValue is a float, but will get converted/truncated
          currOperands.back() = quantPadValue;
        }
        // get new result type
        auto oldType = cast<ValueTensorType>(currOp->getResultTypes()[0]);
        auto intType =
            rewriter.getType<ValueTensorType>(oldType.getSizes(), intDType);
        // rewrite currOp to have new operands and result type
        // store this as oldOpd for next loop
        oldOpd = rewriter
                     .create(loc, (currOp->getName()).getIdentifier(),
                             currOperands, intType, currOp->getAttrs())
                     ->getResult(0);
      }

      // stack is empty, so oldOpd is now the corrected verion of the
      // SrcOp's original operand
      // convert operand -> SrcOp to oldOpd -> newMPTQTOp -> SrcOp
      auto MPTQTOperands = chain.dequantOpd.getDefiningOp()->getOperands();
      auto qTorchType =
          cast<ValueTensorType>(chain.dequantOpd.getType()).getOptionalDtype();
      auto newMPTQTType = rewriter.getType<ValueTensorType>(
          cast<ValueTensorType>(operands[i].getType()).getSizes(), qTorchType);
      operands[i] = Aten_MakePerTensorQuantizedTensorOp::create(
          rewriter, loc, newMPTQTType, oldOpd, MPTQTOperands[1],
          MPTQTOperands[2]);
    }

    rewriter.replaceOpWithNewOp<SrcOp>(op, op.getType(), operands);
    return success();
  }
};

template <typename SrcOp> class QuantizeBias : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> operands(op->getOperands());
    if (operands.size() < 3)
      return failure();

    Value lhsScale;
    if (auto qLhs =
            operands[0].getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>())
      lhsScale = qLhs.getScale();

    Value rhsScale;
    if (auto qRhs =
            operands[1].getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>())
      rhsScale = qRhs.getScale();

    if (!rhsScale || !lhsScale)
      return failure();

    auto resultTy = cast<ValueTensorType>(op.getType());
    if (!isa<mlir::FloatType>(resultTy.getDtype()))
      return failure();

    Value bias = operands[2];
    auto biasTy = dyn_cast<ValueTensorType>(bias.getType());

    if (biasTy) {
      auto biasETy = biasTy.getOptionalDtype();
      if (!biasETy || !isa<mlir::FloatType>(biasETy))
        return failure();
    }

    Value biasScale = AtenMulFloatOp::create(
        rewriter, op.getLoc(), lhsScale.getType(), lhsScale, rhsScale);

    Value zero = Torch::ConstantIntOp::create(
        rewriter, op.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    auto qi32Ty = rewriter.getType<QInt32Type>();

    if (biasTy) {
      auto newBiasTy =
          rewriter.getType<ValueTensorType>(biasTy.getOptionalSizes(), qi32Ty);
      Value dtype = getDtypeIntValueForType(rewriter, op.getLoc(), qi32Ty);
      bias = AtenQuantizePerTensorOp::create(rewriter, op.getLoc(), newBiasTy,
                                             bias, biasScale, zero, dtype);
      bias = AtenIntReprOp::create(
          rewriter, op.getLoc(),
          rewriter.getType<ValueTensorType>(
              biasTy.getOptionalSizes(),
              rewriter.getIntegerType(32, IntegerType::Signed)),
          bias);
      operands[2] = bias;
    }

    auto convTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(),
        rewriter.getIntegerType(32, IntegerType::Signed));
    auto conv = SrcOp::create(rewriter, op.getLoc(), convTy, operands);

    auto convQTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto makeOut = Aten_MakePerTensorQuantizedTensorOp::create(
        rewriter, op.getLoc(), convQTy, conv, biasScale, zero);
    rewriter.replaceOpWithNewOp<AtenDequantizeTensorOp>(op, op.getType(),
                                                        makeOut);

    return success();
  }
};

template <typename SrcOp>
class QuantizeAccumulator : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);

    auto resultTy = dyn_cast_or_null<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype())
      return failure();

    Type resultETy = resultTy.getDtype();
    if (!isa<mlir::FloatType>(resultETy))
      return failure();

    Value lhsScale;
    if (auto defining =
            lhs.template getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      lhsScale = defining.getScale();
    }

    Value rhsScale;
    if (auto defining =
            rhs.template getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
      rhsScale = defining.getScale();
    }

    if (!lhsScale || !rhsScale)
      return failure();

    // Quantize the bias input to the expected result:
    Value zero = Torch::ConstantIntOp::create(
        rewriter, op.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    auto qi32Ty = rewriter.getType<QInt32Type>();
    Value biasScale = AtenMulFloatOp::create(
        rewriter, op.getLoc(), lhsScale.getType(), lhsScale, rhsScale);

    // Update the quantied type:
    llvm::SmallVector<Value> operands(op.getOperands());

    auto newResultTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto conv = SrcOp::create(rewriter, op.getLoc(), newResultTy, operands);

    // Attach the quantize information to the resulting qint32:
    auto intReprTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(),
        rewriter.getIntegerType(32, IntegerType::Signed));
    auto intRepr =
        AtenIntReprOp::create(rewriter, op.getLoc(), intReprTy, conv);

    auto quantTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto quant = Aten_MakePerTensorQuantizedTensorOp::create(
        rewriter, op.getLoc(), quantTy, intRepr, biasScale, zero);
    auto dequant =
        AtenDequantizeTensorOp::create(rewriter, op.getLoc(), resultTy, quant);
    rewriter.replaceOp(op, dequant);

    return success();
  }
};

// Use for ops which do not manipulate scale/zero point of an input.
template <typename SrcOp>
class QuantizeResultLikeOperand : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> operands(op->getOperands());
    Value input = operands[0];

    auto inputType = dyn_cast_or_null<ValueTensorType>(input.getType());
    if (!inputType || !inputType.hasDtype())
      return failure();
    auto qDtype = inputType.getDtype();

    auto resultTy = dyn_cast_or_null<ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasDtype())
      return failure();

    Type resultETy = resultTy.getDtype();
    if (!isa<mlir::FloatType>(resultETy))
      return failure();

    Value inputScale, inputZeroPoint;
    Type definingOpInputType;
    if (auto defining = input.template getDefiningOp<
                        Aten_MakePerTensorQuantizedTensorOp>()) {
      inputScale = defining.getScale();
      inputZeroPoint = defining.getZeroPoint();
      definingOpInputType = defining.getSelf().getType();
    }

    auto inputIntReprType =
        dyn_cast_or_null<ValueTensorType>(definingOpInputType);
    if (!inputScale || !inputZeroPoint || !inputIntReprType ||
        !inputIntReprType.hasDtype())
      return failure();
    auto intReprDtype = inputIntReprType.getDtype();

    // set SrcOp type to use quantized dtype from input
    auto newResultTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qDtype);
    auto newResult =
        SrcOp::create(rewriter, op.getLoc(), newResultTy, operands);

    // int repr to get non quantized int type result
    auto intReprTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(), intReprDtype);
    auto intRepr =
        AtenIntReprOp::create(rewriter, op.getLoc(), intReprTy, newResult);

    // requantize so the scale and zero-point info can be attached
    auto quantTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qDtype);
    auto quant = Aten_MakePerTensorQuantizedTensorOp::create(
        rewriter, op.getLoc(), quantTy, intRepr, inputScale, inputZeroPoint);

    // dequant back to original dtype
    auto dequant =
        AtenDequantizeTensorOp::create(rewriter, op.getLoc(), resultTy, quant);
    rewriter.replaceOp(op, dequant);
    return success();
  }
};

template <typename SrcOp> class RemoveUnused : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    if (result.use_empty()) {
      op.erase();
      return success();
    }
    return failure();
  }
};

class FuseQuantizedOpsPass : public FuseQuantizedOpsBase<FuseQuantizedOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<
        RemoveUnused<AtenDequantizeSelfOp>,
        RemoveUnused<AtenDequantizeTensorOp>,
        RemoveUnused<AtenQuantizePerTensorOp>,
        RemoveUnused<Aten_MakePerTensorQuantizedTensorOp>,
        RemoveUnused<AtenTransposeIntOp>, RemoveUnused<AtenSliceTensorOp>,
        RemoveUnused<AtenReshapeOp>, RemoveUnused<PrimsCollapseOp>,
        RemoveUnused<AtenViewOp>, RemoveUnused<AtenPadOp>,
        RemoveUnused<AtenConstantPadNdOp>,
        QuantizeOperandsPastCommutingOps<AtenConvolutionOp, 5>,
        QuantizeOperandsPastCommutingOps<AtenReluOp, 0>,
        QuantizeOperandsPastCommutingOps<AtenMatmulOp, 2>,
        QuantizeOperandsPastCommutingOps<AtenMmOp, 4>,
        QuantizeAccumulator<AtenMmOp>, QuantizeAccumulator<AtenMatmulOp>,
        QuantizeResultLikeOperand<AtenReluOp>, QuantizeBias<AtenConvolutionOp>>(
        context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createFuseQuantizedOpsPass() {
  return std::make_unique<FuseQuantizedOpsPass>();
}
