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
template <typename SrcOp>
class QuantizeOperands : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Value> operands(op->getOperands());

    bool dequanted = false;
    auto f = [&dequanted](Value operand) {
      if (auto dequant = operand.getDefiningOp<AtenDequantizeTensorOp>()) {
        operand = dequant.getOperand();
        dequanted = true;
      }
      if (auto dequant = operand.getDefiningOp<AtenDequantizeSelfOp>()) {
        operand = dequant.getOperand();
        dequanted = true;
      }
      return operand;
    };

    for (unsigned i : QuantInfo<SrcOp>::operandsToQuantize) {
      operands[i] = f(operands[i]);
    }

    if (!dequanted) {
      return rewriter.notifyMatchFailure(op, "no dequantizations found");
    }

    rewriter.replaceOpWithNewOp<SrcOp>(op, op.getType(), operands);
    return success();
  }
};

template <typename SrcOp>
class QuantizeTransposedOperands : public OpRewritePattern<SrcOp> {
public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {

    llvm::SmallVector<Value> operands(op->getOperands());
    unsigned numOperands = operands.size();
    bool dequanted = false;
    for (unsigned i = 0; i < numOperands; i++) {
      if (auto trans = operands[i].getDefiningOp<AtenTransposeIntOp>()) {
        auto transOperands = trans.getOperands();
        Value dequantOperand;
        if (auto dequant =
                transOperands[0].getDefiningOp<AtenDequantizeSelfOp>()) {
          dequantOperand = dequant.getOperand();
          if (auto quant =
                  dequantOperand
                      .getDefiningOp<Aten_MakePerTensorQuantizedTensorOp>()) {
            auto quantOperands = quant.getOperands();
            auto qType = quantOperands[0]
                             .getType()
                             .cast<ValueTensorType>()
                             .getOptionalDtype();
            auto torchQType =
                quant.getType().cast<ValueTensorType>().getOptionalDtype();
            auto transQTy =
                rewriter.getType<ValueTensorType>(trans.getResult()
                                                      .getType()
                                                      .cast<ValueTensorType>()
                                                      .getOptionalSizes(),
                                                  qType);
            auto newQuantTy =
                rewriter.getType<ValueTensorType>(trans.getResult()
                                                      .getType()
                                                      .cast<ValueTensorType>()
                                                      .getOptionalSizes(),
                                                  torchQType);
            Value newTrans = rewriter.create<AtenTransposeIntOp>(
                op.getLoc(), transQTy, quantOperands[0], transOperands[1],
                transOperands[2]);
            Value newQuant =
                rewriter.create<Aten_MakePerTensorQuantizedTensorOp>(
                    op.getLoc(), newQuantTy, newTrans, quantOperands[1],
                    quantOperands[2]);
            operands[i] = newQuant;
            dequanted = true;
          }
        }
      }
    }
    if (!dequanted) {
      return rewriter.notifyMatchFailure(
          op, "no dequantized transpose inputs found.");
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
    auto biasTy = bias.getType().dyn_cast<ValueTensorType>();

    if (biasTy) {
      auto biasETy = biasTy.getOptionalDtype();
      if (!biasETy || !isa<mlir::FloatType>(biasETy))
        return failure();
    }

    Value biasScale = rewriter.create<AtenMulFloatOp>(
        op.getLoc(), lhsScale.getType(), lhsScale, rhsScale);

    Value zero = rewriter.create<Torch::ConstantIntOp>(
        op.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    auto qi32Ty = rewriter.getType<QInt32Type>();

    if (biasTy) {
      auto newBiasTy =
          rewriter.getType<ValueTensorType>(biasTy.getOptionalSizes(), qi32Ty);
      Value dtype = getDtypeIntValueForType(rewriter, op.getLoc(), qi32Ty);
      bias = rewriter.create<AtenQuantizePerTensorOp>(
          op.getLoc(), newBiasTy, bias, biasScale, zero, dtype);
      bias = rewriter.create<AtenIntReprOp>(
          op.getLoc(),
          rewriter.getType<ValueTensorType>(
              biasTy.getOptionalSizes(),
              rewriter.getIntegerType(32, IntegerType::Signed)),
          bias);
      operands[2] = bias;
    }

    auto convTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(),
        rewriter.getIntegerType(32, IntegerType::Signed));
    auto conv = rewriter.create<SrcOp>(op.getLoc(), convTy, operands);

    auto convQTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto makeOut = rewriter.create<Aten_MakePerTensorQuantizedTensorOp>(
        op.getLoc(), convQTy, conv, biasScale, zero);
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
    Value zero = rewriter.create<Torch::ConstantIntOp>(
        op.getLoc(), rewriter.getType<Torch::IntType>(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0));

    auto qi32Ty = rewriter.getType<QInt32Type>();
    Value biasScale = rewriter.create<AtenMulFloatOp>(
        op.getLoc(), lhsScale.getType(), lhsScale, rhsScale);

    // Update the quantied type:
    llvm::SmallVector<Value> operands(op.getOperands());

    auto newResultTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto conv = rewriter.create<SrcOp>(op.getLoc(), newResultTy, operands);

    // Attach the quantize information to the resulting qint32:
    auto intReprTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(),
        rewriter.getIntegerType(32, IntegerType::Signed));
    auto intRepr = rewriter.create<AtenIntReprOp>(op.getLoc(), intReprTy, conv);

    auto quantTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qi32Ty);
    auto quant = rewriter.create<Aten_MakePerTensorQuantizedTensorOp>(
        op.getLoc(), quantTy, intRepr, biasScale, zero);
    auto dequant =
        rewriter.create<AtenDequantizeTensorOp>(op.getLoc(), resultTy, quant);
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
    auto newResult = rewriter.create<SrcOp>(op.getLoc(), newResultTy, operands);

    // int repr to get non quantized int type result
    auto intReprTy = rewriter.getType<ValueTensorType>(
        resultTy.getOptionalSizes(), intReprDtype);
    auto intRepr =
        rewriter.create<AtenIntReprOp>(op.getLoc(), intReprTy, newResult);

    // requantize so the scale and zero-point info can be attached
    auto quantTy =
        rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qDtype);
    auto quant = rewriter.create<Aten_MakePerTensorQuantizedTensorOp>(
        op.getLoc(), quantTy, intRepr, inputScale, inputZeroPoint);

    // dequant back to original dtype
    auto dequant =
        rewriter.create<AtenDequantizeTensorOp>(op.getLoc(), resultTy, quant);
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
        RemoveUnused<AtenTransposeIntOp>, QuantizeOperands<AtenConvolutionOp>,
        QuantizeOperands<AtenMatmulOp>, QuantizeOperands<AtenReluOp>,
        QuantizeTransposedOperands<AtenMatmulOp>,
        QuantizeAccumulator<AtenMatmulOp>, QuantizeOperands<AtenMmOp>,
        QuantizeTransposedOperands<AtenMmOp>, QuantizeAccumulator<AtenMmOp>,
        QuantizeResultLikeOperand<AtenReluOp>, QuantizeBias<AtenConvolutionOp>>(
        context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
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
