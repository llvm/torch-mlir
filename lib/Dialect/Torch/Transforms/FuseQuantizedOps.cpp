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

    operands[0] = f(operands[0]);
    operands[1] = f(operands[1]);

    if (!dequanted) {
      return rewriter.notifyMatchFailure(op, "no dequantizations found");
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
    if (!resultETy.isa<mlir::FloatType>())
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
    patterns
        .insert<RemoveUnused<AtenDequantizeSelfOp>,
                RemoveUnused<AtenDequantizeTensorOp>,
                RemoveUnused<AtenQuantizePerTensorOp>,
                QuantizeOperands<AtenConvolutionOp>, QuantizeOperands<AtenMmOp>,
                QuantizeAccumulator<AtenMmOp>, QuantizeBias<AtenConvolutionOp>>(
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
