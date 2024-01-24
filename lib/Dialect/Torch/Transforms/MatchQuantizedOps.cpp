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

Type getQuantizedType(MLIRContext *context, Type t) {
  if (t.isSignlessInteger(8))
    return Torch::QUInt8Type::get(context);
  if (t.isInteger(8) || t.isSignedInteger(8))
    return Torch::QInt8Type::get(context);
  if (t.isInteger(32))
    return Torch::QInt32Type::get(context);
  return {};
}

class MatchQuantizeOperator : public OpRewritePattern<OperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(OperatorOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getName() == "torch.quantized_decomposed.quantize_per_tensor") {
      auto resultTy = cast<ValueTensorType>(op.getType(0));
      auto qeTy = getQuantizedType(rewriter.getContext(), resultTy.getDtype());
      if (!qeTy)
        qeTy = resultTy.getDtype();

      auto qTy =
          rewriter.getType<ValueTensorType>(resultTy.getOptionalSizes(), qeTy);
      Value quant = rewriter.create<AtenQuantizePerTensorOp>(
          op.getLoc(), qTy,
          /*self=*/op.getOperand(0), /*scale=*/op.getOperand(1),
          /*zero_point=*/op.getOperand(2), /*dtype=*/op.getOperand(5));

      if (qTy != resultTy) {
        quant = rewriter.create<AtenIntReprOp>(op.getLoc(), resultTy, quant);
      }

      rewriter.replaceOpWithNewOp<AtenClampOp>(
          op, resultTy, quant, op.getOperand(3), op.getOperand(4));
      return success();
    }

    if (op.getName() == "torch.quantized_decomposed.dequantize_per_tensor") {
      auto clamp = rewriter.create<AtenClampOp>(
          op.getLoc(), op.getOperand(0).getType(), op.getOperand(0),
          op.getOperand(3), op.getOperand(4));

      auto clampTy = clamp.getType().cast<Torch::ValueTensorType>();
      if (!clampTy.hasDtype())
        return rewriter.notifyMatchFailure(op,
                                           "dequantization has unknown dtype");

      Type dtype = clampTy.getDtype();
      Type qetype = getQuantizedType(op.getContext(), dtype);
      if (!qetype)
        return rewriter.notifyMatchFailure(op,
                                           "dequantization has unknown qtype");

      Type qTy = Torch::ValueTensorType::get(
          op.getContext(), clampTy.getOptionalSizes(), qetype);
      auto quant = rewriter.create<Aten_MakePerTensorQuantizedTensorOp>(
          op.getLoc(), qTy, clamp, op.getOperand(1), op.getOperand(2));
      rewriter.replaceOpWithNewOp<AtenDequantizeTensorOp>(
          op, op.getResultTypes(), quant);
      return success();
    }

    return failure();
  }
};

class MatchQuantizedCustomOpsPass
    : public MatchQuantizedCustomOpsBase<MatchQuantizedCustomOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<MatchQuantizeOperator>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createMatchQuantizedCustomOpsPass() {
  return std::make_unique<MatchQuantizedCustomOpsPass>();
}
