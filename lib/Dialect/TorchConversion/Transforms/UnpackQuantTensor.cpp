//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class UnpackQuantizedMatmulWeights
    : public OpRewritePattern<ValueTensorLiteralOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ValueTensorLiteralOp constOp,
                                PatternRewriter &rewriter) const override {
    if (!constOp->hasOneUse())
      return failure();

    OpOperand *use = constOp.getResult().use_begin().getOperand();
    auto op = dyn_cast<OperatorOp>(use->getOwner());
    if (!op) {
      return failure();
    }
    if (op.getName().str() != "quant.matmul_rhs_group_quant") {
      return failure();
    }

    if (use->getOperandNumber() != 1) {
      return failure();
    }

    Value rhs = op.getOperand(1);
    Value bitWidth = op.getOperand(4);

    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto constOp = dyn_cast<Torch::ConstantIntOp>(operand.getDefiningOp());
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };
    int unpackedBitWidth;
    if (failed(getConstantIntegerFromDefiningOp(bitWidth, unpackedBitWidth)))
      return failure();

    auto rhsType = rhs.getType().dyn_cast<ValueTensorType>();
    if (!rhsType)
      return failure();

    if (!rhsType.hasDtype())
      return failure();

    Type dType = rhsType.getDtype();
    int dTypeWidth = dType.getIntOrFloatBitWidth();
    if (dTypeWidth == unpackedBitWidth)
      return failure();

    if (!rhsType.hasSizes())
      return failure();

    SmallVector<int64_t> tensorShape(rhsType.getSizes());
    if (tensorShape.back() == kUnknownSize)
      return failure();
    int packRatio = dTypeWidth / unpackedBitWidth;

    tensorShape[tensorShape.size() - 1] *= packRatio;
    Type unpackedElementType;
    if (dType.isSignedInteger())
      unpackedElementType = rewriter.getIntegerType(unpackedBitWidth, true);
    else
      unpackedElementType = rewriter.getIntegerType(unpackedBitWidth, false);
    ValueTensorType newRhsType = ValueTensorType::get(
        rewriter.getContext(), tensorShape, unpackedElementType);

    auto elements = constOp.getValueAttr().dyn_cast<DenseIntElementsAttr>();
    if (!elements)
      return failure();

    auto attrType = RankedTensorType::get(tensorShape, unpackedElementType);

    // TODO: Materialize IR that does the conversion from quantized type to
    //       pure integer type which relys on constant evaluation in backends
    auto data = elements.getRawData();
    std::vector<APInt> newData(data.size() * packRatio,
                               APInt(unpackedBitWidth, 0));
    for (int i = 0, e = data.size(); i < e; ++i) {
      auto el = data[i];
      char mask = (1 << unpackedBitWidth) - 1;
      for (int b = 0; b < packRatio; b++) {
        newData[i * packRatio + b] =
            APInt(unpackedBitWidth, (el & mask) >> (unpackedBitWidth * b));
        mask = mask << unpackedBitWidth;
      }
    }
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(
        constOp, newRhsType,
        DenseElementsAttr::get(attrType, ArrayRef<APInt>(newData)));
    return success();
  }
};
} // namespace

namespace {
class UnpackQuantTensorPass
    : public TorchConversion::UnpackQuantTensorBase<UnpackQuantTensorPass> {
  using UnpackQuantTensorBase<UnpackQuantTensorPass>::UnpackQuantTensorBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<Torch::TorchDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UnpackQuantizedMatmulWeights>(context);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::torch::TorchConversion::createUnpackQuantTensorPass() {
  return std::make_unique<UnpackQuantTensorPass>();
}
