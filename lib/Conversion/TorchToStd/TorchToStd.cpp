//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToStd/TorchToStd.h"

#include "../PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchUtils.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#torch operators), which is in the 100s.

namespace {
// Note: Confusingly, ATen's "dim" means "number of dimensions" which is what
// MLIR calls "rank".
class ConvertAtenDimOp : public OpConversionPattern<AtenDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = rewriter.create<RankOp>(op->getLoc(), operands[0]);
    rewriter.replaceOpWithNewOp<IndexCastOp>(op, op.getType(), rank);
    return success();
  }
};
} // namespace

LogicalResult convertNeIntOp(AtenNeIntOp op, PatternRewriter &rewriter) {
  auto i1 = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::ne,
                                    op->getOperand(0), op->getOperand(1));
  rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(op, op.getType(), i1);
  return success();
}

LogicalResult convertGtIntOp(AtenGtIntOp op, PatternRewriter &rewriter) {
  auto i1 = rewriter.create<CmpIOp>(op->getLoc(), CmpIPredicate::sgt,
                                    op->getOperand(0), op->getOperand(1));
  rewriter.replaceOpWithNewOp<Basicpy::BoolCastOp>(op, op.getType(), i1);
  return success();
}

LogicalResult convertTensorOp(TensorOp op, PatternRewriter &rewriter) {
  auto constant = rewriter.create<ConstantOp>(op->getLoc(), op.value());
  auto vtensor = rewriter.create<FromBuiltinTensorOp>(op->getLoc(), constant);
  Value result = copyTensorToType(rewriter, op->getLoc(),
                                  op.getType().cast<BaseTensorType>(), vtensor);
  rewriter.replaceOp(op, {result});
  return success();
}

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToStd : public ConvertTorchToStdBase<ConvertTorchToStd> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, Basicpy::BasicpyDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, StandardOpsDialect,
                           Basicpy::BasicpyDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    setupValueTensorToBuiltinTensorConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenDimOp>();
    patterns.add<ConvertAtenDimOp>(typeConverter, context);
    target.addIllegalOp<AtenNeIntOp>();
    patterns.add(convertNeIntOp);
    target.addIllegalOp<AtenGtIntOp>();
    patterns.add(convertGtIntOp);
    target.addIllegalOp<TensorOp>();
    patterns.add(convertTensorOp);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToStdPass() {
  return std::make_unique<ConvertTorchToStd>();
}
