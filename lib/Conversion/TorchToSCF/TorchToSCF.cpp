//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertTorchPrimIfYieldOp : public OpConversionPattern<PrimIfYieldOp> {
public:
  using OpConversionPattern<PrimIfYieldOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfYieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchPrimIfOp : public OpConversionPattern<PrimIfOp> {
public:
  using OpConversionPattern<PrimIfOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                newResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert PrimIfOp outputs");
    auto scfIf = rewriter.create<scf::IfOp>(op->getLoc(), newResultTypes,
                                            adaptor.condition(),
                                            /*withElseRegion=*/true);
    auto inlineIfCase = [&](Region &srcRegion, Region &dstRegion) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      rewriter.eraseBlock(&dstRegion.back());
    };
    inlineIfCase(op.thenRegion(), scfIf.getThenRegion());
    inlineIfCase(op.elseRegion(), scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchPrimLoopOp : public OpConversionPattern<PrimLoopOp> {
public:
  using OpConversionPattern<PrimLoopOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimLoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "in ConvertTorchPrimLoopOp\n";
    SmallVector<Type, 1> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                newResultTypes)))
      return rewriter.notifyMatchFailure(
          op, "could not convert PrimLoopOp outputs");
    llvm::errs() << "upper bound: \t" << op.maxTripCount() << "\n";
    auto lowerBound = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    auto step = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
	  scf::ForOp scfFor = rewriter.create<scf::ForOp>(
        op->getLoc(), lowerBound, op.maxTripCount(), step, op.iterArgsInit());
    llvm::errs() << "loop header:\n";
    scfFor.dump();
    auto inlineForCase = [&](Region &srcRegion, Region &dstRegion) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      rewriter.eraseBlock(&dstRegion.back());
    };
    inlineForCase(op.region(), scfFor.getRegion());
    rewriter.replaceOp(op, scfFor.getResults());
    llvm::errs() << "final loop op:\n";
    op->dump();
    scfFor.getResult(0).dump();
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchPrimLoopConditionOp
    : public OpConversionPattern<PrimLoopConditionOp> {
public:
  using OpConversionPattern<PrimLoopConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimLoopConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "in ConvertTorchPrimLoopConditionOp\n";
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, adaptor.shouldContinue(),
                                                  adaptor.iterArgs());
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchToSCF : public ConvertTorchToSCFBase<ConvertTorchToSCF> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, scf::SCFDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<PrimIfOp>();
    patterns.add<ConvertTorchPrimIfOp>(typeConverter, context);
    target.addIllegalOp<PrimIfYieldOp>();
    patterns.add<ConvertTorchPrimIfYieldOp>(typeConverter, context);
    target.addIllegalOp<PrimLoopOp>();
    patterns.add<ConvertTorchPrimLoopOp>(typeConverter, context);
    target.addIllegalOp<PrimLoopConditionOp>();
    patterns.add<ConvertTorchPrimLoopConditionOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToSCFPass() {
  return std::make_unique<ConvertTorchToSCF>();
}
