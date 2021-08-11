//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToSCF/TorchToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "npcomp/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
class ConvertTorchPrimIfYieldOp : public OpConversionPattern<PrimIfYieldOp> {
public:
  using OpConversionPattern<PrimIfYieldOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfYieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
    return success();
  }
};
} // namespace

namespace {
class ConvertTorchPrimIfOp : public OpConversionPattern<PrimIfOp> {
public:
  using OpConversionPattern<PrimIfOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimIfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                newResultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "could not convert PrimIfOp outputs");
    auto scfIf = rewriter.create<scf::IfOp>(
        op->getLoc(), newResultTypes, operands[0], /*withElseRegion=*/true);
    auto inlineIfCase = [&](Region &srcRegion, Region &dstRegion) {
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      rewriter.eraseBlock(&dstRegion.back());
    };
    inlineIfCase(op.thenRegion(), scfIf.thenRegion());
    inlineIfCase(op.elseRegion(), scfIf.elseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
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

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToSCFPass() {
  return std::make_unique<ConvertTorchToSCF>();
}
