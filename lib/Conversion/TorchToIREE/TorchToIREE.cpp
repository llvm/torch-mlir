//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToIREE/TorchToIREE.h"

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "npcomp/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

//===----------------------------------------------------------------------===//
// The patterns
//===----------------------------------------------------------------------===//

namespace {
class ConvertPrimListConstructOp
    : public OpConversionPattern<PrimListConstructOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimListConstructOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(op.getType());
    auto capacity =
        rewriter.create<ConstantIndexOp>(op.getLoc(), op->getNumOperands());
    auto ireeList =
        rewriter.replaceOpWithNewOp<iree::ListCreateOp>(op, type, capacity);
    for (int i = 0, e = operands.size(); i != e; ++i) {
      auto index = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
      rewriter.create<iree::ListSetOp>(op.getLoc(), ireeList, index,
                                       operands[i]);
    }
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace {
class ConvertTorchToIREE : public ConvertTorchToIREEBase<ConvertTorchToIREE> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<iree::IREEDialect>();
    target.addLegalDialect<StandardOpsDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    patterns.add<ConvertPrimListConstructOp>(typeConverter, context);
    target.addIllegalOp<PrimListConstructOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToIREEPass() {
  return std::make_unique<ConvertTorchToIREE>();
}
