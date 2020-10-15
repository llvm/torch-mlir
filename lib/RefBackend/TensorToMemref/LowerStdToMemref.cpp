//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class LowerExtractElementOp : public OpConversionPattern<ExtractElementOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExtractElementOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ExtractElementOp::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<LoadOp>(op, adaptor.aggregate(),
                                        adaptor.indices());
    return success();
  }
};
} // namespace

namespace {
class LowerTensorFromElementsOp
    : public OpConversionPattern<TensorFromElementsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorFromElementsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    int numberOfElements = op.elements().size();
    auto resultType = MemRefType::get(
        {numberOfElements}, op.getType().cast<TensorType>().getElementType());
    Value result = rewriter.create<AllocOp>(op.getLoc(), resultType);
    for (auto element : llvm::enumerate(op.elements())) {
      Value index =
          rewriter.create<ConstantIndexOp>(op.getLoc(), element.index());
      rewriter.create<StoreOp>(op.getLoc(), element.value(), result, index);
    }
    rewriter.replaceOp(op, {result});
    return success();
  }
};
} // namespace

namespace {
class LowerTensorCastOp : public OpConversionPattern<TensorCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorCastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<MemRefCastOp>(op, resultType, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// TODO: Upstream this.
class LowerStdToMemref : public LowerStdToMemrefBase<LowerStdToMemref> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<refback::RefbackDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;

    OwningRewritePatternList patterns;

    ConversionTarget target(*context);

    target.addLegalDialect<StandardOpsDialect>();

    patterns.insert<LowerExtractElementOp>(typeConverter, context);
    target.addIllegalOp<ExtractElementOp>();
    patterns.insert<LowerTensorFromElementsOp>(typeConverter, context);
    target.addIllegalOp<TensorFromElementsOp>();
    patterns.insert<LowerTensorCastOp>(typeConverter, context);
    target.addIllegalOp<TensorCastOp>();

    if (failed(applyPartialConversion(func, target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerStdToMemrefPass() {
  return std::make_unique<LowerStdToMemref>();
}
