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
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendDialect.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendOps.h"

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
class LowerTensorLoadOp : public OpConversionPattern<TensorLoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TensorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// TODO: Upstream this.
class LowerStdToMemref : public LowerStdToMemrefBase<LowerStdToMemref> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<refback::RefBackendDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    typeConverter.addSourceMaterialization([](OpBuilder &builder,
                                              RankedTensorType type,
                                              ValueRange inputs, Location loc) {
      assert(inputs.size() == 1);
      assert(inputs[0].getType().isa<MemRefType>());
      return (Value)builder.create<refback::MemrefToTensorOp>(loc, type,
                                                              inputs[0]);
    });
    typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                              MemRefType type,
                                              ValueRange inputs, Location loc) {
      assert(inputs.size() == 1);
      assert(inputs[0].getType().isa<RankedTensorType>());
      return (Value)builder.create<refback::TensorToMemrefOp>(loc, type,
                                                              inputs[0]);
    });

    OwningRewritePatternList patterns;

    ConversionTarget target(*context);

    target.addLegalDialect<StandardOpsDialect>();

    // The casting ops are introduced by the type converter, so they must be
    // legal.
    target.addLegalOp<refback::MemrefToTensorOp>();
    target.addLegalOp<refback::TensorToMemrefOp>();

    patterns.insert<LowerExtractElementOp>(typeConverter, context);
    target.addIllegalOp<ExtractElementOp>();
    patterns.insert<LowerTensorFromElementsOp>(typeConverter, context);
    target.addIllegalOp<TensorFromElementsOp>();
    patterns.insert<LowerTensorCastOp>(typeConverter, context);
    target.addIllegalOp<TensorCastOp>();
    patterns.insert<LowerTensorLoadOp>(typeConverter, context);
    target.addIllegalOp<TensorLoadOp>();

    if (failed(applyPartialConversion(func, target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerStdToMemrefPass() {
  return std::make_unique<LowerStdToMemref>();
}
