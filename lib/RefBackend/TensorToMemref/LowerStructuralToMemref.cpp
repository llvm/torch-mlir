//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "npcomp/RefBackend/RefBackend.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/RefBackend/IR/RefBackendOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;

//===----------------------------------------------------------------------===//
// Generic "update the types according to the type converter" patterns.
//
// TODO: These should be upstreamed. There's nothing specific to memref type
// conversion about them.
//===----------------------------------------------------------------------===//

namespace {
// This is a type conversion similar to CallOpSignatureConversion.
class LowerIfOpTypes : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 6> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    rewriter.updateRootInPlace(op, [&] {
      for (auto t : llvm::zip(op.getResults(), newResultTypes))
        std::get<0>(t).setType(std::get<1>(t));
    });
    return success();
  }
};
} // namespace

namespace {
// This is a type conversion similar to CallOpSignatureConversion.
class LowerForOpTypes : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 6> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    rewriter.updateRootInPlace(op, [&] {
      for (auto t : llvm::zip(op.getResults(), newResultTypes))
        std::get<0>(t).setType(std::get<1>(t));
      auto bodyArgs = op.getBody()->getArguments();
      for (auto t : llvm::zip(llvm::drop_begin(bodyArgs, 1), newResultTypes))
        std::get<0>(t).setType(std::get<1>(t));
    });
    return success();
  }
};
} // namespace

namespace {
// This is a type conversion similar to CallOpSignatureConversion.
class LowerSelectOpTypes : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SelectOp::Adaptor adaptor(operands);
    rewriter.updateRootInPlace(
        op, [&] { op.getResult().setType(adaptor.true_value().getType()); });
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Further lowerings.
//===----------------------------------------------------------------------===//

namespace {
class LowerTensorToMemrefOp
    : public OpConversionPattern<refback::TensorToMemrefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(refback::TensorToMemrefOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    refback::TensorToMemrefOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};
} // namespace

namespace {
class LowerMemrefToTensorOp
    : public OpConversionPattern<refback::MemrefToTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(refback::MemrefToTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    refback::MemrefToTensorOp::Adaptor adaptor(operands);
    rewriter.replaceOp(op, op.memref());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

namespace {
class LowerStructuralToMemref
    : public LowerStructuralToMemrefBase<LowerStructuralToMemref> {
  void runOnOperation() {
    auto func = getOperation();
    auto *context = &getContext();

    // TODO: move these common type conversions to somewhere common.
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

    // All ops whose results are not tensor types are legal.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return llvm::all_of(op->getResultTypes(),
                          [](Type type) { return !type.isa<TensorType>(); });
    });

    populateFuncOpTypeConversionPattern(patterns, context, typeConverter);
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    patterns.insert<LowerSelectOpTypes>(typeConverter, context);
    patterns.insert<LowerIfOpTypes>(typeConverter, context);
    patterns.insert<LowerForOpTypes>(typeConverter, context);
    patterns.insert<LowerTensorToMemrefOp>(typeConverter, context);
    patterns.insert<LowerMemrefToTensorOp>(typeConverter, context);
    target.addIllegalOp<refback::TensorToMemrefOp>();

    if (failed(applyFullConversion(func, target, patterns)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createLowerStructuralToMemrefPass() {
  return std::make_unique<LowerStructuralToMemref>();
}
