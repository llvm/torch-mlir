//===- BuiltinTensorize.cpp --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/IR/TorchUtils.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
struct FuncBuiltinTensorizePass
    : public FuncBuiltinTensorizeBase<FuncBuiltinTensorizePass> {
  using FuncBuiltinTensorizeBase<
      FuncBuiltinTensorizePass>::FuncBuiltinTensorizeBase;
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    setupValueTensorToBuiltinTensorConversion(target, typeConverter);

    populateFuncOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createFuncBuiltinTensorizePass() {
  return std::make_unique<FuncBuiltinTensorizePass>();
}

namespace {
// In a finalizing conversion, we know that all `!torch.vtensor` have been
// converted to `tensor`, thus, this op becomes an identity.
class FinalizeToBuiltinTensorOp
    : public OpConversionPattern<ToBuiltinTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToBuiltinTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
// In a finalizing conversion, we know that all `!torch.vtensor` have been
// converted to `tensor`, thus, this op becomes an identity.
class FinalizeFromBuiltinTensorOp
    : public OpConversionPattern<FromBuiltinTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromBuiltinTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

namespace {
struct FinalizingBuiltinTensorizePass
    : public FinalizingBuiltinTensorizeBase<FinalizingBuiltinTensorizePass> {
  using FinalizingBuiltinTensorizeBase<
      FinalizingBuiltinTensorizePass>::FinalizingBuiltinTensorizeBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    setupValueTensorToBuiltinTensorConversion(target, typeConverter);
    target.addIllegalOp<ToBuiltinTensorOp, FromBuiltinTensorOp>();

    patterns.add<FinalizeFromBuiltinTensorOp, FinalizeToBuiltinTensorOp>(
        typeConverter, context);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createFinalizingBuiltinTensorizePass() {
  return std::make_unique<FinalizingBuiltinTensorizePass>();
}
