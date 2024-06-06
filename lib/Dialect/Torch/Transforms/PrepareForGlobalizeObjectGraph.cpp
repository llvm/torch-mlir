//===- PrepareForGlobalizeObjectGraph.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertPrimCallMethodToCall : public OpRewritePattern<PrimCallMethodOp> {
public:
  ConvertPrimCallMethodToCall(MLIRContext *context, SymbolTable &symbolTable)
      : OpRewritePattern(context), symbolTable(symbolTable) {}
  LogicalResult matchAndRewrite(PrimCallMethodOp op,
                                PatternRewriter &rewriter) const override {
    auto classType = symbolTable.lookup<ClassTypeOp>(
        cast<NnModuleType>(op.getReceiver().getType()).getClassName());
    assert(classType && "malformed module -- missing ClassTypeOp");
    func::FuncOp func;
    for (auto method : classType.getOps<MethodOp>()) {
      if (method.getName() == op.getName()) {
        func = symbolTable.lookup<func::FuncOp>(method.getFunction());
        break;
      }
    }
    assert(func);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, func, op->getOperands());
    return success();
  }

private:
  SymbolTable &symbolTable;
};
} // namespace

namespace {
class EraseUnusedConstantOp : public OpRewritePattern<func::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class PrepareForGlobalizeObjectGraphPass
    : public PrepareForGlobalizeObjectGraphBase<
          PrepareForGlobalizeObjectGraphPass> {
  void runOnOperation() override {

    SymbolTable symbolTable(getOperation());

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertPrimCallMethodToCall>(context, symbolTable);
    func::CallIndirectOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<EraseUnusedConstantOp>(context);

    // Use applyPatternsAndFoldGreedily because the CallIndirectOp folding
    // makes the ConstantOp unused, which does not work with the visitation
    // order of the dialect conversion infrastructure.
    // TODO: Do this with the dialect conversion infrastructure to avoid doing
    // folding as part of this. Or avoid folding during greedy pattern
    // application. See: https://llvm.org/PR49502
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // Do a dummy full conversion to ensure that the program has been converted
    // to the form we want.
    ConversionTarget target(*context);
    target.addIllegalOp<PrimCallMethodOp>();
    target.addDynamicallyLegalOp<func::ConstantOp>(
        [](func::ConstantOp op) { return !isa<FunctionType>(op.getType()); });
    target.addIllegalOp<func::CallIndirectOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet dummyPatterns(context);

    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(dummyPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createPrepareForGlobalizeObjectGraphPass() {
  return std::make_unique<PrepareForGlobalizeObjectGraphPass>();
}
