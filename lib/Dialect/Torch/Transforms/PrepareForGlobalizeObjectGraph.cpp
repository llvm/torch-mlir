//===- PrepareForGlobalizeObjectGraph.cpp ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
class ConvertPrimCallMethodToCall : public OpRewritePattern<PrimCallMethodOp> {
public:
  ConvertPrimCallMethodToCall(MLIRContext *context, SymbolTable &symbolTable)
      : OpRewritePattern(context), symbolTable(symbolTable) {}
  LogicalResult matchAndRewrite(PrimCallMethodOp op,
                                PatternRewriter &rewriter) const override {
    auto classType = symbolTable.lookup<ClassTypeOp>(
        op.receiver().getType().cast<NnModuleType>().getClassName());
    FuncOp func;
    for (auto method : classType.getOps<MethodOp>()) {
      if (method.name() == op.name()) {
        func = symbolTable.lookup<FuncOp>(method.function());
        break;
      }
    }
    assert(func);
    rewriter.replaceOpWithNewOp<CallOp>(op, func, op->getOperands());
    return success();
  }

private:
  SymbolTable &symbolTable;
};
} // namespace

namespace {
class EraseUnusedConstantOp : public OpRewritePattern<ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
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
    OwningRewritePatternList patterns;
    patterns.insert<ConvertPrimCallMethodToCall>(context, symbolTable);
    CallIndirectOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<EraseUnusedConstantOp>(context);

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
    target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
      return !op.getType().isa<FunctionType>();
    });
    target.addIllegalOp<CallIndirectOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    OwningRewritePatternList dummyPatterns;

    if (failed(applyFullConversion(getOperation(), target,
                                      std::move(dummyPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::NPCOMP::Torch::createPrepareForGlobalizeObjectGraphPass() {
  return std::make_unique<PrepareForGlobalizeObjectGraphPass>();
}
