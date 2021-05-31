//===- ReduceOpVariants.cpp --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

namespace {
// Convert value semantic ops operating on mutable arrays to instead operate on
// immutable tensors.
class ConvertToImmutableTensors : public RewritePattern {
public:
  ConvertToImmutableTensors(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<Torch::OpTrait::HasValueSemantics>())
      return rewriter.notifyMatchFailure(op, "does not have value semantics");

    rewriter.updateRootInPlace(op, [&]() {
      // Convert all operands.
      SmallVector<Value> newOperands;
      for (OpOperand &opOperand : op->getOpOperands()) {
        auto ndArrayType =
            opOperand.get().getType().dyn_cast<Numpy::NdArrayType>();
        if (!ndArrayType)
          continue;
        opOperand.set(rewriter.create<Numpy::CopyToTensorOp>(
            op->getLoc(), ndArrayType.toTensorType(), opOperand.get()));
      }
      // Convert all results.
      rewriter.setInsertionPointAfter(op);
      for (Value result : op->getResults()) {
        auto ndArrayType = result.getType().dyn_cast<Numpy::NdArrayType>();
        if (!ndArrayType)
          continue;
        auto createArray = rewriter.create<Numpy::CreateArrayFromTensorOp>(
            op->getLoc(), result.getType(), result);
        result.replaceAllUsesExcept(createArray, createArray);
        result.setType(ndArrayType.toTensorType());
      }
    });
    return success();
  }
};
} // namespace

namespace {
// Reduce the "trailing underscore inplace variant" to the value semantic
// variant + an overwrite of the original "self" argument.
class ReduceTrailingUnderscoreInplaceVariant : public RewritePattern {
public:
  ReduceTrailingUnderscoreInplaceVariant(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<Torch::OpTrait::IsTrailingUnderscoreInplaceVariant>())
      return rewriter.notifyMatchFailure(op, "is not trailing_ variant");

    SmallVector<StringRef> fragments;
    llvm::SplitString(op->getName().getStringRef(), fragments, ".");
    assert(fragments.size() >= 3 && fragments[2].endswith("_") &&
           "IsTrailingUnderscoreInplaceVariant incorrectly applied");
    fragments[2] = fragments[2].drop_back();
    std::string noUnderscoreName = llvm::join(fragments, ".");

    OperationState state(op->getLoc(), noUnderscoreName);
    state.addTypes(op->getResultTypes());
    state.addOperands(op->getOperands());
    state.addAttributes(op->getAttrDictionary().getValue());
    // Note: No successors or regions. Torch JIT operators don't have any.
    assert(op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
           "Torch JIT operators shouldn't have regions or successors");

    Operation *newOp = rewriter.createOperation(state);
    auto tensor = rewriter.create<Numpy::CopyToTensorOp>(
        op->getLoc(),
        newOp->getResult(0).getType().cast<Numpy::NdArrayType>().toTensorType(),
        newOp->getResult(0));
    rewriter.create<Numpy::OverwriteArrayOp>(op->getLoc(), tensor,
                                             op->getOperand(0));
    rewriter.replaceOp(op, op->getOperand(0));

    return success();
  }
};
} // namespace

namespace {
class ReduceOpVariantsPass : public ReduceOpVariantsBase<ReduceOpVariantsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertToImmutableTensors>(context);
    patterns.add<ReduceTrailingUnderscoreInplaceVariant>(context);

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      if (op->hasTrait<Torch::OpTrait::HasValueSemantics>()) {
        auto isNdArray = [](Type t) { return t.isa<Numpy::NdArrayType>(); };
        return llvm::none_of(op->getOperandTypes(), isNdArray) &&
               llvm::none_of(op->getResultTypes(), isNdArray);
      }
      if (op->hasTrait<Torch::OpTrait::IsTrailingUnderscoreInplaceVariant>()) {
        return false;
      }
      return true;
    });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createReduceOpVariantsPass() {
  return std::make_unique<ReduceOpVariantsPass>();
}
