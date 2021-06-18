//===- MaximizeValueSemantics.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

class AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users;
    // See if our limited form of analysis is even applicatble.
    for (Operation *user : copy.getResult().getUsers()) {
      // We can only analyze within a single basic block.
      if (user->getBlock() != copy->getBlock())
        return failure();
      // We can only analyze these ops.
      if (!isa<CopyToValueTensorOp, OverwriteTensorOp>(user))
        return failure();
      users.push_back(user);
    }
    // Sort by order in the block, so we can abstractly interpret the ops.
    llvm::sort(users, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    // Do an abstract interpretation within the block.
    // We track the current value tensor that holds the same contents as the
    // non-value tensor at each program point as we walk forward.
    Value currentlyHeldValueTensor = copy.getOperand();
    for (Operation *user : users) {
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(user)) {
        rewriter.replaceOp(copyToValueTensor, {currentlyHeldValueTensor});
      } else if (auto overwriteTensor = dyn_cast<OverwriteTensorOp>(user)) {
        currentlyHeldValueTensor = overwriteTensor.value();
        rewriter.eraseOp(overwriteTensor);
      } else {
        llvm_unreachable("only those ops supported!");
      }
    }
    rewriter.eraseOp(copy);
    return success();
  }
};

class RewriteNonValueTensorNeverMutatedOrAliased
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users;
    // See if our limited form of analysis is even applicatble.
    for (Operation *user : copy.getResult().getUsers()) {
      if (!isa<CopyToValueTensorOp>(user))
        return failure();
      users.push_back(user);
    }
    for (Operation *user : users)
      rewriter.replaceOp(user, copy.getOperand());
    return success();
  }
};

namespace {

class MaximizeValueSemanticsPass
    : public MaximizeValueSemanticsBase<MaximizeValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock,
                    RewriteNonValueTensorNeverMutatedOrAliased>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Torch::createMaximizeValueSemanticsPass() {
  return std::make_unique<MaximizeValueSemanticsPass>();
}
