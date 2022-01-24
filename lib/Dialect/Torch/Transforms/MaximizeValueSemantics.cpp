//===- MaximizeValueSemantics.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
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
} // namespace

namespace {
// Calculate a forward slice starting from a CopyToNonValueTensorOp
// and ending at CopyToValueTensorOp's. If all intervening ops
// are just view-like operations (i.e. no mutation), then we can trivially
// convert them all to value semantics.
class RewriteViewLikeSubgraph
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    // Find a subgraph starting with this CopyToNonValueTensorOp, and
    // terminating at CopyToValueTensorOp's or ReturnOp's, possibly with
    // intervening view-like ops.
    // This also catches the special case of a CopyToNonValueTensorOp that
    // trivially feeds into CopyToValueTensorOp's.
    SmallVector<Operation *> viewLikeOps;
    SmallVector<CopyToValueTensorOp> copyToValueTensorOps;
    SmallVector<mlir::ReturnOp> returnOps;
    auto workList = llvm::to_vector<6>(copy.getResult().getUsers());
    // We currently only support view-like ops with one tensor input and one
    // tensor output, meaning that the tensor use-def chains form a tree.
    // This will not be the case for an op like `torch.aten.view_as`, so
    // we will need to add a set to prune duplicate visitation.
    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(op)) {
        copyToValueTensorOps.push_back(copyToValueTensor);
      } else if (auto returnOp = dyn_cast<mlir::ReturnOp>(op)) {
        returnOps.push_back(returnOp);
      } else if (isa<AtenSqueezeOp, AtenSqueezeDimOp, AtenUnsqueezeOp,
                     AtenFlattenUsingIntsOp, AtenTransposeIntOp,
                     TensorStaticInfoCastOp, AtenBroadcastToOp, AtenToDtypeOp,
                     AtenContiguousOp, AtenPermuteOp, AtenViewOp, AtenExpandOp,
                     AtenFill_ScalarOp, AtenSliceTensorOp, AtenSelectIntOp,
                     AtenTOp>(op)) {
        // AtenContiguousOp might return a view, so this is conservatively
        // correct. We could potentially be more precise and identify the cases
        // that it does not return a view and treat those as having value
        // semantics.
        viewLikeOps.push_back(op);
        llvm::append_range(workList, op->getResult(0).getUsers());
      } else {
        return rewriter.notifyMatchFailure(
            copy, "can only handle these transitive user ops");
      }
    }

    if (copyToValueTensorOps.empty() && viewLikeOps.empty())
      return rewriter.notifyMatchFailure(copy, "no types to change");

    // All uses of `copy` will be updated by the logic below.
    copy.replaceAllUsesWith(copy.getOperand());
    // All CopyToValueTensorOp operands will be changed to the correct type
    // by the logic below.
    for (CopyToValueTensorOp op : copyToValueTensorOps)
      rewriter.replaceOp(op, op.getOperand());
    // Keep track of the original types of any view-like ops, so that we can
    // correctly copy them back to their mlir::ReturnOp's expected types.
    DenseMap<Value, Type> originalTypes;
    for (Operation *op : viewLikeOps) {
      rewriter.updateRootInPlace(op, [&]() {
        if (auto nonValueTensorType =
                op->getResult(0).getType().dyn_cast<NonValueTensorType>()) {
          originalTypes[op->getResult(0)] = nonValueTensorType;
          op->getResult(0).setType(nonValueTensorType.getWithValueSemantics());
        }
      });
    }
    // For ReturnOp's, we need to update the operands to their original types.
    for (mlir::ReturnOp op : returnOps) {
      for (int i = 0, e = op->getNumOperands(); i < e; i++) {
        OpOperand &operand = op->getOpOperand(i);
        auto it = originalTypes.find(operand.get());
        if (it == originalTypes.end())
          continue;
        auto originalType = it->second.cast<BaseTensorType>();
        rewriter.setInsertionPoint(op);
        Value newReturnValue = copyTensorToType(rewriter, op->getLoc(),
                                                originalType, operand.get());
        operand.set(newReturnValue);
      }
    }
    return success();
  }
};
} // namespace

namespace {

class MaximizeValueSemanticsPass
    : public MaximizeValueSemanticsBase<MaximizeValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock,
                    RewriteViewLikeSubgraph>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createMaximizeValueSemanticsPass() {
  return std::make_unique<MaximizeValueSemanticsPass>();
}
