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

static bool isViewLikeOp(Operation *op) {
  // AtenContiguousOp might return a view, so this is conservatively
  // correct. We could potentially be more precise and identify the cases
  // that it does not return a view and treat those as having value
  // semantics.
  return isa<AtenBroadcastToOp, AtenContiguousOp, AtenExpandOp,
             AtenFlattenUsingIntsOp, AtenPermuteOp, AtenReshapeOp,
             AtenSelectIntOp, AtenSliceTensorOp, AtenSqueezeDimOp,
             AtenSqueezeOp, AtenTOp, AtenToDtypeOp, AtenTransposeIntOp,
             AtenUnsqueezeOp, AtenViewOp, TensorStaticInfoCastOp>(op);
}

namespace {
class AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  struct InterpretedOps {
    SmallVector<Operation *> copyLikeOps;
    SmallVector<Operation *> viewLikeOps;
    SmallVector<OverwriteTensorContentsOp> overwriteTensorContentsOps;
    Optional<mlir::ReturnOp> returnOp;
  };

  // Check that graph rewriting is possible by doing an abstract
  // interpretation within a single basic block. If rewriting is
  // possible, the interpreted ops are returned split into their
  // respective categories.
  static FailureOr<InterpretedOps>
  abstractlyInterpretSlice(CopyToNonValueTensorOp copyToNonValueTensor,
                           SmallVector<Operation *> nonValueTensorUsers,
                           PatternRewriter &rewriter) {
    // Sort by order in the block, so we can abstractly interpret the ops.
    llvm::sort(nonValueTensorUsers, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    // We track the available aliases at each point as well as split the
    // users into view-like, copy-to-value, and overwrite ops as we walk
    // forward.
    //
    // We also need to track all seen aliases to make sure that we only rewrite
    // those operands of a ReturnOp, if present (a ReturnOp can return tensors
    // from multiple different slices).
    InterpretedOps result;
    result.copyLikeOps.push_back(copyToNonValueTensor);
    DenseSet<Value> availableAliases{copyToNonValueTensor.result()};
    DenseSet<Value> seenAliases{copyToNonValueTensor.result()};
    for (Operation *user : nonValueTensorUsers) {
      if (isViewLikeOp(user)) {
        Value operand = user->getOperand(0);
        if (!availableAliases.contains(operand)) {
          return rewriter.notifyMatchFailure(
              copyToNonValueTensor,
              "operand of view-like op is not a valid tensor alias");
        }

        // View-like ops produce a new alias available to later ops.
        availableAliases.insert(user->getResult(0));
        seenAliases.insert(user->getResult(0));
        result.viewLikeOps.push_back(user);
      } else if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(user)) {
        if (!availableAliases.contains(copyToValueTensor.operand())) {
          return rewriter.notifyMatchFailure(
              copyToNonValueTensor,
              "operand of copyToValueTensorOp is not a valid tensor alias");
        }
        result.copyLikeOps.push_back(copyToValueTensor);
      } else if (auto overwrite = dyn_cast<OverwriteTensorContentsOp>(user)) {
        Value overwritten = overwrite.overwritten();
        if (!availableAliases.contains(overwritten)) {
          return rewriter.notifyMatchFailure(
              copyToNonValueTensor, "overwritten tensor is not a valid alias");
        }

        // To simplify the analysis, we only support the case where the
        // only aliases used after an overwrite are the aliases generated
        // after plus the alias being overwritten.
        availableAliases.clear();
        availableAliases.insert(overwritten);
        result.overwriteTensorContentsOps.push_back(overwrite);
      } else if (auto returnOp = dyn_cast<mlir::ReturnOp>(user)) {
        for (Value operand : returnOp->getOperands()) {
          if (!seenAliases.contains(operand))
            continue;
          if (!availableAliases.contains(operand)) {
            return rewriter.notifyMatchFailure(
                copyToNonValueTensor,
                "operand of ReturnOp is not a valid tensor alias");
          }
        }
        result.returnOp = returnOp;
      } else {
        return rewriter.notifyMatchFailure(
            copyToNonValueTensor,
            "unsupported op encountered during abstract analysis");
      }
    }
    return result;
  }

  // Rewrite slice composed of the interpreted ops so that the slice uses
  // value semantics everywhere.
  static void rewriteSlice(const InterpretedOps &ops,
                           PatternRewriter &rewriter) {

    DenseMap<int, Type> originalReturnTypes;
    if (ops.returnOp.hasValue()) {
      auto returnOp = ops.returnOp.getValue();
      for (auto operand : llvm::enumerate(returnOp->getOperands())) {
        auto type = operand.value().getType();
        if (!type.isa<NonValueTensorType>())
          continue;
        originalReturnTypes[operand.index()] = type;
      }
    }
    // The rewriting for the overwrite op involves replacing all uses of its
    // non-value tensor operand with its value tensor operand. Since the
    // rewriting of other ops can potentially change the non-value tensor
    // operand to a value tensor, this rewriting MUST happen first to avoid
    // wrongly replacing operands that were previously not a view of the
    // overwritten tensor.
    for (OverwriteTensorContentsOp overwrite :
         llvm::reverse(ops.overwriteTensorContentsOps)) {
      Value overwritten = overwrite.overwritten();
      assert(overwritten.getType().dyn_cast<NonValueTensorType>() &&
             "the analysis assumes that overwritten remains a nonValueTensor "
             "throughout the rewriting");
      overwritten.replaceUsesWithIf(
          overwrite.value(), [&](const OpOperand &operand) {
            return !operand.getOwner()->isBeforeInBlock(overwrite);
          });
      rewriter.eraseOp(overwrite);
    }

    for (Operation *copyLikeOp : ops.copyLikeOps)
      rewriter.replaceOp(copyLikeOp, copyLikeOp->getOperand(0));

    // Replace return type of view-like ops with value-semantics type variant.
    for (Operation *viewLikeOp : ops.viewLikeOps) {
      rewriter.updateRootInPlace(viewLikeOp, [&] {
        Value result = viewLikeOp->getResult(0);
        auto resultType = result.getType().dyn_cast<NonValueTensorType>();
        assert(resultType && "all view-like ops considered must have result of "
                             "type `NonValueTensorType` before rewriting");
        result.setType(resultType.getWithValueSemantics());
      });
    }
    if (ops.returnOp.hasValue()) {
      auto returnOp = ops.returnOp.getValue();
      for (int i = 0, e = returnOp->getNumOperands(); i < e; i++) {
        OpOperand &operand = returnOp->getOpOperand(i);
        auto it = originalReturnTypes.find(i);
        if (it == originalReturnTypes.end())
          continue;
        auto originalType = it->second.cast<NonValueTensorType>();
        rewriter.setInsertionPoint(returnOp);
        Value newReturnValue = copyTensorToType(rewriter, returnOp->getLoc(),
                                                originalType, operand.get());
        operand.set(newReturnValue);
      }
    }
  }

  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    // Find a subgraph starting with this CopyToNonValueTensorOp, and
    // terminating at CopyToValueTensorOp's, possibly with intervening view-like
    // ops and overwrites. This also catches the special case of a
    // CopyToNonValueTensorOp that trivially feeds into CopyToValueTensorOp's.
    SmallVector<Operation *> nonValueTensorUsers;
    auto workList = llvm::to_vector(copy.result().getUsers());
    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      if (op->getBlock() != copy->getBlock()) {
        return rewriter.notifyMatchFailure(
            copy, "can only analyze within a single basic block");
      }
      nonValueTensorUsers.push_back(op);

      if (isViewLikeOp(op)) {
        auto isTensor = [](const Value operand) {
          return operand.getType().isa<BaseTensorType>();
        };

        // We currently only support view-like ops with one tensor input and one
        // tensor output, meaning that the tensor use-def chains form a tree.
        // This will not be the case for an op like `torch.aten.view_as`, so
        // we will need to add a set to prune duplicate visitation.
        if (llvm::count_if(op->getOperands(), isTensor) != 1 ||
            llvm::count_if(op->getResults(), isTensor) != 1 ||
            !isTensor(op->getOperand(0)) || !isTensor(op->getResult(0))) {
          return rewriter.notifyMatchFailure(
              copy, "unsupported: view-like ops must have one tensor input and "
                    "one tensor output, and the tensor input/output must be "
                    "the first operand/result");
        }

        llvm::append_range(workList, op->getResult(0).getUsers());
      }
    }

    // Nothing to do if there is just a ReturnOp -- we know that we won't be
    // rewriting anything, since we must preserve the ReturnOp's original type.
    if (llvm::hasSingleElement(nonValueTensorUsers) &&
        isa<mlir::ReturnOp>(nonValueTensorUsers[0])) {
      return failure();
    }

    FailureOr<InterpretedOps> interpretedOps = abstractlyInterpretSlice(
        copy, std::move(nonValueTensorUsers), rewriter);
    if (failed(LogicalResult(interpretedOps)))
      return failure();
    rewriteSlice(*interpretedOps, rewriter);
    return success();
  }
};
} // namespace

namespace {
// Calculate a forward slice starting from a CopyToNonValueTensorOp
// and ending at CopyToValueTensorOp's. If all intervening ops
// are just view-like operations (i.e. no mutation), then we can trivially
// convert them all to value semantics.
// This pattern handles the case where views span multiple basic blocks,
// which is currently not supported by
// `AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock`.
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
      } else if (isViewLikeOp(op)) {
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
