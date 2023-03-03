//===- MaximizeValueSemantics.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

static Value assertNonValueTensor(Value tensor) {
  assert(tensor.getType().isa<NonValueTensorType>() &&
         "tensor is expected to be a non-value tensor");
  return tensor;
}

// TODO: We currently ignore step information. This can be fixed by using
// a more precise domain like Presburger formulas.
struct AliasSliceInfo {
  // The dimension that the slice is taken along. We currently only support
  // slicing along one dimension. A value of -1 indicates that the tensor is
  // not sliced.
  int64_t dim;
  int64_t start;
  int64_t end;
  // A flag indicating whether the slice is approximated. If it is, we
  // do not perform any more slicing analysis on it.
  bool approximated = false;

  bool isOverlapping(const AliasSliceInfo &other) {
    // If there is no slicing on either side, then the aliases are
    // overlapping as one must be a slice of the other.
    if (dim == -1 || other.dim == -1)
      return true;

    // If the slices are on different dimensions, then the aliases are
    // overlapping.
    if (dim != other.dim)
      return true;

    int64_t start1 = start;
    int64_t start2 = other.start;
    int64_t end1 = end;
    int64_t end2 = other.end;

    // If the slices are disjoint, then the aliases are not overlapping.
    if (end1 <= start2 || end2 <= start1)
      return false;
    // Otherwise, the aliases are overlapping.
    return true;
  }
};

namespace {
class AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Used to represent all of the interpreted ops that have at least
  // one non-value tensor as input or output.
  struct InterpretedOps {
    SmallVector<Operation *> copyLikeOps;
    SmallVector<Operation *> viewLikeOps;
    SmallVector<OverwriteTensorContentsOp> overwriteTensorContentsOps;
    std::optional<mlir::func::ReturnOp> returnOp;
  };

  // Check that graph rewriting is possible by doing an abstract
  // interpretation within a single basic block. If rewriting is
  // possible, the interpreted ops are returned split into their
  // respective categories.
  static FailureOr<InterpretedOps> abstractlyInterpretSlice(
      CopyToNonValueTensorOp copyToNonValueTensor,
      const DenseMap<Operation *, SmallVector<Value>> &nonValueTensorsUsedByOp,
      PatternRewriter &rewriter) {
    // Sort by order in the block, so we can abstractly interpret the ops.
    SmallVector<Operation *> nonValueTensorUsers(
        llvm::make_first_range(nonValueTensorsUsedByOp));
    llvm::sort(nonValueTensorUsers, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    // We track the available aliases at each point as well as split the
    // users into view-like, copy-to-value, and overwrite ops as we walk
    // forward.
    InterpretedOps result;
    result.copyLikeOps.push_back(copyToNonValueTensor);
    DenseSet<Value> availableAliases{
        assertNonValueTensor(copyToNonValueTensor.getResult())};
    DenseMap<Value, AliasSliceInfo> aliasSliceInfo;
    for (Operation *user : nonValueTensorUsers) {
      for (Value operand : nonValueTensorsUsedByOp.lookup(user)) {
        if (!availableAliases.contains(operand)) {
          return rewriter.notifyMatchFailure(
              copyToNonValueTensor,
              "operand of op is not a valid tensor alias");
        }
      }
      if (isViewLikeOp(user)) {
        Value userResult = user->getResult(0);
        // View-like ops produce a new alias available to later ops.
        // However, if the view-like op has been partially converted
        // to use value semantics (which happens for example with ops
        // that take two aliases as input), then it is possible that the
        // op no longer generates an alias.
        if (userResult.getType().isa<NonValueTensorType>()) {
          availableAliases.insert(userResult);
          calculateAliasInfo(userResult, aliasSliceInfo);
        }
        result.viewLikeOps.push_back(user);
      } else if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(user)) {
        result.copyLikeOps.push_back(copyToValueTensor);
      } else if (auto overwrite = dyn_cast<OverwriteTensorContentsOp>(user)) {
        removePossiblyMutatedAliases(availableAliases, aliasSliceInfo,
                                     overwrite);
        result.overwriteTensorContentsOps.push_back(overwrite);
      } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(user)) {
        result.returnOp = returnOp;
      } else {
        return rewriter.notifyMatchFailure(
            copyToNonValueTensor, "unsupported op `" +
                                      user->getName().getStringRef() +
                                      "` encountered during abstract analysis");
      }
    }
    return result;
  }

  // Rewrite slice composed of the interpreted ops so that the slice uses
  // value semantics everywhere.
  static void rewriteSlice(const InterpretedOps &ops,
                           PatternRewriter &rewriter) {

    DenseMap<int, Type> originalReturnTypes;
    if (ops.returnOp.has_value()) {
      auto returnOp = ops.returnOp.value();
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
      Value overwritten = assertNonValueTensor(overwrite.getOverwritten());
      overwritten.replaceUsesWithIf(
          overwrite.getValue(), [&](const OpOperand &operand) {
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
        if (resultType)
          result.setType(resultType.getWithValueSemantics());
      });
    }
    if (ops.returnOp.has_value()) {
      auto returnOp = ops.returnOp.value();
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
    DenseMap<Operation *, SmallVector<Value>> nonValueTensorsUsedByOp;

    // Some view-like ops take more than one non-value tensor as input (such as
    // `aten.view_as`). For these ops, we assume that the tensor view that gets
    // returned by the op is a view of the first operand of the op.

    // View-like ops that return a non-value tensor and have a view of the
    // operand of `copy.to_tensor` as the first operand.
    DenseSet<Operation *> validViewLikeOps;
    // View-like ops that return a non-value tensor and have a view of the
    // operand of `copy.to_tensor` as an operand other than the first operand.
    DenseSet<Operation *> viewLikeOpsToCheck;

    using OpOperandRefs = SmallVector<std::reference_wrapper<OpOperand>>;
    OpOperandRefs workList(copy.getResult().getUses());
    while (!workList.empty()) {
      OpOperand &operand = workList.pop_back_val();
      Operation *op = operand.getOwner();
      if (op->getBlock() != copy->getBlock()) {
        return rewriter.notifyMatchFailure(
            copy, "can only analyze within a single basic block");
      }

      if (isViewLikeOp(op)) {
        // We currently only support view-like ops with one tensor output.
        if (op->getNumResults() != 1 ||
            !op->getResult(0).getType().isa<BaseTensorType>()) {
          return rewriter.notifyMatchFailure(
              copy, "unsupported: view-like ops must have one tensor output, "
                    "and the tensor output must be the first result");
        }

        Value opResult = op->getResult(0);
        // There are cases where a view-like op will be partially converted to
        // value semantics, resulting in at least one of the inputs being a
        // non-value tensor and the output being a value tensor. If this is the
        // case then there is no need to look at the users of the result of the
        // op.
        if (opResult.getType().isa<NonValueTensorType>()) {
          if (operand.getOperandNumber() == 0) {
            validViewLikeOps.insert(op);
            llvm::append_range(workList, opResult.getUses());
          } else {
            viewLikeOpsToCheck.insert(op);
          }
        }
      }

      nonValueTensorsUsedByOp[op].push_back(
          assertNonValueTensor(operand.get()));
    }

    // Nothing to do if there is just a ReturnOp -- we know that we won't be
    // rewriting anything, since we must preserve the ReturnOp's original type.
    if (llvm::hasSingleElement(nonValueTensorsUsedByOp) &&
        isa<mlir::func::ReturnOp>(nonValueTensorsUsedByOp.begin()->first)) {
      return failure();
    }

    if (llvm::any_of(viewLikeOpsToCheck, [&](Operation *op) {
          return !validViewLikeOps.contains(op);
        })) {
      return rewriter.notifyMatchFailure(
          copy, "if a view-like op returns a non-value tensor, the first "
                "operand must be a view of the operand of the `copy.to_tensor` "
                "op");
    }

    FailureOr<InterpretedOps> interpretedOps =
        abstractlyInterpretSlice(copy, nonValueTensorsUsedByOp, rewriter);
    if (failed(LogicalResult(interpretedOps)))
      return failure();
    rewriteSlice(*interpretedOps, rewriter);
    return success();
  }

private:
  static void
  calculateAliasInfo(Value tensor,
                     DenseMap<Value, AliasSliceInfo> &aliasSliceInfo) {
    Operation *op = tensor.getDefiningOp();

    // If op is not a view-like op, then the tensor is an alias of itself.
    if (!isViewLikeOp(op)) {
      aliasSliceInfo[tensor] = AliasSliceInfo{-1, 0, INT64_MAX, false};
      return;
    }

    // Check if the alias information of the operation is already available.
    if (aliasSliceInfo.count(tensor))
      return;

    if (auto staticInfoCast = dyn_cast<TensorStaticInfoCastOp>(op)) {
      // If op is static info cast, we assume it does not perform any
      // reshaping or slicing.
      calculateAliasInfo(staticInfoCast.getOperand(), aliasSliceInfo);
      aliasSliceInfo[tensor] = aliasSliceInfo[staticInfoCast.getOperand()];
      return;
    } else if (auto sliceOp = dyn_cast<AtenSliceTensorOp>(op)) {
      // Get the alias information of the input tensor.
      calculateAliasInfo(sliceOp.getSelf(), aliasSliceInfo);
      if (!aliasSliceInfo.count(sliceOp.getSelf()))
        return;

      AliasSliceInfo &aliasInfo = aliasSliceInfo[sliceOp.getSelf()];

      // We mark the slice as approximate unless we can statically determine
      // the slicing information.
      aliasSliceInfo[tensor] =
          AliasSliceInfo{aliasInfo.dim, aliasInfo.start, aliasInfo.end, true};
      AliasSliceInfo &newSliceInfo = aliasSliceInfo[tensor];

      // Shape of slice is not statically analyzable.
      if (aliasInfo.approximated)
        return;
      // If operation is already sliced, we take an approximation of alias with
      // only the dimension from the first slice.
      // TODO: Support multiple slices.
      if (aliasInfo.dim != -1)
        return;

      int64_t start, end, dim, step;
      if (!matchPattern(sliceOp.getStart(), m_TorchConstantInt(&start)) ||
          !matchPattern(sliceOp.getEnd(), m_TorchConstantInt(&end)) ||
          !matchPattern(sliceOp.getDim(), m_TorchConstantInt(&dim)) ||
          !matchPattern(sliceOp.getStep(), m_TorchConstantInt(&step)))
        return;

      // We currently do not support negative strides.
      if (step < 0)
        return;

      if (start < 0)
        start += INT64_MAX;
      if (end < 0)
        end += INT64_MAX;

      // Get length of the dimension being sliced.
      // We can directly set the slice information since we assume only
      // one slice if performed for now.
      // TODO: Support multiple slices.
      newSliceInfo.dim = dim;
      newSliceInfo.start = start;
      newSliceInfo.end = end;
      newSliceInfo.approximated = false;
      return;
    } else {
      // TODO: Support other view-like ops.
      return;
    }
  }

  static void
  removePossiblyMutatedAliases(DenseSet<Value> &availableAliases,
                               DenseMap<Value, AliasSliceInfo> &aliasSliceInfo,
                               OverwriteTensorContentsOp overwrite) {
    Value tensor = overwrite.getOverwritten();
    if (!aliasSliceInfo.count(tensor)) {
      // If no alias information about overwritten tensor is available, we
      // conservatively assume that all available aliases are mutated.
      availableAliases.clear();
      availableAliases.insert(assertNonValueTensor(overwrite.getOverwritten()));
      return;
    }

    AliasSliceInfo tensorInfo = aliasSliceInfo[tensor];

    // Remove all aliases that overlap with the overwritten tensor.
    DenseSet<Value> newAvailableAliases;
    for (Value alias : availableAliases) {
      if (aliasSliceInfo.count(alias) &&
          !tensorInfo.isOverlapping(aliasSliceInfo[alias]))
        newAvailableAliases.insert(alias);
    }

    availableAliases = newAvailableAliases;
    availableAliases.insert(assertNonValueTensor(tensor));
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
    SmallVector<mlir::func::ReturnOp> returnOps;
    auto workList = llvm::to_vector<6>(copy.getResult().getUsers());
    // We currently only support view-like ops with one tensor input and one
    // tensor output, meaning that the tensor use-def chains form a tree.
    // This will not be the case for an op like `torch.aten.view_as`, so
    // we will need to add a set to prune duplicate visitation.
    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(op)) {
        copyToValueTensorOps.push_back(copyToValueTensor);
      } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
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

    // All CopyToValueTensorOp operands will be changed to the correct type
    // by the logic below.
    for (CopyToValueTensorOp op : copyToValueTensorOps)
      rewriter.replaceOp(op, op.getOperand());
    // All uses of `copy` will be updated by the logic below.
    copy.replaceAllUsesWith(copy.getOperand());
    // Keep track of the original types of any view-like ops, so that we can
    // correctly copy them back to their mlir::func::ReturnOp's expected types.
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
    for (mlir::func::ReturnOp op : returnOps) {
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

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createMaximizeValueSemanticsPass() {
  return std::make_unique<MaximizeValueSemanticsPass>();
}
