//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "SimplificationUtils.h"
#include "mlir/IR/IRMapping.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class FoldPrimUncheckedCastOp : public OpRewritePattern<PrimUncheckedCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimUncheckedCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!isValidSubtype(op.getX().getType(), op.getResult().getType())) {
      return rewriter.notifyMatchFailure(
          op, "input tensor type is not a valid subtype of result type");
    }
    rewriter.replaceOp(op, op.getX());
    return success();
  }
};
} // namespace

namespace {

class FullyUnrollPrimLoopOp : public OpRewritePattern<PrimLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimLoopOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    if (!op.isForLike())
      return rewriter.notifyMatchFailure(op, "Loop is not for-like");
    int64_t maxTripCount;
    if (!matchPattern(op.getMaxTripCount(), m_TorchConstantInt(&maxTripCount)))
      return rewriter.notifyMatchFailure(
          op, "Expected `maxTripCount` to be a constant int");
    ;
    SmallVector<Value> indices;
    for (int64_t i = 0; i < maxTripCount; i++) {
      // TODO: Add convenience builder.
      indices.push_back(rewriter.create<ConstantIntOp>(
          loc, rewriter.getIntegerAttr(IntegerType::get(context, 64), i)));
    }
    Block *beforeBlock = op->getBlock();
    Block *afterBlock = rewriter.splitBlock(op->getBlock(), op->getIterator());

    SmallVector<Block *> blocksToMerge;
    IRMapping bvm;
    // TODO: Helper for region().front()
    auto condition =
        cast<PrimLoopConditionOp>(op.getRegion().front().getTerminator());
    for (int64_t i = 0; i < maxTripCount; i++) {
      SmallVector<Value> iterArgs;
      if (i == 0) {
        llvm::append_range(iterArgs, op.getIterArgsInit());
      } else {
        llvm::append_range(
            iterArgs, llvm::map_range(condition.getIterArgs(),
                                      [&](Value v) { return bvm.lookup(v); }));
      }
      bvm.clear();
      bvm.map(op.getRegion().front().getArgument(0), indices[i]);
      bvm.map(op.getRegion().front().getArguments().slice(1), iterArgs);

      op.getRegion().cloneInto(afterBlock->getParent(),
                               afterBlock->getIterator(), bvm);
      Block *clonedBlock = bvm.lookup(&op.getRegion().front());
      rewriter.eraseOp(clonedBlock->getTerminator());
      blocksToMerge.push_back(clonedBlock);
    }

    blocksToMerge.push_back(afterBlock);
    for (Block *block : blocksToMerge)
      rewriter.mergeBlocks(block, beforeBlock);
    if (maxTripCount == 0) {
      rewriter.replaceOp(op, op.getIterArgsInit());
    } else {
      rewriter.replaceOp(op, llvm::to_vector<6>(llvm::map_range(
                                 condition.getIterArgs(),
                                 [&](Value v) { return bvm.lookup(v); })));
    }
    return success();
  }
};
} // namespace

namespace {
class AbstractlyInterpretListOpsWithinABlock
    : public OpRewritePattern<PrimListConstructOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListConstructOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    auto allUsers = llvm::to_vector<6>(op->getUsers());

    // Sort the users into program order.
    auto getParentInBlock = [&](Operation *op) {
      while (op->getBlock() != block)
        op = op->getParentOp();
      return op;
    };
    // Use a stable sort for deterministic results when users are nested in two
    // regions of the same parent op.
    llvm::stable_sort(allUsers, [&](Operation *lhs, Operation *rhs) {
      return getParentInBlock(lhs)->isBeforeInBlock(getParentInBlock(rhs));
    });

    // We cannot interpret all ops. So first do a check to see up until which
    // point we can interpret.
    int numUsersToInterpret = 0;
    for (int i = 0, e = allUsers.size(); i != e; i++, numUsersToInterpret++) {
      Operation *user = allUsers[i];
      // If a user potentially mutates the list, then we require it to be in the
      // same block for our simple abstract interpretation to work (we can't,
      // for example, handle an "append" operation in a loop or other region).
      // However, if the op is read-only, then from the purpose of our abstract
      // interpretation, we can handle it effectively as though it was at the
      // same position as the corresponding parent op in the block under
      // consideration.
      if (potentiallyMutatesListOperands(user)) {
        if (user->getBlock() != block)
          break;
      }
    }

    // Truncate the list of users to the number of users we're going to
    // interpret.
    allUsers.resize(numUsersToInterpret);
    auto usersToInterpret = ArrayRef(allUsers).take_front(numUsersToInterpret);

    // For each mutating op (which must be in the same block), we save the
    // current state of the list as a vector of Value's. These will then
    // be converted to PrimListConstructOp's at the correct program points.
    SmallVector<SmallVector<Value>> listLiterals;
    SmallVector<Value> runningList;
    llvm::append_range(runningList, op->getOperands());
    bool generatedNewLiteral = false;
    for (Operation *user : usersToInterpret) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        if (!append.use_empty())
          return rewriter.notifyMatchFailure(
              op, "Expected `AtenAppendTOp` to not have users");
        if (append.getSelf() == op) {
          runningList.push_back(append.getEl());
          generatedNewLiteral = true;
        }
        listLiterals.push_back(runningList);
        continue;
      }
      if (auto insert = dyn_cast<AtenInsertTOp>(user)) {
        if (!insert.use_empty())
          return rewriter.notifyMatchFailure(
              op, "Expected `AtenInsertTOp` to not have users");
        int64_t index;
        if (!matchPattern(insert.getIdx(), m_TorchConstantInt(&index)))
          return rewriter.notifyMatchFailure(
              op, "Expected `idx` of `AtenInsertTOp` to be a constant int");
        // The index might be statically out of bounds.
        if (index < 0 || index > static_cast<int64_t>(runningList.size()))
          return rewriter.notifyMatchFailure(
              op, "Index in `AtenInsertTOp` is out of bounds");
        if (insert.getSelf() == op) {
          runningList.insert(runningList.begin() + index, insert.getEl());
          generatedNewLiteral = true;
        }
        listLiterals.push_back(runningList);
        continue;
      }
      if (auto setItem = dyn_cast<Aten_SetItemTOp>(user)) {
        if (!setItem.use_empty())
          return rewriter.notifyMatchFailure(
              op, "Expected `Aten_SetItemTOp` to not have users");
        std::optional<int64_t> indexOpt = matchLegalConstantIndexIntoListOfSize(
            setItem.getIdx(), runningList.size());
        // The index might be statically out of bounds.
        if (!indexOpt)
          return rewriter.notifyMatchFailure(
              op, "Index in `Aten_SetItemTOp` is out of bounds");
        if (setItem.getL() == op) {
          runningList[*indexOpt] = setItem.getEl();
          generatedNewLiteral = true;
        }
        listLiterals.push_back(runningList);
        continue;
      }
      // If this user potentially mutates the list and isn't handled above, then
      // we can't abstractly interpret any further.
      if (potentiallyMutatesListOperands(user))
        break;
    }

    if (!generatedNewLiteral)
      return rewriter.notifyMatchFailure(op, "No new literal created");

    // Rewrite all users to use the appropriate list literals.
    Value latestLiteral = rewriter.create<PrimListConstructOp>(
        op->getLoc(), op.getType(), op->getOperands());
    int nextLiteral = 0;
    for (Operation *user : usersToInterpret) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        rewriter.setInsertionPoint(append);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            append->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (append.getSelf() == op)
          rewriter.eraseOp(append);
        continue;
      }
      if (auto insert = dyn_cast<AtenInsertTOp>(user)) {
        rewriter.setInsertionPoint(insert);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            insert->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (insert.getSelf() == op)
          rewriter.eraseOp(insert);
        continue;
      }
      if (auto setItem = dyn_cast<Aten_SetItemTOp>(user)) {
        rewriter.setInsertionPoint(setItem);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            setItem->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (setItem.getL() == op)
          rewriter.eraseOp(setItem);
        continue;
      }
      for (OpOperand &opOperand : user->getOpOperands()) {
        if (opOperand.get() == op.getResult()) {
          opOperand.set(latestLiteral);
        }
      }
    }

    // Any remaining uses should use the updated value of the latest literal.
    rewriter.replaceOp(op, latestLiteral);
    return success();
  }
};
} // namespace

namespace {
class FoldListAppendChainWithinABlock
    : public OpRewritePattern<PrimListConstructOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListConstructOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    Operation *curOp = op;
    auto curOpUsers = llvm::to_vector<6>(curOp->getUsers());
    llvm::SmallVector<Operation *, 20> opsToDelete;
    SmallVector<Value> runningList;
    llvm::append_range(runningList, op->getOperands());
    while (curOpUsers.size() == 1) {
      if (auto append = dyn_cast<AtenAppendTOp>(curOpUsers[0])) {
        if (append->getBlock() != block)
          break;
        runningList.push_back(append.getEl());
        opsToDelete.push_back(curOp);
        curOp = append;
        curOpUsers = llvm::to_vector<6>(curOp->getUsers());
      } else
        break;
    }

    if (curOp == op)
      return rewriter.notifyMatchFailure(
          op, "Chain of append to list not detected");
    rewriter.setInsertionPoint(curOp);
    rewriter.replaceOp(curOp, rewriter.create<PrimListConstructOp>(
                                  curOp->getLoc(), op.getType(), runningList));

    llvm::for_each(llvm::reverse(opsToDelete),
                   [&](Operation *op) { rewriter.eraseOp(op); });

    return success();
  }
};
} // namespace

LogicalResult Torch::updateCalculateOpResultTypes(Operation *calculateOp,
                                                  int resultNum,
                                                  Type newResultType,
                                                  PatternRewriter &rewriter) {
  Location loc = calculateOp->getLoc();
  auto result = calculateOp->getResult(resultNum);
  Type originalResultType = result.getType();
  Type updatedType;
  if (auto originalBaseTensorType =
          dyn_cast<BaseTensorType>(originalResultType)) {
    // If we didn't get any new information, there is nothing left for us to do.
    updatedType = meetTensorTypes(originalBaseTensorType,
                                  cast<BaseTensorType>(newResultType));
    if (!updatedType || updatedType == originalBaseTensorType)
      return rewriter.notifyMatchFailure(
          calculateOp, "New type information does not refine old type");
  } else if (auto originalResultType =
                 dyn_cast<Torch::NumberType>(result.getType())) {
    if (!isa<Torch::FloatType, Torch::IntType>(newResultType)) {
      return rewriter.notifyMatchFailure(
          calculateOp,
          "Refinement of `NumberType` must be a `FloatType` or `IntType`");
    }
    updatedType = newResultType;
  } else {
    return rewriter.notifyMatchFailure(calculateOp,
                                       "Unimplemented: Expected result type to "
                                       "be `BaseTensorType` or `NumberType`");
  }

  // Update all the uses of the result type to the new type, if possible. Insert
  // a TensorStaticInfoCastOp for any users that might require the exact
  // previous type.
  Value originalTypedValue;
  for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
    if (use.getOwner()
            ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
      continue;
    }
    if (!originalTypedValue) {
      rewriter.setInsertionPointAfter(calculateOp);
      if (isa<BaseTensorType>(originalResultType)) {
        originalTypedValue = rewriter.create<TensorStaticInfoCastOp>(
            loc, originalResultType, result);
      } else if (isa<Torch::NumberType>(originalResultType)) {
        originalTypedValue =
            rewriter.create<DerefineOp>(loc, originalResultType, result);
      } else {
        return rewriter.notifyMatchFailure(
            calculateOp, "Unimplemented: Expected result type to "
                         "be `BaseTensorType` or `NumberType`");
      }
    }
    use.set(originalTypedValue);
  }
  result.setType(updatedType);

  // Update the value yielded from the body to match the new result type. If we
  // can refine the def in place, do that, otherwise insert a
  // TensorStaticInfoCastOp.
  Operation *yieldValues = calculateOp->getRegion(0).front().getTerminator();
  OpOperand &use = yieldValues->getOpOperand(resultNum);
  Value def = use.get();
  Value newYieldedValue;
  if (isa<OpResult>(def) &&
      cast<OpResult>(def)
          .getDefiningOp()
          ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
    newYieldedValue = def;
  } else {
    rewriter.setInsertionPoint(yieldValues);
    if (isa<BaseTensorType>(updatedType)) {
      newYieldedValue =
          rewriter.create<TensorStaticInfoCastOp>(loc, updatedType, def);
    } else {
      newYieldedValue =
          rewriter.create<PrimUncheckedCastOp>(loc, updatedType, def);
    }
  }
  use.set(newYieldedValue);
  newYieldedValue.setType(updatedType);

  return success();
}

void mlir::torch::Torch::populateFoldPrimUncheckedCastOpPattern(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<FoldPrimUncheckedCastOp>(context);
}

void mlir::torch::Torch::populateFullyUnrollPrimLoopOpPattern(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<FullyUnrollPrimLoopOp>(context);
}

void mlir::torch::Torch::populateAbstractlyInterpretListOpsWithinABlockPattern(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<AbstractlyInterpretListOpsWithinABlock>(context);
}

void mlir::torch::Torch::populateFoldListAppendChainWithinABlockPattern(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.insert<FoldListAppendChainWithinABlock>(context);
}
