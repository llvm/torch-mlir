//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "SimplifyAbstractInterpCalculationsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// TODO: Only unroll inside the shape calculation region.
// Maybe do this by only applying patterns and folding greedily on the ops
// inside the region + the shape.calculate op itself?
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

      op.getRegion().cloneInto(afterBlock->getParent(), afterBlock->getIterator(),
                            bvm);
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
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    auto tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasSizes())
      return rewriter.notifyMatchFailure(op, "unranked tensor");
    int64_t rank = tensorType.getSizes().size();
    SmallVector<Value> sizes;
    for (int i = 0; i < rank; i++) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      sizes.push_back(rewriter.create<AtenSizeIntOp>(loc, self, dim));
    }

    Value sizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), sizes);
    rewriter.replaceOp(op, sizeList);
    return success();
  }
};
} // namespace

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

static LogicalResult refineShapeCalculateResult(ShapeCalculateOp op,
                                                int resultNum,
                                                PatternRewriter &rewriter) {
  auto yieldShapes = op.getCalculation().front().getTerminator();
  auto shape = yieldShapes->getOperand(resultNum);
  auto result = op->getResult(resultNum);

  // If the yielded shape is not a list literal, we can't analyze it.
  // AbstractlyInterpretListOpsWithinABlock should already have converted as
  // much as possible to literals.
  auto listConstruct = shape.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return rewriter.notifyMatchFailure(
        op, "Expected result from ShapeCalculateOp calculation to be a "
            "`PrimListConstructOp`");
  llvm::BitVector clobberedElements(listConstruct->getNumOperands());
  // Analyze the users to determine if we can refine the shape.
  for (Operation *user : listConstruct->getUsers()) {
    // If an op doesn't mutate the list, then we can handle it.
    if (!potentiallyMutatesListOperands(user))
      continue;
    // We can handle Aten_SetItemTOp specially, since we know that it doesn't
    // change the size of the list. It might clobber some elements, which then
    // become dimensions with unknown size.
    if (auto setItem = dyn_cast<Aten_SetItemTOp>(user)) {
      // If the index is statically known, we can clobber only a single index.
      // Otherwise, we conservatively clobber all of them.
      std::optional<int64_t> indexOpt = matchLegalConstantIndexIntoListOfSize(
          setItem.getIdx(), listConstruct->getNumOperands());
      if (indexOpt)
        clobberedElements.set(*indexOpt);
      else
        clobberedElements.set();
      continue;
    }
    // An unhandled op! We can't make any assumptions about the shape.
    return rewriter.notifyMatchFailure(op, "Unhandled op that mutates lists");
  }

  // Construct the list of sizes implied by the yielded shape.
  SmallVector<int64_t> sizes;
  for (auto operand : llvm::enumerate(listConstruct->getOperands())) {
    int64_t size;
    if (matchPattern(operand.value(), m_TorchConstantInt(&size)) &&
        !clobberedElements[operand.index()])
      sizes.push_back(size);
    else
      sizes.push_back(kUnknownSize);
  }

  auto originalResultType = result.getType().cast<BaseTensorType>();
  auto impliedTypesFromShape =
      originalResultType.cast<BaseTensorType>()
          .getWithSizesAndDtype(ArrayRef(sizes),
                                originalResultType.getOptionalDtype())
          .cast<BaseTensorType>();

  return updateCalculateOpResultTypes(op, resultNum, impliedTypesFromShape,
                                      rewriter);
}

namespace {
// This pattern propagates information out of the shape calculation region and
// into the ShapeCalculateOp result types.
class RefineShapeCalculateOp : public OpRewritePattern<ShapeCalculateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ShapeCalculateOp op,
                                PatternRewriter &rewriter) const override {
    LogicalResult result = failure();
    for (int i = 0, e = op->getNumResults(); i != e; i++)
      if (succeeded(refineShapeCalculateResult(op, i, rewriter)))
        result = success();
    return result;
  }
};
} // namespace

namespace {
class SimplifyShapeCalculationsPass
    : public SimplifyShapeCalculationsBase<SimplifyShapeCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<FullyUnrollPrimLoopOp>(context);
    patterns.insert<AbstractlyInterpretListOpsWithinABlock>(context);
    patterns.insert<DecomposeAtenSizeOp>(context);
    patterns.insert<RefineShapeCalculateOp>(context);
    patterns.insert<FoldPrimUncheckedCastOp>(context);

    PrimIfOp::getCanonicalizationPatterns(patterns, context);
    Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    AtenSizeOp::getCanonicalizationPatterns(patterns, context);
    AtenLenTOp::getCanonicalizationPatterns(patterns, context);
    AtenAddTOp::getCanonicalizationPatterns(patterns, context);

    // TODO: Debug visitation order to make this more efficient.
    // A single linear scan should suffice.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createSimplifyShapeCalculationsPass() {
  return std::make_unique<SimplifyShapeCalculationsPass>();
}
