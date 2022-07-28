//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

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
      return failure();
    int64_t maxTripCount;
    if (!matchPattern(op.maxTripCount(), m_TorchConstantInt(&maxTripCount)))
      return failure();
    SmallVector<Value> indices;
    for (int64_t i = 0; i < maxTripCount; i++) {
      // TODO: Add convenience builder.
      indices.push_back(rewriter.create<ConstantIntOp>(
          loc, rewriter.getIntegerAttr(IntegerType::get(context, 64), i)));
    }
    Block *beforeBlock = op->getBlock();
    Block *afterBlock = rewriter.splitBlock(op->getBlock(), op->getIterator());

    SmallVector<Block *> blocksToMerge;
    BlockAndValueMapping bvm;
    // TODO: Helper for region().front()
    auto condition =
        cast<PrimLoopConditionOp>(op.region().front().getTerminator());
    for (int64_t i = 0; i < maxTripCount; i++) {
      SmallVector<Value> iterArgs;
      if (i == 0) {
        llvm::append_range(iterArgs, op.iterArgsInit());
      } else {
        llvm::append_range(
            iterArgs, llvm::map_range(condition.iterArgs(),
                                      [&](Value v) { return bvm.lookup(v); }));
      }
      bvm.clear();
      bvm.map(op.region().front().getArgument(0), indices[i]);
      bvm.map(op.region().front().getArguments().slice(1), iterArgs);

      op.region().cloneInto(afterBlock->getParent(), afterBlock->getIterator(),
                            bvm);
      Block *clonedBlock = bvm.lookup(&op.region().front());
      rewriter.eraseOp(clonedBlock->getTerminator());
      blocksToMerge.push_back(clonedBlock);
    }

    blocksToMerge.push_back(afterBlock);
    for (Block *block : blocksToMerge)
      rewriter.mergeBlocks(block, beforeBlock);
    if (maxTripCount == 0) {
      rewriter.replaceOp(op, op.iterArgsInit());
    } else {
      rewriter.replaceOp(op, llvm::to_vector<6>(llvm::map_range(
                                 condition.iterArgs(),
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
    auto usersToInterpret =
        makeArrayRef(allUsers).take_front(numUsersToInterpret);

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
          return failure();
        if (append.self() == op) {
          runningList.push_back(append.el());
          generatedNewLiteral = true;
        }
        listLiterals.push_back(runningList);
        continue;
      }
      if (auto insert = dyn_cast<AtenInsertTOp>(user)) {
        if (!insert.use_empty())
          return failure();
        int64_t index;
        if (!matchPattern(insert.idx(), m_TorchConstantInt(&index)))
          return failure();
        // The index might be statically out of bounds.
        if (index < 0 || index > static_cast<int64_t>(runningList.size()))
          return failure();
        if (insert.self() == op) {
          runningList.insert(runningList.begin() + index, insert.el());
          generatedNewLiteral = true;
        }
        listLiterals.push_back(runningList);
        continue;
      }
      if (auto setItem = dyn_cast<Aten_SetItemTOp>(user)) {
        if (!setItem.use_empty())
          return failure();
        llvm::Optional<int64_t> indexOpt =
            matchLegalConstantIndexIntoListOfSize(setItem.idx(),
                                                  runningList.size());
        // The index might be statically out of bounds.
        if (!indexOpt)
          return failure();
        if (setItem.l() == op) {
          runningList[*indexOpt] = setItem.el();
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
      return failure();

    // Rewrite all users to use the appropriate list literals.
    Value latestLiteral = rewriter.create<PrimListConstructOp>(
        op->getLoc(), op.getType(), op->getOperands());
    int nextLiteral = 0;
    for (Operation *user : usersToInterpret) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        rewriter.setInsertionPoint(append);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            append->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (append.self() == op)
          rewriter.eraseOp(append);
        continue;
      }
      if (auto insert = dyn_cast<AtenInsertTOp>(user)) {
        rewriter.setInsertionPoint(insert);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            insert->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (insert.self() == op)
          rewriter.eraseOp(insert);
        continue;
      }
      if (auto setItem = dyn_cast<Aten_SetItemTOp>(user)) {
        rewriter.setInsertionPoint(setItem);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            setItem->getLoc(), op.getType(), listLiterals[nextLiteral++]);
        if (setItem.l() == op)
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
    Value self = op.self();
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
    if (!isValidSubtype(op.x().getType(), op.result().getType())) {
      return rewriter.notifyMatchFailure(
          op, "input tensor type is not a valid subtype of result type");
    }
    rewriter.replaceOp(op, op.x());
    return success();
  }
};
} // namespace

static void refineShapeCalculateResult(ShapeCalculateOp op, int resultNum,
                                       PatternRewriter &rewriter,
                                       bool &madeChange) {
  auto yieldValues = op.body().front().getTerminator();
  auto yieldShapes = op.shapeCalculation().front().getTerminator();
  auto shape = yieldShapes->getOperand(resultNum);
  auto result = op->getResult(resultNum);

  // If the yielded shape is not a list literal, we can't analyze it.
  // AbstractlyInterpretListOpsWithinABlock should already have converted as
  // much as possible to literals.
  auto listConstruct = shape.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return;
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
      llvm::Optional<int64_t> indexOpt = matchLegalConstantIndexIntoListOfSize(
          setItem.idx(), listConstruct->getNumOperands());
      if (indexOpt)
        clobberedElements.set(*indexOpt);
      else
        clobberedElements.set();
      continue;
    }
    // An unhandled op! We can't make any assumptions about the shape.
    return;
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

  // Calculate the updated type incorporating the new shape information.
  Type originalResultType = result.getType();
  auto impliedTypesFromShape =
      originalResultType.cast<BaseTensorType>().getWithSizesAndDtype(
          makeArrayRef(sizes), nullptr);
  auto updatedType =
      meetTensorTypes(originalResultType.cast<BaseTensorType>(),
                      impliedTypesFromShape.cast<BaseTensorType>());
  // If we didn't get any new information, there is nothing left for us to do.
  if (!updatedType || updatedType == originalResultType)
    return;

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
      rewriter.setInsertionPointAfter(op);
      originalTypedValue = rewriter.create<TensorStaticInfoCastOp>(
          op->getLoc(), originalResultType, result);
    }
    use.set(originalTypedValue);
  }
  result.setType(updatedType);
  madeChange = true;

  // Update the value yielded from the body to match the new result type. If we
  // can refine the def in place, do that, otherwise insert a
  // TensorStaticInfoCastOp.
  OpOperand &use = op.body().front().getTerminator()->getOpOperand(resultNum);
  Value def = use.get();
  Value newYieldedValue;
  if (def.isa<OpResult>() &&
      def.cast<OpResult>()
          .getDefiningOp()
          ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
    newYieldedValue = def;
  } else {
    rewriter.setInsertionPoint(yieldValues);
    newYieldedValue =
        rewriter.create<TensorStaticInfoCastOp>(op->getLoc(), updatedType, def);
  }
  use.set(newYieldedValue);
  newYieldedValue.setType(updatedType);
}

namespace {
// This pattern propagates information out of the shape calculation region and
// into the ShapeCalculateOp result types.
class RefineShapeCalculateOp : public OpRewritePattern<ShapeCalculateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ShapeCalculateOp op,
                                PatternRewriter &rewriter) const override {
    bool madeChange = false;
    for (int i = 0, e = op->getNumResults(); i != e; i++)
      refineShapeCalculateResult(op, i, rewriter, madeChange);
    return success(madeChange);
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
    config.maxIterations = GreedyRewriteConfig::kNoIterationLimit;
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
