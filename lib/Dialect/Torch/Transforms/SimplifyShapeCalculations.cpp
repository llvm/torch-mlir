//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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
    auto users = llvm::to_vector<6>(op->getUsers());
    for (Operation *user : users) {
      // If a user potentially mutates the list, then we require it to be in the
      // same block for our simple abstract interpretation to work (we can't,
      // for example, handle an "append" operation in a loop or other region).
      // However, if the op is read-only, then from the purpose of our abstract
      // interpretation, we can handle it effectively as though it was at the
      // same position as the corresponding parent op in the block under
      // consideration.
      if (potentiallyMutatesListOperands(user)) {
        if (user->getBlock() != block)
          return failure();
      }
      // TODO: Correctly handle CFG/branch-based control flow.
    }

    auto getParentInBlock = [&](Operation *op) {
      while (op->getBlock() != block)
        op = op->getParentOp();
      return op;
    };
    llvm::sort(users, [&](Operation *lhs, Operation *rhs) {
      return getParentInBlock(lhs)->isBeforeInBlock(getParentInBlock(rhs));
    });

    // For each mutating op (which must be in the same block), we save the
    // current state of the list as a vector of Value's. These will then
    // be converted to PrimListConstructOp's at the correct program points.
    SmallVector<SmallVector<Value>> listLiterals;
    SmallVector<Value> runningList;
    llvm::append_range(runningList, op->getOperands());
    for (Operation *user : users) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        if (!append.use_empty())
          return failure();
        runningList.push_back(append.el());
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
        runningList.insert(runningList.begin() + index, insert.el());
        listLiterals.push_back(runningList);
        continue;
      }
      // If this user potentially mutates the list and isn't handled above, then
      // we can't abstractly interpret any further.
      if (potentiallyMutatesListOperands(user))
        break;
    }

    if (listLiterals.empty())
      return failure();

    // Rewrite all users to use the appropriate list literals.
    Value latestLiteral = op;
    int nextLiteral = 0;
    for (Operation *user : users) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        rewriter.setInsertionPoint(append);
        latestLiteral = rewriter.replaceOpWithNewOp<PrimListConstructOp>(
            append, append.getType(), listLiterals[nextLiteral++]);
        continue;
      }
      if (auto insert = dyn_cast<AtenInsertTOp>(user)) {
        rewriter.setInsertionPoint(insert);
        latestLiteral = rewriter.create<PrimListConstructOp>(
            op->getLoc(), insert.self().getType(), listLiterals[nextLiteral++]);
        rewriter.eraseOp(insert);
        continue;
      }
      for (OpOperand &opOperand : user->getOpOperands()) {
        if (opOperand.get() == op.getResult()) {
          opOperand.set(latestLiteral);
        }
      }
    }

    return success();
  }
};
} // namespace

// TODO: Copypasta from DecomposeComplexOps
// Helper funtion to get rank of `Base tensor type`.
// -1 is returned if the tensorRank can't be determined.
static int getTensorRank(Value tensor) {
  int tensorRank = -1;
  BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();

  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> tensorShape = tensorType.getSizes();
    tensorRank = tensorShape.size();
  }
  return tensorRank;
}

// TODO: Copypasta from DecomposeComplexOps
namespace {
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.self();
    MLIRContext *context = op.getContext();
    int64_t rank = getTensorRank(self);
    if (rank < 0)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
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
// This pattern propagates information out of the shape calculation region and
// into the ShapeCalculateOp result types.
class RefineShapeCalculateOp : public OpRewritePattern<ShapeCalculateOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ShapeCalculateOp op,
                                PatternRewriter &rewriter) const override {
    auto yieldShapes = cast<ShapeCalculateYieldShapesOp>(
        op.shapeCalculation().front().getTerminator());
    SmallVector<SmallVector<int64_t>> shapes;
    SmallVector<bool> hasShapes;
    for (Value shapeValue : yieldShapes->getOperands()) {
      auto listConstruct = shapeValue.getDefiningOp<PrimListConstructOp>();
      if (!listConstruct || isListPotentiallyMutated(listConstruct)) {
        shapes.emplace_back();
        hasShapes.push_back(false);
        continue;
      }
      hasShapes.push_back(true);
      SmallVector<int64_t> &shape = shapes.emplace_back();
      for (Value dimension : listConstruct->getOperands()) {
        int64_t constantSize;
        if (matchPattern(dimension, m_TorchConstantInt(&constantSize)))
          shape.push_back(constantSize);
        else
          shape.push_back(kUnknownSize);
      }
    }
    assert(shapes.size() == hasShapes.size());
    SmallVector<Type> updatedResultTypes;
    for (int i = 0, e = shapes.size(); i < e; i++) {
      if (!hasShapes[i]) {
        updatedResultTypes.push_back(Type());
        continue;
      }

      Type resultType = op->getResult(i).getType();
      auto impliedTypeFromShape =
          resultType.cast<BaseTensorType>().getWithSizesAndDtype(
              makeArrayRef(shapes[i]), nullptr);
      auto updatedType =
          meetTensorTypes(resultType.cast<BaseTensorType>(),
                          impliedTypeFromShape.cast<BaseTensorType>());
      updatedResultTypes.push_back(updatedType);
    }
    // Update uses of the results.
    bool madeChange = false;
    for (int i = 0, e = shapes.size(); i < e; i++) {
      if (!hasShapes[i])
        continue;
      Value result = op->getResult(i);
      auto originalType = result.getType();
      auto updatedType = updatedResultTypes[i];
      if (!updatedType || updatedType == originalType)
        continue;
      Value originalTypedValue;
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()
                ->hasTrait<
                    mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
          continue;
        }
        if (!originalTypedValue) {
          rewriter.setInsertionPointAfter(op);
          originalTypedValue = rewriter.create<TensorStaticInfoCastOp>(
              op->getLoc(), originalType, result);
        }
        use.set(originalTypedValue);
      }
      result.setType(updatedType);
      madeChange = true;
    }
    if (!madeChange)
      return rewriter.notifyMatchFailure(op, "No updates to make.");

    // Update the body.
    // TODO: Rename ShapeCalculateYieldOp as ShapeCalculateYieldValuesOp?
    auto yieldValues =
        cast<ShapeCalculateYieldOp>(op.body().front().getTerminator());
    for (int i = 0, e = shapes.size(); i < e; i++) {
      if (!hasShapes[i])
        continue;
      OpOperand &use = yieldValues->getOpOperand(i);
      Value def = use.get();
      auto updatedType = updatedResultTypes[i];
      Value newYieldedValue;
      // If we can't refine the def in place, then create a new one.
      if (def.isa<OpResult>() &&
          def.cast<OpResult>()
              .getDefiningOp()
              ->hasTrait<mlir::torch::Torch::OpTrait::AllowsTypeRefinement>()) {
        newYieldedValue = def;
      } else {
        rewriter.setInsertionPoint(yieldValues);
        newYieldedValue = rewriter.create<TensorStaticInfoCastOp>(
            op->getLoc(), updatedType, def);
      }
      use.set(newYieldedValue);
      newYieldedValue.setType(updatedType);
    }
    return success();
  }
};
} // namespace

namespace {
class SimplifyShapeCalculationsPass
    : public SimplifyShapeCalculationsBase<SimplifyShapeCalculationsPass> {
  void runOnOperation() override {
    // XXX: This pass needs tests!
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<FullyUnrollPrimLoopOp>(context);
    patterns.insert<AbstractlyInterpretListOpsWithinABlock>(context);
    patterns.insert<DecomposeAtenSizeOp>(context);
    patterns.insert<RefineShapeCalculateOp>(context);

    PrimIfOp::getCanonicalizationPatterns(patterns, context);
    Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    AtenSizeOp::getCanonicalizationPatterns(patterns, context);
    AtenLenTOp::getCanonicalizationPatterns(patterns, context);

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

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createSimplifyShapeCalculationsPass() {
  return std::make_unique<SimplifyShapeCalculationsPass>();
}
