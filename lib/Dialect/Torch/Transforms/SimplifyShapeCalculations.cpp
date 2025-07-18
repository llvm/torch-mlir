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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    auto tensorType = cast<BaseTensorType>(self.getType());
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
class InferTensorOp : public OpRewritePattern<AtenTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op.getContext();
    auto loc = op.getLoc();
    auto result = op.getResult();
    auto resultType = cast<BaseTensorType>(result.getType());
    if (resultType.hasSizes() && resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "The result of aten.tensor is already a BaseTensorType.");
    }

    auto inputList = op.getOperand(0);
    auto listConstruct = inputList.getDefiningOp<PrimListConstructOp>();
    if (!listConstruct) {
      return rewriter.notifyMatchFailure(
          op, "The operand 0 of aten.tensor is not PrimListConstructOp.");
    }

    // Currently only support the 1d input list.
    SmallVector<int64_t> sizes;
    sizes.push_back(listConstruct->getOperands().size());
    FailureOr<Type> torchType;
    auto eleType = listConstruct->getOperands()[0].getType();
    if (isa<Torch::IntType>(eleType)) {
      torchType = getTypeForScalarType(op->getContext(),
                                       torch_upstream::ScalarType::Long);
    } else if (isa<Torch::FloatType>(eleType)) {
      torchType = getTypeForScalarType(op->getContext(),
                                       torch_upstream::ScalarType::Float);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Currently only support Int and Float Type.");
    }
    auto newResultType = ValueTensorType::get(context, sizes, *torchType);

    Value originalTypedValue;
    for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
      if (!originalTypedValue) {
        rewriter.setInsertionPointAfter(op);
        originalTypedValue =
            rewriter.create<TensorStaticInfoCastOp>(loc, resultType, result);
      }
      use.set(originalTypedValue);
    }

    result.setType(newResultType);

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

  // For the cases when the result is none, the implied type will remain the
  // same as original result type.
  Type impliedTypesFromShape = result.getType();
  if (!isa<Torch::NoneType>(impliedTypesFromShape)) {
    BaseTensorType originalResultType = cast<BaseTensorType>(result.getType());
    impliedTypesFromShape =
        cast<BaseTensorType>(originalResultType)
            .getWithSizesAndDtype(ArrayRef(sizes),
                                  originalResultType.getOptionalDtype());
  }
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
    populateFullyUnrollPrimLoopOpPattern(patterns, context);
    populateAbstractlyInterpretListOpsWithinABlockPattern(patterns, context);
    populateFoldPrimUncheckedCastOpPattern(patterns, context);
    patterns.insert<DecomposeAtenSizeOp>(context);
    patterns.insert<RefineShapeCalculateOp>(context);
    patterns.insert<InferTensorOp>(context);

    PrimIfOp::getCanonicalizationPatterns(patterns, context);
    Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    AtenSizeOp::getCanonicalizationPatterns(patterns, context);
    AtenLenTOp::getCanonicalizationPatterns(patterns, context);
    AtenAddTOp::getCanonicalizationPatterns(patterns, context);
    AtenSliceTOp::getCanonicalizationPatterns(patterns, context);

    // TODO: Debug visitation order to make this more efficient.
    // A single linear scan should suffice.
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
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
