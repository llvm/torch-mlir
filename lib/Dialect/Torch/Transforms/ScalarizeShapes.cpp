//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

LogicalResult getListOperands(Value value, SmallVector<Value> &vals) {
  auto list = value.getDefiningOp<Torch::PrimListConstructOp>();
  if (!list)
    return failure();

  for (auto operand : list.getOperands())
    vals.push_back(operand);

  return success();
}

LogicalResult getListFromTensor(Value value, SmallVector<Value> &vals) {
  auto tensor = value.getDefiningOp<Torch::AtenTensorOp>();
  if (!tensor)
    return failure();

  return getListOperands(tensor.getData(), vals);
}
} // namespace

namespace {
class PropagateAtenShapeToTensorPattern
    : public OpRewritePattern<Aten_ShapeAsTensorOp> {
public:
  using OpRewritePattern<Aten_ShapeAsTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ShapeAsTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto self = op.getSelf();
    auto selfTy = cast<BaseTensorType>(self.getType());
    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "self has unknown rank");

    int64_t rank = selfTy.getSizes().size();
    SmallVector<Value> dims;
    for (int64_t i = 0; i < rank; ++i) {
      auto iv = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      dims.push_back(rewriter.create<Torch::AtenSizeIntOp>(
          loc, rewriter.getType<Torch::IntType>(), self, iv));
    }

    auto dimList = rewriter.create<Torch::PrimListConstructOp>(
        loc,
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        dims);

    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), dimList, cstNone, cstNone, cstFalse);
    return success();
  }
};
} // namespace

namespace {
class PropagateAtenIndexSelectPattern
    : public OpRewritePattern<AtenIndexSelectOp> {
public:
  using OpRewritePattern<AtenIndexSelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexSelectOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    SmallVector<Value> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "requires a constant dim");

    DenseElementsAttr idx;
    if (!matchPattern(op.getIndex(), m_Constant(&idx)))
      return rewriter.notifyMatchFailure(op, "requires a constant index");

    auto selfTy = cast<BaseTensorType>(op.getSelf().getType());
    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "requires known rank");

    auto selfShape = selfTy.getSizes();
    int64_t selfRank = selfShape.size();
    dim = dim < 0 ? dim + selfRank : dim;
    int64_t dimLength = elements.size();
    if (selfShape[dim] != dimLength)
      return rewriter.notifyMatchFailure(
          op, "dim length does not match number of elements");

    for (int64_t i = 0; i < selfRank; ++i) {
      if (i == dim)
        continue;
      if (selfShape[i] != 1)
        return rewriter.notifyMatchFailure(op,
                                           "expects unary non-dim dimension");
    }

    SmallVector<Value> selected;
    if (idx.isSplat()) {
      int64_t indexInt = idx.getSplatValue<APInt>().getSExtValue();
      indexInt = indexInt < 0 ? indexInt + dimLength : indexInt;
      selected.resize(idx.getNumElements(), elements[indexInt]);
    } else {
      for (APInt val : idx.getValues<APInt>()) {
        int64_t indexInt = val.getSExtValue();
        selected.push_back(elements[indexInt]);
      }
    }

    auto eTy = elements.front().getType();

    auto dimList = rewriter.create<Torch::PrimListConstructOp>(
        loc, rewriter.getType<Torch::ListType>(eTy), selected);

    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), dimList, cstNone, cstNone, cstFalse);
    return success();
  }
};
} // namespace

namespace {
// Conversion attempts to handle some common propagatable slice cases, namely
// splatted values, no-op slices, known list of values, or any case where a
// new construction can be generated from a previous set of scalars allowing
// the parent tensor to be bypassed.
class PropagateAtenSliceTensorPattern
    : public OpRewritePattern<AtenSliceTensorOp> {
public:
  using OpRewritePattern<AtenSliceTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSliceTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    SmallVector<Value> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    int64_t dim, start, end, step;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "requires a constant dim");

    if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
      return rewriter.notifyMatchFailure(op, "requires a constant start");

    if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end)))
      return rewriter.notifyMatchFailure(op, "requires a constant end");

    if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
      return rewriter.notifyMatchFailure(op, "requires a constant step");

    if (step < 0)
      return rewriter.notifyMatchFailure(op, "requires a positive step value");

    auto selfTy = cast<BaseTensorType>(op.getSelf().getType());
    auto selfShape = selfTy.getSizes();
    int64_t selfRank = selfShape.size();

    // Correct for negative indexing:
    dim = dim < 0 ? dim + selfRank : dim;

    int64_t dimLength = elements.size();
    start = start < 0 ? start + dimLength : start;
    end = end < 0 ? end + dimLength : end;

    start = start < 0 ? 0 : start;
    end = end < 0 ? 0 : end;
    end = end > dimLength ? dimLength : end;

    if (selfShape[dim] != dimLength)
      return rewriter.notifyMatchFailure(
          op, "dim length does not match number of elements");

    for (int64_t i = 0; i < selfRank; ++i) {
      if (i == dim)
        continue;
      if (selfShape[i] != 1)
        return rewriter.notifyMatchFailure(op,
                                           "expects unary non-dim dimension");
    }

    SmallVector<Value> selected;
    for (int i = start; i < end; i += step)
      selected.push_back(elements[i]);

    auto eTy = elements.front().getType();
    auto dimList = rewriter.create<Torch::PrimListConstructOp>(
        loc, rewriter.getType<Torch::ListType>(eTy), selected);

    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), dimList, cstNone, cstNone, cstFalse);
    return success();
  }
};
} // namespace

namespace {
class PropagateAtenItemPattern : public OpRewritePattern<AtenItemOp> {
public:
  using OpRewritePattern<AtenItemOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenItemOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    if (elements.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected no elements");

    rewriter.replaceOp(op, elements[0]);
    return success();
  }
};
} // namespace

namespace {
class FoldAtenTensorSplatPattern : public OpRewritePattern<AtenTensorOp> {
public:
  using OpRewritePattern<AtenTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTensorOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> elements;
    if (failed(getListOperands(op.getData(), elements)))
      return failure();

    if (elements.size() < 1)
      return rewriter.notifyMatchFailure(op, "no elements");

    auto front = elements.front();
    for (auto element : elements)
      if (element != front)
        return rewriter.notifyMatchFailure(op, "multiple elements found");

    if (elements.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected no elements");

    auto resultTy = cast<BaseTensorType>(op.getType());
    if (!resultTy.hasSizes() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "dynamic output shape");

    auto loc = op.getLoc();
    llvm::SmallVector<Value> sizes;
    for (auto size : resultTy.getSizes())
      sizes.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(size)));

    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getType<Torch::IntType>(), 1);
    Value sizeList = rewriter.create<Torch::PrimListConstructOp>(
        loc,
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        one);

    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    rewriter.replaceOpWithNewOp<AtenFullOp>(op, resultTy, sizeList, front, none,
                                            none, none, cstFalse);
    return success();
  }
};
} // namespace

namespace {
template <typename T> class RemoveUnusedPattern : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    for (auto use : op->getResults())
      if (!use.use_empty())
        return failure();

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class ScalarizeShapesPass : public ScalarizeShapesBase<ScalarizeShapesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PropagateAtenIndexSelectPattern, PropagateAtenItemPattern,
                    PropagateAtenShapeToTensorPattern,
                    PropagateAtenSliceTensorPattern, FoldAtenTensorSplatPattern,
                    RemoveUnusedPattern<Torch::AtenSizeIntOp>,
                    RemoveUnusedPattern<Torch::AtenSliceTensorOp>,
                    RemoveUnusedPattern<Torch::AtenTensorOp>,
                    RemoveUnusedPattern<Torch::ConstantBoolOp>,
                    RemoveUnusedPattern<Torch::ConstantIntOp>,
                    RemoveUnusedPattern<Torch::ConstantNoneOp>,
                    RemoveUnusedPattern<Torch::PrimListConstructOp>>(context);

    context->getLoadedDialect<mlir::arith::ArithDialect>()
        ->getCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createScalarizeShapesPass() {
  return std::make_unique<ScalarizeShapesPass>();
}
