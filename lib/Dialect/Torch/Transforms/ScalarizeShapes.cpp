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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

LogicalResult materializeFolds(ImplicitLocOpBuilder b,
                               ArrayRef<OpFoldResult> fold,
                               SmallVector<Value> &values) {
  for (auto f : fold) {
    if (auto val = dyn_cast<Value>(f)) {
      values.push_back(val);
      continue;
    }

    if (auto attr = dyn_cast<Attribute>(f)) {
      if (auto val = dyn_cast<FloatAttr>(attr)) {
        values.push_back(b.create<Torch::ConstantFloatOp>(
            b.getType<Torch::FloatType>(), val));
        continue;
      }

      if (auto val = dyn_cast<IntegerAttr>(attr)) {
        values.push_back(
            b.create<Torch::ConstantIntOp>(b.getType<Torch::IntType>(), val));
        continue;
      }
    }

    return failure();
  }

  return success();
}

LogicalResult getListOperands(Value value, SmallVector<Value> &vals) {
  auto list = value.getDefiningOp<Torch::PrimListConstructOp>();
  if (!list)
    return failure();

  for (auto operand : list.getOperands())
    vals.push_back(operand);

  return success();
}

LogicalResult getListFromTensor(Value value, SmallVector<Value> &vals) {
  constexpr int64_t kMaxFold = 16;
  if (auto tensor = value.getDefiningOp<Torch::AtenTensorOp>())
    return getListOperands(tensor.getData(), vals);

  if (auto full = value.getDefiningOp<Torch::AtenFullOp>()) {
    auto ty = cast<ValueTensorType>(full.getType());
    if (!ty.areAllSizesKnown() || ty.getSizes().size() != 1)
      return failure();

    if (ty.getSizes()[0] > kMaxFold)
      return failure();

    vals.resize(vals.size() + ty.getSizes()[0], full.getFillValue());
    return success();
  }

  return failure();
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
class PropagateAtenCatPattern : public OpRewritePattern<AtenCatOp> {
public:
  using OpRewritePattern<AtenCatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCatOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    constexpr int64_t kMaxFold = 16;

    auto resultTy = dyn_cast<ValueTensorType>(op.getType());
    if (!resultTy.hasSizes() || resultTy.getSizes().size() != 1 ||
        !resultTy.areAllSizesKnown())
      return failure();

    if (resultTy.getSizes().front() > kMaxFold)
      return failure();

    if (!resultTy.hasDtype())
      return failure();

    SmallVector<Value> tensors;
    if (failed(getListOperands(op.getTensors(), tensors)))
      return failure();

    SmallVector<OpFoldResult> scalars;
    for (auto element : tensors) {
      llvm::SmallVector<Value> delisted;
      if (succeeded(getListFromTensor(element, delisted))) {
        for (auto scalar : delisted)
          scalars.push_back(scalar);
        continue;
      }

      DenseElementsAttr attr;
      if (matchPattern(element, m_Constant(&attr))) {
        if (attr.isSplat()) {
          scalars.resize(scalars.size() + attr.getNumElements(),
                         attr.getSplatValue<Attribute>());
          continue;
        }

        for (auto e : attr.getValues<Attribute>()) {
          scalars.push_back(e);
        }
        continue;
      }

      return rewriter.notifyMatchFailure(op, "unknown op fold type");
    }

    for (auto &scalar : scalars) {
      if (auto attr = dyn_cast<Attribute>(scalar)) {
        if (auto iattr = dyn_cast<IntegerAttr>(attr)) {
          auto i64 = iattr.getValue().getSExtValue();
          scalar = rewriter.getI64IntegerAttr(i64);
        }
      }
    }

    SmallVector<Value> values;
    if (failed(materializeFolds(b, scalars, values)))
      return rewriter.notifyMatchFailure(op, "unable to materialize constants");

    Type eTy = b.getType<Torch::FloatType>();
    if (isa<mlir::IntegerType>(resultTy.getDtype()))
      eTy = rewriter.getType<Torch::IntType>();

    auto elementsList = b.create<Torch::PrimListConstructOp>(
        rewriter.getType<Torch::ListType>(eTy), values);

    Value cstNone = b.create<Torch::ConstantNoneOp>();
    Value cstFalse =
        b.create<Torch::ConstantBoolOp>(rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), elementsList, cstNone, cstNone, cstFalse);

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
    ImplicitLocOpBuilder b(loc, rewriter);

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
    ImplicitLocOpBuilder b(loc, rewriter);

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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
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
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

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
    SmallVector<Value> sizes;
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
    rewriter.replaceOpWithNewOp<AtenFullOp>(
        op, resultTy, sizeList, elements.front(), none, none, none, cstFalse);
    return success();
  }
};
} // namespace

namespace {
class FoldAtenSqueezePattern : public OpRewritePattern<AtenSqueezeOp> {
public:
  using OpRewritePattern<AtenSqueezeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ValueTensorType>(op.getType());
    if (!resultTy.hasSizes() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "Unknown result shape");

    if (auto atenFull = op.getSelf().getDefiningOp<AtenFullOp>()) {
      SmallVector<Value> sizes;
      for (int i = 0, s = resultTy.getSizes().size(); i < s; ++i)
        sizes.push_back(rewriter.create<Torch::ConstantIntOp>(
            op.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(i)));

      Value sizeList = rewriter.create<Torch::PrimListConstructOp>(
          op.getLoc(),
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          sizes);

      Value none = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
      rewriter.replaceOpWithNewOp<Torch::AtenFullOp>(op, resultTy, sizeList,
                                                     atenFull.getFillValue(),
                                                     none, none, none, none);
      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
class FoldAtenWhereSelf : public OpRewritePattern<AtenWhereSelfOp> {
public:
  using OpRewritePattern<AtenWhereSelfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereSelfOp op,
                                PatternRewriter &rewriter) const override {

    auto getRoot = [](Value v) {
      while (true) {
        if (auto numToTensor =
                v.getDefiningOp<Torch::PrimNumToTensorScalarOp>()) {
          v = numToTensor.getA();
          continue;
        }

        break;
      }

      return v;
    };

    auto self = getRoot(op.getSelf());
    auto other = getRoot(op.getOther());

    if (self == other) {
      rewriter.replaceOp(op, op.getSelf());
      return success();
    }

    auto selfSize = self.getDefiningOp<Torch::AtenSizeIntOp>();
    auto otherSize = other.getDefiningOp<Torch::AtenSizeIntOp>();

    if (selfSize && otherSize) {
      if (selfSize.getSelf() != otherSize.getSelf())
        return failure();

      if (selfSize.getDim() != otherSize.getDim())
        return failure();

      rewriter.replaceOp(op, op.getSelf());
      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
class FoldAtenUnsqueezePattern : public OpRewritePattern<AtenUnsqueezeOp> {
public:
  using OpRewritePattern<AtenUnsqueezeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenUnsqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ValueTensorType>(op.getType());
    if (!resultTy.hasSizes() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "Unknown result shape");

    if (auto atenFull = op.getSelf().getDefiningOp<AtenFullOp>()) {
      SmallVector<Value> sizes;
      for (int i = 0, s = resultTy.getSizes().size(); i < s; ++i)
        sizes.push_back(rewriter.create<Torch::ConstantIntOp>(
            op.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getI64IntegerAttr(i)));

      Value sizeList = rewriter.create<Torch::PrimListConstructOp>(
          op.getLoc(),
          rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
          sizes);

      Value none = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
      rewriter.replaceOpWithNewOp<Torch::AtenFullOp>(op, resultTy, sizeList,
                                                     atenFull.getFillValue(),
                                                     none, none, none, none);
      return success();
    }

    return failure();
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
    patterns
        .insert<PropagateAtenCatPattern, PropagateAtenIndexSelectPattern,
                PropagateAtenItemPattern, PropagateAtenShapeToTensorPattern,
                PropagateAtenSliceTensorPattern, FoldAtenTensorSplatPattern,
                FoldAtenSqueezePattern, FoldAtenUnsqueezePattern,
                FoldAtenWhereSelf, RemoveUnusedPattern<Torch::AtenSizeIntOp>,
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
