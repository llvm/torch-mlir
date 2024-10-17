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
#include "mlir/IR/Iterators.h"
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

LogicalResult constructListFromLiteral(PatternRewriter &rewriter,
                                       ValueTensorLiteralOp literalOp,
                                       SmallVector<Value> &vals) {
  // only supports splat ValueTensorLiterals for now. TODO: add support for
  // small non-splat valuetensorliterals.
  auto ty = dyn_cast<ValueTensorType>(literalOp.getType());
  if (!ty || !ty.hasSizes())
    return failure();
  auto attr = dyn_cast_or_null<SplatElementsAttr>(literalOp.getValue());
  if (!attr)
    return failure();
  auto attrInt = dyn_cast<IntegerAttr>(attr.getSplatValue<Attribute>());
  if (!attrInt)
    return failure();
  IntegerType intty = cast<IntegerType>(attrInt.getType());
  if (!intty.isSignedInteger())
    return failure();
  Value materializedVal = rewriter.create<Torch::ConstantIntOp>(
      literalOp.getLoc(), attrInt.getSInt());
  vals.resize(vals.size() + ty.getSizes()[0], materializedVal);
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
class PropagateValueTensorLiteralPattern
    : public OpRewritePattern<Torch::ValueTensorLiteralOp> {
public:
  using OpRewritePattern<Torch::ValueTensorLiteralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Torch::ValueTensorLiteralOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: if size is less than 16, materialize each int as a constant int op,
    // pass to a list construct and pass to a TensorOp If splat, return a full
    // op.
    return failure();
  }
};
} // namespace

/// ------ Propagation Patterns ------ ///
// The general goal of these patterns is to convert SomeTensorOp to [scalarOps
// -> PrimListOfInts -> AtenTensorOp] Since these tensorized shape calculation
// ops are chained together, sequences like OpA -> OpB will propagate OpA first:
// [scalarOpsA -> ListA -> TensorA] -> OpB. Then OpB will be able to
// getListFromTensor(A), and further propagate scalarization.

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
class PropagateAtenWhereSelfPattern : public OpRewritePattern<AtenWhereSelfOp> {
public:
  using OpRewritePattern<AtenWhereSelfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereSelfOp op,
                                PatternRewriter &rewriter) const override {
    Value condition = op.getCondition();
    Value self = op.getSelf();
    Value other = op.getOther();
    auto conditionTy = dyn_cast<Torch::ValueTensorType>(condition.getType());
    if (!conditionTy || !conditionTy.hasSizes() ||
        conditionTy.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "bad condition type");
    auto selfTy = dyn_cast<Torch::ValueTensorType>(self.getType());
    if (!selfTy || !selfTy.hasSizes() || selfTy.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "bad self type");
    auto otherTy = dyn_cast<Torch::ValueTensorType>(other.getType());
    if (!otherTy || !otherTy.hasSizes() || otherTy.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "bad other type");
    int64_t conditionSize = selfTy.getSizes()[0];
    int64_t selfSize = selfTy.getSizes()[0];
    int64_t otherSize = otherTy.getSizes()[0];

    if (selfSize != otherSize || selfSize != conditionSize)
      return rewriter.notifyMatchFailure(
          op,
          "unimplemented: support for propogating with implicit broadcasting.");

    constexpr int64_t kMaxFold = 16;
    if (selfSize == Torch::kUnknownSize || selfSize > kMaxFold)
      return rewriter.notifyMatchFailure(op,
                                         "arguments are dynamic or too big");

    SmallVector<Value> conditionList, selfList, otherList;
    if (failed(getListFromTensor(condition, conditionList)) ||
        (int64_t)conditionList.size() != conditionSize)
      return failure();

    // If one of these tensors is a value tensor literal op, we will need to
    // create constant ints in the IR to form a list. Before calling
    // constructListFromLiteral, we must be certain that the conversion can no
    // longer fail, otherwise we will cause an infinite loop of creating a
    // constant and removing it.
    LogicalResult selfFromList = getListFromTensor(self, selfList);
    LogicalResult otherFromList = getListFromTensor(other, otherList);

    if (failed(selfFromList) && failed(otherFromList))
      return rewriter.notifyMatchFailure(
          op, "At least one operand must succeed at constructing a list");

    auto selfLiteral = self.getDefiningOp<Torch::ValueTensorLiteralOp>();
    auto otherLiteral = other.getDefiningOp<Torch::ValueTensorLiteralOp>();
    if (succeeded(selfFromList) && otherLiteral &&
        failed(constructListFromLiteral(rewriter, otherLiteral, otherList)))
      return failure();
    if (succeeded(otherFromList) && selfLiteral &&
        failed(constructListFromLiteral(rewriter, selfLiteral, selfList)))
      return failure();
    if ((int64_t)selfList.size() != selfSize ||
        (int64_t)otherList.size() != otherSize)
      // this should only occur if we did not generate IR with
      // constructListFromLiteral
      return failure();

    Location loc = op.getLoc();
    SmallVector<Value> whereVals;
    auto rank0IntTy = rewriter.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>({}), selfTy.getDtype());
    auto rank0BoolTy = rewriter.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>({}), conditionTy.getDtype());
    for (uint64_t i = 0; i < selfList.size(); i++) {
      Value rank0Cond = rewriter.create<Torch::PrimNumToTensorScalarOp>(
          loc, rank0BoolTy, conditionList[i]);
      Value rank0Self = rewriter.create<Torch::PrimNumToTensorScalarOp>(
          loc, rank0IntTy, selfList[i]);
      Value rank0Other = rewriter.create<Torch::PrimNumToTensorScalarOp>(
          loc, rank0IntTy, otherList[i]);
      Value rank0Where = rewriter.create<AtenWhereSelfOp>(
          loc, rank0IntTy, rank0Cond, rank0Self, rank0Other);
      whereVals.push_back(rewriter.create<AtenItemOp>(
          loc, rewriter.getType<Torch::IntType>(), rank0Where));
    }
    Value list = rewriter.create<Torch::PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(whereVals[0].getType()), whereVals);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        op.getLoc(), rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), list, cstNone, cstNone, cstFalse);
    return success();
  }
};
} // namespace

namespace {
class PropagateAtenEqTensorPattern : public OpRewritePattern<AtenEqTensorOp> {
public:
  using OpRewritePattern<AtenEqTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEqTensorOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value other = op.getOther();
    auto selfTy = dyn_cast<Torch::ValueTensorType>(self.getType());
    if (!selfTy || !selfTy.hasSizes() || selfTy.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "bad self type");
    auto otherTy = dyn_cast<Torch::ValueTensorType>(other.getType());
    if (!otherTy || !otherTy.hasSizes() || otherTy.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "bad other type");
    int64_t selfSize = selfTy.getSizes()[0];
    int64_t otherSize = otherTy.getSizes()[0];

    if (selfSize != otherSize)
      return rewriter.notifyMatchFailure(
          op,
          "unimplemented: support for propogating with implicit broadcasting.");

    constexpr int64_t kMaxFold = 16;
    if (selfSize == Torch::kUnknownSize || selfSize > kMaxFold ||
        otherSize == Torch::kUnknownSize || otherSize > kMaxFold)
      return rewriter.notifyMatchFailure(op,
                                         "self or other is dynamic or too big");

    SmallVector<Value> selfList, otherList;
    // If one of these tensors is a value tensor literal op, we will need to
    // create constant ints in the IR to form a list. Before calling
    // constructListFromLiteral, we must be certain that the conversion can no
    // longer fail, otherwise we will cause an infinite loop of creating a
    // constant and removing it.
    LogicalResult selfFromList = getListFromTensor(self, selfList);
    LogicalResult otherFromList = getListFromTensor(other, otherList);

    if (failed(selfFromList) && failed(otherFromList))
      return rewriter.notifyMatchFailure(
          op, "At least one operand must succeed at constructing a list");

    auto selfLiteral = self.getDefiningOp<Torch::ValueTensorLiteralOp>();
    auto otherLiteral = other.getDefiningOp<Torch::ValueTensorLiteralOp>();
    if (succeeded(selfFromList) && otherLiteral &&
        failed(constructListFromLiteral(rewriter, otherLiteral, otherList)))
      return failure();
    if (succeeded(otherFromList) && selfLiteral &&
        failed(constructListFromLiteral(rewriter, selfLiteral, selfList)))
      return failure();
    if ((int64_t)selfList.size() != selfSize ||
        (int64_t)otherList.size() != otherSize)
      // this should only occur if we did not generate IR with
      // constructListFromLiteral
      return failure();

    SmallVector<Value> eqVals;
    for (uint64_t i = 0; i < selfList.size(); i++) {
      Value boolResult =
          rewriter.create<AtenEqIntOp>(op.getLoc(), selfList[i], otherList[i]);
      eqVals.push_back(rewriter.create<AtenIntBoolOp>(op.getLoc(), boolResult));
    }
    Value list = rewriter.create<Torch::PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(eqVals[0].getType()), eqVals);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        op.getLoc(), rewriter.getBoolAttr(false));
    rewriter.replaceOpWithNewOp<Torch::AtenTensorOp>(
        op, op.getType(), list, cstNone, cstNone, cstFalse);
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
    Value self = op.getSelf();
    auto selfTy = cast<ValueTensorType>(self.getType());

    // Rank 0 item op prop
    if (selfTy.getSizes().size() == 0) {
      auto numToTensor = self.getDefiningOp<Torch::PrimNumToTensorScalarOp>();
      auto squeezeDim = self.getDefiningOp<AtenSqueezeDimOp>();
      if (!squeezeDim && !numToTensor)
        return rewriter.notifyMatchFailure(op,
                                           "unhandled item of rank 0 operand");
      if (numToTensor) {
        rewriter.replaceOp(op, numToTensor.getA());
        return success();
      }
      rewriter.replaceOpWithNewOp<AtenItemOp>(op, op.getType(),
                                              squeezeDim.getSelf());
      return success();
    }

    // Rank 1 item op prop
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    if (elements.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected no elements");

    rewriter.replaceOp(op, elements[0]);
    return success();
  }
};
} // namespace

/// ------ Fold Patterns ------ ///
// These are shape-specific folding patterns

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
        return rewriter.notifyMatchFailure(op, "sizes not of same tensor");
      int64_t dimSelf, dimOther;
      if ((selfSize.getDim() != otherSize.getDim()) &&
          (!matchPattern(selfSize.getDim(), m_TorchConstantInt(&dimSelf)) ||
           !matchPattern(otherSize.getDim(), m_TorchConstantInt(&dimOther)) ||
           (dimSelf != dimOther)))
        return rewriter.notifyMatchFailure(op, "sizes not of same dim");

      rewriter.replaceOp(op, op.getSelf());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unable to fold");
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
    auto squeezeOp = op.getSelf().getDefiningOp<AtenSqueezeDimOp>();
    if (squeezeOp && resultTy.getSizes().size() == 1) {
      rewriter.replaceOp(op, squeezeOp.getSelf());
      return success();
    }

    return failure();
  }
};
} // namespace

/// ------ Canonicalization Patterns ------ ///

namespace {
// This is a specific pattern for converting views like [?,...,?,lastDim] ->
// [?,...,?,factor0,factor1] to unflatten, and views like
// [?,...,?,factor0,factor1] -> [?,...,?,lastDim] to flatten, whenever it is
// possible to infer that all but last shared dim match
// TODO: move this to an actual canonicalizer for view after deleting the
// conflicting decompositions for flatten/unflatten -> view.
class CanonicalizeAtenViewPattern : public OpRewritePattern<AtenViewOp> {
public:
  using OpRewritePattern<AtenViewOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenViewOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> viewSizes;
    if (failed(getListOperands(op.getSize(), viewSizes)))
      return rewriter.notifyMatchFailure(
          op, "view size must be from a list construct");
    auto selfTy = dyn_cast<Torch::ValueTensorType>(op.getSelf().getType());
    if (!selfTy || !selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "missing input type or sizes");
    auto resultTy = dyn_cast<Torch::ValueTensorType>(op.getType());
    if (!resultTy || !resultTy.hasSizes() ||
        resultTy.getSizes().size() != viewSizes.size())
      return rewriter.notifyMatchFailure(op, "missing result type or sizes");
    int64_t inRank = selfTy.getSizes().size();
    int64_t outRank = resultTy.getSizes().size();

    SmallVector<int64_t> sizes(selfTy.getSizes());
    int64_t endMatchingDim = -1;
    // input sizes vs. provided view sizes comparison loop
    for (int64_t i = 0; i < std::min(outRank, inRank); i++) {
      int64_t providedSize;
      bool providedStatic =
          matchPattern(viewSizes[i], m_TorchConstantInt(&providedSize));
      // if sizes[i] is static, it must match a constant in viewSizes[i]
      if (sizes[i] != Torch::kUnknownSize) {
        if (!providedStatic)
          return rewriter.notifyMatchFailure(
              op, "unsupported: found static input dim, but unable to match "
                  "provided view size on a constant. See position : " +
                      std::to_string(i));
        if (providedSize != sizes[i]) {
          endMatchingDim = i;
          break;
        }
        continue;
      }
      // the remaining assumes sizes[i] is dynamic
      // if provided dim is static, we can't verify it is a flatten/unflatten
      // unless -1
      if (i == outRank - 1 && providedStatic && providedSize == -1) {
        endMatchingDim = i;
        break;
      }
      if (providedStatic)
        return rewriter.notifyMatchFailure(
            op, "unexpected static view dim corresponding to dynamic input dim "
                "at position : " +
                    std::to_string(i));
      auto sizeIntOp = viewSizes[i].getDefiningOp<AtenSizeIntOp>();
      // if we don't have a size int op on self, fail
      if (!sizeIntOp || sizeIntOp.getSelf() != op.getSelf())
        return rewriter.notifyMatchFailure(
            op, "expected dynamic view dim to come from a corresponding "
                "size.int op. See position : " +
                    std::to_string(i));
      int64_t dim;
      // if the dim of the size int op doesn't match, fail
      if (!matchPattern(sizeIntOp.getDim(), m_TorchConstantInt(&dim)) ||
          dim != i)
        return rewriter.notifyMatchFailure(
            op,
            "size int op dim cannot be matched to current dim at position : " +
                std::to_string(i));
      // passing the previous checks means viewSizes[i] = aten.size.int(self,
      // i), so continue
    }
    // if all dims match and the ranks are equal, fold
    if (endMatchingDim == -1 && inRank == outRank) {
      rewriter.replaceOp(op, op.getSelf());
      return success();
    }
    if (endMatchingDim > -1 && inRank > outRank) {
      // only support flattening last dim
      if (endMatchingDim != outRank - 1)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: output has more than back dim mismatching");
      // flatten
      Value start =
          rewriter.create<Torch::ConstantIntOp>(op.getLoc(), endMatchingDim);
      Value end =
          rewriter.create<Torch::ConstantIntOp>(op.getLoc(), inRank - 1);
      rewriter.replaceOpWithNewOp<AtenFlattenUsingIntsOp>(
          op, resultTy, op.getSelf(), start, end);
      return success();
    }
    if (endMatchingDim > -1 && inRank < outRank) {
      // only support unflattening last dim
      if (endMatchingDim != inRank - 1)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: input has more than back dim mismatching");
      // unflatten
      Value dim =
          rewriter.create<Torch::ConstantIntOp>(op.getLoc(), endMatchingDim);
      Value primList = rewriter.create<Torch::PrimListConstructOp>(
          op.getLoc(), op.getSize().getType(),
          ArrayRef<Value>(viewSizes.begin() + endMatchingDim, viewSizes.end()));
      rewriter.replaceOpWithNewOp<AtenUnflattenIntOp>(
          op, resultTy, op.getSelf(), dim, primList);
      return success();
    }
    // examples that might reach this:
    // input shape = [10, 5]; view sizes = [5, 10] (or dynamic variants)
    // input shape = [dim0, dim1]; view sizes = [dim0, dim1, 1, 1] (unsqueezes)
    // input shape = [dim0, dim1, 1, 1] view sizes = [dim0, dim1] (squeezes)
    return rewriter.notifyMatchFailure(
        op, "unhandled case: endMatchingDim=" + std::to_string(endMatchingDim) +
                ", inRank=" + std::to_string(inRank) +
                ", outRank=" + std::to_string(outRank));
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

bool isSourceOpForShapeScalarization(Operation *op) {
  return llvm::isa<AtenSizeIntOp, Torch::ConstantIntOp, Torch::ConstantBoolOp,
                   Aten_ShapeAsTensorOp, Torch::ValueTensorLiteralOp>(op);
}

bool isPrimListOfInts(Operation *op) {
  auto primListOp = dyn_cast<Torch::PrimListConstructOp>(op);
  if (!primListOp)
    return false;
  auto listType = dyn_cast<Torch::ListType>(primListOp.getType());
  if (!listType)
    return false;
  return llvm::isa<Torch::IntType>(listType.getContainedType());
}

void populateScalarizationFoldPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldAtenSqueezePattern, FoldAtenUnsqueezePattern,
                  FoldAtenWhereSelf, FoldAtenTensorSplatPattern>(
      patterns.getContext());
}

void populateScalarizationCanonicalizePatterns(RewritePatternSet &patterns) {
  patterns.add<CanonicalizeAtenViewPattern>(patterns.getContext());
}

void populateScalarizationPropagationPatterns(RewritePatternSet &patterns) {
  patterns.insert<PropagateAtenCatPattern, PropagateAtenIndexSelectPattern,
                  PropagateAtenItemPattern, PropagateAtenShapeToTensorPattern,
                  PropagateAtenSliceTensorPattern, PropagateAtenEqTensorPattern,
                  PropagateAtenWhereSelfPattern>(patterns.getContext());
}

void populateScalarizationRemovePatterns(RewritePatternSet &patterns) {
  patterns.insert<RemoveUnusedPattern<Torch::AtenIntBoolOp>,
                  RemoveUnusedPattern<Torch::AtenEqIntOp>,
                  RemoveUnusedPattern<Torch::PrimNumToTensorScalarOp>,
                  RemoveUnusedPattern<Torch::AtenFullOp>,
                  RemoveUnusedPattern<Torch::AtenUnsqueezeOp>,
                  RemoveUnusedPattern<Torch::AtenSqueezeDimOp>,
                  RemoveUnusedPattern<Torch::AtenSizeIntOp>,
                  RemoveUnusedPattern<Torch::AtenSliceTensorOp>,
                  RemoveUnusedPattern<Torch::AtenTensorOp>,
                  RemoveUnusedPattern<Torch::ConstantBoolOp>,
                  RemoveUnusedPattern<Torch::ConstantIntOp>,
                  RemoveUnusedPattern<Torch::ConstantNoneOp>,
                  RemoveUnusedPattern<Torch::PrimListConstructOp>>(
      patterns.getContext());
}

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

    // populate patterns
    populateScalarizationPropagationPatterns(patterns);
    populateScalarizationFoldPatterns(patterns);
    populateScalarizationCanonicalizePatterns(patterns);
    populateScalarizationRemovePatterns(patterns);
    context->getLoadedDialect<mlir::arith::ArithDialect>()
        ->getCanonicalizationPatterns(patterns);

    // walk func op to get a subgraph of shape computations
    auto funcOp = getOperation();
    llvm::SetVector<Operation *> shapeCalculationOps;
    funcOp.walk<WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](Operation *op) {
          // walking the func op in reverse order, start adding ops when we
          // reach an anchor point (a prim list of ints)
          if (isPrimListOfInts(op)) {
            shapeCalculationOps.insert(op);
            return;
          }
          // add view ops for now until the decompositions for flatten and
          // unflatten are removed.
          if (isa<AtenViewOp>(op)) {
            shapeCalculationOps.insert(op);
            return;
          }
          // all ops which feed into the list-like anchor ops get added to the
          // set. Stop when we feed into a source op for shape scalarization
          for (OpOperand &use : op->getUses()) {
            Operation *userOp = use.getOwner();
            if (shapeCalculationOps.contains(userOp) &&
                !isSourceOpForShapeScalarization(userOp) &&
                !isa<AtenViewOp>(userOp)) {
              shapeCalculationOps.insert(op);
              return;
            }
          }
        });

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(shapeCalculationOps.getArrayRef(),
                                      std::move(patterns), config))) {
      return signalPassFailure();
    }

    // TODO: walk through the func op again, and repopulate the shape
    // calculation ops. If any of those shape calculation ops are still tensors,
    // fail or warn.
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createScalarizeShapesPass() {
  return std::make_unique<ScalarizeShapesPass>();
}
