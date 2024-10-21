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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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
                               SmallVectorImpl<Value> &values) {
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
            b.create<Torch::ConstantIntOp>(val.getValue().getSExtValue()));
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

LogicalResult getListFromTensor(Value value, SmallVector<OpFoldResult> &vals) {
  constexpr int64_t kMaxFold = 16;
  if (auto tensor = value.getDefiningOp<Torch::AtenTensorOp>()) {
    SmallVector<Value> unfolded;
    LogicalResult gotList = getListOperands(tensor.getData(), unfolded);
    vals = getAsOpFoldResult(unfolded);
    return gotList;
  }

  if (auto full = value.getDefiningOp<Torch::AtenFullOp>()) {
    auto ty = cast<ValueTensorType>(full.getType());
    if (!ty.areAllSizesKnown() || ty.getSizes().size() != 1)
      return failure();

    if (ty.getSizes()[0] > kMaxFold)
      return failure();

    vals.resize(vals.size() + ty.getSizes()[0],
                getAsOpFoldResult(full.getFillValue()));
    return success();
  }

  if (auto unsqueeze = value.getDefiningOp<Torch::AtenUnsqueezeOp>()) {
    Value usqSelf = unsqueeze.getSelf();
    if (auto numToTensor =
            usqSelf.getDefiningOp<Torch::PrimNumToTensorScalarOp>()) {
      vals.push_back(getAsOpFoldResult(numToTensor.getA()));
      return success();
    }
  }

  // A common rank 0 tensor producer
  if (auto numToTensor =
          value.getDefiningOp<Torch::PrimNumToTensorScalarOp>()) {
    vals.push_back(getAsOpFoldResult(numToTensor.getA()));
    return success();
  }

  // Last supported case: ValueTensorLiteralOp
  auto literalOp = value.getDefiningOp<Torch::ValueTensorLiteralOp>();
  if (!literalOp)
    return failure();

  // Check the type.
  auto ty = cast<ValueTensorType>(literalOp.getType());
  if (!ty.hasSizes() || ty.getSizes().size() > 1)
    return failure();
  // make sure the type is not unsigned here before trying to materialize
  auto intTy = dyn_cast_or_null<IntegerType>(ty.getDtype());
  if (!intTy || intTy.isUnsigned())
    return failure();

  // if we have a rank 0 literal, we will be adding one element to the list
  int64_t listSize = ty.getSizes().size() == 1 ? ty.getSizes().front() : 1;

  // check for a splat or dense attr
  auto splattr = dyn_cast_or_null<SplatElementsAttr>(literalOp.getValue());
  auto denseAttr = dyn_cast_or_null<DenseIntElementsAttr>(literalOp.getValue());

  if (!splattr && !denseAttr)
    return failure();

  // These are not mutually exclusive, so try splat first.
  if (splattr) {
    auto attr = splattr.getSplatValue<Attribute>();
    vals.resize((int64_t)vals.size() + listSize, attr);
    return success();
  }

  // remaining case: denseAttr
  if ((int64_t)denseAttr.getValues<Attribute>().size() != listSize)
    return failure();
  for (auto e : denseAttr.getValues<Attribute>())
    vals.push_back(e);
  return success();
}

Value constructAtenTensorOpFromList(ImplicitLocOpBuilder b, mlir::Type resultTy,
                                    SmallVector<Value> &listValues) {
  auto dimList = b.create<Torch::PrimListConstructOp>(
      b.getType<Torch::ListType>(listValues.front().getType()), listValues);
  Value cstNone = b.create<Torch::ConstantNoneOp>();
  Value cstFalse = b.create<Torch::ConstantBoolOp>(b.getBoolAttr(false));
  return b.create<Torch::AtenTensorOp>(resultTy, dimList, cstNone, cstNone,
                                       cstFalse);
}
} // namespace

/// ------ Propagation Patterns ------ ///
// The general goal of these patterns is to convert SomeTensorOp to [scalarOps
// -> PrimListOfInts -> AtenTensorOp] Since these tensorized shape calculation
// ops are chained together, sequences like OpA -> OpB will propagate OpA first:
// [scalarOpsA -> ListA -> TensorA] -> OpB. Then OpB will be able to
// getListFromTensor(A), and further propagate scalarization.

namespace {
class PropagateAtenBroadcastToPattern
    : public OpRewritePattern<AtenBroadcastToOp> {
public:
  using OpRewritePattern<AtenBroadcastToOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBroadcastToOp op,
                                PatternRewriter &rewriter) const override {
    constexpr int64_t kMaxFold = 16;
    // for tensor<si64>, or tensor<1xsi64>, broadcasted to tensor<nxsi64>, grab
    // the element and convert to a full op.
    auto ty = cast<ValueTensorType>(op.getType());
    if (!ty.areAllSizesKnown() || ty.getSizes().size() != 1)
      return failure();

    if (ty.getSizes()[0] > kMaxFold)
      return failure();

    SmallVector<OpFoldResult> fillFold;
    if (failed(getListFromTensor(op.getSelf(), fillFold)) ||
        fillFold.size() != 1)
      return failure();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value, 1> fillVals;
    if (failed(materializeFolds(b, fillFold, fillVals)))
      return failure();

    Value size = b.create<Torch::ConstantIntOp>(ty.getSizes().front());
    Value sizeList = b.create<Torch::PrimListConstructOp>(
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        size);
    Value none = b.create<Torch::ConstantNoneOp>();
    Value cstFalse = b.create<Torch::ConstantBoolOp>(false);
    rewriter.replaceOpWithNewOp<AtenFullOp>(op, ty, sizeList, fillVals.front(),
                                            none, none, none, cstFalse);
    return success();
  }
};
} // namespace

namespace {
class PropagateAtenShapeToTensorPattern
    : public OpRewritePattern<Aten_ShapeAsTensorOp> {
public:
  using OpRewritePattern<Aten_ShapeAsTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ShapeAsTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto self = op.getSelf();
    auto selfTy = cast<BaseTensorType>(self.getType());
    if (!selfTy.hasSizes())
      return rewriter.notifyMatchFailure(op, "self has unknown rank");

    int64_t rank = selfTy.getSizes().size();
    SmallVector<OpFoldResult> dims;
    for (int64_t i = 0; i < rank; ++i) {
      auto iv = b.create<Torch::ConstantIntOp>(i);
      dims.push_back(b.createOrFold<Torch::AtenSizeIntOp>(
          rewriter.getType<Torch::IntType>(), self, iv));
    }
    SmallVector<Value> materializedDims;
    if (failed(materializeFolds(b, dims, materializedDims))) {
      return failure();
    }

    Value result =
        constructAtenTensorOpFromList(b, op.getType(), materializedDims);
    rewriter.replaceOp(op, result);
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
      llvm::SmallVector<OpFoldResult> delisted;
      if (failed(getListFromTensor(element, delisted)))
        return rewriter.notifyMatchFailure(op, "unknown op fold type");

      for (auto scalar : delisted)
        scalars.push_back(scalar);
    }

    SmallVector<Value> values;
    if (failed(materializeFolds(b, scalars, values)) || values.empty())
      return rewriter.notifyMatchFailure(op, "unable to materialize constants");

    Value result = constructAtenTensorOpFromList(b, resultTy, values);
    rewriter.replaceOp(op, result);
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

    SmallVector<OpFoldResult> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "requires a constant dim");

    SmallVector<OpFoldResult> idxFolds;
    if (failed(getListFromTensor(op.getIndex(), idxFolds)))
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

    SmallVector<OpFoldResult> selected;
    for (auto idx : idxFolds) {
      auto attr = dyn_cast_or_null<IntegerAttr>(dyn_cast<Attribute>(idx));
      if (!attr)
        return failure();
      int64_t indexInt = attr.getValue().getSExtValue();
      indexInt = indexInt < 0 ? indexInt + dimLength : indexInt;
      if (indexInt < 0 || indexInt >= dimLength)
        return failure();
      selected.push_back(elements[indexInt]);
    }

    SmallVector<Value> materializedSelected;
    if (failed(materializeFolds(b, selected, materializedSelected)))
      return failure();

    Value result =
        constructAtenTensorOpFromList(b, op.getType(), materializedSelected);
    rewriter.replaceOp(op, result);
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

    SmallVector<OpFoldResult> elements;
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

    SmallVector<OpFoldResult> selected;
    for (int i = start; i < end; i += step)
      selected.push_back(elements[i]);

    SmallVector<Value> values;
    if (failed(materializeFolds(b, selected, values)))
      return failure();

    Value result = constructAtenTensorOpFromList(b, op.getType(), values);
    rewriter.replaceOp(op, result);
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

    SmallVector<OpFoldResult> conditionFolds, selfFolds, otherFolds;
    if (failed(getListFromTensor(condition, conditionFolds)) ||
        failed(getListFromTensor(self, selfFolds)) ||
        failed(getListFromTensor(other, otherFolds)))
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    SmallVector<Value> conditionList, selfList, otherList;
    if (failed(materializeFolds(b, conditionFolds, conditionList)) ||
        failed(materializeFolds(b, selfFolds, selfList)) ||
        failed(materializeFolds(b, otherFolds, otherList)))
      return failure();

    SmallVector<Value> whereVals;
    auto rank0IntTy = rewriter.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>({}), selfTy.getDtype());
    auto rank0BoolTy = rewriter.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>({}), conditionTy.getDtype());
    for (uint64_t i = 0; i < selfList.size(); i++) {
      Value rank0Cond = b.create<Torch::PrimNumToTensorScalarOp>(
          rank0BoolTy, conditionList[i]);
      Value rank0Self =
          b.create<Torch::PrimNumToTensorScalarOp>(rank0IntTy, selfList[i]);
      Value rank0Other =
          b.create<Torch::PrimNumToTensorScalarOp>(rank0IntTy, otherList[i]);
      Value rank0Where = b.create<AtenWhereSelfOp>(rank0IntTy, rank0Cond,
                                                   rank0Self, rank0Other);
      whereVals.push_back(
          b.create<AtenItemOp>(rewriter.getType<Torch::IntType>(), rank0Where));
    }
    Value result = constructAtenTensorOpFromList(b, op.getType(), whereVals);
    rewriter.replaceOp(op, result);
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

    SmallVector<OpFoldResult> selfFolds, otherFolds;
    if (failed(getListFromTensor(self, selfFolds)) ||
        failed(getListFromTensor(other, otherFolds)))
      return rewriter.notifyMatchFailure(op, "failed to get list from tensor");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> selfList, otherList;
    if (failed(materializeFolds(b, selfFolds, selfList)) ||
        failed(materializeFolds(b, otherFolds, otherList)))
      return rewriter.notifyMatchFailure(op, "failed to materialize folds");

    SmallVector<OpFoldResult> eqBoolFolds;
    for (uint64_t i = 0; i < selfList.size(); i++) {
      OpFoldResult eqInt =
          b.createOrFold<AtenEqIntOp>(selfList[i], otherList[i]);
      if (auto eqIntVal = dyn_cast<Value>(eqInt))
        eqInt = b.createOrFold<AtenIntBoolOp>(eqIntVal);
      // if eqInt was an Attribute, it will materialize to a constant int op,
      // which is what we want.
      eqBoolFolds.push_back(eqInt);
    }
    SmallVector<Value> eqVals;
    if (failed(materializeFolds(b, eqBoolFolds, eqVals))) {
      return failure();
    }

    Value result = constructAtenTensorOpFromList(b, op.getType(), eqVals);
    rewriter.replaceOp(op, result);
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
    SmallVector<OpFoldResult> elements;
    Value self = op.getSelf();
    auto selfTy = cast<ValueTensorType>(self.getType());
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

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
      return rewriter.notifyMatchFailure(op, "expected one element");

    SmallVector<Value, 1> materialized;
    if (failed(materializeFolds(b, elements, materialized)))
      return failure();

    rewriter.replaceOp(op, materialized.front());
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
  patterns
      .insert<PropagateAtenCatPattern, PropagateAtenIndexSelectPattern,
              PropagateAtenItemPattern, PropagateAtenShapeToTensorPattern,
              PropagateAtenSliceTensorPattern, PropagateAtenEqTensorPattern,
              PropagateAtenWhereSelfPattern, PropagateAtenBroadcastToPattern>(
          patterns.getContext());
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
    // don't load torch canonicalization patterns, since these may lead to
    // issues with propagation

    // walk func op bottom-up to collect a SetVector of shape-related operations
    // When we pass this SetVector to the pattern rewrite driver, it will
    // process the operations top-down, thereby propagating scalarization
    // starting from sources.
    auto funcOp = getOperation();
    llvm::SetVector<Operation *> shapeCalculationOps;
    funcOp.walk<WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](Operation *op) {
          // Walking bottom-up, start adding ops when we reach an anchor point
          // (a prim list of ints)
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
          // Insert the op if any of it's consumers have already been identified
          // as a shape calculation op. To avoid adding the producer of
          // something like a size.int op, don't add ops when their consumer is
          // a source op for shape scalarization. Here is some sample IR:
          // ------
          // %0 = aten.matmul %arg0, %arg1 : ... -> !torch.vtensor<[?,?,?],f32>
          // %1 = aten.size.int %0, %int0 : !torch.int
          // %2 = prim.ListConstruct %1 : (!torch.int) -> !torch.list<int>
          // return %2 : !torch.list<int>
          // ------
          // In this example, don't add the matmul (%0), or it's producers, to
          // shapeCalculationOps. It's consumer (%1) is indeed a shape
          // calculation op, but the size.int op is an elementary unit of shape
          // computation. No futher gathering of producers is necessary to
          // reduce this. Similarly, don't add the `self` of a view op.
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
    // When propagating, we need to go back and clean up aten.Tensor ops that
    // have been futher propagated. It is also necessary to add newly created
    // ops for custom folding after scalarizing a where.self op.
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    if (failed(applyOpPatternsAndFold(shapeCalculationOps.getArrayRef(),
                                      std::move(patterns), config))) {
      return signalPassFailure();
    }

    // TODO: Warn when failing to process operations in the worklist.
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createScalarizeShapesPass() {
  return std::make_unique<ScalarizeShapesPass>();
}
