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
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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
        values.push_back(
            b.create<Torch::ConstantFloatOp>(APFloat(val.getValueAsDouble())));
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

  if (listSize > kMaxFold)
    return failure();

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
    dim = toPositiveDim(dim, selfRank);
    if (!isValidDim(dim, selfRank))
      return failure();
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

    auto selfTy = cast<BaseTensorType>(op.getSelf().getType());
    auto resultTy = cast<BaseTensorType>(op.getType());
    if (!selfTy.areAllSizesKnown() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "requires static sizes");

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

    auto selfShape = selfTy.getSizes();
    auto resultShape = resultTy.getSizes();
    int64_t selfRank = selfShape.size();

    // Correct for negative indexing:
    dim = toPositiveDim(dim, selfRank);
    if (!isValidDim(dim, selfRank))
      return failure();

    int64_t dimLength = selfShape[dim];
    start = start < 0 ? start + dimLength : start;
    end = end < 0 ? end + dimLength : end;
    end = (end < 0) ? -1 : end;
    end = (end < 0 && step > 0) ? 0 : end;

    start = start < 0 ? 0 : start;
    end = end > dimLength ? dimLength : end;

    int64_t frontDimProd = 1, backDimProd = 1;
    for (int64_t i = 0; i < selfRank; i++) {
      if (i < dim)
        frontDimProd *= selfShape[i];
      if (i > dim)
        backDimProd *= selfShape[i];
    }
    int64_t fullDimProd = frontDimProd * dimLength * backDimProd;
    if (fullDimProd != (int64_t)elements.size())
      return rewriter.notifyMatchFailure(op, "unexpected number of elements.");

    // [d0,d1] i -> (i//d1, i % d1) -> (i//d1) * d1 + (i % d1)
    // [d0,d1,d2] i -> (i//d2, i%d2) -> ((i//(d1*d2), (i//d2) % d1, i % d2)

    auto isSliceIdx = [&](int64_t i) {
      int64_t dimidx = (i / backDimProd) % dimLength;
      bool onStep = ((dimidx - start) % step == 0);
      bool beforeEnd = (step < 0 && dimidx > end);
      beforeEnd = beforeEnd || (step > 0 && dimidx < end);
      bool afterBegin = (step < 0 && dimidx <= start);
      afterBegin = afterBegin || (step > 0 && dimidx >= start);
      return onStep && beforeEnd && afterBegin;
    };

    auto flipIdx = [&](int64_t i) {
      int64_t frontIdx = (i / (backDimProd * dimLength));
      int64_t dimIdx = (i / (backDimProd)) % dimLength;
      int64_t flipDimIdx = dimLength - 1 - dimIdx;
      int64_t backIdx = i % (backDimProd);
      return frontIdx * (dimLength * backDimProd) + flipDimIdx * (backDimProd) +
             backIdx;
    };
    SmallVector<OpFoldResult> selected;
    for (int64_t i = 0; i < (int64_t)elements.size(); i++) {
      if (!isSliceIdx(i))
        continue;
      int64_t index = (step > 0) ? i : flipIdx(i);
      selected.push_back(elements[index]);
    }

    fullDimProd = (fullDimProd * resultShape[dim]) / selfShape[dim];
    if ((int64_t)selected.size() != fullDimProd)
      return rewriter.notifyMatchFailure(
          op, "Constructed slice values have an incompatable number of "
              "elements to match the provided return type.");

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
class PropagateAtenTransposeIntPattern
    : public OpRewritePattern<AtenTransposeIntOp> {
public:
  using OpRewritePattern<AtenTransposeIntOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTransposeIntOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto selfTy = cast<BaseTensorType>(op.getSelf().getType());
    auto resultTy = cast<BaseTensorType>(op.getType());
    if (!selfTy.areAllSizesKnown() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "requires static sizes");

    SmallVector<OpFoldResult> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    int64_t dim0, dim1;
    if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0)))
      return failure();
    if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
      return failure();

    ArrayRef<int64_t> selfSizes = selfTy.getSizes();
    int64_t rank = selfSizes.size();

    dim0 = toPositiveDim(dim0, rank);
    dim1 = toPositiveDim(dim1, rank);
    if (!isValidDim(dim0, rank) || !isValidDim(dim0, rank))
      return failure();

    if (dim0 == dim1) {
      rewriter.replaceOp(op, op.getSelf());
      return success();
    }

    if (dim0 > dim1) {
      // swap dim0 and dim1
      dim0 = dim0 + dim1;
      dim1 = dim0 - dim1;
      dim0 -= dim1;
    }

    // A generic transpose will look like...
    // [frontDimsFlat, dim0, midDimsFlat, dim1, backDimsFlat] -> .
    // [frontDimsFlat, dim1, midDimsFlat, dim0, backDimsFlat] .
    // If any of front, mid, or back don't actually exist (e.g. dim0 = 0, or
    // dim1 = dim0 + 1), the reassociation of completely flattened indices will
    // remain unaffected by the artificially unsqueezed dims.
    // --------
    // Setting some notation, let D0,D1,D2,D3,D4 be the respective dim sizes of
    // "self". Let D'j be the transpose dim sizes, and Djk = Dj*Dk. Let fl_trans
    // and fl_self be 1-D flattened tensors. Then:
    // --------
    // fl_trans[i] =
    // = trans[i/D'1234, i/(D'234) % D'1, i/(D'34) % D'2, i/D'4 % D'3, i % D'4]
    // = trans[i/D1234, i/D214 % D3, i/D14 % D2, i/D4 % D1, i % D4]
    // = self[i/D1234, i/D4 % D1, i/D14 % D2, i/D214 % D3, i % D4]
    // = fl_self[dot.prod(indices, (D1234,D234,D34,D4,1))] .
    // --------
    // reassoc(i) = (i/(D1234)) * D1234 +
    //              (i/D4 % D1) * D234 +
    //              (i/(D14) % D2) * D34 +
    //              (i/(D214) % D3) * D4 +
    //              (i % D4) .

    SmallVector<int64_t, 5> D(5, 1);
    int64_t i = -1;
    // D[0] corresponds to flattened front dims
    while (++i < dim0)
      D[0] *= selfSizes[i];
    // D[1] is the earliest transpose dim
    D[1] = selfSizes[i];
    // D[2] corresponds to flattened middle dims
    while (++i < dim1)
      D[2] *= selfSizes[i];
    // D[3] is the later transpose dim
    D[3] = selfSizes[i];
    // D[4] corresponds to flattened back dims
    while (++i < rank)
      D[4] *= selfSizes[i];

    int64_t D1234 = D[1] * D[2] * D[3] * D[4];
    int64_t fullDP = D[0] * D1234;
    if (fullDP != (int64_t)elements.size())
      return failure();
    auto reassoc = [&](int64_t i) {
      return (i / D1234) * D1234 + ((i / D[4]) % D[1]) * D[2] * D[3] * D[4] +
             ((i / (D[1] * D[4])) % D[2]) * D[3] * D[4] +
             ((i / (D[2] * D[1] * D[4])) % D[3]) * D[4] + (i % D[4]);
    };
    SmallVector<OpFoldResult> transposedFolds;
    transposedFolds.reserve(fullDP);
    for (int64_t i = 0; i < fullDP; i++)
      transposedFolds.push_back(elements[reassoc(i)]);

    SmallVector<Value> transposedVals;
    if (failed(materializeFolds(b, transposedFolds, transposedVals)))
      return failure();

    Value result = constructAtenTensorOpFromList(b, resultTy, transposedVals);
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
    if (selfTy.getSizes().empty()) {
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

namespace {

LogicalResult convertOpFoldResults(ImplicitLocOpBuilder &b,
                                   SmallVector<OpFoldResult> &converted,
                                   SmallVector<OpFoldResult> &elements,
                                   Type inputDtype, Type resultDtype) {
  auto inputIsInt = dyn_cast<mlir::IntegerType>(inputDtype);
  auto resultIsInt = dyn_cast<mlir::IntegerType>(resultDtype);
  if (!inputIsInt && !isa<mlir::FloatType>(inputDtype))
    return failure();
  if (!resultIsInt && !isa<mlir::FloatType>(resultDtype))
    return failure();

  // if dtypes are both int or both float, no conversion needed
  if (static_cast<bool>(inputIsInt) == static_cast<bool>(resultIsInt)) {
    converted = elements;
    return success();
  }

  if (resultIsInt) {
    for (auto &e : elements) {
      auto eValue = dyn_cast<Value>(e);
      if (eValue) {
        converted.push_back(b.createOrFold<AtenIntScalarOp>(eValue));
        continue;
      }
      auto eAttr = dyn_cast<Attribute>(e);
      auto eFloatAttr = dyn_cast_or_null<FloatAttr>(eAttr);
      if (!eFloatAttr)
        return failure();

      converted.push_back(IntegerAttr::get(
          resultDtype, static_cast<int64_t>(eFloatAttr.getValueAsDouble())));
    }
    return success();
  }

  // result is float
  for (auto &e : elements) {
    auto eValue = dyn_cast<Value>(e);
    if (eValue) {
      converted.push_back(b.createOrFold<AtenFloatScalarOp>(eValue));
      continue;
    }
    auto eAttr = dyn_cast<Attribute>(e);
    auto eIntAttr = dyn_cast<IntegerAttr>(eAttr);
    if (!eIntAttr)
      return failure();

    auto eInt = (inputIsInt.isSigned()) ? eIntAttr.getValue().getSExtValue()
                                        : eIntAttr.getValue().getZExtValue();
    converted.push_back(FloatAttr::get(resultDtype, static_cast<double>(eInt)));
  }
  return success();
}

class PropagateAtenToDtypePattern : public OpRewritePattern<AtenToDtypeOp> {
public:
  using OpRewritePattern<AtenToDtypeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToDtypeOp op,
                                PatternRewriter &rewriter) const override {
    bool nonBlocking, copyArg;
    // The non_blocking arg must be `False`.
    if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking)) ||
        nonBlocking)
      return failure();
    // The copy arg must be `False`.
    if (!matchPattern(op.getCopy(), m_TorchConstantBool(&copyArg)) || copyArg)
      return failure();
    // The memory_format arg must be `none`.
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType()))
      return failure();

    auto inputType = dyn_cast<ValueTensorType>(op.getSelf().getType());
    auto resultType = dyn_cast<ValueTensorType>(op.getType());
    if (!inputType || !resultType || !inputType.hasDtype() ||
        !resultType.hasDtype())
      return failure();
    auto inputDtype = inputType.getDtype();
    auto resultDtype = resultType.getDtype();

    SmallVector<OpFoldResult> elements;
    if (failed(getListFromTensor(op.getSelf(), elements)))
      return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<OpFoldResult> converted;
    if (failed(convertOpFoldResults(b, converted, elements, inputDtype,
                                    resultDtype)))
      return rewriter.notifyMatchFailure(
          op, "Unhandled attribute type encountered.");

    SmallVector<Value> vals;
    if (failed(materializeFolds(b, converted, vals)))
      return failure();

    Value result = constructAtenTensorOpFromList(b, op.getType(), vals);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
template <typename AtenViewLikeOp>
class PropagateAtenViewLikePattern : public OpRewritePattern<AtenViewLikeOp> {
public:
  using OpRewritePattern<AtenViewLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenViewLikeOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> selfFolds;
    if (failed(getListFromTensor(op.getSelf(), selfFolds)))
      return failure();
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<Value> selfVals;
    if (failed(materializeFolds(b, selfFolds, selfVals)))
      return failure();
    Value result = constructAtenTensorOpFromList(b, op.getType(), selfVals);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {

template <typename OpTy> struct ArithmeticHelper {
  static LogicalResult getAlphaAndVerify(OpTy &op, int64_t &alpha) {
    alpha = 1;
    return success();
  }
};

template <> struct ArithmeticHelper<AtenAddTensorOp> {
  static LogicalResult getAlphaAndVerify(AtenAddTensorOp &op, int64_t &alpha) {
    if (!matchPattern(op.getAlpha(), m_TorchConstantInt(&alpha)) || alpha != 1)
      return failure();
    return success();
  }
};

template <> struct ArithmeticHelper<AtenSubTensorOp> {
  static LogicalResult getAlphaAndVerify(AtenSubTensorOp &op, int64_t &alpha) {
    if (!matchPattern(op.getAlpha(), m_TorchConstantInt(&alpha)) || alpha != 1)
      return failure();
    return success();
  }
};

template <typename OpTy, typename ScalarOpTy>
class PropagateAtenArithmeticPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Check type
    auto resultTy = cast<ValueTensorType>(op.getType());
    if (resultTy.getSizes().size() > 1)
      return rewriter.notifyMatchFailure(op, "unsupported: rank > 1");
    if (!resultTy.hasDtype() || !isa<mlir::IntegerType>(resultTy.getDtype()))
      return rewriter.notifyMatchFailure(op, "not an int type");

    int64_t alpha;
    if (failed(ArithmeticHelper<OpTy>::getAlphaAndVerify(op, alpha)))
      return rewriter.notifyMatchFailure(op, "alpha must be 1");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<OpFoldResult> selfFold, otherFold;
    if (failed(getListFromTensor(op.getSelf(), selfFold)) ||
        failed(getListFromTensor(op.getOther(), otherFold)) ||
        selfFold.size() != otherFold.size())
      return failure();
    SmallVector<Value> selfVals, otherVals;
    if (failed(materializeFolds(b, selfFold, selfVals)) ||
        failed(materializeFolds(b, otherFold, otherVals)))
      return failure();
    SmallVector<OpFoldResult> resultFolds;
    for (uint64_t i = 0; i < selfVals.size(); i++) {
      resultFolds.push_back(b.createOrFold<ScalarOpTy>(
          selfVals[i].getType(), selfVals[i], otherVals[i]));
    }
    SmallVector<Value> resultVals;
    if (failed(materializeFolds(b, resultFolds, resultVals)))
      return failure();

    if (resultTy.getSizes().empty()) {
      rewriter.replaceOpWithNewOp<Torch::PrimNumToTensorScalarOp>(
          op, resultTy, resultVals.front());
      return success();
    }

    Value result = constructAtenTensorOpFromList(b, resultTy, resultVals);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy, typename ScalarOpTy>
class PropagateAtenUnaryPattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Check type
    auto resultTy = cast<ValueTensorType>(op.getType());
    if (resultTy.getSizes().size() > 1)
      return rewriter.notifyMatchFailure(op, "unsupported: rank > 1");
    if (!resultTy.hasDtype() || !isa<mlir::IntegerType>(resultTy.getDtype()))
      return rewriter.notifyMatchFailure(op, "not an int type");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    SmallVector<OpFoldResult> selfFold;
    if (failed(getListFromTensor(op.getSelf(), selfFold)))
      return failure();
    SmallVector<Value> selfVals;
    if (failed(materializeFolds(b, selfFold, selfVals)))
      return failure();
    SmallVector<OpFoldResult> resultFolds;
    for (uint64_t i = 0; i < selfVals.size(); i++) {
      resultFolds.push_back(
          b.createOrFold<ScalarOpTy>(selfVals[i].getType(), selfVals[i]));
    }
    SmallVector<Value> resultVals;
    if (failed(materializeFolds(b, resultFolds, resultVals)))
      return failure();

    if (resultTy.getSizes().size() == 0) {
      rewriter.replaceOpWithNewOp<Torch::PrimNumToTensorScalarOp>(
          op, resultTy, resultVals.front());
      return success();
    }

    Value result = constructAtenTensorOpFromList(b, resultTy, resultVals);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace
/// ------ Fold Patterns ------ ///
// These are shape-specific folding patterns

namespace {
class FoldAtenEqIntPattern : public OpRewritePattern<AtenEqIntOp> {
public:
  using OpRewritePattern<AtenEqIntOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEqIntOp op,
                                PatternRewriter &rewriter) const override {
    // replaces (size.int == 0) with false and adds an assert
    // these comparisons are getting generated because onnx.Reshape considers 0
    // to mean "don't change this dim". However, if the size we are passing to
    // onnx.Reshape is a tensor dim, this is definitely never supposed to be
    // interpreted as "don't change this dim".
    int64_t otherInt;
    if (!matchPattern(op.getB(), m_TorchConstantInt(&otherInt)) ||
        otherInt != 0)
      return failure();

    // in case the shape is a product of two ints, check each
    if (auto mulOp = op.getA().getDefiningOp<AtenMulIntOp>()) {
      Value self = mulOp.getA();
      Value other = mulOp.getB();
      Value selfEq = rewriter.create<AtenEqIntOp>(op.getLoc(), self, op.getB());
      Value otherEq =
          rewriter.create<AtenEqIntOp>(op.getLoc(), other, op.getB());
      rewriter.replaceOpWithNewOp<Aten__Or__BoolOp>(op, selfEq, otherEq);
      return success();
    }

    // if lhs is size.int op, assert size > 0 and replace with false.
    if (auto sizeOp = op.getA().getDefiningOp<AtenSizeIntOp>()) {
      Value selfGtOther = rewriter.create<AtenGtIntOp>(
          op.getLoc(), op.getType(), op.getA(), op.getB());
      rewriter.create<Torch::RuntimeAssertOp>(
          op.getLoc(), selfGtOther,
          rewriter.getStringAttr("Expected dim size > 0."));
      Value cstFalse =
          rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
      rewriter.replaceOp(op, cstFalse);
      return success();
    }

    return failure();
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
    if (resultTy.getSizes().size() == 0) {
      rewriter.replaceOpWithNewOp<Torch::PrimNumToTensorScalarOp>(
          op, op.getType(), elements.front());
      return success();
    }

    auto loc = op.getLoc();
    SmallVector<Value> sizes;
    for (auto size : resultTy.getSizes())
      sizes.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(size)));

    Value sizeList = rewriter.create<Torch::PrimListConstructOp>(
        loc,
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        sizes);

    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    rewriter.replaceOpWithNewOp<AtenFullOp>(
        op, resultTy, sizeList, elements.front(), none, none, none, cstFalse);
    return success();
  }
};
} // namespace

namespace {
template <typename SqueezeOp>
class FoldAtenSqueezePattern : public OpRewritePattern<SqueezeOp> {
public:
  using OpRewritePattern<SqueezeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultTy = cast<ValueTensorType>(op.getType());
    if (!resultTy.hasSizes() || !resultTy.areAllSizesKnown())
      return rewriter.notifyMatchFailure(op, "Unknown result shape");

    Value self = op.getSelf();
    if (auto atenFull = self.getDefiningOp<AtenFullOp>()) {
      // in the rank 0 case, just return the rank 0 scalar
      if (resultTy.getSizes().size() == 0) {
        rewriter.replaceOpWithNewOp<Torch::PrimNumToTensorScalarOp>(
            op, resultTy, atenFull.getFillValue());
        return success();
      }
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
// fold ridiculous patterns like size.int -> float.scalar -> int.scalar
class FoldAtenIntScalarPattern : public OpRewritePattern<AtenIntScalarOp> {
public:
  using OpRewritePattern<AtenIntScalarOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIntScalarOp op,
                                PatternRewriter &rewriter) const override {
    auto floatScalarOp = op.getA().getDefiningOp<AtenFloatScalarOp>();
    if (!floatScalarOp)
      return failure();
    auto sizeOp = floatScalarOp.getA().getDefiningOp<AtenSizeIntOp>();
    if (!sizeOp)
      return failure();
    rewriter.replaceOp(op, floatScalarOp.getA());
    return success();
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
    int64_t leftMatchEnd = 0;
    // compare input sizes with provided dims from left
    for (; leftMatchEnd < std::min(outRank, inRank); leftMatchEnd++) {
      int64_t providedSize;
      bool providedStatic = matchPattern(viewSizes[leftMatchEnd],
                                         m_TorchConstantInt(&providedSize));
      // static dim case
      if (sizes[leftMatchEnd] != Torch::kUnknownSize) {
        // if can't infer equality of dims, set end index and break
        if (!providedStatic || providedSize != sizes[leftMatchEnd])
          break;
        continue;
      }
      // the remaining assumes sizes[leftMatchEnd] is dynamic
      // if provided dim is static, we can't match.
      if (providedStatic)
        break;
      auto sizeIntOp = viewSizes[leftMatchEnd].getDefiningOp<AtenSizeIntOp>();
      // if we don't have a size int op on self, break
      if (!sizeIntOp || sizeIntOp.getSelf() != op.getSelf())
        break;
      int64_t dim;
      // if the dim of the size int op doesn't match, fail
      if (!matchPattern(sizeIntOp.getDim(), m_TorchConstantInt(&dim)) ||
          dim != leftMatchEnd)
        break;
    }

    int64_t rightMatchEnd = 0;
    // compare input sizes with provided dims from right
    for (; rightMatchEnd < std::min(outRank, inRank) - leftMatchEnd;
         rightMatchEnd++) {
      int64_t providedSize;
      bool providedStatic = matchPattern(viewSizes[outRank - 1 - rightMatchEnd],
                                         m_TorchConstantInt(&providedSize));
      // static dim case
      if (sizes[inRank - 1 - rightMatchEnd] != Torch::kUnknownSize) {
        // if can't infer equality of dims, set end index and break
        if (!providedStatic ||
            providedSize != sizes[inRank - 1 - rightMatchEnd])
          break;
        continue;
      }
      // the remaining assumes sizes[inRank - 1 - rightMatchEnd] is dynamic
      // if provided dim is static, we can't match.
      if (providedStatic)
        break;
      auto sizeIntOp =
          viewSizes[outRank - 1 - rightMatchEnd].getDefiningOp<AtenSizeIntOp>();
      // if we don't have a size int op on self, break
      if (!sizeIntOp || sizeIntOp.getSelf() != op.getSelf())
        break;
      int64_t dim;
      // if the dim of the size int op doesn't match, break
      if (!matchPattern(sizeIntOp.getDim(), m_TorchConstantInt(&dim)) ||
          dim != inRank - 1 - rightMatchEnd)
        break;
    }
    // the unmatched input dims start at leftMatchEnd, and end before inRank -
    // rightMatchEnd
    int64_t inputUnmatched = (inRank - rightMatchEnd) - leftMatchEnd;
    int64_t outputUnmatched = (outRank - rightMatchEnd) - leftMatchEnd;
    // if too many dims are unmatched in input/output, cannot canonicalize.
    if (inputUnmatched > 1 && outputUnmatched > 1)
      return rewriter.notifyMatchFailure(
          op,
          "View op is not simple enough to canonicalize.\n# Unmatched Input "
          "dims = " +
              std::to_string(inputUnmatched) +
              "\n# Unmatched Output Dims = " + std::to_string(outputUnmatched) +
              "\nStarting unmatched index = " + std::to_string(leftMatchEnd));

    // if all dims match, return self.
    if (inputUnmatched == outputUnmatched &&
        (inputUnmatched == 1 || inputUnmatched == 0)) {
      rewriter.replaceOpWithNewOp<Torch::TensorStaticInfoCastOp>(
          op, op.getType(), op.getSelf());
      return success();
    }
    // if input has 1 unmatched dim, and output has multiple, unflatten
    if (inputUnmatched == 1 && outputUnmatched > 1) {
      Value dimVal =
          rewriter.create<Torch::ConstantIntOp>(op.getLoc(), leftMatchEnd);
      SmallVector<Value> unflattenSizes(viewSizes.begin() + leftMatchEnd,
                                        viewSizes.end() - rightMatchEnd);
      // try to convert a single dynamic size input to -1
      int64_t dynCount = 0;
      int64_t dynIdx = 0;
      for (auto [i, v] : llvm::enumerate(unflattenSizes)) {
        int64_t szeInt;
        if (!matchPattern(v, m_TorchConstantInt(&szeInt))) {
          dynCount++;
          dynIdx = i;
          continue;
        }
        // if we have a -1 already, make dynCount invalid and break
        if (szeInt == -1) {
          dynCount = -1;
          break;
        }
      }
      // if only one size is dynamic, make it -1
      if (dynCount == 1)
        unflattenSizes[dynIdx] =
            rewriter.create<Torch::ConstantIntOp>(op.getLoc(), -1);

      Value unflattenList = rewriter.create<Torch::PrimListConstructOp>(
          op.getLoc(), op.getSize().getType(), unflattenSizes);
      rewriter.replaceOpWithNewOp<AtenUnflattenIntOp>(
          op, op.getType(), op.getSelf(), dimVal, unflattenList);
      return success();
    }
    // if multiple unmatched input dims map to one output dim, flatten
    if (inputUnmatched > 1 && outputUnmatched == 1) {
      Value startDim =
          rewriter.create<Torch::ConstantIntOp>(op.getLoc(), leftMatchEnd);
      // note: flatten end is inclusive for some reason.
      int64_t endInt = inRank - rightMatchEnd - 1;
      Value endDim = rewriter.create<Torch::ConstantIntOp>(op.getLoc(), endInt);
      rewriter.replaceOpWithNewOp<AtenFlattenUsingIntsOp>(
          op, op.getType(), op.getSelf(), startDim, endDim);
      return success();
    }
    // the remaining cases involve maximal matching dims, but mismatched ranks.
    // This could only occur if squeezing or unsqueezing.
    return rewriter.notifyMatchFailure(
        op, "unhandled view op canonicalization to squeeze/unsqueeze.");
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

bool isItemForSliceOp(Operation *op) {
  auto itemOp = dyn_cast_or_null<AtenItemOp>(op);
  if (!itemOp)
    return false;
  for (OpOperand &use : op->getUses()) {
    Operation *userOp = use.getOwner();
    if (isa<AtenSliceTensorOp>(userOp))
      return true;
  }
  return false;
}

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

bool isAnchorOp(Operation *op) {
  return isa<Torch::RuntimeAssertOp>(op) || isa<AtenArangeStartStepOp>(op) ||
         isPrimListOfInts(op) || isItemForSliceOp(op);
}

// The argument to this function, op, is the use of some source op, srcOp. If
// this function returns true, we want to invalidate srcOp as a target for shape
// scalarization.
bool isInvalidValidViewConsumer(Operation *op,
                                SetVector<Operation *> &workList) {
  // if the consumer isn't a view op, don't invalidate it
  auto view = dyn_cast_or_null<AtenViewOp>(op);
  if (!view)
    return false;
  auto resultTy = dyn_cast<ValueTensorType>(view.getType());
  if (!resultTy || !resultTy.hasDtype())
    return true;
  // if the view op doesn't return integer types, then srcOp is not a shape
  // tensor. note: prim lists will always get added before reaching this
  // function call.
  if (!isa<mlir::IntegerType>(resultTy.getDtype()))
    return true;
  // check uses of the view op.
  // If the view op has a use in our worklist, then it needs to be scalarized.
  for (OpOperand &use : op->getUses()) {
    Operation *userOp = use.getOwner();
    if (workList.contains(userOp))
      return false;
  }
  // invalidate, since the view op was added as a one-off for canonicalization.
  return true;
}

void populateScalarizationFoldPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldAtenSqueezePattern<AtenSqueezeOp>,
                  FoldAtenSqueezePattern<AtenSqueezeDimOp>,
                  FoldAtenIntScalarPattern, FoldAtenUnsqueezePattern,
                  FoldAtenWhereSelf, FoldAtenTensorSplatPattern,
                  FoldAtenEqIntPattern>(patterns.getContext());
}

void populateScalarizationCanonicalizePatterns(RewritePatternSet &patterns) {
  patterns.add<CanonicalizeAtenViewPattern>(patterns.getContext());
}

void populateScalarizationPropagationPatterns(RewritePatternSet &patterns) {
  patterns.add<PropagateAtenViewLikePattern<AtenViewOp>>(patterns.getContext(),
                                                         /*benefit=*/10);
  patterns.insert<PropagateAtenViewLikePattern<AtenFlattenUsingIntsOp>,
                  PropagateAtenViewLikePattern<AtenUnflattenIntOp>>(
      patterns.getContext());
  // A note on division: onnx.Div from int, int -> int types rounds towards
  // zero. The torch DivTensorOp actually doesn't allow returning an int dtype,
  // but this was artificially plummbed through. Unfortunately, there is no
  // scalar trunc div op in torch; however, we can safely assume all operands
  // are positive so floor divide should be a sufficient scalar replacement.
  patterns.insert<
      PropagateAtenCatPattern, PropagateAtenIndexSelectPattern,
      PropagateAtenItemPattern, PropagateAtenShapeToTensorPattern,
      PropagateAtenSliceTensorPattern, PropagateAtenEqTensorPattern,
      PropagateAtenWhereSelfPattern, PropagateAtenBroadcastToPattern,
      PropagateAtenTransposeIntPattern, PropagateAtenToDtypePattern,
      PropagateAtenUnaryPattern<AtenNegOp, AtenNegIntOp>,
      PropagateAtenArithmeticPattern<AtenAddTensorOp, AtenAddIntOp>,
      PropagateAtenArithmeticPattern<AtenSubTensorOp, AtenSubIntOp>,
      PropagateAtenArithmeticPattern<AtenMulTensorOp, AtenMulIntOp>,
      PropagateAtenArithmeticPattern<AtenRemainderTensorOp, AtenRemainderIntOp>,
      PropagateAtenArithmeticPattern<AtenDivTensorOp, AtenFloordivIntOp>>(
      patterns.getContext());
}

void populateScalarizationRemovePatterns(RewritePatternSet &patterns) {
  patterns.insert<RemoveUnusedPattern<Torch::AtenIntBoolOp>,
                  RemoveUnusedPattern<Torch::AtenEqIntOp>,
                  RemoveUnusedPattern<Torch::AtenToDtypeOp>,
                  RemoveUnusedPattern<Torch::PrimNumToTensorScalarOp>,
                  RemoveUnusedPattern<Torch::AtenFullOp>,
                  RemoveUnusedPattern<Torch::AtenUnsqueezeOp>,
                  RemoveUnusedPattern<Torch::AtenSqueezeDimOp>,
                  RemoveUnusedPattern<Torch::AtenSizeIntOp>,
                  RemoveUnusedPattern<Torch::AtenSliceTensorOp>,
                  RemoveUnusedPattern<Torch::AtenTensorOp>,
                  RemoveUnusedPattern<Torch::AtenFloatScalarOp>,
                  RemoveUnusedPattern<Torch::AtenIntScalarOp>,
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
          if (isAnchorOp(op)) {
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
          // reduce this. Similarly, don't always add the `self` of a view op.
          for (OpOperand &use : op->getUses()) {
            Operation *userOp = use.getOwner();
            if (shapeCalculationOps.contains(userOp) &&
                !isSourceOpForShapeScalarization(userOp) &&
                !isInvalidValidViewConsumer(userOp, shapeCalculationOps)) {
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
