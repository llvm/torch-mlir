//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include <unordered_set>
#include <vector>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

static SmallVector<int64_t> getReduceOutputShape(ArrayRef<int64_t> inputShape,
                                                 ArrayRef<int64_t> dims) {
  std::unordered_set<int64_t> dimsSet(dims.begin(), dims.end());
  SmallVector<int64_t> reduceResultShape;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (dimsSet.find(i) == dimsSet.end()) {
      reduceResultShape.push_back(inputShape[i]);
    }
  }
  return reduceResultShape;
}

static Value createInitialValueForReduceOp(Operation *op, Type elementTy,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<AtenSumOp, AtenSumDimIntListOp, AtenFrobeniusNormDimOp,
          AtenLinalgVectorNormOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (isa<mlir::IntegerType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenAmaxOp, AtenMaxOp, AtenMaxDimOp, AtenArgmaxOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getInf(cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                           /*negative=*/true)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (isa<mlir::IntegerType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenAminOp, AtenMinOp, AtenMinDimOp, AtenArgminOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getInf(cast<mlir::FloatType>(elementTy).getFloatSemantics(),
                           /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (isa<mlir::IntegerType>(elementTy)) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenProdOp>(op)) {
    if (isa<mlir::FloatType>(elementTy)) {
      APFloat one(cast<mlir::FloatType>(elementTy).getFloatSemantics(), 1);
      auto constAttr = DenseElementsAttr::get(constType, one);
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (isa<mlir::IntegerType>(elementTy)) {
      APInt one(elementTy.getIntOrFloatBitWidth(), 1);
      auto constAttr = DenseElementsAttr::get(constType, one);
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenAllOp>(op)) {
    auto constAttr =
        DenseElementsAttr::get(constType, {APInt(/*numBits=*/1, 1)});
    return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                  constAttr);
  }

  if (isa<AtenAnyOp, AtenAnyDimOp>(op)) {
    auto constAttr =
        DenseElementsAttr::get(constType, {APInt(/*numBits=*/1, 0)});
    return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                  constAttr);
  }

  op->emitError("unimplemented lowering in "
                "createInitialValueForReduceOp");
  return nullptr;
}

static Value createReduceOpWithSingleRegionOp(Operation *op, Value input,
                                              Type outTy,
                                              ArrayRef<int64_t> dims,
                                              PatternRewriter &rewriter) {
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy)
    return nullptr;
  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue)
    return nullptr;

  stablehlo::ReduceOp reduce = rewriter.create<stablehlo::ReduceOp>(
      op->getLoc(), outTy, input, initValue,
      rewriter.getDenseI64ArrayAttr(dims));

  Block &block = reduce.getBody().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());
  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());
  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value result;
    if (isa<AtenAmaxOp, AtenMaxOp, AtenMaxDimOp>(op)) {
      result = rewriter.create<stablehlo::MaxOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else if (isa<AtenAminOp, AtenMinOp, AtenMinDimOp>(op)) {
      result = rewriter.create<stablehlo::MinOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else if (isa<AtenSumOp, AtenSumDimIntListOp, AtenFrobeniusNormDimOp,
                   AtenLinalgVectorNormOp>(op)) {
      result = rewriter.create<stablehlo::AddOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else if (isa<AtenAllOp>(op)) {
      result = rewriter.create<stablehlo::AndOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else if (isa<AtenAnyOp, AtenAnyDimOp>(op)) {
      result = rewriter.create<stablehlo::OrOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else if (isa<AtenProdOp>(op)) {
      result = rewriter.create<stablehlo::MulOp>(
          op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    } else {
      op->emitError("unimplemented lowering in "
                    "createReduceOpWithSingleRegionOp");
      return nullptr;
    }
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), result);
  }
  return reduce.getResults()[0];
}

// Util for converting AtenMaxDimOp/AtenMinDimOp
static std::optional<ValueRange>
createReduceOpReturnIndices(ConversionPatternRewriter &rewriter, Operation *op,
                            Value &input, ArrayRef<Value> inputShapeVec,
                            int64_t dim, size_t dimSizeIndexBits) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  if (!inputTy) {
    return std::nullopt;
  }
  if (!inputTy.getElementType().isIntOrFloat()) {
    return std::nullopt;
  }
  auto inputShape = inputTy.getShape();
  auto inputElemTy = inputTy.getElementType();

  Value initValue = createInitialValueForReduceOp(op, inputElemTy, rewriter);
  if (!initValue)
    return std::nullopt;
  Value initIndex;
  if (dimSizeIndexBits == 32) {
    initIndex = hlo::getConstTensor<int32_t>(rewriter, op, {0}, {}).value();
  } else {
    initIndex = hlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).value();
  }

  auto outputShape = getReduceOutputShape(inputShape, {dim});
  auto outputTy = RankedTensorType::get(outputShape, inputElemTy);
  auto outputIndexTy =
      RankedTensorType::get(outputShape, rewriter.getIntegerType(64));

  auto inputShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), inputShapeVec);
  auto indexTensor = rewriter.create<stablehlo::DynamicIotaOp>(
      op->getLoc(),
      RankedTensorType::get(inputShape,
                            rewriter.getIntegerType(dimSizeIndexBits)),
      inputShapeTensor, static_cast<uint64_t>(dim));

  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op->getLoc(), TypeRange{outputTy, outputIndexTy},
      ValueRange{input, indexTensor},
      ValueRange{
          initValue,
          initIndex,
      },
      rewriter.getDenseI64ArrayAttr(dim));

  Block &block = stablehloReduceOp.getBody().emplaceBlock();

  // Add block arguments
  auto blockValArgumentType =
      RankedTensorType::get({}, inputTy.getElementType());
  auto blockIdxArgumentType =
      RankedTensorType::get({}, rewriter.getIntegerType(dimSizeIndexBits));
  auto compareResultType = RankedTensorType::get({}, rewriter.getI1Type());
  block.addArgument(blockValArgumentType, op->getLoc());
  block.addArgument(blockIdxArgumentType, op->getLoc());

  block.addArgument(blockValArgumentType, op->getLoc());
  block.addArgument(blockIdxArgumentType, op->getLoc());

  auto *firstValArg = block.args_begin();
  auto *firstIdxArg = std::next(firstValArg);
  auto *secondValArg = std::next(firstIdxArg);
  auto *secondIdxArg = std::next(secondValArg);

  stablehlo::ComparisonTypeAttr compareTypeAttr;
  if (isa<mlir::FloatType>(inputTy.getElementType())) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::FLOAT);
  } else if (isa<mlir::IntegerType>(inputTy.getElementType())) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::SIGNED);
  }
  stablehlo::ComparisonDirectionAttr compareGeDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::GE);
  stablehlo::ComparisonDirectionAttr compareLeDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::LE);
  stablehlo::ComparisonDirectionAttr compareEqDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::EQ);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value compareResult;
    if (isa<AtenMaxDimOp>(op)) {
      compareResult = rewriter.create<stablehlo::CompareOp>(
          op->getLoc(), compareResultType, *firstValArg, *secondValArg,
          compareGeDirectionAttr, compareTypeAttr);
    } else if (isa<AtenMinDimOp>(op)) {
      compareResult = rewriter.create<stablehlo::CompareOp>(
          op->getLoc(), compareResultType, *firstValArg, *secondValArg,
          compareLeDirectionAttr, compareTypeAttr);
    } else {
      op->emitError("unimplement lowering of createReduceOpReturnIndices");
      return std::nullopt;
    }
    Value retValResult = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareResult, *firstValArg, *secondValArg);

    // get smaller index value if compared nums are equal.
    Value compareEqResult = rewriter.create<stablehlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareEqDirectionAttr, compareTypeAttr);
    Value minIdx = rewriter.create<stablehlo::MinOp>(op->getLoc(), *firstIdxArg,
                                                     *secondIdxArg);
    Value idxWithGeVal = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareResult, *firstIdxArg, *secondIdxArg);
    Value retIdxResult = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

    rewriter.create<stablehlo::ReturnOp>(
        op->getLoc(), ValueRange{retValResult, retIdxResult});
  }
  return stablehloReduceOp.getResults();
}

static Value reshapeReduceResultWhenKeepDim(ConversionPatternRewriter &rewriter,
                                            Location loc, Value reduceResult,
                                            ArrayRef<Value> inputShapeVec,
                                            Type outType,
                                            ArrayRef<int64_t> dims,
                                            size_t dimSizeIndexBits) {
  SmallVector<Value> outShapeVec(inputShapeVec);
  Value one = rewriter.create<arith::ConstantOp>(
      loc,
      rewriter.getIntegerAttr(rewriter.getIntegerType(dimSizeIndexBits), 1));
  for (auto dim : dims) {
    outShapeVec[dim] = one;
  }
  auto outShapeTensor =
      rewriter.create<tensor::FromElementsOp>(loc, outShapeVec);
  return rewriter.create<stablehlo::DynamicReshapeOp>(
      loc, outType, reduceResult, outShapeTensor);
}

namespace {
template <typename AtenOpT>
class ConvertAtenReductionOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(false && "Unimplemented");
    return failure();
  };
};

template <typename AtenOpT>
class ConvertAtenReduceAllDimsOp : public ConvertAtenReductionOp<AtenOpT> {
public:
  using ConvertAtenReductionOp<AtenOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    auto outTy = dyn_cast<RankedTensorType>(
        ConvertAtenReductionOp<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (!inputTy || !outTy) {
      return rewriter.notifyMatchFailure(
          op, "only Tensor types supported in StableHLO");
    }

    auto inputElemTy = inputTy.getElementType();
    if (!inputElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }
    if (inputElemTy != outTy.getElementType()) {
      // use output type as computation type
      input = rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input,
                                                    outTy.getElementType());
    }

    SmallVector<int64_t> dims =
        llvm::to_vector(llvm::seq<int64_t>(0, inputTy.getRank()));
    Value result =
        createReduceOpWithSingleRegionOp(op, input, outTy, dims, rewriter);
    if (!result) {
      return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenReduceOneDimOp : public ConvertAtenReductionOp<AtenOpT> {
public:
  using ConvertAtenReductionOp<AtenOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    auto outTy = dyn_cast<RankedTensorType>(
        ConvertAtenReductionOp<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (!inputTy || !outTy) {
      return rewriter.notifyMatchFailure(
          op, "only Tensor types supported in StableHLO");
    }

    auto inputElemTy = inputTy.getElementType();
    if (!inputElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }
    if (inputElemTy != outTy.getElementType()) {
      // use output type as computation type
      input = rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input,
                                                    outTy.getElementType());
    }

    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
    }

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(
          op, "non-const integer `dim` is not supported");
    }
    dim = toPositiveDim(dim, inputTy.getRank());
    SmallVector<int64_t> reduceResultShape =
        getReduceOutputShape(inputTy.getShape(), {dim});

    Value reduceResult = createReduceOpWithSingleRegionOp(
        op, input,
        RankedTensorType::get(reduceResultShape, outTy.getElementType()), {dim},
        rewriter);
    if (!reduceResult) {
      return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
    }

    if (keepDim) {
      const auto &options = ConvertAtenReductionOp<AtenOpT>::getOptions();
      auto outShapeInfo = hlo::getDimSizesOfTensor(rewriter, op, input,
                                                   options.dimSizeIndexBits);
      if (failed(outShapeInfo)) {
        return rewriter.notifyMatchFailure(
            op, "failed to get dimension sizes of the input");
      }
      reduceResult = reshapeReduceResultWhenKeepDim(
          rewriter, op->getLoc(), reduceResult, *outShapeInfo, outTy, {dim},
          options.dimSizeIndexBits);
    }
    rewriter.replaceOp(op, reduceResult);
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenReduceDimsOp : public ConvertAtenReductionOp<AtenOpT> {
public:
  using ConvertAtenReductionOp<AtenOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    auto outTy = dyn_cast<RankedTensorType>(
        ConvertAtenReductionOp<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (!inputTy || !outTy) {
      return rewriter.notifyMatchFailure(
          op, "only Tensor types supported in StableHLO");
    }

    auto inputElemTy = inputTy.getElementType();
    if (!inputElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }
    if (inputElemTy != outTy.getElementType()) {
      // use output type as computation type
      input = rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input,
                                                    outTy.getElementType());
    }

    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
    }

    SmallVector<int64_t> inputDims;
    SmallVector<int64_t> dims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(inputDims))) {
      return rewriter.notifyMatchFailure(
          op, "non-const integer `dim` is not supported");
    }
    if (inputDims.size() == 0) {
      dims = llvm::to_vector(llvm::seq<int64_t>(0, inputTy.getRank()));
    } else {
      for (auto d : inputDims) {
        d = toPositiveDim(d, inputTy.getRank());
        // Drop invalid dims
        if (isValidDim(d, inputTy.getRank())) {
          dims.push_back(d);
        }
      }
      llvm::sort(dims.begin(), dims.end());
    }
    SmallVector<int64_t> reduceResultShape =
        getReduceOutputShape(inputTy.getShape(), dims);

    Value reduceResult = createReduceOpWithSingleRegionOp(
        op, input,
        RankedTensorType::get(reduceResultShape, outTy.getElementType()), dims,
        rewriter);
    if (!reduceResult) {
      return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
    }

    if (keepDim) {
      const auto &options = ConvertAtenReductionOp<AtenOpT>::getOptions();
      auto outShapeInfo = hlo::getDimSizesOfTensor(rewriter, op, input,
                                                   options.dimSizeIndexBits);
      if (failed(outShapeInfo)) {
        return rewriter.notifyMatchFailure(
            op, "failed to get dimension sizes of the input");
      }
      reduceResult = reshapeReduceResultWhenKeepDim(
          rewriter, op->getLoc(), reduceResult, *outShapeInfo, outTy, dims,
          options.dimSizeIndexBits);
    }
    rewriter.replaceOp(op, reduceResult);
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenReduceWithIndicesOp : public ConvertAtenReductionOp<AtenOpT> {
public:
  using ConvertAtenReductionOp<AtenOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTy) {
      return rewriter.notifyMatchFailure(
          op, "only Tensor types supported in StableHLO");
    }
    auto inputElemTy = inputTy.getElementType();
    if (!inputElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    RankedTensorType valResultType = cast<RankedTensorType>(
        ConvertAtenReductionOp<AtenOpT>::getTypeConverter()->convertType(
            op.getResult(0).getType()));
    RankedTensorType idxResultType = cast<RankedTensorType>(
        ConvertAtenReductionOp<AtenOpT>::getTypeConverter()->convertType(
            op.getResult(1).getType()));
    Type idxElementType = idxResultType.getElementType();
    if (!isa<mlir::IntegerType>(idxElementType)) {
      return op.emitError("indices result should to be integer tyep");
    }

    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(op, "non-int dim unsupported");
    }
    dim = toPositiveDim(dim, inputTy.getRank());
    if (!isValidDim(dim, inputTy.getRank())) {
      return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
    }
    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
    }

    const auto &options = ConvertAtenReductionOp<AtenOpT>::getOptions();
    auto inputShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(inputShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    auto inputShapeVec = *inputShapeInfo;

    if (op.getResult(1).use_empty()) {
      llvm::SmallVector<int64_t> outputShape(inputTy.getShape());
      outputShape.erase(outputShape.begin() + dim);
      Value reduceResult = createReduceOpWithSingleRegionOp(
          op, input, RankedTensorType::get(outputShape, inputElemTy),
          ArrayRef<int64_t>{dim}, rewriter);
      if (!reduceResult) {
        return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
      }

      if (keepDim) {
        reduceResult = reshapeReduceResultWhenKeepDim(
            rewriter, op->getLoc(), reduceResult, inputShapeVec, valResultType,
            {dim}, options.dimSizeIndexBits);
      }
      rewriter.replaceOp(op, {reduceResult, Value()});
      return success();
    } else {
      ValueRange stablehloReduceResults =
          createReduceOpReturnIndices(rewriter, op, input, inputShapeVec, dim,
                                      options.dimSizeIndexBits)
              .value();
      if (keepDim) {
        stablehloReduceResults[0] = reshapeReduceResultWhenKeepDim(
            rewriter, op->getLoc(), stablehloReduceResults[0], inputShapeVec,
            valResultType, {dim}, options.dimSizeIndexBits);
        stablehloReduceResults[1] = reshapeReduceResultWhenKeepDim(
            rewriter, op->getLoc(), stablehloReduceResults[1], inputShapeVec,
            idxResultType, {dim}, options.dimSizeIndexBits);
      }
      rewriter.replaceOp(
          op, {stablehloReduceResults[0], stablehloReduceResults[1]});
      return success();
    }
  };
};
} // namespace

// AtenSumDimIntListOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenSumDimIntListOp>::matchAndRewrite(
    AtenSumDimIntListOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  auto outTy =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  if (inputTy.getElementType() != outTy.getElementType()) {
    // Use output element type as computation type.
    auto dstElemTy = outTy.getElementType();
    input =
        rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input, dstElemTy);
    inputTy = dyn_cast<RankedTensorType>(input.getType());
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> inputDims;
  SmallVector<int64_t> dims;
  if (failed(checkNotNone(rewriter, op, op.getDim()))) {
    inputDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, inputTy.getRank()));
  } else {
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(inputDims))) {
      return rewriter.notifyMatchFailure(
          op, "non-const integer `dim` is not supported");
    }
    if (inputDims.size() == 0) {
      inputDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, inputTy.getRank()));
    }
  }
  for (auto d : inputDims) {
    d = toPositiveDim(d, inputTy.getRank());
    // Drop invalid dims
    if (isValidDim(d, inputTy.getRank())) {
      dims.push_back(d);
    }
  }
  llvm::sort(dims.begin(), dims.end());

  SmallVector<int64_t> reduceResultShape =
      getReduceOutputShape(inputTy.getShape(), dims);

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }

  Value reduceResult = createReduceOpWithSingleRegionOp(
      op, input,
      RankedTensorType::get(reduceResultShape, outTy.getElementType()), dims,
      rewriter);
  if (!reduceResult) {
    return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
  }

  if (keepDim) {
    const auto &options = getOptions();
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    reduceResult = reshapeReduceResultWhenKeepDim(
        rewriter, op->getLoc(), reduceResult, *outShapeInfo, outTy, dims,
        options.dimSizeIndexBits);
  }
  rewriter.replaceOp(op, reduceResult);
  return success();
}
} // namespace

// AtenFrobeniusNormDimOp
// aten.frobenius_norm.dim => stablehlo.reduce(calculate square sum along given
// dims) + stablehlo.sqrt
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenFrobeniusNormDimOp>::matchAndRewrite(
    AtenFrobeniusNormDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  const TorchToStablehloOptions &options = getOptions();

  Value input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return op.emitError(
        "only ranked tensor input supported in AtenFrobeniusNormDimOp");
  }
  auto inputRank = inputType.getRank();
  auto inputElemType = inputType.getElementType();
  if (!isa<mlir::FloatType>(inputElemType)) {
    return op.emitError(
        "only float dtype allowed in input tensor of AtenFrobeniusNormDimOp");
  }

  SmallVector<int64_t> dims;
  if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims))) {
    return rewriter.notifyMatchFailure(
        op, "non-const integer `dim` is not supported");
  }
  for (auto &dim : dims) {
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank)) {
      return rewriter.notifyMatchFailure(op,
                                         "invalid dimension detected in `dim`");
    }
  }
  // Sort the dims in ascending order, making the conversion
  // stable with unordered dims.
  std::sort(dims.begin(), dims.end());

  SmallVector<int64_t> reduceResultShape =
      getReduceOutputShape(inputType.getShape(), dims);

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(
        op, "non-const bool `keepdim` is not supported");
  }

  auto squareOp = rewriter.create<stablehlo::MulOp>(op->getLoc(), input, input);

  Value reduceResult = createReduceOpWithSingleRegionOp(
      op, squareOp.getResult(),
      RankedTensorType::get(reduceResultShape, inputElemType), dims, rewriter);
  if (!reduceResult) {
    return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
  }

  Value output = rewriter.create<stablehlo::SqrtOp>(op->getLoc(), reduceResult);

  if (keepDim) {
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    output = reshapeReduceResultWhenKeepDim(
        rewriter, op->getLoc(), output, *outShapeInfo,
        getTypeConverter()->convertType(op.getType()), dims,
        options.dimSizeIndexBits);
  }
  rewriter.replaceOp(op, output);
  return success();
}
} // namespace

// AtenLinalgVectorNormOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenLinalgVectorNormOp>::matchAndRewrite(
    AtenLinalgVectorNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  const TorchToStablehloOptions &options = getOptions();

  Value input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return op.emitError(
        "only ranked tensor input supported in AtenLinalgVectorNormOp");
  }
  int64_t inputRank = inputType.getRank();

  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  auto outElemType = outType.getElementType();
  if (!isa<mlir::FloatType>(outElemType)) {
    return op.emitError("only float dtype allowed in AtenLinalgVectorNormOp");
  }

  if (inputType.getElementType() != outType.getElementType()) {
    input =
        rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input, outElemType);
  }

  Value ord =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getOrd(), outElemType);

  SmallVector<int64_t> dims;
  if (failed(checkNotNone(rewriter, op, op.getDim()))) {
    dims = llvm::to_vector<4>(llvm::seq<int64_t>(0, inputRank));
  } else {
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims))) {
      return rewriter.notifyMatchFailure(
          op, "non-const integer `dim` is not supported");
    }

    for (auto &dim : dims) {
      dim = toPositiveDim(dim, inputRank);
      if (!isValidDim(dim, inputRank)) {
        return rewriter.notifyMatchFailure(
            op, "invalid dimension detected in `dim`");
      }
    }
    // Sort the dims in ascending order, making the conversion
    // stable with unordered dims.
    std::sort(dims.begin(), dims.end());
  }

  SmallVector<int64_t> reduceResultShape =
      getReduceOutputShape(inputType.getShape(), dims);

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(
        op, "non-const bool `keepdim` is not supported");
  }

  Value absValue = rewriter.create<stablehlo::AbsOp>(op->getLoc(), input);
  Value powValue = rewriter.create<chlo::BroadcastPowOp>(op->getLoc(), absValue,
                                                         ord, nullptr);

  Value reduceResult = createReduceOpWithSingleRegionOp(
      op, powValue, RankedTensorType::get(reduceResultShape, outElemType), dims,
      rewriter);
  if (!reduceResult) {
    return op->emitError("createReduceOpWithSingleRegionOp return nullptr");
  }

  auto scalarType = RankedTensorType::get({}, outElemType);
  auto constantOne = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), scalarType,
      DenseElementsAttr::get(
          scalarType,
          APFloat(cast<mlir::FloatType>(outElemType).getFloatSemantics(), 1)));
  auto reciprocalOrd = rewriter.create<stablehlo::DivOp>(
      op->getLoc(), scalarType, constantOne, ord);
  Value output = rewriter.create<chlo::BroadcastPowOp>(
      op->getLoc(), reduceResult, reciprocalOrd, nullptr);

  if (keepDim) {
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    output = reshapeReduceResultWhenKeepDim(rewriter, op->getLoc(), output,
                                            *outShapeInfo, outType, dims,
                                            options.dimSizeIndexBits);
  }
  rewriter.replaceOp(op, output);
  return success();
}
} // namespace

void mlir::torch::torch_to_stablehlo::populateReductionOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();
#define INSERT_ATEN_REDUCTION_OP_PATTERN(AtenOp)                               \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReductionOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenSumDimIntListOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenFrobeniusNormDimOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenLinalgVectorNormOp);
#undef INSERT_ATEN_REDUCTION_OP_PATTERN

#define INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenOp)                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReduceAllDimsOp<AtenOp>>(typeConverter, context,     \
                                                   options)
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenMaxOp);
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenMinOp);
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenSumOp);
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenProdOp);
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenAllOp);
  INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN(AtenAnyOp);
#undef INSERT_ATEN_REDUCTION_ALL_DIMS_OP_PATTERN

#define INSERT_ATEN_REDUCTION_ONE_DIM_OP_PATTERN(AtenOp)                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReduceOneDimOp<AtenOp>>(typeConverter, context,      \
                                                  options)
  INSERT_ATEN_REDUCTION_ONE_DIM_OP_PATTERN(AtenAnyDimOp);
#undef INSERT_ATEN_REDUCTION_ONE_DIM_OP_PATTERN

#define INSERT_ATEN_REDUCTION_DIMS_OP_PATTERN(AtenOp)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReduceDimsOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATEN_REDUCTION_DIMS_OP_PATTERN(AtenAmaxOp);
  INSERT_ATEN_REDUCTION_DIMS_OP_PATTERN(AtenAminOp);
#undef INSERT_ATEN_REDUCTION_DIMS_OP_PATTERN

#define INSERT_ATEN_REDUCTION_WITH_INDICES_PATTERN(AtenOp)                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReduceWithIndicesOp<AtenOp>>(typeConverter, context, \
                                                       options)
  INSERT_ATEN_REDUCTION_WITH_INDICES_PATTERN(AtenMaxDimOp);
  INSERT_ATEN_REDUCTION_WITH_INDICES_PATTERN(AtenMinDimOp);
#undef INSERT_ATEN_REDUCTION_WITH_INDICES_PATTERN
}
