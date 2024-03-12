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
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include <unordered_set>
#include <vector>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

static Value createInitialValueForReduceOp(Operation *op, Type elementTy,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<AtenSumOp, AtenSumDimIntListOp, AtenFrobeniusNormDimOp,
          AtenLinalgVectorNormOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenMaxOp, AtenMaxDimOp, AtenArgmaxOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getInf(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/true)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenMinOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getInf(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  op->emitError("unimplemented lowering in "
                "createInitialValueForReduceOp");
  return nullptr;
}

// Util for converting AtenArgmaxOp and AtenMaxDimOp
static std::optional<ValueRange>
getMaxInDim(ConversionPatternRewriter &rewriter, Operation *op, Value &input,
            ArrayRef<Value> inputShapeVec, int64_t dim,
            size_t dimSizeIndexBits) {
  auto inputTy = input.getType().template cast<RankedTensorType>();
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

  std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
  outputShape.erase(outputShape.begin() + dim);
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
  if (inputTy.getElementType().isa<mlir::FloatType>()) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::FLOAT);
  } else if (inputTy.getElementType().isa<mlir::IntegerType>()) {
    compareTypeAttr = stablehlo::ComparisonTypeAttr::get(
        rewriter.getContext(), stablehlo::ComparisonType::SIGNED);
  }
  stablehlo::ComparisonDirectionAttr compareGeDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::GE);
  stablehlo::ComparisonDirectionAttr compareEqDirectionAttr =
      stablehlo::ComparisonDirectionAttr::get(
          rewriter.getContext(), stablehlo::ComparisonDirection::EQ);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value compareGeResult = rewriter.create<stablehlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareGeDirectionAttr, compareTypeAttr);
    Value retValResult = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareGeResult, *firstValArg, *secondValArg);

    // get smaller index value if compared nums are equal.
    Value compareEqResult = rewriter.create<stablehlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareEqDirectionAttr, compareTypeAttr);
    Value minIdx = rewriter.create<stablehlo::MinOp>(op->getLoc(), *firstIdxArg,
                                                     *secondIdxArg);
    Value idxWithGeVal = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareGeResult, *firstIdxArg, *secondIdxArg);
    Value retIdxResult = rewriter.create<stablehlo::SelectOp>(
        op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

    rewriter.create<stablehlo::ReturnOp>(
        op->getLoc(), mlir::ValueRange{retValResult, retIdxResult});
  }
  return stablehloReduceOp.getResults();
}

namespace {
template <typename AtenOpT>
class ConvertAtenReductionOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

// AtenArgmaxOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenArgmaxOp>::matchAndRewrite(
    AtenArgmaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }

  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported!
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenArgmaxOp to StableHLO");
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

  const auto &options = getOptions();
  auto inputShapeInfo =
      hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto stablehloReduceResults = getMaxInDim(rewriter, op, input, inputShapeVec,
                                            dim, options.dimSizeIndexBits)
                                    .value();

  if (keepDim) {
    auto outShapeVec = inputShapeVec;
    outShapeVec[dim] = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));

    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op, typeConverter->convertType(op.getType()), stablehloReduceResults[1],
        outShapeTensor);
    return success();
  }

  rewriter.replaceOp(op, stablehloReduceResults[1]);
  return success();
}
} // namespace

// AtenMaxDimOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenMaxDimOp>::matchAndRewrite(
    AtenMaxDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenMaxDimOp to StableHLO");
  }

  RankedTensorType valResultType = getTypeConverter()
                                       ->convertType(op.getResult(0).getType())
                                       .template cast<RankedTensorType>();
  RankedTensorType idxResultType = getTypeConverter()
                                       ->convertType(op.getResult(1).getType())
                                       .template cast<RankedTensorType>();
  Type idxElementType = idxResultType.getElementType();
  if (!idxElementType.isa<mlir::IntegerType>()) {
    return op.emitError("Aten.max.dim needs integer-like result");
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

  const auto &options = getOptions();
  auto inputShapeInfo =
      hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto stablehloReduceResults = getMaxInDim(rewriter, op, input, inputShapeVec,
                                            dim, options.dimSizeIndexBits)
                                    .value();

  if (keepDim) {
    auto outShapeVec = inputShapeVec;
    outShapeVec[dim] = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);

    auto stablehloReduceValueResult =
        rewriter.create<stablehlo::DynamicReshapeOp>(
            op->getLoc(), valResultType, stablehloReduceResults[0],
            outShapeTensor);
    auto stablehloReduceIndexResult =
        rewriter.create<stablehlo::DynamicReshapeOp>(
            op->getLoc(), idxResultType, stablehloReduceResults[1],
            outShapeTensor);
    rewriter.replaceOp(
        op, {stablehloReduceValueResult, stablehloReduceIndexResult});
    return success();
  }

  rewriter.replaceOp(op,
                     {stablehloReduceResults[0], stablehloReduceResults[1]});
  return success();
}
} // namespace

// AtenSumOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenSumOp>::matchAndRewrite(
    AtenSumOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  if (inputTy.getElementType() != outTy.getElementType()) {
    // Use output element type as computation type.
    auto dstElemTy = outTy.getElementType();
    input =
        rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input, dstElemTy);
    inputTy = input.getType().dyn_cast<RankedTensorType>();
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenSumOp to StableHLO");
  }

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    dims.push_back(i);
  }
  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue)
    return failure();

  llvm::sort(dims.begin(), dims.end());
  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op.getLoc(), RankedTensorType::get({}, outTy.getElementType()), input,
      initValue, rewriter.getDenseI64ArrayAttr(dims));

  Block &block = stablehloReduceOp.getBody().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outTy,
                                              stablehloReduceOp.getResults());
  return success();
}
} // namespace

// AtenMaxOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenMaxOp>::matchAndRewrite(
    AtenMaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenMaxOp to StableHLO");
  }

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    dims.push_back(i);
  }

  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue)
    return failure();
  llvm::sort(dims.begin(), dims.end());
  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op.getLoc(), RankedTensorType::get({}, inputElemTy), input, initValue,
      rewriter.getDenseI64ArrayAttr(dims));

  Block &block = stablehloReduceOp.getBody().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value maxResult = rewriter.create<stablehlo::MaxOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), maxResult);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()),
      stablehloReduceOp.getResults());
  return success();
}
} // namespace

// AtenMinOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenMinOp>::matchAndRewrite(
    AtenMinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenMinOp to StableHLO");
  }

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    dims.push_back(i);
  }

  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue)
    return failure();
  llvm::sort(dims.begin(), dims.end());
  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op.getLoc(), RankedTensorType::get({}, inputElemTy), input, initValue,
      rewriter.getDenseI64ArrayAttr(dims));

  Block &block = stablehloReduceOp.getBody().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value minResult = rewriter.create<stablehlo::MinOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), minResult);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()),
      stablehloReduceOp.getResults());
  return success();
}
} // namespace

// AtenSumDimIntListOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenSumDimIntListOp>::matchAndRewrite(
    AtenSumDimIntListOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in StableHLO");
  }
  if (inputTy.getElementType() != outTy.getElementType()) {
    // Use output element type as computation type.
    auto dstElemTy = outTy.getElementType();
    input =
        rewriter.create<stablehlo::ConvertOp>(op->getLoc(), input, dstElemTy);
    inputTy = input.getType().dyn_cast<RankedTensorType>();
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op, "IntegerType with bitwidth 8 unsupported in convertion from "
            "AtenSumDimIntListOp to StableHLO");
  }

  SmallVector<int64_t> inputDims;
  SmallVector<int64_t> dims;
  if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(inputDims))) {
    return rewriter.notifyMatchFailure(op, "non-int dim list unsupported");
  }
  if (inputDims.size() == 0) {
    inputDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, inputTy.getRank()));
  }

  for (auto d : inputDims) {
    d = toPositiveDim(d, inputTy.getRank());
    // Drop invalid dims
    if (isValidDim(d, inputTy.getRank())) {
      dims.push_back(d);
    }
  }

  std::unordered_set<int64_t> dimsSet(dims.begin(), dims.end());
  SmallVector<int64_t> reduceResultShape;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    if (dimsSet.find(i) == dimsSet.end()) {
      reduceResultShape.push_back(inputTy.getDimSize(i));
    }
  }

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }
  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue)
    return failure();

  llvm::sort(dims.begin(), dims.end());
  auto stablehloReduceOp = rewriter.create<stablehlo::ReduceOp>(
      op.getLoc(),
      RankedTensorType::get(reduceResultShape, outTy.getElementType()), input,
      initValue, rewriter.getDenseI64ArrayAttr(dims));

  Region &region = stablehloReduceOp.getBody();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult);
  }

  if (keepDim) {
    const auto &options = getOptions();
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    auto outShapeVec = *outShapeInfo;
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    for (int64_t i : dims) {
      outShapeVec[i] = one;
    }
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()),
        stablehloReduceOp.getResult(0), outShapeTensor);
    return success();
  }
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outTy,
                                              stablehloReduceOp.getResults());
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
  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return op.emitError(
        "only ranked tensor input supported in AtenFrobeniusNormDimOp");
  }
  auto inputRank = inputType.getRank();
  auto inputElemType = inputType.getElementType();
  if (!inputElemType.isa<mlir::FloatType>()) {
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

  std::unordered_set<int64_t> dimsSet(dims.begin(), dims.end());
  SmallVector<int64_t> reduceResultShape;
  for (int64_t i = 0; i < inputRank; i++) {
    if (dimsSet.find(i) == dimsSet.end()) {
      reduceResultShape.push_back(inputType.getDimSize(i));
    }
  }

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(
        op, "non-const bool `keepdim` is not supported");
  }

  auto squareOp = rewriter.create<stablehlo::MulOp>(op->getLoc(), input, input);

  auto initValue = createInitialValueForReduceOp(op, inputElemType, rewriter);
  if (!initValue) {
    return failure();
  }

  auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
      op->getLoc(), RankedTensorType::get(reduceResultShape, inputElemType),
      squareOp.getResult(), initValue, rewriter.getDenseI64ArrayAttr(dims));

  Region &region = reduceOp.getBody();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputElemType);

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto firstArgument = *block.args_begin();
  auto secondArgument = *block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    auto addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), firstArgument, secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult.getResult());
  }

  auto output =
      rewriter.create<stablehlo::SqrtOp>(op->getLoc(), reduceOp.getResult(0));

  if (keepDim) {
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    auto outShapeVec = *outShapeInfo;
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    for (int64_t i : dims) {
      outShapeVec[i] = one;
    }
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), output,
        outShapeTensor);
    return success();
  }
  rewriter.replaceOp(op, output.getResult());
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
  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return op.emitError(
        "only ranked tensor input supported in AtenLinalgVectorNormOp");
  }
  int64_t inputRank = inputType.getRank();

  auto outType =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  auto outElemType = outType.getElementType();
  if (!outElemType.isa<mlir::FloatType>()) {
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

  std::unordered_set<int64_t> dimsSet(dims.begin(), dims.end());
  SmallVector<int64_t> reduceResultShape;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    if (dimsSet.find(i) == dimsSet.end()) {
      reduceResultShape.push_back(inputType.getDimSize(i));
    }
  }

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(
        op, "non-const bool `keepdim` is not supported");
  }

  auto initValue = createInitialValueForReduceOp(op, outElemType, rewriter);
  if (!initValue) {
    return failure();
  }

  Value absValue = rewriter.create<stablehlo::AbsOp>(op->getLoc(), input);
  Value powValue = rewriter.create<chlo::BroadcastPowOp>(op->getLoc(), absValue,
                                                         ord, nullptr);

  auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
      op->getLoc(), RankedTensorType::get(reduceResultShape, outElemType),
      powValue, initValue, rewriter.getDenseI64ArrayAttr(dims));

  Region &region = reduceOp.getBody();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, outElemType);

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto firstArgument = *block.args_begin();
  auto secondArgument = *block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    auto addResult = rewriter.create<stablehlo::AddOp>(
        op->getLoc(), firstArgument, secondArgument);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult.getResult());
  }
  auto constantOne = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), blockArgumentTy,
      DenseElementsAttr::get(
          blockArgumentTy,
          APFloat(outElemType.cast<mlir::FloatType>().getFloatSemantics(), 1)));
  auto reciprocalOrd = rewriter.create<stablehlo::DivOp>(
      op->getLoc(), blockArgumentTy, constantOne, ord);
  auto output = rewriter.create<chlo::BroadcastPowOp>(
      op->getLoc(), reduceOp.getResult(0), reciprocalOrd, nullptr);

  if (keepDim) {
    auto outShapeInfo =
        hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    auto outShapeVec = *outShapeInfo;
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    for (int64_t i : dims) {
      outShapeVec[i] = one;
    }
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), output,
        outShapeTensor);
    return success();
  }

  rewriter.replaceOp(op, output.getResult());
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
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenArgmaxOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenMaxDimOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenSumDimIntListOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenSumOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenMaxOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenMinOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenFrobeniusNormDimOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenLinalgVectorNormOp);
#undef INSERT_ATEN_REDUCTION_OP_PATTERN
}
