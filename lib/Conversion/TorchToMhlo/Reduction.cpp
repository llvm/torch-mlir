//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "../PassDetail.h"
#include "./MhloLegalizeUtils.h"
#include "./PopulatePatterns.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_mhlo;

static Value createInitialValueForReduceOp(Operation *op, Type elementTy,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<AtenSumOp, AtenSumDimIntListOp, AtenFrobeniusNormDimOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<mhlo::ConstantOp>(op->getLoc(), constType,
                                               constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<mhlo::ConstantOp>(op->getLoc(), constType,
                                               constAttr);
    }
  }

  if (isa<AtenMaxOp, AtenMaxDimOp, AtenArgmaxOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getLargest(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/true)});
      return rewriter.create<mhlo::ConstantOp>(op->getLoc(), constType,
                                               constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<mhlo::ConstantOp>(op->getLoc(), constType,
                                               constAttr);
    }
  }

  op->emitError("unimplemented lowering in "
                "createInitialValueForReduceOp");
  return nullptr;
}

// Util for converting AtenArgmaxOp and AtenMaxDimOp
static llvm::Optional<ValueRange>
getMaxInDim(ConversionPatternRewriter &rewriter, Operation *op, Value &input,
            ArrayRef<Value> inputShapeVec, int64_t dim,
            size_t dimSizeIndexBits) {
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return llvm::None;
  }
  if (!inputTy.getElementType().isIntOrFloat()) {
    return llvm::None;
  }
  auto inputShape = inputTy.getShape();
  auto inputElemTy = inputTy.getElementType();

  Value initValue = createInitialValueForReduceOp(op, inputElemTy, rewriter);
  if (!initValue) return llvm::None;
  Value initIndex;
  if (dimSizeIndexBits == 32) {
    initIndex = mhlo::getConstTensor<int32_t>(rewriter, op, {0}, {}).value();
  } else {
    initIndex = mhlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).value();
  }

  DenseIntElementsAttr dimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI64Type()), dim);

  auto inputShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), inputShapeVec);
  auto indexTensor = rewriter.create<mhlo::DynamicIotaOp>(
      op->getLoc(),
      RankedTensorType::get(inputShape,
                            rewriter.getIntegerType(dimSizeIndexBits)),
      inputShapeTensor, static_cast<uint64_t>(dim));

  auto mhloReduceOp = rewriter.create<mhlo::ReduceOp>(
      op->getLoc(), ValueRange{input, indexTensor},
      ValueRange{
          initValue,
          initIndex,
      },
      dimensions);

  Block &block = mhloReduceOp.body().emplaceBlock();

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

  mhlo::ComparisonTypeAttr compareTypeAttr;
  if (inputTy.getElementType().isa<mlir::FloatType>()) {
    compareTypeAttr = mhlo::ComparisonTypeAttr::get(
        rewriter.getContext(), mhlo::ComparisonType::FLOAT);
  } else if (inputTy.getElementType().isa<mlir::IntegerType>()) {
    compareTypeAttr = mhlo::ComparisonTypeAttr::get(
        rewriter.getContext(), mhlo::ComparisonType::SIGNED);
  }
  mhlo::ComparisonDirectionAttr compareGeDirectionAttr =
      mhlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                         mhlo::ComparisonDirection::GE);
  mhlo::ComparisonDirectionAttr compareEqDirectionAttr =
      mhlo::ComparisonDirectionAttr::get(rewriter.getContext(),
                                         mhlo::ComparisonDirection::EQ);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value compareGeResult = rewriter.create<mhlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareGeDirectionAttr, compareTypeAttr);
    Value retValResult = rewriter.create<mhlo::SelectOp>(
        op->getLoc(), compareGeResult, *firstValArg, *secondValArg);

    // get smaller index value if compared nums are equal.
    Value compareEqResult = rewriter.create<mhlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareEqDirectionAttr, compareTypeAttr);
    Value minIdx =
        rewriter.create<mhlo::MinOp>(op->getLoc(), *firstIdxArg, *secondIdxArg);
    Value idxWithGeVal = rewriter.create<mhlo::SelectOp>(
        op->getLoc(), compareGeResult, *firstIdxArg, *secondIdxArg);
    Value retIdxResult = rewriter.create<mhlo::SelectOp>(
        op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

    rewriter.create<mhlo::ReturnOp>(
        op->getLoc(), mlir::ValueRange{retValResult, retIdxResult});
  }
  return mhloReduceOp.getResults();
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
  Value input = adaptor.self();
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "only Tensor types supported in MHLO");
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
            "AtenArgmaxOp to MHLO");
  }

  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op, "non-int dim unsupported");
  }
  dim = toPositiveDim(dim, inputTy.getRank());
  if (!isValidDim(dim, inputTy.getRank())) {
    return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
  }

  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }

  const auto &options = getOptions();
  auto inputShapeInfo =
      mhlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto mhloReduceResults = getMaxInDim(rewriter, op, input, inputShapeVec, dim,
                                       options.dimSizeIndexBits)
                               .value();

  if (keepDim) {
    auto outShapeVec = inputShapeVec;
    outShapeVec[dim] = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));

    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        op, typeConverter->convertType(op.getType()), mhloReduceResults[1],
        outShapeTensor);
    return success();
  }

  rewriter.replaceOp(op, mhloReduceResults[1]);
  return success();
}
} // namespace

// AtenMaxDimOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenMaxDimOp>::matchAndRewrite(
    AtenMaxDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "only Tensor types supported in MHLO");
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
            "AtenMaxDimOp to MHLO");
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
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op, "non-int dim unsupported");
  }
  dim = toPositiveDim(dim, inputTy.getRank());
  if (!isValidDim(dim, inputTy.getRank())) {
    return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
  }
  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }

  const auto &options = getOptions();
  auto inputShapeInfo =
      mhlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto mhloReduceResults = getMaxInDim(rewriter, op, input, inputShapeVec, dim,
                                       options.dimSizeIndexBits)
                               .value();

  if (keepDim) {
    auto outShapeVec = inputShapeVec;
    outShapeVec[dim] = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(
            rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);

    auto mhloReduceValueResult = rewriter.create<mhlo::DynamicReshapeOp>(
        op->getLoc(), valResultType, mhloReduceResults[0], outShapeTensor);
    auto mhloReduceIndexResult = rewriter.create<mhlo::DynamicReshapeOp>(
        op->getLoc(), idxResultType, mhloReduceResults[1], outShapeTensor);
    rewriter.replaceOp(op, {mhloReduceValueResult, mhloReduceIndexResult});
    return success();
  }

  rewriter.replaceOp(op, {mhloReduceResults[0], mhloReduceResults[1]});
  return success();
}
} // namespace

// AtenSumOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenSumOp>::matchAndRewrite(
    AtenSumOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "only Tensor types supported in MHLO");
  }
  auto dtype = adaptor.dtype();
  if (!dtype.getType().isa<Torch::NoneType>()) {
    auto dstElemTy = getTypeConverter()
                         ->convertType(op.getType())
                         .template dyn_cast<RankedTensorType>()
                         .getElementType();
    input = rewriter.create<mhlo::ConvertOp>(op->getLoc(), input, dstElemTy);
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
            "AtenSumOp to MHLO");
  }

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    dims.push_back(i);
  }

  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue) return failure();

  auto mhloReduceOp = rewriter.create<mhlo::ReduceOp>(
      op.getLoc(), input, initValue, rewriter.getI64TensorAttr(dims));

  Block &block = mhloReduceOp.body().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<mhlo::AddOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), addResult);
  }

  rewriter.replaceOp(op, mhloReduceOp.getResults());
  return success();
}
} // namespace

// AtenMaxOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenMaxOp>::matchAndRewrite(
    AtenMaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "only Tensor types supported in MHLO");
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
            "AtenMaxOp to MHLO");
  }

  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    dims.push_back(i);
  }

  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue) return failure();
  auto mhloReduceOp = rewriter.create<mhlo::ReduceOp>(
      op.getLoc(), input, initValue, rewriter.getI64TensorAttr(dims));

  Block &block = mhloReduceOp.body().emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value maxResult = rewriter.create<mhlo::MaxOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), maxResult);
  }

  rewriter.replaceOp(op, mhloReduceOp.getResults());
  return success();
}
} // namespace

// AtenSumDimIntListOp
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenSumDimIntListOp>::matchAndRewrite(
    AtenSumDimIntListOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(op, "only Tensor types supported in MHLO");
  }
  auto dtype = adaptor.dtype();
  if (!dtype.getType().isa<Torch::NoneType>()) {
    auto dstElemTy = getTypeConverter()
                         ->convertType(op.getType())
                         .template dyn_cast<RankedTensorType>()
                         .getElementType();
    input = rewriter.create<mhlo::ConvertOp>(op->getLoc(), input, dstElemTy);
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
            "AtenSumDimIntListOp to MHLO");
  }

  SmallVector<int64_t> inputDims;
  SmallVector<int64_t> dims;
  if (!matchPattern(op.dim(), m_TorchConstantIntList(inputDims))) {
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

  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }
  Value initValue =
      createInitialValueForReduceOp(op, inputTy.getElementType(), rewriter);
  if (!initValue) return failure();

  auto mhloReduceOp = rewriter.create<mhlo::ReduceOp>(
      op.getLoc(), input, initValue, rewriter.getI64TensorAttr(dims));

  Region &region = mhloReduceOp.body();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputTy.getElementType());

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value addResult = rewriter.create<mhlo::AddOp>(
        op->getLoc(), blockArgumentTy, *firstArgument, *secondArgument);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), addResult);
  }

  if (keepDim) {
    const auto &options = getOptions();
    auto outShapeInfo = mhlo::getDimSizesOfTensor(rewriter, op, input,
                                                  options.dimSizeIndexBits);
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
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()),
        mhloReduceOp.getResult(0), outShapeTensor);
    return success();
  }
  rewriter.replaceOp(op, mhloReduceOp.getResults());
  return success();
}
} // namespace

// AtenFrobeniusNormDimOp
// aten.frobenius_norm.dim => mhlo.reduce(calculate square sum along given dims)
//                            + mhlo.sqrt
namespace {
template <>
LogicalResult ConvertAtenReductionOp<AtenFrobeniusNormDimOp>::matchAndRewrite(
    AtenFrobeniusNormDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  const TorchToMhloOptions &options = getOptions();

  Value input = adaptor.self();
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
  if (!matchPattern(op.dim(), m_TorchConstantIntList(dims))) {
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

  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(
        op, "non-const bool `keepdim` is not supported");
  }

  auto initValue = createInitialValueForReduceOp(op, inputElemType, rewriter);
  if (!initValue) {
    return failure();
  }

  auto squareSumReduceOp = rewriter.create<mhlo::ReduceOp>(
      op->getLoc(), input, initValue, rewriter.getI64TensorAttr(dims));

  Region &region = squareSumReduceOp.body();
  Block &block = region.emplaceBlock();
  auto blockArgumentTy = RankedTensorType::get({}, inputElemType);

  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArgument = block.args_begin();
  auto secondArgument = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    auto constantOrd2 = rewriter.create<mhlo::ConstantOp>(
        op->getLoc(), blockArgumentTy,
        DenseElementsAttr::get(blockArgumentTy, llvm::ArrayRef<float>{2.0}));
    auto abs = rewriter.create<mhlo::AbsOp>(op->getLoc(), *secondArgument);
    auto squareResult = rewriter.create<mhlo::PowOp>(
        op->getLoc(), abs, constantOrd2);
    auto addResult = rewriter.create<mhlo::AddOp>(op->getLoc(), squareResult,
                                                  *firstArgument);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), addResult.getResult());
  }

  auto output = rewriter.create<mhlo::SqrtOp>(op->getLoc(),
                                              squareSumReduceOp.getResult(0));

  if (keepDim) {
    auto outShapeInfo = mhlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
    if (failed(outShapeInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to get dimension sizes of the input");
    }
    auto outShapeVec = *outShapeInfo;
    auto one = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getIntegerAttr(
                          rewriter.getIntegerType(options.dimSizeIndexBits), 1));
    for (int64_t i : dims) {
      outShapeVec[i] = one;
    }
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        op, getTypeConverter()->convertType(op.getType()), output,
        outShapeTensor);
    return success();
  }
  rewriter.replaceOp(op, output.getResult());
  return success();
}
} // namespace

void mlir::torch::torch_to_mhlo::populateReductionOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToMhloOptions &options) {
  MLIRContext *context = patterns.getContext();
#define INSERT_ATEN_REDUCTION_OP_PATTERN(AtenOp)                               \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenReductionOp<AtenOp>>(typeConverter, context, options)
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenArgmaxOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenMaxDimOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenSumDimIntListOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenSumOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenMaxOp);
  INSERT_ATEN_REDUCTION_OP_PATTERN(AtenFrobeniusNormDimOp);
#undef INSERT_ATEN_REDUCTION_OP_PATTERN
}
