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
#include <iostream>
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

static Value createInitialValueForAtenPoolingOp(Operation *op, Type elementTy,
                                                PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  // Avg pooling
  if (isa<AtenAvgPool1dOp, AtenAdaptiveAvgPool2dOp, AtenAvgPool2dOp,
          AtenCumsumOp>(op)) {
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

  // Max pooling
  if (isa<AtenMaxPool2dOp, AtenMaxPool2dWithIndicesOp>(op)) {
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
  op->emitError("unimplemented lowering in AtenPoolingOp");
  return nullptr;
}

// AtenMaxPool2dOp
template <>
LogicalResult ConvertAtenOp<AtenMaxPool2dOp>::matchAndRewrite(
    AtenMaxPool2dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputElemTy = inputTy.getElementType();

  auto inputRank = inputTy.getRank();
  auto outTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

  if (inputRank <= 2) {
    return op.emitError(
        "max_pooling2d only supports inputs with rank higher than 2");
  }
  SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;
  bool ceilMode = false;

  if (!(matchPattern(op.getKernelSize(),
                     m_TorchListOfConstantInts(kernelSize)))) {
    return rewriter.notifyMatchFailure(
        op, "non-const int kernel size unsupported!");
  }
  if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
    return rewriter.notifyMatchFailure(op, "non-const int stride unsupported!");
  }
  if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int padding unsupported!");
  }
  if (!(matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int dilation unsupported!");
  }
  if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const bool ceil_mode unsupported!");
  }

  // prepend 1 to kernelSize, stride, dilation until they are of same rank as
  // input
  SmallVector<int64_t> stablehloStride(inputRank, 1);
  SmallVector<int64_t> stablehloDilation(inputRank, 1);
  SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
  SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);
  std::copy(dilation.begin(), dilation.end(),
            stablehloDilation.begin() + inputRank - 2);
  std::copy(stride.begin(), stride.end(),
            stablehloStride.begin() + inputRank - 2);
  std::copy(kernelSize.begin(), kernelSize.end(),
            stablehloKernelSize.begin() + inputRank - 2);

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  stablehloPadding[stablehloPadding.size() - 4] = padding[0];
  stablehloPadding[stablehloPadding.size() - 3] = padding[0];
  stablehloPadding[stablehloPadding.size() - 2] = padding[1];
  stablehloPadding[stablehloPadding.size() - 1] = padding[1];

  auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
  auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
  DenseI64ArrayAttr baseDilations;
  auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      stablehloPadding);
  auto reduceWindowOp = rewriter.create<stablehlo::ReduceWindowOp>(
      op->getLoc(), outTy, input, initVal, windowDimensions, windowStrides,
      baseDilations, windowDilations, pad);

  Block &block = reduceWindowOp.getBody().emplaceBlock();

  auto blockArgumentTy = RankedTensorType::get({}, inputElemTy);
  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArg = block.args_begin();
  auto secondArg = block.args_rbegin();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value result =
        rewriter.create<stablehlo::MaxOp>(op->getLoc(), *firstArg, *secondArg);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), result);
  }

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

// AtenMaxPool2dWithIndicesOp
template <>
LogicalResult ConvertAtenOp<AtenMaxPool2dWithIndicesOp>::matchAndRewrite(
    AtenMaxPool2dWithIndicesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputElemTy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  auto outValTy =
      getTypeConverter()->convertType(op.getType(0)).cast<RankedTensorType>();
  auto outIdxTy =
      getTypeConverter()->convertType(op.getType(1)).cast<RankedTensorType>();

  if (inputRank <= 2) {
    return op.emitError(
        "max_pooling2d only supports inputs with rank higher than 2");
  }
  SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;
  bool ceilMode = false;

  if (!(matchPattern(op.getKernelSize(),
                     m_TorchListOfConstantInts(kernelSize)))) {
    return rewriter.notifyMatchFailure(
        op, "non-const int kernel size unsupported!");
  }
  if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
    return rewriter.notifyMatchFailure(op, "non-const int stride unsupported!");
  }
  if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int padding unsupported!");
  }
  if (!(matchPattern(op.getDilation(), m_TorchListOfConstantInts(dilation)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int dilation unsupported!");
  }
  if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const bool ceil_mode unsupported!");
  }

  // prepend 1 to kernelSize, stride, dilation until they are of same rank as
  // input
  SmallVector<int64_t> stablehloStride(inputRank, 1);
  SmallVector<int64_t> stablehloDilation(inputRank, 1);
  SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
  SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);
  std::copy(dilation.begin(), dilation.end(),
            stablehloDilation.begin() + inputRank - 2);
  std::copy(stride.begin(), stride.end(),
            stablehloStride.begin() + inputRank - 2);
  std::copy(kernelSize.begin(), kernelSize.end(),
            stablehloKernelSize.begin() + inputRank - 2);

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  stablehloPadding[stablehloPadding.size() - 4] = padding[0];
  stablehloPadding[stablehloPadding.size() - 3] = padding[0];
  stablehloPadding[stablehloPadding.size() - 2] = padding[1];
  stablehloPadding[stablehloPadding.size() - 1] = padding[1];

  auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
  auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
  DenseI64ArrayAttr baseDilations;
  auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      stablehloPadding);

  const auto &options = getOptions();
  auto inputShapeInfo =
      hlo::getDimSizesOfTensor(rewriter, op, input, options.dimSizeIndexBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto inputShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), inputShapeVec);

  SmallVector<Value> initIndexShapeVec;
  for (int64_t i = 0; i < inputRank - 2; i++)
    initIndexShapeVec.push_back(inputShapeVec[i]);
  initIndexShapeVec.push_back(rewriter.create<mlir::arith::MulIOp>(
      op->getLoc(), inputShapeVec[inputRank - 1],
      inputShapeVec[inputRank - 2]));
  auto initIndexShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), initIndexShapeVec);

  SmallVector<int64_t> initIndexShapeForType(inputShape.begin(),
                                             inputShape.end() - 2);
  if (inputShape[inputRank - 1] == ShapedType::kDynamic ||
      inputShape[inputRank - 2] == ShapedType::kDynamic) {
    initIndexShapeForType.push_back(ShapedType::kDynamic);
  } else {
    initIndexShapeForType.push_back(inputShape[inputRank - 1] *
                                    inputShape[inputRank - 2]);
  }

  auto initIndexTensor =
      rewriter
          .create<stablehlo::DynamicIotaOp>(
              op->getLoc(),
              RankedTensorType::get(initIndexShapeForType,
                                    rewriter.getI64Type()),
              initIndexShapeTensor, static_cast<uint64_t>(inputRank - 2))
          .getResult();

  auto indexTensor =
      rewriter
          .create<stablehlo::DynamicReshapeOp>(
              op->getLoc(),
              RankedTensorType::get(inputShape, rewriter.getI64Type()),
              initIndexTensor, inputShapeTensor)
          .getResult();

  Value initIdx = hlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).value();

  auto reduceWindowOp = rewriter.create<stablehlo::ReduceWindowOp>(
      op->getLoc(), mlir::TypeRange{outValTy, outIdxTy},
      mlir::ValueRange{input, indexTensor}, mlir::ValueRange{initVal, initIdx},
      windowDimensions, windowStrides, baseDilations, windowDilations, pad);

  Block &block = reduceWindowOp.getBody().emplaceBlock();

  // Add bb argument
  auto blockValArgumentType = RankedTensorType::get({}, inputElemTy);
  auto blockIdxArgumentType = RankedTensorType::get({}, rewriter.getI64Type());
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

    // Get smaller index if compared values are equal.
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

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

namespace {
template <typename AtenOpT, int Dim>
class ConvertAtenAvgPoolOp : public ConvertAtenOp<AtenOpT> {
public:
  using ConvertAtenOp<AtenOpT>::ConvertAtenOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputTy = input.getType().cast<RankedTensorType>();
    Type inputElemTy = inputTy.getElementType();
    int64_t inputRank = inputTy.getRank();
    RankedTensorType outTy = ConvertAtenOp<AtenOpT>::getTypeConverter()
                                 ->convertType(op.getType())
                                 .template cast<RankedTensorType>();
    auto outShape = outTy.getShape();

    if (inputRank <= Dim) {
      return op.emitError(
          "avg_pooling1d/2d only supports inputs with rank higher than 1/2");
    }
    SmallVector<int64_t, Dim> padding, kernelSize, stride;
    bool ceilMode = false;
    bool countIncludePad = true;

    if (!(matchPattern(op.getKernelSize(),
                       m_TorchListOfConstantInts(kernelSize)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const int kernel size unsupported!");
    }
    if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int stride unsupported!");
    }
    if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int padding unsupported!");
    }
    if (!(matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool ceil_mode unsupported!");
    }
    if (!(matchPattern(op.getCountIncludePad(),
                       m_TorchConstantBool(&countIncludePad)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool count_include_pad unsupported!");
    }

    if constexpr (std::is_same<AtenOpT, AtenAvgPool2dOp>()) {
      if (succeeded(checkNotNone(rewriter, op, op.getDivisorOverride())))
        return rewriter.notifyMatchFailure(
            op, "only None divisor_override supported for now!");
    }

    // Prepend 1 to kernelSize, stride, dilation until they are of same rank
    // as input
    SmallVector<int64_t> stablehloStride(inputRank, 1);
    SmallVector<int64_t> stablehloDilation(inputRank, 1);
    SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
    SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);

    std::copy(stride.begin(), stride.end(),
              stablehloStride.begin() + inputRank - Dim);
    std::copy(kernelSize.begin(), kernelSize.end(),
              stablehloKernelSize.begin() + inputRank - Dim);
    if (Dim == 1) {
      stablehloPadding[stablehloPadding.size() - 2] = padding[0];
      stablehloPadding[stablehloPadding.size() - 1] = padding[0];
    } else {
      stablehloPadding[stablehloPadding.size() - 4] = padding[0];
      stablehloPadding[stablehloPadding.size() - 3] = padding[0];
      stablehloPadding[stablehloPadding.size() - 2] = padding[1];
      stablehloPadding[stablehloPadding.size() - 1] = padding[1];
    }

    Value initVal =
        createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

    auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
    auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
    DenseI64ArrayAttr baseDilations;
    auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
    DenseIntElementsAttr pad = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
            rewriter.getI64Type()),
        stablehloPadding);

    auto reduceWindowSum = rewriter.create<stablehlo::ReduceWindowOp>(
        op->getLoc(), outTy, input, initVal, windowDimensions, windowStrides,
        baseDilations, windowDilations, pad);

    Block &sumBlock = reduceWindowSum.getBody().emplaceBlock();

    // Add bb argument
    auto blockArgumentType = RankedTensorType::get({}, inputElemTy);
    sumBlock.addArgument(blockArgumentType, op->getLoc());
    sumBlock.addArgument(blockArgumentType, op->getLoc());
    auto firstArg = *sumBlock.args_begin();
    auto secondArg = *sumBlock.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sumBlock);

      Value sumResult =
          rewriter.create<stablehlo::AddOp>(op->getLoc(), firstArg, secondArg);
      rewriter.create<stablehlo::ReturnOp>(op->getLoc(), sumResult);
    }

    // Use kernel size as the divisor
    if (countIncludePad) {
      Value divisor;
      if (Dim == 1) {
        divisor =
            hlo::getConstTensor<int64_t>(rewriter, op, {kernelSize[0]}, {})
                .value();
      } else {
        divisor = hlo::getConstTensor<int64_t>(
                      rewriter, op, {kernelSize[0] * kernelSize[1]}, {})
                      .value();
      }
      divisor = hlo::promoteType(rewriter, op.getLoc(), divisor, outTy);
      DenseI64ArrayAttr bcastDimensions;
      rewriter.replaceOpWithNewOp<mlir::chlo::BroadcastDivOp>(
          op, outTy, reduceWindowSum.getResult(0), divisor, bcastDimensions);
      return success();
    }

    // Use another mhlo.ReduceWindowOp to get the divisor
    Value windowSizeConst =
        hlo::getConstTensor<float>(rewriter, op, {1.0}, {}).value();
    windowSizeConst =
        hlo::promoteType(rewriter, op.getLoc(), windowSizeConst, outTy);
    const auto &options = ConvertAtenOp<AtenOpT>::getOptions();
    auto inputShapeVec = *hlo::getDimSizesOfTensor(rewriter, op, input,
                                                   options.dimSizeIndexBits);
    auto inputShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), inputShapeVec);

    windowSizeConst = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
        op->getLoc(),
        RankedTensorType::get(inputTy.getShape(), outTy.getElementType()),
        windowSizeConst, inputShapeTensor, rewriter.getDenseI64ArrayAttr({}));

    Value zero = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);
    auto reduceWindowSize = rewriter.create<stablehlo::ReduceWindowOp>(
        op->getLoc(), RankedTensorType::get(outShape, inputElemTy),
        windowSizeConst, zero, windowDimensions, windowStrides, baseDilations,
        windowDilations, pad);

    Block &sizeBlock = reduceWindowSize.getBody().emplaceBlock();

    // Add bb argument
    blockArgumentType = RankedTensorType::get({}, inputElemTy);
    sizeBlock.addArgument(blockArgumentType, op->getLoc());
    sizeBlock.addArgument(blockArgumentType, op->getLoc());
    firstArg = *sizeBlock.args_begin();
    secondArg = *sizeBlock.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&sizeBlock);

      Value sumResult =
          rewriter.create<stablehlo::AddOp>(op->getLoc(), firstArg, secondArg);
      rewriter.create<stablehlo::ReturnOp>(op->getLoc(), sumResult);
    }

    rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
        op, outTy, reduceWindowSum.getResult(0), reduceWindowSize.getResult(0));
    return success();
  }
};
} // namespace

// AtenCumsumOp
template <>
LogicalResult ConvertAtenOp<AtenCumsumOp>::matchAndRewrite(
    AtenCumsumOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto outTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  input = hlo::promoteType(rewriter, op.getLoc(), input, outTy);
  inputTy = input.getType().cast<RankedTensorType>();
  auto inputElemTy = inputTy.getElementType();
  auto inputRank = inputTy.getRank();
  auto inputShape = inputTy.getShape();

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: dim must be a constant int");
  }
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank)) {
    return rewriter.notifyMatchFailure(op, "dim is out of range");
  }
  if (inputTy.isDynamicDim(dim)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: cumsum dim must be static");
  }

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
  stablehloKernelSize[dim] = inputShape[dim];
  SmallVector<int64_t> stablehloStride(inputRank, 1);
  SmallVector<int64_t> stablehloDilation(inputRank, 1);
  SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);
  stablehloPadding[dim * 2] = inputShape[dim] - 1;

  auto windowDimensions = rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
  auto windowStrides = rewriter.getDenseI64ArrayAttr(stablehloStride);
  DenseI64ArrayAttr baseDilations;
  auto windowDilations = rewriter.getDenseI64ArrayAttr(stablehloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      stablehloPadding);

  auto reduceWindowSum = rewriter.create<stablehlo::ReduceWindowOp>(
      op->getLoc(), outTy, input, initVal, windowDimensions, windowStrides,
      baseDilations, windowDilations, pad);

  Block &sumBlock = reduceWindowSum.getBody().emplaceBlock();

  // Add bb argument
  auto blockArgumentType = RankedTensorType::get({}, inputElemTy);
  sumBlock.addArgument(blockArgumentType, op->getLoc());
  sumBlock.addArgument(blockArgumentType, op->getLoc());
  auto *firstArg = sumBlock.args_begin();
  auto *secondArg = std::next(firstArg);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&sumBlock);

    Value sumResult =
        rewriter.create<stablehlo::AddOp>(op->getLoc(), *firstArg, *secondArg);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), sumResult);
  }

  rewriter.replaceOp(op, reduceWindowSum.getResults());
  return success();
}

void mlir::torch::torch_to_stablehlo::populatePoolingOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenAvgPool1dOp>();
  patterns.add<ConvertAtenOp<AtenAvgPool1dOp>>(typeConverter, context, options);
  target.addIllegalOp<AtenMaxPool2dOp>();
  patterns.add<ConvertAtenOp<AtenMaxPool2dOp>>(typeConverter, context, options);
  target.addIllegalOp<AtenAvgPool2dOp>();
  patterns.add<ConvertAtenOp<AtenAvgPool2dOp>>(typeConverter, context, options);
  target.addIllegalOp<AtenMaxPool2dWithIndicesOp>();
  patterns.add<ConvertAtenOp<AtenMaxPool2dWithIndicesOp>>(typeConverter,
                                                          context, options);
  target.addIllegalOp<AtenCumsumOp>();
  patterns.add<ConvertAtenOp<AtenCumsumOp>>(typeConverter, context, options);
#define INSERT_ATEN_AVGPOOL_PATTERN(AtenOp, Dim)                               \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAvgPoolOp<AtenOp, Dim>>(typeConverter, context,      \
                                                  options)
  INSERT_ATEN_AVGPOOL_PATTERN(AtenAvgPool1dOp, 1);
  INSERT_ATEN_AVGPOOL_PATTERN(AtenAvgPool2dOp, 2);
#undef INSERT_ATEN_AVGPOOL_PATTERN
}
