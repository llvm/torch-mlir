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
#include "./PopulatePattern.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
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

static Value createInitialValueForAtenPoolingOp(Operation *op, Type elementTy,
                                                PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  // mean pooling
  if (isa<AtenAdaptiveAvgPool2dOp>(op)) {
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

  // max pooling
  if (isa<AtenMaxPool2dOp, AtenMaxPool2dWithIndicesOp>(op)) {
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
  op->emitError("unimplemented lowering in AtenPoolingOp");
  return nullptr;
}

namespace {
template <typename AtenOpT>
class ConvertAtenPoolingOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult ConvertAtenPoolingOp<AtenMaxPool2dOp>::matchAndRewrite(
    AtenMaxPool2dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputElemTy = inputTy.getElementType();
  // auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  auto outTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

  if (inputRank <= 2) {
    return op.emitError(
        "max_pooling2d only supports inputs with higher rank than 2");
  }
  SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;
  bool ceilMode = false;

  if (!(matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSize)))) {
    return rewriter.notifyMatchFailure(
        op, "non-const int kernel size unsupported!");
  }
  if (!(matchPattern(op.stride(), m_TorchConstantIntList(stride)))) {
    return rewriter.notifyMatchFailure(op, "non-const int stride unsupported!");
  }
  if (!(matchPattern(op.padding(), m_TorchConstantIntList(padding)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int padding unsupported!");
  }
  if (!(matchPattern(op.dilation(), m_TorchConstantIntList(dilation)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int dilation unsupported!");
  }
  if (!(matchPattern(op.ceil_mode(), m_TorchConstantBool(&ceilMode)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const bool ceil_mode unsupported!");
  }

  // prepend 1 to kernelSize, stride, dilation until they are of same rank as
  // input
  SmallVector<int64_t> mhloStride(inputRank, 1);
  SmallVector<int64_t> mhloDilation(inputRank, 1);
  SmallVector<int64_t> mhloKernelSize(inputRank, 1);
  SmallVector<int64_t> mhloPadding(inputRank * 2, 0);
  std::copy(dilation.begin(), dilation.end(),
            mhloDilation.begin() + inputRank - 2);
  std::copy(stride.begin(), stride.end(), mhloStride.begin() + inputRank - 2);
  std::copy(kernelSize.begin(), kernelSize.end(),
            mhloKernelSize.begin() + inputRank - 2);

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  mhloPadding[mhloPadding.size() - 4] = padding[0];
  mhloPadding[mhloPadding.size() - 3] = padding[0];
  mhloPadding[mhloPadding.size() - 2] = padding[1];
  mhloPadding[mhloPadding.size() - 1] = padding[1];

  DenseIntElementsAttr windowDimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloKernelSize.size())},
                            rewriter.getI64Type()),
      mhloKernelSize);
  DenseIntElementsAttr windowStrides = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloStride.size())},
                            rewriter.getI64Type()),
      mhloStride);
  DenseIntElementsAttr baseDilations;
  DenseIntElementsAttr windowDilations = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloDilation.size())},
                            rewriter.getI64Type()),
      mhloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
                            rewriter.getI64Type()), mhloPadding);
  auto reduceWindowOp = rewriter.create<mhlo::ReduceWindowOp>(
      op->getLoc(), outTy, input, initVal, windowDimensions, windowStrides,
      baseDilations, windowDilations, pad);

  Block &block = reduceWindowOp.body().emplaceBlock();

  auto blockArgumentTy = RankedTensorType::get({}, inputElemTy);
  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArg = block.args_begin();
  auto secondArg = block.args_rbegin();

  {
    mlir::IRRewriter::InsertPoint prevIP = rewriter.saveInsertionPoint();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value result =
        rewriter.create<mhlo::MaxOp>(op->getLoc(), *firstArg, *secondArg);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), result);

    rewriter.restoreInsertionPoint(prevIP);
  }

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

template <>
LogicalResult ConvertAtenPoolingOp<AtenMaxPool2dWithIndicesOp>::matchAndRewrite(
    AtenMaxPool2dWithIndicesOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
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
        "max_pooling2d only supports inputs with higher rank than 2");
  }
  SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;
  bool ceilMode = false;

  if (!(matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSize)))) {
    return rewriter.notifyMatchFailure(
        op, "non-const int kernel size unsupported!");
  }
  if (!(matchPattern(op.stride(), m_TorchConstantIntList(stride)))) {
    return rewriter.notifyMatchFailure(op, "non-const int stride unsupported!");
  }
  if (!(matchPattern(op.padding(), m_TorchConstantIntList(padding)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int padding unsupported!");
  }
  if (!(matchPattern(op.dilation(), m_TorchConstantIntList(dilation)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int dilation unsupported!");
  }
  if (!(matchPattern(op.ceil_mode(), m_TorchConstantBool(&ceilMode)))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const bool ceil_mode unsupported!");
  }

  // prepend 1 to kernelSize, stride, dilation until they are of same rank as
  // input
  SmallVector<int64_t> mhloStride(inputRank, 1);
  SmallVector<int64_t> mhloDilation(inputRank, 1);
  SmallVector<int64_t> mhloKernelSize(inputRank, 1);
  SmallVector<int64_t> mhloPadding(inputRank * 2, 0);
  std::copy(dilation.begin(), dilation.end(),
            mhloDilation.begin() + inputRank - 2);
  std::copy(stride.begin(), stride.end(), mhloStride.begin() + inputRank - 2);
  std::copy(kernelSize.begin(), kernelSize.end(),
            mhloKernelSize.begin() + inputRank - 2);

  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);

  mhloPadding[mhloPadding.size() - 4] = padding[0];
  mhloPadding[mhloPadding.size() - 3] = padding[0];
  mhloPadding[mhloPadding.size() - 2] = padding[1];
  mhloPadding[mhloPadding.size() - 1] = padding[1];

  DenseIntElementsAttr windowDimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloKernelSize.size())},
                            rewriter.getI64Type()),
      mhloKernelSize);
  DenseIntElementsAttr windowStrides = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloStride.size())},
                            rewriter.getI64Type()),
      mhloStride);
  DenseIntElementsAttr baseDilations;
  DenseIntElementsAttr windowDilations = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(mhloDilation.size())},
                            rewriter.getI64Type()),
      mhloDilation);
  DenseIntElementsAttr pad = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
          rewriter.getI64Type()),
      mhloPadding);

  SmallVector<int64_t> initIndexShape(inputShape.begin(), inputShape.end() - 1);
  initIndexShape[initIndexShape.size() - 1] *= inputShape[inputRank - 1];
  auto initIndexShapeConst =
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(initIndexShape),
                           {static_cast<int64_t>(initIndexShape.size())})
          .getValue();
  auto indexShapeConst =
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(inputShape),
                           {static_cast<int64_t>(inputShape.size())})
          .getValue();
  auto indexTensor = rewriter.create<mhlo::DynamicIotaOp>(
      op->getLoc(),
      RankedTensorType::get(initIndexShape, rewriter.getI64Type()),
      initIndexShapeConst, static_cast<uint64_t>(inputRank - 2)).getResult();
  indexTensor = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), RankedTensorType::get(inputShape, rewriter.getI64Type()),
      indexTensor, indexShapeConst).getResult();

  Value initIdx =
      mhlo::getConstTensor<int64_t>(rewriter, op, {0}, {}).getValue();

  auto reduceWindowOp = rewriter.create<mhlo::ReduceWindowOp>(
      op->getLoc(), mlir::TypeRange{outValTy, outIdxTy}, mlir::ValueRange{input, indexTensor},
      mlir::ValueRange{initVal, initIdx}, windowDimensions, windowStrides, baseDilations,
      windowDilations, pad);

  Block &block = reduceWindowOp.body().emplaceBlock();

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
    mlir::IRRewriter::InsertPoint prevIP = rewriter.saveInsertionPoint();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);

    Value compareGeResult = rewriter.create<mhlo::CompareOp>(
        op->getLoc(), compareResultType, *firstValArg, *secondValArg,
        compareGeDirectionAttr, compareTypeAttr);
    Value retValResult = rewriter.create<mhlo::SelectOp>(
        op->getLoc(), compareGeResult, *firstValArg, *secondValArg);

    // get smaller index if compared values are equal.
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

    rewriter.restoreInsertionPoint(prevIP);
  }

  rewriter.replaceOp(op, reduceWindowOp.getResults());
  return success();
}

// FIXME: delete this funtion and add support for lowering AvgPool2dOp after upstreaming
template <>
LogicalResult ConvertAtenPoolingOp<AtenAdaptiveAvgPool2dOp>::matchAndRewrite(
    AtenAdaptiveAvgPool2dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputElemTy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  auto outTy = getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

  SmallVector<int64_t> outShape;
  if (!matchPattern(op.output_size(), m_TorchConstantIntList(outShape))) {
    return rewriter.notifyMatchFailure(
        op, "Non-const output_size for adaptive pooling unsupported.");
  }
  // Since upstream torch-mlir has updated,
  // the AdaptiveAvgPooling2dOp will be decomposed as AvgPooling2dOp,
  // which is not generated in our torch-mlir.
  // For the purpose of lowering ResNet18 model from end2end,
  // we only handle the special case in ResNet, i.e. outputshape is (1, 1)

  // Sum elements in the window
  if (outShape[0] != 1 || outShape[1] != 1) {
    return op->emitError("Now only support (1, 1) output_size");
  }
  DenseIntElementsAttr pad, windowStrides, baseDilations, windowDilations;
  SmallVector<int64_t> kernelSize(inputRank, 1);
  kernelSize[inputRank - 2] = inputShape[inputRank - 2];
  kernelSize[inputRank - 1] = inputShape[inputRank - 1];
  DenseIntElementsAttr windowDimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(kernelSize.size())},
                            rewriter.getI64Type()),
      kernelSize);
  Value initVal = createInitialValueForAtenPoolingOp(op, inputElemTy, rewriter);
  auto reduceWindowOp = rewriter.create<mhlo::ReduceWindowOp>(
      op->getLoc(), outTy, input, initVal, windowDimensions, windowStrides,
      baseDilations, windowDilations, pad);

  Block &block = reduceWindowOp.body().emplaceBlock();

  auto blockArgumentTy = RankedTensorType::get({}, inputElemTy);
  block.addArgument(blockArgumentTy, op->getLoc());
  block.addArgument(blockArgumentTy, op->getLoc());

  auto *firstArg = block.args_begin();
  auto secondArg = block.args_rbegin();

  {
    mlir::IRRewriter::InsertPoint prevIP = rewriter.saveInsertionPoint();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value result =
        rewriter.create<mhlo::AddOp>(op->getLoc(), *firstArg, *secondArg);
    rewriter.create<mhlo::ReturnOp>(op->getLoc(), result);

    rewriter.restoreInsertionPoint(prevIP);
  }

  // divide the sum by the number of elements in the window
  int64_t windowSize = inputShape[inputRank - 2] * inputShape[inputRank - 1];
  Value divisor = mhlo::getConstTensor<float>(rewriter, op, {static_cast<float>(windowSize)}, {}).getValue();
  divisor = mhlo::promoteAndBroadcast(rewriter, divisor, outTy);
  rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, reduceWindowOp.getResult(0), divisor);
  return success();
}
} // namespace

void mlir::torch::torch_to_mhlo::populatePoolingOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMaxPool2dOp>();
  patterns.add<ConvertAtenPoolingOp<AtenMaxPool2dOp>>(typeConverter, context);
  target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
  patterns.add<ConvertAtenPoolingOp<AtenAdaptiveAvgPool2dOp>>(typeConverter, context);
  target.addIllegalOp<AtenMaxPool2dWithIndicesOp>();
  patterns.add<ConvertAtenPoolingOp<AtenMaxPool2dWithIndicesOp>>(typeConverter, context);
}
