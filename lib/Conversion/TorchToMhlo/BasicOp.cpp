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

static llvm::Optional<Value> squeeze(Operation *op, Value input,
                                     llvm::ArrayRef<int64_t> dims,
                                     PatternRewriter &rewriter) {
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return llvm::None;
  }

  auto inputRank = inputTy.getRank();
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return llvm::None;
  }

  SmallVector<int64_t> outShape;
  DenseSet<int64_t> squeezeDims;
  for (auto dim : dims) {
    dim = dim < 0 ? (dim + inputRank) : dim;

    // drop invalid dims
    if (dim < inputRank) {
      squeezeDims.insert(dim);
    }
  }
  bool needSqueeze = false;
  for (auto en : llvm::enumerate(inputTy.getShape())) {
    if (en.value() == 1 &&
        squeezeDims.contains(static_cast<int64_t>(en.index()))) {
      needSqueeze = true;
      continue;
    }
    outShape.push_back(en.value());
  }

  if (!needSqueeze)
    return input;

  llvm::Optional<Value> outShapeConst =
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(outShape),
                           {static_cast<int64_t>(outShape.size())});
  auto result =
      rewriter
          .create<mhlo::DynamicReshapeOp>(
              op->getLoc(),
              RankedTensorType::get(outShape, inputTy.getElementType()), input,
              outShapeConst.getValue())
          .getResult();
  return result;
}

namespace {

// Binary op legalizations for comparator ops.
template <typename AtenOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    Value rhs = adaptor.other();
    RankedTensorType lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsTy = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported in MHLO");

    RankedTensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                                   ->convertType(op.getType())
                                   .template cast<RankedTensorType>();

    Type lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsAsTensor, lhsElemTy, {}))) {
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
      }
    }

    Value lhsTensor = lhs;
    Value rhsTensor = rhsTy ? rhs : rhsAsTensor;
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, lhsTy);

    mhlo::ComparisonTypeAttr compareTypeAttr;
    mhlo::ComparisonDirectionAttr compareDirectionAttr;

    if (lhsElemTy.isa<mlir::FloatType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          op->getContext(), mhlo::ComparisonType::FLOAT);
    } else if (lhsElemTy.isa<mlir::IntegerType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          op->getContext(), mhlo::ComparisonType::SIGNED);
    }

    if (std::is_same<AtenOpT, AtenLtTensorOp>() ||
        std::is_same<AtenOpT, AtenLtScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::LT);
    } else if (std::is_same<AtenOpT, AtenGtTensorOp>() ||
               std::is_same<AtenOpT, AtenGtScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::GT);
    } else if (std::is_same<AtenOpT, AtenEqTensorOp>() ||
               std::is_same<AtenOpT, AtenEqScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::EQ);
    } else if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
               std::is_same<AtenOpT, AtenNeScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::NE);
    }

    rewriter.replaceOpWithNewOp<mhlo::CompareOp>(
        op, outType, lhsTensor, rhsTensor, compareDirectionAttr,
        compareTypeAttr);
    return success();
  }
};

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only Tensor types supported in MHLO");

    if (selfTy.getElementType().isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<MhloOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          self);
      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
  }
};

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template<>
LogicalResult ConvertAtenOp<AtenReciprocalOp>::matchAndRewrite(AtenReciprocalOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto outTy = getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  if (!inputTy.getElementType().isa<mlir::FloatType>()) {
    return op.emitError(
          "Only floating-point datatype legalization supported for AtenReciprocalOp");
  }
  Value oneTensor = mhlo::getConstTensor<float>(rewriter, op, {static_cast<float>(1.0)}, {}).getValue();
  oneTensor = mhlo::promoteAndBroadcast(rewriter, oneTensor, inputTy);
  rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, outTy, oneTensor, input);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.input();
  // shape = [N C H W]
  auto inputTy = input.getType().cast<RankedTensorType>();
  Value weight = adaptor.weight();
  Value bias = adaptor.bias();
  Value runningMean = adaptor.running_mean();
  Value runningVar = adaptor.running_var();
  // momentum is ignored
  Value momentum = adaptor.momentum();
  (void)momentum;

  // init weight, bias, runningVar, runningMean if they are none
  auto initNoneValue = [&](Value &input, bool zero) {
    SmallVector<APFloat> constVec(inputTy.getShape()[1],
                                  APFloat::getZero(inputTy.getElementType()
                                                       .cast<mlir::FloatType>()
                                                       .getFloatSemantics()));
    if (!zero) {
      for (auto &item : constVec) {
        item = APFloat(inputTy.getElementType()
                           .cast<mlir::FloatType>()
                           .getFloatSemantics(),
                       1);
      }
    }
    auto constType = RankedTensorType::get({inputTy.getShape()[1]},
                                           inputTy.getElementType());
    auto constAttr = DenseElementsAttr::get(constType, constVec);
    input =
        rewriter.create<mhlo::ConstantOp>(op.getLoc(), constType, constAttr);
  };
  if (failed(checkNotNone(rewriter, op, weight))) {
    initNoneValue(weight, false);
  }
  if (failed(checkNotNone(rewriter, op, bias))) {
    initNoneValue(bias, true);
  }
  if (failed(checkNotNone(rewriter, op, runningVar))) {
    initNoneValue(runningVar, false);
  }
  if (failed(checkNotNone(rewriter, op, runningMean))) {
    initNoneValue(runningMean, true);
  }

  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto biasTy = bias.getType().cast<RankedTensorType>();
  auto runningMeanTy = runningMean.getType().cast<RankedTensorType>();
  auto runningVarTy = runningVar.getType().cast<RankedTensorType>();
  if (inputTy.getRank() <= 2) {
    return rewriter.notifyMatchFailure(op,
                                       "input should have rank larger than 2");
  }
  if (weightTy.getRank() != 1 || biasTy.getRank() != 1 ||
      runningMeanTy.getRank() != 1 || runningVarTy.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "expect weight, bias, running_mean and running_var to be rank 1");
  }
  if (!inputTy.getElementType().template isa<mlir::FloatType>() ||
      !weightTy.getElementType().template isa<mlir::FloatType>() ||
      !biasTy.getElementType().template isa<mlir::FloatType>() ||
      !runningMeanTy.getElementType().template isa<mlir::FloatType>() ||
      !runningVarTy.getElementType().template isa<mlir::FloatType>()) {
    return op.emitError(
        "Only float element type is supported in MHLO BatchNormOp");
  }

  double eps = 0.0;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op, "non-float(double) eps unsupported");
  }
  bool training = false;
  if (!matchPattern(op.training(), m_TorchConstantBool(&training))) {
    return rewriter.notifyMatchFailure(op, "non-bool training unsupported");
  }
  // TODO: handle cudnnEnabled parameter. Here, we just ignore it!
  bool cudnnEnabled = false;
  if (!matchPattern(op.cudnn_enabled(), m_TorchConstantBool(&cudnnEnabled))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-bool cudnn_enabled unsupported");
  }
  if (training) {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    Type batchMeanOrVarTy =
        RankedTensorType::get(weightTy.getShape(), inputTy.getElementType());
    auto batchNormTrainingResult = rewriter.create<mhlo::BatchNormTrainingOp>(
        op.getLoc(), outputTy, batchMeanOrVarTy, batchMeanOrVarTy, input,
        weight, bias, rewriter.getF32FloatAttr(eps),
        rewriter.getI64IntegerAttr(1));
    rewriter.replaceOp(op, batchNormTrainingResult.getResult(0));
    return success();
  } else {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mhlo::BatchNormInferenceOp>(
        op, outputTy, input, weight, bias, runningMean, runningVar,
        rewriter.getFloatAttr(inputTy.getElementType(), eps),
        rewriter.getI64IntegerAttr(1));
    return success();
  }
}
// AtenSqueezeDimOp
template <>
LogicalResult ConvertAtenOp<AtenSqueezeDimOp>::matchAndRewrite(
    AtenSqueezeDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only tensor types are currently supported");
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return op.emitError("dim must be a Scalar constant");
  }

  auto result = squeeze(op, input, {dim}, rewriter);
  if (!result) {
    return op.emitError("Unsupported squeeze operation lowering");
  }
  rewriter.replaceOp(op, result.getValue());
  return success();
}

// AtenSqueezeOp
template <>
LogicalResult ConvertAtenOp<AtenSqueezeOp>::matchAndRewrite(
    AtenSqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only tensor types are currently supported");
  }

  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> dims;
  for (auto en : llvm::enumerate(inputTy.getShape())) {
    if (en.value() == 1) {
      dims.push_back(static_cast<int64_t>(en.index()));
    }
  }

  auto result = squeeze(op, input, dims, rewriter);
  if (!result) {
    return op.emitError("Unsupported squeeze operation lowering");
  }
  rewriter.replaceOp(op, result.getValue());
  return success();
}

// AtenUnsqueezeOp
template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().template cast<RankedTensorType>();
  auto inputRank = inputTy.getRank();

  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");
  }
  dim = dim >= 0 ? dim : dim + inputRank + 1;
  if (dim > inputRank) {
    return rewriter.notifyMatchFailure(op, "dim is invalid");
  }

  SmallVector<int64_t> outShape;
  for (auto en : llvm::enumerate(inputTy.getShape())) {
    if (static_cast<int64_t>(en.index()) == dim) {
      outShape.push_back(1);
    }
    outShape.push_back(en.value());
  }
  if (dim == inputRank) {
    outShape.push_back(1);
  }

  llvm::Optional<Value> outShapeConst =
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(outShape),
                           {static_cast<int64_t>(outShape.size())});

  rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
      op, RankedTensorType::get(outShape, inputTy.getElementType()),
      input, outShapeConst.getValue());
  return success();
}

// ValueTensorLiteralOp
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType resultType = getTypeConverter()
                                    ->convertType(op->getResult(0).getType())
                                    .cast<RankedTensorType>();

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  if (auto elements = op.valueAttr().dyn_cast<DenseIntElementsAttr>()) {
    Type builtinTensorElemTy = resultType.getElementType();
    unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();

    DenseElementsAttr valueAttr =
        elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
          return APInt(bitWidth, v.getSExtValue());
        });

    rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, resultType, valueAttr);
    return success();
  }

  rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, resultType,
                                                adaptor.value());
  return success();
}

// AtenTanhOp
template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<mhlo::TanhOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  } else {
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

// AtenIndexSelectOp
template <>
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  Value index = adaptor.index();
  RankedTensorType resultType = getTypeConverter()
                                    ->convertType(op->getResult(0).getType())
                                    .cast<RankedTensorType>();

  int64_t dimInt;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dimInt)))
    return op->emitError("unimplemented: dim is not constant");
  uint64_t batchDims = 0;

  rewriter.replaceOpWithNewOp<mhlo::TorchIndexSelectOp>(
      op, resultType, input, index, dimInt, batchDims);
  return success();
}

bool skipMultiplyAlpha(Value alphaValue) {
  double doubleValue;
  auto isFloat = matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(alphaValue, m_TorchConstantInt(&intValue));

  return ((isFloat && doubleValue == 1.0) || (isInt && intValue == 1.0));
}

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    TensorType lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    TensorType rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    TensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(op.getType())
                             .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsType) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsAsTensor, outElemTy,
                                               outType.getShape())))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
    }
    Value lhsTensor = lhs;
    Value rhsTensor = rhsType ? rhs : rhsAsTensor;

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    // Handle alpha.
    Value multTensor;
    if (skipMultiplyAlpha(op.alpha())) {
      multTensor = rhsTensor;
    } else {
      Value alphaTensor;
      if (failed(mhlo::torchAlphaToMhloTensor(rewriter, op.getOperation(),
                                              op.alpha(), alphaTensor,
                                              outElemTy, outType.getShape(),
                                              /*checkForUnity=*/false))) {
        return op.emitError("Currently only scalar constants are supported for "
                            "alpha in conversion to MHLO operation");
      }

      multTensor = rewriter.create<mhlo::MulOp>(op.getLoc(), outType, rhsTensor,
                                                alphaTensor);
    }

    rewriter.replaceOpWithNewOp<MhloOpT>(op, outType, lhsTensor, multTensor);
    return success();
  }
};

// Binary op legalizations for Mul variants.
template <typename AtenOpT>
class ConvertAtenMulOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    TensorType rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value lhsTensor = lhs;
    Value rhsTensor;
    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      if (!rhsType) {
        if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                                 rhsTensor, outElemTy,
                                                 outType.getShape())))
          return op.emitError(
              "Currently only scalar constants are supported for "
              "conversion in MHLO operation");
      } else {
        rhsTensor = rhs;
      }
    }

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, outType, lhsTensor, rhsTensor);
    return success();
  }
};

// Binary op legalizations for Div variants.
template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported.");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();
    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value lhsTensor = lhs;
    Value rhsTensor;
    if (!rhsTy) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsTensor, outElemTy,
                                               outType.getShape())))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
    } else {
      rhsTensor = rhs;
    }

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, outType, lhsTensor, rhsTensor);
    return success();
  }
};

// AtenViewOp
class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    auto inType = self.getType().cast<RankedTensorType>();
    if (!inType)
      return op.emitError("Only ranked tensor types are currently supported");

    if (!inType.getElementType().isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    SmallVector<int64_t> outShape;
    if (!matchPattern(op.size(), m_TorchConstantIntList(outShape)))
      return op.emitError("size must consist of Scalar constants");

    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    // Output shape maybe dynamic, change it to static!
    if (!outType.hasStaticShape()) {
      outType = RankedTensorType::get(outShape, inType.getElementType());
    }

    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, outType, self);

    return success();
  }
};

// AtenTransposeIntOp
class ConvertAtenTransposeIntOp
    : public OpConversionPattern<AtenTransposeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTransposeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    int64_t dim0;
    if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0))) {
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    }
    int64_t dim1;
    if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1))) {
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");
    }

    auto inType = self.getType().cast<RankedTensorType>();
    auto inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

    dim0 = toPositiveDim(dim0, inputRank);
    if (!isValidDim(dim0, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim0 out of range");
    }
    dim1 = toPositiveDim(dim1, inputRank);
    if (!isValidDim(dim1, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim1 out of range");
    }

    SmallVector<int64_t> permValues(inputRank);
    std::iota(std::begin(permValues), std::end(permValues), 0);
    std::swap(permValues[dim0], permValues[dim1]);
    DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(permValues.size())},
                              rewriter.getI64Type()),
        permValues);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(op, outType, self,
                                                   permutation);
    return success();
  }
};

// AtenPermuteOp
class ConvertAtenPermuteOp : public OpConversionPattern<AtenPermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    // Not a ranked tensor type
    auto inType = self.getType().dyn_cast<RankedTensorType>();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    if (!inType)
      return op.emitError("Only ranked tensor types with static shapes are "
                          "currently supported");

    SmallVector<int64_t> permValues;
    if (!matchPattern(adaptor.dims(), m_TorchConstantIntList(permValues)))
      return rewriter.notifyMatchFailure(
          op, "Only constant dimensions are currently supported");

    int64_t inRank = inType.getRank();
    for (auto &d : permValues) {
      d = toPositiveDim(d, inRank);
      if (!isValidDim(d, inRank))
        return op.emitError("Not all dims are valid");
    }

    DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(permValues.size())},
                              rewriter.getI64Type()),
        permValues);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(op, outType, self,
                                                   permutation);
    return success();
  }
};

// AtenBroadcastToOp
class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

    Value bcastOp = mhlo::promoteAndBroadcast(rewriter, self, outType);
    rewriter.replaceOp(op, bcastOp);
    return success();
  }
};

// AtenSliceTensorOp
class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    RankedTensorType inType = self.getType().cast<RankedTensorType>();
    RankedTensorType outType = getTypeConverter()
                                   ->convertType(op->getResult(0).getType())
                                   .cast<RankedTensorType>();

    int64_t inRank = inType.getRank();
    ArrayRef<int64_t> inShape = inType.getShape();

    SmallVector<int64_t> startIndicesValues(inRank, 0);
    SmallVector<int64_t> limitIndicesValues(inRank);
    for (int64_t i = 0; i < inRank; ++i) {
      limitIndicesValues[i] = inShape[i];
    }
    SmallVector<int64_t> stridesValues(inRank, 1);

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return op->emitError("unimplemented: dim is not constant");
    int64_t start;
    if (!matchPattern(op.start(), m_TorchConstantInt(&start)))
      return op->emitError("unimplemented: start is not constant");
    int64_t end;
    if (!matchPattern(op.end(), m_TorchConstantInt(&end)))
      return op->emitError("unimplemented: end is not constant");
    int64_t step;
    if (!matchPattern(op.step(), m_TorchConstantInt(&step)))
      return op->emitError("unimplemented: step is not constant");
    startIndicesValues[dim] = start;
    limitIndicesValues[dim] = end;
    stridesValues[dim] = step;

    DenseIntElementsAttr startIndices = DenseIntElementsAttr::get(
        RankedTensorType::get({inRank}, rewriter.getI64Type()),
        startIndicesValues);
    DenseIntElementsAttr limitIndices = DenseIntElementsAttr::get(
        RankedTensorType::get({inRank}, rewriter.getI64Type()),
        limitIndicesValues);
    DenseIntElementsAttr strides = DenseIntElementsAttr::get(
        RankedTensorType::get({inRank}, rewriter.getI64Type()), stridesValues);
    rewriter.replaceOpWithNewOp<mhlo::SliceOp>(op, outType, self, startIndices,
                                               limitIndices, strides);
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.self();
  auto lhsTy = lhs.getType().cast<RankedTensorType>();
  auto lhsElemTy = lhsTy.getElementType();

  int64_t lhsSize = 1;
  for (auto &en : llvm::enumerate(lhsTy.getShape())) {
    lhsSize *= en.value();
  }
  auto constTy = RankedTensorType::get(lhsTy.getShape(), lhsElemTy);
  DenseElementsAttr constAttr;
  if (lhsElemTy.isa<mlir::FloatType>()) {
    std::vector<APFloat> constVec(
        lhsSize,
        APFloat::getZero(lhsElemTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false));
    constAttr = DenseElementsAttr::get(constTy, constVec);
  } else if (lhsElemTy.isa<mlir::IntegerType>()) {
    std::vector<APInt> constVec(
        lhsSize, APInt::getZero(lhsElemTy.getIntOrFloatBitWidth()));
    constAttr = DenseElementsAttr::get(constTy, constVec);
  }
  Value rhs =
      rewriter.create<mhlo::ConstantOp>(op.getLoc(), constTy, constAttr);

  rewriter.replaceOpWithNewOp<mhlo::MaxOp>(op, lhs, rhs);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenFlattenUsingIntsOp>::matchAndRewrite(
    AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();

  int64_t startDim, endDim;
  if (!matchPattern(op.start_dim(), m_TorchConstantInt(&startDim))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const int start_dim unsupported");
  }
  if (!matchPattern(op.end_dim(), m_TorchConstantInt(&endDim))) {
    return rewriter.notifyMatchFailure(op, "non-const int end_dim unsupported");
  }

  startDim = toPositiveDim(startDim, inputRank);
  if (!isValidDim(startDim, inputRank)) {
    return rewriter.notifyMatchFailure(op, "invalid start_dim");
  }
  endDim = toPositiveDim(endDim, inputRank);
  if (!isValidDim(endDim, inputRank)) {
    return rewriter.notifyMatchFailure(op, "invalid end_dim");
  }
  SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < startDim; i++) {
    outShape.push_back(inputShape[i]);
  }
  int64_t flattenDimSize = 1;
  for (int64_t i = startDim; i <= endDim; i++) {
    flattenDimSize *= inputShape[i];
  }
  outShape.push_back(flattenDimSize);
  for (int64_t i = endDim + 1; i < inputRank; i++) {
    outShape.push_back(inputShape[i]);
  }
  auto outShapeConst =
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(outShape),
                           {static_cast<int64_t>(outShape.size())})
          .getValue();
  rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), input, outShapeConst);
  return success();
}
} // namespace

void mlir::torch::torch_to_mhlo::populateBasicOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenViewOp>();
  patterns.add<ConvertAtenViewOp>(typeConverter, context);

  target.addIllegalOp<AtenTransposeIntOp>();
  patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);

  target.addIllegalOp<AtenPermuteOp>();
  patterns.add<ConvertAtenPermuteOp>(typeConverter, context);

  target.addIllegalOp<AtenBroadcastToOp>();
  patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);

  target.addIllegalOp<AtenSliceTensorOp>();
  patterns.add<ConvertAtenSliceTensorOp>(typeConverter, context);

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, MhloOp)                            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, MhloOp>>(typeConverter,        \
                                                         context);
  INSERT_UNARY_FPONLY_PATTERN(AtenLogOp, mhlo::LogOp)
  INSERT_UNARY_FPONLY_PATTERN(AtenExpOp, mhlo::ExpOp)
  INSERT_UNARY_FPONLY_PATTERN(AtenCloneOp, mhlo::CopyOp)
  INSERT_UNARY_FPONLY_PATTERN(AtenSqrtOp, mhlo::SqrtOp)
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, MhloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, MhloOp>>(typeConverter, context);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, mhlo::AddOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, mhlo::AddOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, mhlo::SubOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, mhlo::SubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp)
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_MUL_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMulOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_MUL_PATTERN(AtenMulTensorOp);
  INSERT_BINARY_MUL_PATTERN(AtenMulScalarOp);
#undef INSERT_BINARY_MUL_PATTERN

#define INSERT_BINARY_DIV_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenDivOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp);
  INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp);
#undef INSERT_BINARY_DIV_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
  INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATENOP_PATTERN(AtenTanhOp);
  INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
  INSERT_ATENOP_PATTERN(AtenReciprocalOp);
  // Squeeze Ops
  INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeOp);
  INSERT_ATENOP_PATTERN(AtenSqueezeDimOp);
  // BatchNormOp
  INSERT_ATENOP_PATTERN(AtenBatchNormOp);
  // AtenReluOp
  INSERT_ATENOP_PATTERN(AtenReluOp);
  INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
#undef INSERT_ATENOP_PATTERN
}
