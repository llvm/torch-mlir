//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTcp/TorchToTcp.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

bool IsMultiplyAlphaOne(Value alphaValue) {
  double doubleValue;
  auto isFloat = matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(alphaValue, m_TorchConstantInt(&intValue));

  return ((isFloat && doubleValue == 1.0) || (isInt && intValue == 1.0));
}

SignednessAttr
getTcpSignednessAttr(MLIRContext *context,
                     IntegerType::SignednessSemantics signednessInfo) {
  if (signednessInfo == IntegerType::SignednessSemantics::Signless)
    return SignednessAttr::get(context, Signedness::Signless);
  if (signednessInfo == IntegerType::SignednessSemantics::Signed)
    return SignednessAttr::get(context, Signedness::Signed);
  return SignednessAttr::get(context, Signedness::Unsigned);
}

template <typename AtenOpT, typename TcpOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();

    RankedTensorType resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    auto inputAType = op.getSelf()
                          .getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();
    auto outputType = op.getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();

    if (isa<AtenAddScalarOp>(op) || isa<AtenSubScalarOp>(op)) {
      RankedTensorType tensorResultType =
          RankedTensorType::get({}, adaptor.getOther().getType());
      rhs = torch_to_tcp::scalarToTcpTensor(rewriter, op, tensorResultType,
                                            adaptor.getOther());
      if (adaptor.getOther().getType().template isa<mlir::FloatType>())
        // FP rhs is treated as fp64
        rhs = torch_to_tcp::castTensorToDtype(rewriter, rewriter.getF64Type(),
                                              outputType, rhs,
                                              resultType.getElementType());
      else if (adaptor.getOther().getType().template isa<mlir::IntegerType>())
        // INT rhs is treated as si64
        rhs = torch_to_tcp::castTensorToDtype(
            rewriter, rewriter.getIntegerType(64, true), outputType, rhs,
            resultType.getElementType());
      else
        return rewriter.notifyMatchFailure(op, "Unsupported rhs data type");
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    } else {
      auto inputBType = op.getOther()
                            .getType()
                            .template dyn_cast<torch::Torch::ValueTensorType>()
                            .getDtype();
      rhs = torch_to_tcp::castTensorToDtype(rewriter, inputBType, outputType,
                                            rhs, resultType.getElementType());
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    }

    lhs = torch_to_tcp::castTensorToDtype(rewriter, inputAType, outputType, lhs,
                                          resultType.getElementType());
    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
        rewriter, lhs, rhs, resultType.getElementType());

    if (!IsMultiplyAlphaOne(op.getAlpha())) {
      RankedTensorType tensorResultType =
          RankedTensorType::get({}, adaptor.getAlpha().getType());
      Value alpha = torch_to_tcp::scalarToTcpTensor(
          rewriter, op, tensorResultType, adaptor.getAlpha());
      if (adaptor.getAlpha().getType().template isa<mlir::FloatType>())
        // FP alpha is treated as fp64
        alpha = torch_to_tcp::castTensorToDtype(rewriter, rewriter.getF64Type(),
                                                outputType, alpha,
                                                resultType.getElementType());
      else if (adaptor.getAlpha().getType().template isa<mlir::IntegerType>())
        // INT alpha is treated as si64
        alpha = torch_to_tcp::castTensorToDtype(
            rewriter, rewriter.getIntegerType(64, true), outputType, alpha,
            resultType.getElementType());
      else
        return rewriter.notifyMatchFailure(op, "Unsupported alpha data type");
      alpha = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, alpha, rhs, resultType.getElementType());
      rhs = rewriter.create<MulOp>(op->getLoc(), resultType, alpha, rhs);
    }

    rewriter.replaceOpWithNewOp<TcpOpT>(op, resultType, lhs, rhs);
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenMulOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();

    RankedTensorType resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    auto inputAType = op.getSelf()
                          .getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();
    auto outputType = op.getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();

    if (isa<AtenMulScalarOp>(op)) {
      RankedTensorType tensorResultType =
          RankedTensorType::get({}, adaptor.getOther().getType());
      rhs = torch_to_tcp::scalarToTcpTensor(rewriter, op, tensorResultType,
                                            adaptor.getOther());
      if (adaptor.getOther().getType().template isa<mlir::FloatType>())
        // FP rhs is treated as fp64
        rhs = torch_to_tcp::castTensorToDtype(rewriter, rewriter.getF64Type(),
                                              outputType, rhs,
                                              resultType.getElementType());
      else if (adaptor.getOther().getType().template isa<mlir::IntegerType>())
        // INT rhs is treated as si64
        rhs = torch_to_tcp::castTensorToDtype(
            rewriter, rewriter.getIntegerType(64, true), outputType, rhs,
            resultType.getElementType());
      else
        return rewriter.notifyMatchFailure(op, "Unsupported rhs data type");
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    } else {
      auto inputBType = op.getOther()
                            .getType()
                            .template dyn_cast<torch::Torch::ValueTensorType>()
                            .getDtype();
      rhs = torch_to_tcp::castTensorToDtype(rewriter, inputBType, outputType,
                                            rhs, resultType.getElementType());
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    }

    lhs = torch_to_tcp::castTensorToDtype(rewriter, inputAType, outputType, lhs,
                                          resultType.getElementType());
    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
        rewriter, lhs, rhs, resultType.getElementType());

    rewriter.replaceOpWithNewOp<tcp::MulOp>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern<AtenBatchNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();

    Value weight = adaptor.getWeight();
    RankedTensorType weightType = weight.getType().dyn_cast<RankedTensorType>();

    Value bias = adaptor.getBias();
    RankedTensorType biasType = bias.getType().dyn_cast<RankedTensorType>();

    Value runningMean = adaptor.getRunningMean();
    RankedTensorType runningMeanType =
        runningMean.getType().dyn_cast<RankedTensorType>();

    Value runningVar = adaptor.getRunningVar();
    RankedTensorType runningVarType =
        runningVar.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenBatchNormOp>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    if (!inputType || !weightType || !biasType || !runningMeanType ||
        !runningVarType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "only Ranked Tensor types are supported in TCP");

    if (runningMeanType.getNumElements() == 0 ||
        runningVarType.getNumElements() == 0)
      return rewriter.notifyMatchFailure(
          op, "zero element running_mean and running_var not supported");

    double eps = 0.0;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
      return rewriter.notifyMatchFailure(op,
                                         "non-float(double) eps unsupported");

    Value epsVal;
    if (auto result = torch_to_tcp::getConstTensor<float>(
            rewriter, op, llvm::ArrayRef(static_cast<float>(eps)), {}))
      epsVal = *result;
    else
      return rewriter.notifyMatchFailure(op,
                                         "failed to get constTensor for eps");

    // momentum is ignored
    Value momentum = adaptor.getMomentum();
    (void)momentum;

    // cudnnEnabled is ignored
    Value cudnnEnabled = adaptor.getCudnnEnabled();
    (void)cudnnEnabled;

    bool training = false;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training)))
      return rewriter.notifyMatchFailure(op, "non-bool training unsupported");
    if (training)
      return rewriter.notifyMatchFailure(
          op, "only inference mode batch_norm lowering supported");

    // PyTorch inputs are [NCHW], and BatchNorm parameters are [C] length
    // vectors. `axisInOutput = 1` allows [C] -> [1, C, 1, 1] expansion
    // followed by a broadcast.
    runningMean = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, runningMean, input, inputType.getElementType(),
        /*axisInOutput=*/1);
    runningVar = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, runningVar, input, inputType.getElementType(),
        /*axisInOutput=*/1);
    weight = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, weight, input, inputType.getElementType(),
        /*axisInOutput=*/1);
    bias = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, bias, input, inputType.getElementType(), /*axisInOutput=*/1);
    epsVal = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, epsVal, input, inputType.getElementType());

    Value op1SubInputMean = rewriter.create<tcp::SubOp>(op.getLoc(), resultType,
                                                        input, runningMean);
    Value op2AddVarEpsilon = rewriter.create<tcp::AddOp>(
        op.getLoc(), resultType, runningVar, epsVal);
    Value op3SqrtOp2 =
        rewriter.create<tcp::SqrtOp>(op.getLoc(), resultType, op2AddVarEpsilon);
    Value op4DivOp1Op3 = rewriter.create<tcp::DivFOp>(
        op.getLoc(), resultType, op1SubInputMean, op3SqrtOp2);
    Value op5MulWeightOp4 = rewriter.create<tcp::MulOp>(op.getLoc(), resultType,
                                                        weight, op4DivOp1Op3);
    Value op6AddOp5Bias = rewriter.create<tcp::AddOp>(op.getLoc(), resultType,
                                                      op5MulWeightOp4, bias);

    rewriter.replaceOp(op, {op6AddOp5Bias});
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();

    RankedTensorType resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    // TODO: Add integer conversions once `tcp.divsi` and `tcp.divui` are
    // added
    if (resultType.getElementType().isa<mlir::IntegerType>()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating point division supported for now");
    }

    auto inputAType = op.getSelf()
                          .getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();
    auto outputType = op.getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();

    if (isa<AtenDivScalarOp>(op)) {
      RankedTensorType tensorResultType =
          RankedTensorType::get({}, adaptor.getOther().getType());
      rhs = torch_to_tcp::scalarToTcpTensor(rewriter, op, tensorResultType,
                                            adaptor.getOther());
      if (adaptor.getOther().getType().template isa<mlir::FloatType>())
        // FP rhs is treated as fp64
        rhs = torch_to_tcp::castTensorToDtype(rewriter, rewriter.getF64Type(),
                                              outputType, rhs,
                                              resultType.getElementType());
      else if (adaptor.getOther().getType().template isa<mlir::IntegerType>())
        // INT rhs is treated as si64
        rhs = torch_to_tcp::castTensorToDtype(
            rewriter, rewriter.getIntegerType(64, true), outputType, rhs,
            resultType.getElementType());
      else
        return rewriter.notifyMatchFailure(op, "Unsupported rhs data type");
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    } else {
      auto inputBType = op.getOther()
                            .getType()
                            .template dyn_cast<torch::Torch::ValueTensorType>()
                            .getDtype();
      rhs = torch_to_tcp::castTensorToDtype(rewriter, inputBType, outputType,
                                            rhs, resultType.getElementType());
      rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
          rewriter, rhs, lhs, resultType.getElementType());
    }

    lhs = torch_to_tcp::castTensorToDtype(rewriter, inputAType, outputType, lhs,
                                          resultType.getElementType());
    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
        rewriter, lhs, rhs, resultType.getElementType());

    rewriter.replaceOpWithNewOp<tcp::DivFOp>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenClampOp : public OpConversionPattern<AtenClampOp> {
public:
  using OpConversionPattern<AtenClampOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenClampOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    auto elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op,
          "Clamp input tensor must have integer or floating-point datatype");

    Value minValue = op.getMin();
    Value maxValue = op.getMax();
    if (checkNotNone(rewriter, op, minValue).failed() &&
        checkNotNone(rewriter, op, maxValue).failed()) {
      return rewriter.notifyMatchFailure(
          op, "clamp op requires at least one of min or max");
    }

    auto setMinMaxAttrs = [&](Value value, FloatAttr &floatAttr,
                              IntegerAttr &intAttr) {
      double floatValue;
      int64_t intValue;
      if (matchPattern(value, m_TorchConstantFloat(&floatValue))) {
        if (elementType.isa<mlir::FloatType>())
          floatAttr = rewriter.getF32FloatAttr(floatValue);
        else if (elementType.isa<mlir::IntegerType>())
          intAttr =
              rewriter.getI64IntegerAttr(static_cast<int64_t>(floatValue));
      } else if (matchPattern(value, m_TorchConstantInt(&intValue))) {
        if (elementType.isa<mlir::FloatType>())
          floatAttr = rewriter.getF32FloatAttr(static_cast<float>(intValue));
        else if (elementType.isa<mlir::IntegerType>())
          intAttr = rewriter.getI64IntegerAttr(intValue);
      } else {
        llvm_unreachable("only float or integer constants are supported as min "
                         "/ max values");
      }
    };

    FloatAttr minFloatAttr, maxFloatAttr;
    IntegerAttr minIntAttr, maxIntAttr;
    if (checkNotNone(rewriter, op, minValue).succeeded()) {
      setMinMaxAttrs(minValue, minFloatAttr, minIntAttr);
    }
    if (checkNotNone(rewriter, op, maxValue).succeeded()) {
      setMinMaxAttrs(maxValue, maxFloatAttr, maxIntAttr);
    }

    rewriter.replaceOpWithNewOp<tcp::ClampOp>(op, inputType, input,
                                              minFloatAttr, maxFloatAttr,
                                              minIntAttr, maxIntAttr);
    return success();
  }
};

class ConvertAtenReluOp : public OpConversionPattern<AtenReluOp> {
public:
  using OpConversionPattern<AtenReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    auto elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Relu input tensor must have integer or floating-point datatype");

    FloatAttr minFloatAttr, maxFloatAttr;
    IntegerAttr minIntAttr, maxIntAttr;
    if (elementType.isa<mlir::FloatType>())
      minFloatAttr = rewriter.getF32FloatAttr(0.0f);
    else
      minIntAttr = rewriter.getI64IntegerAttr(0);

    rewriter.replaceOpWithNewOp<tcp::ClampOp>(op, inputType, input,
                                              minFloatAttr, maxFloatAttr,
                                              minIntAttr, maxIntAttr);
    return success();
  }
};

class ConvertAtenAbsOp : public OpConversionPattern<AtenAbsOp> {
public:
  using OpConversionPattern<AtenAbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenAbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    auto elementType = inputType.getElementType();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Abs input tensor must have integer or floating-point datatype");

    rewriter.replaceOpWithNewOp<tcp::AbsOp>(op, inputType, input);
    return success();
  }
};

template <typename AtenOpT, typename TcpOpT>
class ConvertAtenUnaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    if (!inputType.getElementType().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "Input tensor must have floating-point datatype");

    rewriter.replaceOpWithNewOp<TcpOpT>(op, inputType, input);
    return success();
  }
};

class ConvertAtenAtan2Op : public OpConversionPattern<AtenAtan2Op> {
public:
  using OpConversionPattern<AtenAtan2Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenAtan2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenAtan2Op>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    if (!lhsType.getElementType().isa<mlir::FloatType>() ||
        !rhsType.getElementType().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "Input tensors must have floating-point datatype");

    auto inputAType = op.getSelf()
                          .getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();
    auto inputBType = op.getOther()
                          .getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();
    auto outputType = op.getType()
                          .template dyn_cast<torch::Torch::ValueTensorType>()
                          .getDtype();

    rhs = torch_to_tcp::castTensorToDtype(rewriter, inputBType, outputType, rhs,
                                          resultType.getElementType());
    rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
        rewriter, rhs, lhs, resultType.getElementType());

    lhs = torch_to_tcp::castTensorToDtype(rewriter, inputAType, outputType, lhs,
                                          resultType.getElementType());
    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShapeAndType(
        rewriter, lhs, rhs, resultType.getElementType());

    rewriter.replaceOpWithNewOp<tcp::Atan2Op>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenToDtypeOp : public OpConversionPattern<AtenToDtypeOp> {
public:
  using OpConversionPattern<AtenToDtypeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenToDtypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    Value input = op.getSelf();
    auto inputType = input.getType().dyn_cast<torch::Torch::ValueTensorType>();
    auto outputType = op.getType().dyn_cast<torch::Torch::ValueTensorType>();
    RankedTensorType resultType =
        OpConversionPattern<AtenToDtypeOp>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "Expected Input/Output to be ValueTensorType");

    auto elementType = inputType.getDtype();
    if (!elementType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Input tensor must have integer or floating-point datatype");

    // The non_blocking arg should be a constant `False`.
    bool nonBlocking;
    if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking)) ||
        nonBlocking) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: non_blocking arg must "
                                         "be a constant with False value");
    }

    // The copy arg should be a constant `False`.
    bool copy;
    if (!matchPattern(op.getCopy(), m_TorchConstantBool(&copy)) || copy) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: copy arg must be a constant with False value");
    }

    // Only `none`, `contiguous` and `preserve` memory_format is supported.
    if (!op.getMemoryFormat().getType().isa<Torch::NoneType>()) {
      int64_t memoryFormat;
      if (!matchPattern(op.getMemoryFormat(),
                        m_TorchConstantInt(&memoryFormat)) ||
          (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
           memoryFormat != torch_upstream::MemoryFormat::Preserve))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: the memory format should be specified in "
                "an integer constant with none, contiguous or preserve value");
    }

    if (inputType.getDtype().isa<mlir::FloatType>() &&
        outputType.getDtype().isa<mlir::FloatType>())
      // FP -> FP
      rewriter.replaceOpWithNewOp<tcp::CastOp>(
          op, resultType, adaptor.getSelf(), SignednessAttr{},
          SignednessAttr{});
    else if (inputType.getDtype().isa<mlir::FloatType>()) {
      // FP -> INT
      if (auto intType = outputType.getDtype().dyn_cast<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<tcp::CastOp>(
            op, resultType, adaptor.getSelf(), SignednessAttr{},
            getTcpSignednessAttr(context, intType.getSignedness()));
      else
        return rewriter.notifyMatchFailure(
            op, "expect output type to be signless/signed/unsigned integer");
    }
    else if (outputType.getDtype().isa<mlir::FloatType>()) {
      // INT -> FP
      if (auto intType = inputType.getDtype().dyn_cast<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<tcp::CastOp>(
            op, resultType, adaptor.getSelf(),
            getTcpSignednessAttr(context, intType.getSignedness()),
            SignednessAttr{});
      else
        return rewriter.notifyMatchFailure(
            op, "expect input type to be signless/signed/unsigned integer");
    }
    else {
      // INT -> INT
      auto inIntType = inputType.getDtype().dyn_cast<mlir::IntegerType>();
      auto outIntType = outputType.getDtype().dyn_cast<mlir::IntegerType>();
      if (inIntType && outIntType)
        rewriter.replaceOpWithNewOp<tcp::CastOp>(
            op, resultType, adaptor.getSelf(),
            getTcpSignednessAttr(context, inIntType.getSignedness()),
            getTcpSignednessAttr(context, outIntType.getSignedness()));
      else
        return rewriter.notifyMatchFailure(op,
                                           "invalid input/output data type");
    }
    return success();
  }
};

} // namespace

void torch_to_tcp::populateElementwisePatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenToDtypeOp>();
  patterns.add<ConvertAtenToDtypeOp>(typeConverter, context);

  target.addIllegalOp<AtenClampOp>();
  patterns.add<ConvertAtenClampOp>(typeConverter, context);
  target.addIllegalOp<AtenReluOp>();
  patterns.add<ConvertAtenReluOp>(typeConverter, context);

  target.addIllegalOp<AtenAddTensorOp>();
  target.addIllegalOp<AtenSubTensorOp>();
  target.addIllegalOp<AtenAddScalarOp>();
  target.addIllegalOp<AtenSubScalarOp>();
  patterns.add<ConvertAtenAddSubOp<AtenAddTensorOp, tcp::AddOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenAddSubOp<AtenSubTensorOp, tcp::SubOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenAddSubOp<AtenAddScalarOp, tcp::AddOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenAddSubOp<AtenSubScalarOp, tcp::SubOp>>(typeConverter,
                                                                 context);

  target.addIllegalOp<AtenMulTensorOp>();
  target.addIllegalOp<AtenMulScalarOp>();
  patterns.add<ConvertAtenMulOp<AtenMulTensorOp>>(typeConverter, context);
  patterns.add<ConvertAtenMulOp<AtenMulScalarOp>>(typeConverter, context);

  target.addIllegalOp<AtenDivTensorOp>();
  target.addIllegalOp<AtenDivScalarOp>();
  patterns.add<ConvertAtenDivOp<AtenDivTensorOp>>(typeConverter, context);
  patterns.add<ConvertAtenDivOp<AtenDivScalarOp>>(typeConverter, context);

  target.addIllegalOp<AtenCeilOp>();
  target.addIllegalOp<AtenFloorOp>();
  target.addIllegalOp<AtenSqrtOp>();
  target.addIllegalOp<AtenSigmoidOp>();
  target.addIllegalOp<AtenTanhOp>();
  target.addIllegalOp<AtenSinOp>();
  target.addIllegalOp<AtenCosOp>();
  target.addIllegalOp<AtenLogOp>();
  target.addIllegalOp<AtenNegOp>();
  target.addIllegalOp<AtenAtanOp>();
  patterns.add<ConvertAtenUnaryOp<AtenFloorOp, tcp::FloorOp>>(typeConverter,
                                                              context);
  patterns.add<ConvertAtenUnaryOp<AtenCeilOp, tcp::CeilOp>>(typeConverter,
                                                            context);
  patterns.add<ConvertAtenUnaryOp<AtenSqrtOp, tcp::SqrtOp>>(typeConverter,
                                                            context);
  patterns.add<ConvertAtenUnaryOp<AtenSigmoidOp, tcp::SigmoidOp>>(typeConverter,
                                                                  context);
  patterns.add<ConvertAtenUnaryOp<AtenTanhOp, tcp::TanhOp>>(typeConverter,
                                                            context);
  patterns.add<ConvertAtenUnaryOp<AtenSinOp, tcp::SinOp>>(typeConverter,
                                                          context);
  patterns.add<ConvertAtenUnaryOp<AtenCosOp, tcp::CosOp>>(typeConverter,
                                                          context);
  patterns.add<ConvertAtenUnaryOp<AtenLogOp, tcp::LogOp>>(typeConverter,
                                                          context);
  patterns.add<ConvertAtenUnaryOp<AtenNegOp, tcp::NegOp>>(typeConverter,
                                                          context);
  patterns.add<ConvertAtenUnaryOp<AtenAtanOp, tcp::AtanOp>>(typeConverter,
                                                            context);

  target.addIllegalOp<AtenAbsOp>();
  patterns.add<ConvertAtenAbsOp>(typeConverter, context);

  target.addIllegalOp<AtenBatchNormOp>();
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);

  target.addIllegalOp<AtenAtan2Op>();
  patterns.add<ConvertAtenAtan2Op>(typeConverter, context);
}
