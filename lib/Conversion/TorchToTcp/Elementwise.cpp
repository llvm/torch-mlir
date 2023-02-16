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

bool skipMultiplyAlpha(Value alphaValue) {
  double doubleValue;
  auto isFloat = matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(alphaValue, m_TorchConstantInt(&intValue));

  return ((isFloat && doubleValue == 1.0) || (isInt && intValue == 1.0));
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
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, lhs, rhs);
    rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, rhs, lhs);

    if (!skipMultiplyAlpha(op.getAlpha()))
      return rewriter.notifyMatchFailure(
          op, "torch ops with alpha != 1 is not yet supported in "
              "Torch to TCP conversion");

    rewriter.replaceOpWithNewOp<TcpOpT>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenMulOp : public OpConversionPattern<AtenMulTensorOp> {
public:
  using OpConversionPattern<AtenMulTensorOp>::OpConversionPattern;
  using OpAdaptor = typename AtenMulTensorOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenMulTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenMulTensorOp>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, lhs, rhs);
    rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, rhs, lhs);

    rewriter.replaceOpWithNewOp<tcp::MulOp>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern<AtenBatchNormOp>::OpConversionPattern;
  using OpAdaptor = typename AtenBatchNormOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();

    Value weight = adaptor.getWeight();
    RankedTensorType weightType = weight.getType().dyn_cast<RankedTensorType>();

    Value bias = adaptor.getBias();
    RankedTensorType biasType = bias.getType().dyn_cast<RankedTensorType>();

    Value running_mean = adaptor.getRunningMean();
    RankedTensorType runningMeanType = running_mean.getType().dyn_cast<RankedTensorType>();

    Value running_var = adaptor.getRunningVar();
    RankedTensorType runningVarType = running_var.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenBatchNormOp>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!inputType || !weightType || !biasType ||
          !runningMeanType || !runningVarType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    assert(runningMeanType.getNumElements() != 0 && runningVarType.getNumElements() != 0);

    double eps = 0.0;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps))) {
      return rewriter.notifyMatchFailure(op, "non-float(double) eps unsupported");
    }

    Value epsVal;
    if (auto result = torch_to_tcp::getConstTensor<float>(rewriter, op, llvm::makeArrayRef(static_cast<float>(eps)), {}))
      epsVal = *result;
    else
      return rewriter.notifyMatchFailure(op, "failed to get constTensor for eps");

    // PyTorch inputs are [NCHW], and BatchNorm parameters are [C] length vectors
    // axisInOutput = 1 allows [C] -> [1, C, 1, 1] expansion followed by a broadcast
    running_mean = torch_to_tcp::broadcast1DToNDAndMatchShape(rewriter, running_mean, input, /*axisInOutput=*/1);
    running_var = torch_to_tcp::broadcast1DToNDAndMatchShape(rewriter, running_var, input, /*axisInOutput=*/1);
    weight = torch_to_tcp::broadcast1DToNDAndMatchShape(rewriter, weight, input, /*axisInOutput=*/1);
    bias = torch_to_tcp::broadcast1DToNDAndMatchShape(rewriter, bias, input, /*axisInOutput=*/1);
    epsVal = torch_to_tcp::broadcast0DToNDAndMatchShape(rewriter, epsVal, input);

    Value op1SubInputMean = rewriter.create<tcp::SubOp>(op.getLoc(), resultType, input, running_mean);
    Value op2AddVarEpsilon = rewriter.create<tcp::AddOp>(op.getLoc(), resultType, running_var, epsVal);
    Value op3SqrtOp2 = rewriter.create<tcp::SqrtOp>(op.getLoc(), resultType, op2AddVarEpsilon);
    Value op4DivOp1Op3 = rewriter.create<tcp::DivFOp>(op.getLoc(), resultType, op1SubInputMean, op3SqrtOp2);
    Value op5MulWeightOp4 = rewriter.create<tcp::MulOp>(op.getLoc(), resultType, weight, op4DivOp1Op3);
    Value op6AddOp5Bias = rewriter.create<tcp::AddOp>(op.getLoc(), resultType, op5MulWeightOp4, bias);

    rewriter.replaceOp(op, {op6AddOp5Bias});
    return success();
  }
};

class ConvertAtenDivFOp : public OpConversionPattern<AtenDivTensorOp> {
public:
  using OpConversionPattern<AtenDivTensorOp>::OpConversionPattern;
  using OpAdaptor = typename AtenDivTensorOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenDivTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    RankedTensorType resultType =
        OpConversionPattern<AtenDivTensorOp>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();

    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    lhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, lhs, rhs);
    rhs = torch_to_tcp::broadcastInLeadingDimsToMatchShape(rewriter, rhs, lhs);

    rewriter.replaceOpWithNewOp<tcp::DivFOp>(op, resultType, lhs, rhs);
    return success();
  }
};

class ConvertAtenTanhOp : public OpConversionPattern<AtenTanhOp> {
public:
  using OpConversionPattern<AtenTanhOp>::OpConversionPattern;
  using OpAdaptor = typename AtenTanhOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenTanhOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    if (!inputType.getElementType().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "Tanh input tensor must have floating-point datatype");

    rewriter.replaceOpWithNewOp<tcp::TanhOp>(op, inputType, input);
    return success();
  }
};

class ConvertAtenSigmoidOp : public OpConversionPattern<AtenSigmoidOp> {
public:
  using OpConversionPattern<AtenSigmoidOp>::OpConversionPattern;
  using OpAdaptor = typename AtenSigmoidOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenSigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    if (!inputType.getElementType().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "Sigmoid input tensor must have floating-point datatype");

    rewriter.replaceOpWithNewOp<tcp::SigmoidOp>(op, inputType, input);
    return success();
  }
};

class ConvertAtenSqrtOp : public OpConversionPattern<AtenSqrtOp> {
public:
  using OpConversionPattern<AtenSqrtOp>::OpConversionPattern;
  using OpAdaptor = typename AtenSqrtOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    if (!inputType.getElementType().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "Sqrt input tensor must have floating-point datatype");

    rewriter.replaceOpWithNewOp<tcp::SqrtOp>(op, inputType, input);
    return success();
  }
};

class ConvertAtenClampOp : public OpConversionPattern<AtenClampOp> {
public:
  using OpConversionPattern<AtenClampOp>::OpConversionPattern;
  using OpAdaptor = typename AtenClampOp::Adaptor;

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
  using OpAdaptor = typename AtenReluOp::Adaptor;

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

} // namespace

void torch_to_tcp::populateElementwisePatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenTanhOp>();
  patterns.add<ConvertAtenTanhOp>(typeConverter, context);
  target.addIllegalOp<AtenClampOp>();
  patterns.add<ConvertAtenClampOp>(typeConverter, context);
  target.addIllegalOp<AtenReluOp>();
  patterns.add<ConvertAtenReluOp>(typeConverter, context);

  target.addIllegalOp<AtenSigmoidOp>();
  patterns.add<ConvertAtenSigmoidOp>(typeConverter, context);

  target.addIllegalOp<AtenAddTensorOp>();
  target.addIllegalOp<AtenSubTensorOp>();
  patterns.add<ConvertAtenAddSubOp<AtenAddTensorOp, tcp::AddOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenAddSubOp<AtenSubTensorOp, tcp::SubOp>>(typeConverter,
                                                                 context);

  target.addIllegalOp<AtenMulTensorOp>();
  patterns.add<ConvertAtenMulOp>(typeConverter, context);

  target.addIllegalOp<AtenDivTensorOp>();
  patterns.add<ConvertAtenDivFOp>(typeConverter, context);

  target.addIllegalOp<AtenSqrtOp>();
  patterns.add<ConvertAtenSqrtOp>(typeConverter, context);

  target.addIllegalOp<AtenBatchNormOp>();
  patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
}
