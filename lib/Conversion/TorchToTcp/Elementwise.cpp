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
}
