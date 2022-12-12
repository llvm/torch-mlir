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

class ConvertAtenAddOp : public OpConversionPattern<AtenAddTensorOp> {
public:
  using OpConversionPattern<AtenAddTensorOp>::OpConversionPattern;
  using OpAdaptor = typename AtenAddTensorOp::Adaptor;

  LogicalResult
  matchAndRewrite(AtenAddTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();

    Value rhs = adaptor.getOther();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    auto lhsRank = lhsType.getRank();
    auto rhsRank = rhsType.getRank();
    if (lhsRank < rhsRank) {
      int64_t rankIncrease = rhsRank - lhsRank;
      lhs =
          torch_to_tcp::broadcastRankInLeadingDims(rewriter, lhs, rankIncrease);
      lhs = torch_to_tcp::broadcastShapeInLeadingDims(rewriter, lhs, rhs,
                                                      rankIncrease);
    }
    if (lhsRank > rhsRank) {
      int64_t rankIncrease = lhsRank - rhsRank;
      rhs =
          torch_to_tcp::broadcastRankInLeadingDims(rewriter, rhs, rankIncrease);
      rhs = torch_to_tcp::broadcastShapeInLeadingDims(rewriter, rhs, lhs,
                                                      rankIncrease);
    }

    if (!skipMultiplyAlpha(op.getAlpha()))
      return rewriter.notifyMatchFailure(
          op, "torch.add with alpha != 1 is not yet supported in "
              "Torch to TCP conversion");

    RankedTensorType resultType =
        OpConversionPattern<AtenAddTensorOp>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tcp::AddOp>(op, resultType, lhs, rhs);
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

} // namespace

void torch_to_tcp::populateElementwisePatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenTanhOp>();
  patterns.add<ConvertAtenTanhOp>(typeConverter, context);

  target.addIllegalOp<AtenAddTensorOp>();
  patterns.add<ConvertAtenAddOp>(typeConverter, context);
}
