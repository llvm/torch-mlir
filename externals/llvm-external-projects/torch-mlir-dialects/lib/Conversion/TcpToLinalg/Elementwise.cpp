//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

using namespace mlir;
using namespace mlir::tcp;

namespace {

Value createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    RankedTensorType resultTensorType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  auto resultRank = resultTensorType.getRank();

  // In order to populate the resultDimSizes, we only need to look at one of
  // the tensorOperands, since all the operands are expected to have the same
  // shape.
  SmallVector<OpFoldResult> resultDimSizes =
      mlir::tensor::createDimValues(b, loc, tensorOperands[0]);

  // Add indexing maps for all the tensor operands and for the result.
  SmallVector<AffineMap> indexingMaps{tensorOperands.size() + 1,
                                      b.getMultiDimIdentityMap(resultRank)};

  SmallVector<utils::IteratorType> iteratorTypes(resultRank,
                                                 utils::IteratorType::parallel);

  Value emptyTensor = b.create<tensor::EmptyOp>(
      loc, resultDimSizes, resultTensorType.getElementType());
  return b
      .create<linalg::GenericOp>(loc, emptyTensor.getType(), tensorOperands,
                                 emptyTensor, indexingMaps, iteratorTypes,
                                 bodyBuilder)
      .getResult(0);
}

FailureOr<Value>
createLinalgPayloadForElementwiseOp(Operation *op,
                                    RankedTensorType resultTensorType,
                                    OpBuilder &b, ValueRange payloadArgs) {
  Location loc = op->getLoc();
  auto elemType = resultTensorType.getElementType();

  if (isa<TanhOp>(op))
    return {b.create<math::TanhOp>(loc, payloadArgs[0])};

  if (auto clampOp = dyn_cast<ClampOp>(op)) {
    // This implementation always performs the max followed by min.
    // TODO: Is this going to work for degenerative floating point numbers?
    Value result = payloadArgs[0];
    if (elemType.isa<mlir::FloatType>()) {
      auto minFloat = clampOp.getMinFloat();
      auto maxFloat = clampOp.getMaxFloat();
      if (minFloat)
        result = b.create<arith::MaxFOp>(
            loc, result,
            b.create<arith::ConstantFloatOp>(loc, *minFloat, b.getF32Type()));
      if (maxFloat)
        result = b.create<arith::MinFOp>(
            loc, result,
            b.create<arith::ConstantFloatOp>(loc, *maxFloat, b.getF32Type()));
    } else if (elemType.isa<mlir::IntegerType>()) {
      auto minInt = clampOp.getMinInt();
      auto maxInt = clampOp.getMaxInt();
      if (minInt)
        result = b.create<arith::MaxSIOp>(
            loc, result,
            b.create<arith::ConstantIntOp>(loc, *minInt, b.getIntegerType(64)));
      if (maxInt)
        result = b.create<arith::MinSIOp>(
            loc, result,
            b.create<arith::ConstantIntOp>(loc, *maxInt, b.getIntegerType(64)));
    } else {
      llvm_unreachable(
          "unsupported element type in createLinalgPayloadForElementwiseOp");
    }
    return result;
  }

  if (isa<SigmoidOp>(op)) {
    auto elemType = resultTensorType.getElementType();
    auto one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elemType, 1));
    auto negate = b.create<arith::NegFOp>(loc, payloadArgs[0]);
    auto exp = b.create<math::ExpOp>(loc, negate);
    auto sum = b.create<arith::AddFOp>(loc, exp, one);
    return {b.create<arith::DivFOp>(loc, one, sum)};
  }

  if (isa<SqrtOp>(op)) {
    return {b.create<math::SqrtOp>(loc, payloadArgs[0])};
  }

  if (isa<AddOp>(op)) {
    if (elemType.isa<mlir::FloatType>())
      return {b.create<arith::AddFOp>(loc, payloadArgs[0], payloadArgs[1])};
    else if (elemType.isa<mlir::IntegerType>())
      return {b.create<arith::AddIOp>(loc, payloadArgs[0], payloadArgs[1])};
    else
      llvm_unreachable(
          "unsupported element type in createLinalgPayloadForElementwiseOp");
  }

  if (isa<SubOp>(op)) {
    if (elemType.isa<mlir::FloatType>())
      return {b.create<arith::SubFOp>(loc, payloadArgs[0], payloadArgs[1])};
    else if (elemType.isa<mlir::IntegerType>())
      return {b.create<arith::SubIOp>(loc, payloadArgs[0], payloadArgs[1])};
    else
      llvm_unreachable(
          "unsupported element type in createLinalgPayloadForElementwiseOp");
  }

  if (isa<MulOp>(op)) {
    if (elemType.isa<mlir::FloatType>())
      return {b.create<arith::MulFOp>(loc, payloadArgs[0], payloadArgs[1])};
    else if (elemType.isa<mlir::IntegerType>())
      return {b.create<arith::MulIOp>(loc, payloadArgs[0], payloadArgs[1])};
    else
      llvm_unreachable(
          "unsupported element type in createLinalgPayloadForElementwiseOp");
  }

  if (isa<DivFOp>(op)) {
    if (elemType.isa<mlir::FloatType>())
      return {b.create<arith::DivFOp>(loc, payloadArgs[0], payloadArgs[1])};
    else
      llvm_unreachable(
          "unsupported element type in createLinalgPayloadForElementwiseOp");
  }

  return op->emitError(
      "unimplemented lowering in createLinalgPayloadForElementwiseOp");
}

template <typename TcpOpT>
class ConvertElementwiseOp : public OpConversionPattern<TcpOpT> {
public:
  using OpConversionPattern<TcpOpT>::OpConversionPattern;
  using OpAdaptor = typename TcpOpT::Adaptor;

  LogicalResult
  matchAndRewrite(TcpOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTensorType = OpConversionPattern<TcpOpT>::getTypeConverter()
                                ->convertType(op->getResult(0).getType())
                                .template cast<RankedTensorType>();
    auto tensorOperands = llvm::to_vector<6>(
        llvm::make_filter_range(adaptor.getOperands(), [](Value v) {
          return v.getType().isa<RankedTensorType>();
        }));

    // Create Linalg payload
    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      FailureOr<Value> result = createLinalgPayloadForElementwiseOp(
          op, resultTensorType, b, payloadArgs);
      // TODO: Check for failure once GenericOp::build supports a body builder
      // that can return a LogicalResult.
      b.create<linalg::YieldOp>(loc, *result);
    };

    Value generic = createElementwiseLinalgGeneric(
        rewriter, loc, tensorOperands, resultTensorType, bodyBuilder);
    rewriter.replaceOp(op, generic);
    return success();
  }
};

} // namespace

void mlir::TcpToLinalg::populateElementwisePatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalDialect<TcpDialect>();
  patterns.add<ConvertElementwiseOp<AddOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<ClampOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<MulOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<DivFOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<SubOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<TanhOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<SigmoidOp>>(typeConverter, context);
  patterns.add<ConvertElementwiseOp<SqrtOp>>(typeConverter, context);
}
