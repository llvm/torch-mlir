//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenMaxPool2dOp : public OpConversionPattern<AtenMaxPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    Value ceilMode = adaptor.ceil_mode();

    Type elementType = self.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    SmallVector<int64_t, 2> paddingInts;
    if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int paddings");
    SmallVector<int64_t, 2> kernelSizeInts;
    if (!matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSizeInts)))
      return rewriter.notifyMatchFailure(op, "only support kernel size ints");

    Value falseValue = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(rewriter.getIntegerType(1), 0));
    Value ceilModeFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, ceilMode, falseValue);
    rewriter.create<cf::AssertOp>(
        loc, ceilModeFalse,
        rewriter.getStringAttr("only ceil_mode false is supported"));

    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput = torch_to_linalg::getPaddedTensor(op, rewriter, self,
                                                         paddingIncludingNC);

    Value N = getDimOp(rewriter, loc, self, 0);
    Value C = getDimOp(rewriter, loc, self, 1);
    Value H = getDimOp(rewriter, loc, self, 2);
    Value W = getDimOp(rewriter, loc, self, 3);

    SmallVector<Value> paddingIntValues =
        getAsConstantIntValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> kernelSizeIntValues =
        getAsConstantIntValues(rewriter, loc, kernelSizeInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    Value Hout = torch_to_linalg::getOutputDimForConvOps(
        rewriter, loc, H, paddingIntValues[0], dilationIntValues[0],
        kernelSizeIntValues[0], strideIntValues[0]);
    Value Wout = torch_to_linalg::getOutputDimForConvOps(
        rewriter, loc, W, paddingIntValues[1], dilationIntValues[1],
        kernelSizeIntValues[1], strideIntValues[1]);

    // Initialize output tensor with smallest floating point value
    Value outTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, C, Hout, Wout}, elementType);
    auto initialAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getSmallest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative*/ true));
    Value initValue = rewriter.create<arith::ConstantOp>(loc, initialAttr);
    Value outTensorInitialized =
        rewriter.create<linalg::FillOp>(loc, initValue, outTensor).getResult(0);

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
    Value windowTensor = rewriter.create<linalg::InitTensorOp>(
        loc, getAsConstantIndexValues(rewriter, loc, kernelSizeInts),
        elementType);

    Value maxPool2d = rewriter
                          .create<linalg::PoolingNchwMaxOp>(
                              loc, outTensorInitialized.getType(),
                              ValueRange{paddedInput, windowTensor},
                              outTensorInitialized, stridesAttr, dilationAttr)
                          .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, maxPool2d);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAdaptiveAvgPool2dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.self(); /* in form of N*C*H*W */
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type elementType = inputType.getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    auto inputRank = inputType.getRank();
    if (inputRank != 4)
      return rewriter.notifyMatchFailure(op, "input should be rank 4");

    SmallVector<int64_t, 2> expects{1, 1};
    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    if (!isConstantIntListMatching(op.output_size(), expects))
      return rewriter.notifyMatchFailure(
          op, "only support output_size with H and W both equal to constant 1");

    Value N = getDimOp(rewriter, loc, input, 0);
    Value C = getDimOp(rewriter, loc, input, 1);
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, C}, elementType);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
    Value initTensor0 =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);

    SmallVector<AffineExpr, 2> ncExprs;
    ncExprs.push_back(mlir::getAffineDimExpr(0, context));
    ncExprs.push_back(mlir::getAffineDimExpr(1, context));
    auto ncIndexingMap = AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0, ncExprs, context);
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(4), // input
        ncIndexingMap,                      // output
    };
    SmallVector<StringRef, 4> iteratorTypesSum{"parallel", "parallel",
                                               "reduction", "reduction"};
    Value sumPool2d = rewriter
                          .create<linalg::GenericOp>(
                              loc, initTensor0.getType(), input, initTensor0,
                              /*indexingMaps=*/indexingMaps,
                              /*iteratorTypes=*/iteratorTypesSum,
                              [&](OpBuilder &b, Location loc, ValueRange args) {
                                Value input = args[0], sum = args[1];
                                Value result = rewriter.create<arith::AddFOp>(
                                    loc, sum, input);
                                b.create<linalg::YieldOp>(loc, result);
                              })
                          .getResult(0);

    // Calculate H*W so that avg can be got from sum / (H*W)
    Value H = getDimOp(rewriter, loc, input, 2);
    Value W = getDimOp(rewriter, loc, input, 3);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(
          loc, IntegerType::get(context, 64), v);
    };
    Value HtimesW = rewriter.create<arith::MulIOp>(loc, castIndexToInt(H),
                                                   castIndexToInt(W));
    Value HtimesWf =
        rewriter.create<arith::SIToFPOp>(loc, elementType, HtimesW);

    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
    Value outputTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, C, c1Index, c1Index}, elementType);
    SmallVector<AffineMap, 2> indexingMapsAvg{
        ncIndexingMap, rewriter.getMultiDimIdentityMap(4)};
    SmallVector<StringRef, 4> iteratorTypesAvg(4, "parallel");
    Value avgPool2d =
        rewriter
            .create<linalg::GenericOp>(
                loc, outputTensor.getType(), sumPool2d, outputTensor,
                /*indexingMaps=*/indexingMapsAvg,
                /*iteratorTypes=*/iteratorTypesAvg,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value avg = b.create<arith::DivFOp>(loc, args[0], HtimesWf);
                  b.create<linalg::YieldOp>(loc, avg);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, avgPool2d);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populatePoolingPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMaxPool2dOp>();
  patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
  target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
  patterns.add<ConvertAtenAdaptiveAvgPool2dOp>(typeConverter, context);
}
