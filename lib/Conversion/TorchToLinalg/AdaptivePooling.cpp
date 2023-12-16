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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenAdaptiveAvgPool1dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool1dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool1dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    const TypeConverter *typeconverter = getTypeConverter();

    // input tensor and output shape
    Value input = adaptor.getSelf();
    Value outputShape = adaptor.getOutputSize();
    SmallVector<Value> outShapeVector;
    getListConstructElements(outputShape, outShapeVector);
    outShapeVector =
        getTypeConvertedValues(rewriter, loc, typeconverter, outShapeVector);
    Value Hin = getDimOp(rewriter, loc, input, 2);
    Value Hout = outShapeVector[0];
    Value HoutIndex = castIntToIndex(rewriter, loc, Hout);
    RankedTensorType InputType = input.getType().cast<RankedTensorType>();
    RankedTensorType OutputType =
        typeconverter->convertType(op.getResult().getType())
            .cast<RankedTensorType>();

    // get rank of input (same as rank of output)
    int64_t rank = input.getType().cast<RankedTensorType>().getRank();
    // input operand should be NCH (i.e. rank 3)
    if (rank != 3) {
      return rewriter.notifyMatchFailure(op, "only supports input type NCH");
    }

    // get elementType of input tensor
    Type elementType = InputType.getElementType();

    // make an iteration space of size Kmax = 1 + ceildiv (Hin - 1) , Hout
    Type dummyType = rewriter.getI1Type();
    Value Kiter;
    if (InputType.isDynamicDim(rank - 1) || OutputType.isDynamicDim(rank - 1)) {
      Value constantOne = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value HinPlusOne = rewriter.create<arith::SubIOp>(loc, Hin, constantOne);
      Value KmaxMinusOne = rewriter.create<arith::CeilDivSIOp>(loc, HinPlusOne, HoutIndex);
      Value Kmax = rewriter.create<arith::AddIOp>(loc, constantOne, KmaxMinusOne);
      Kiter = rewriter.create<tensor::EmptyOp>(
          loc, RankedTensorType::get({ShapedType::kDynamic}, dummyType), Kmax);
    } else {
      int64_t HinInt = InputType.getShape()[rank - 1];
      int64_t HoutInt = OutputType.getShape()[rank - 1];
      int64_t correction = ( (HinInt - 1) % HoutInt == 0) ? 0 : 1; 
      int64_t Kmax = 1 + (HinInt - 1) / HoutInt + correction;
      Kiter = rewriter.create<tensor::EmptyOp>(
          loc, RankedTensorType::get({Kmax+1}, dummyType), ValueRange{});
    }

    // need to buffer input, else there will possibly be an out of bounds access
    // later BuffVal = 0 for avg pooling and -inf for max pooling
    Value BuffVal = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));
    SmallVector<int64_t> lowPadding = {0,0,0};
    SmallVector<int64_t> highPadding = {0,0,1}; 
    Value BuffInput = torch_to_linalg::getPaddedTensor(
        op, rewriter, input, lowPadding, highPadding, BuffVal);

    // make a list of DynamicOutputSizes
    SmallVector<Value> dynOutputSizes;
    Value kwTensor;
    for (unsigned i = 0; i < rank - 1; i++) {
      if (OutputType.isDynamicDim(i)) {
        dynOutputSizes.push_back(getDimOp(rewriter, loc, input, i));
      }
    }
    if (OutputType.isDynamicDim(rank - 1)) {
      dynOutputSizes.push_back(HoutIndex);
      kwTensor = rewriter.create<tensor::EmptyOp>(loc, RankedTensorType::get({ShapedType::kDynamic}, elementType), HoutIndex); 
    }
    else {
      kwTensor = rewriter.create<tensor::EmptyOp>(loc, RankedTensorType::get({OutputType.getShape()[rank - 1]}, elementType), ValueRange{}); 
    }

    // initialize an output tensor
    Value EmptyOutput =
        rewriter.create<tensor::EmptyOp>(loc, OutputType, dynOutputSizes);
    Value InitOutput = rewriter.create<linalg::FillOp>(loc, BuffVal, EmptyOutput).getResult(0);
    // initialize a kernel-width tensor (Avg - only)

    // setup indexing maps and iterator types for linalg generic op
    // for output (d0,d1,d2,d3) -> (d0,d1,d2)
    // for Kiter (d0,d1,d2,d3) -> (d3)
    SmallVector<AffineExpr> KiterExprs, outputExprs, kwTensorExprs;
    for (unsigned i = 0; i < 3; i++) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    kwTensorExprs.push_back(rewriter.getAffineDimExpr(2));
    KiterExprs.push_back(rewriter.getAffineDimExpr(3));
    SmallVector<AffineMap> indexingMaps =
        AffineMap::inferFromExprList({KiterExprs, outputExprs, kwTensorExprs});
    SmallVector<utils::IteratorType> iteratorTypes(
        3, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);

    Value indexOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto SumPool = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/TypeRange({OutputType, kwTensor.getType()}),
        /*inputs=*/ValueRange({Kiter}),
        /*outputs=*/ValueRange({InitOutput, kwTensor}),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value res = args[1];
          Value ind0 = b.create<linalg::IndexOp>(loc, 0);
          Value ind1 = b.create<linalg::IndexOp>(loc, 1);
          Value ind2 = b.create<linalg::IndexOp>(loc, 2);
          Value ind3 = b.create<linalg::IndexOp>(loc, 3);
          // compute start and end indices
          // st = s1( s0(ind2 * Hin) // Hout )
          Value s0 = b.create<arith::MulIOp>(loc, ind2, Hin);
          Value s1 = b.create<arith::FloorDivSIOp>(loc, s0, HoutIndex);
          // en = e4( 1 + e3( e2( e1( e0(ind2 + 1) * Hin ) - 1 ) // Hout ) )
          Value e0 = b.create<arith::AddIOp>(loc, ind2, indexOne);
          Value e1 = b.create<arith::MulIOp>(loc, e0, Hin);
          Value e2 = b.create<arith::SubIOp>(loc, e1, indexOne);
          Value e3 = b.create<arith::FloorDivSIOp>(loc, e2, HoutIndex);
          Value e4 = b.create<arith::AddIOp>(loc, indexOne, e3);
          // get input element @ st + ind3:
          Value windex = b.create<arith::AddIOp>(loc, s1, ind3);
          Value inElt = b.create<tensor::ExtractOp>(
              loc, elementType, BuffInput,
              ValueRange({ind0, ind1, windex}));
          // Check if we extracted at windex < end index
          Value cond =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate(6), windex, e4);
          // if inElt is in bounds, include it in the computation
          // else, use BuffVal = 0 (for max pool use -infinity)
          Value out1 = b.create<arith::SelectOp>(loc, cond, inElt, BuffVal);
          // compute Kernel size: we store this to kwTensor 
          Value kw = b.create<arith::SubIOp>(loc, e4, s1);
          Value kwint = castIndexToInt64(b, loc, kw);
          Value kwf = b.create<arith::SIToFPOp>(loc, elementType, kwint);
          // accumulate out2 to res = args[1]
          Value out3 = b.create<arith::AddFOp>(loc, res, out1);
          b.create<linalg::YieldOp>(loc, ValueRange({out3, kwf}));
        });

    SmallVector<AffineMap> indexingMaps1 =
        AffineMap::inferFromExprList({kwTensorExprs, outputExprs});
    SmallVector<utils::IteratorType> iteratorTypes1(
        3, utils::IteratorType::parallel);
    auto Output = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/OutputType,
        /*inputs=*/SumPool.getResultTensors()[1],
        /*outputs=*/SumPool.getResultTensors()[0],
        /*indexingMaps=*/indexingMaps1,
        /*iteratorTypes=*/iteratorTypes1,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value q = b.create<arith::DivFOp>(loc, args[1], args[0]);
          b.create<linalg::YieldOp>(loc, q);
        }); 
    
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, OutputType,
                                                Output.getResultTensors());
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateAdaptivePoolingPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenAdaptiveAvgPool1dOp>();
  patterns.add<ConvertAtenAdaptiveAvgPool1dOp>(typeConverter, context);
}
