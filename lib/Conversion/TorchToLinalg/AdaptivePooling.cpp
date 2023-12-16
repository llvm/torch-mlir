//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <iostream>

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
    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    // make an iteration space of size Kmax = ceildiv Hin, Hout
    Type dummyType = rewriter.getI1Type();
    Value Kiter;
    if (InputType.isDynamicDim(rank - 1) || OutputType.isDynamicDim(rank - 1)) {
      Value Kmax = rewriter.create<arith::CeilDivSIOp>(loc, Hin, HoutIndex);
      Kiter = rewriter.create<tensor::EmptyOp>(
          loc, RankedTensorType::get({ShapedType::kDynamic}, dummyType), Kmax);
    } else {
      int64_t Kmax =
          InputType.getShape()[rank - 1] / OutputType.getShape()[rank - 1];
      Kiter = rewriter.create<tensor::EmptyOp>(
          loc, RankedTensorType::get({Kmax}, dummyType), ValueRange{});
    }

    // need to buffer input, else there will possibly be an out of bounds access
    // later BuffVal = 0 for avg pooling and -inf for max pooling
    Value BuffVal = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));
    auto BuffInput = rewriter.create<tensor::PadOp>(
        loc, input.getType(), input, ArrayRef<int64_t>({0, 0, 0}),
        ArrayRef<int64_t>({0, 0, 1}), ValueRange{}, ValueRange{});
    {
      SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
      SmallVector<Location> blockArgLocs(rank, loc);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&BuffInput.getRegion(), BuffInput.getRegion().end(),
                           blockArgTypes, blockArgLocs);
      rewriter.create<tensor::YieldOp>(loc, BuffVal);
    }

    // make a list of DynamicOutputSizes
    SmallVector<Value> DynOutputSizes;
    for (unsigned i = 0; i < rank - 1; i++) {
      if (OutputType.isDynamicDim(i)) {
        DynOutputSizes.push_back(getDimOp(rewriter, loc, input, i));
      }
    }
    if (OutputType.isDynamicDim(rank - 1)) {
      DynOutputSizes.push_back(HoutIndex);
    }

    // initialize an output tensor
    Value EmptyOutput =
        rewriter.create<tensor::EmptyOp>(loc, OutputType, DynOutputSizes);
    Value InitOutput = rewriter.create<linalg::FillOp>(loc, BuffVal, EmptyOutput).getResult(0);

    // setup indexing maps and iterator types for linalg generic op
    // for output (d0,d1,d2,d3) -> (d0,d1,d2)
    // for Kiter (d0,d1,d2,d3) -> (d3)
    SmallVector<AffineExpr> KiterExprs, outputExprs;
    for (unsigned i = 0; i < 3; i++) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    KiterExprs.push_back(rewriter.getAffineDimExpr(3));
    SmallVector<AffineMap> indexingMaps =
        AffineMap::inferFromExprList({KiterExprs, outputExprs});
    SmallVector<utils::IteratorType> iteratorTypes(
        3, utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction);

    Value indexOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto Output = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/OutputType,
        /*inputs=*/ValueRange({Kiter}),
        /*outputs=*/InitOutput,
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
              loc, elementType, BuffInput.getResult(),
              ValueRange({ind0, ind1, windex}));
          // Check if we extracted at windex < end index
          Value cond =
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate(6), windex, e4);
          // if inElt is in bounds, include it in the computation
          // else, use BuffVal = 0 (for max pool use -infinity)
          Value out1 = b.create<arith::SelectOp>(loc, cond, inElt, BuffVal);
          // compute Kernel size: it may be better to divide by this outside op
          Value kw = b.create<arith::SubIOp>(loc, e4, s1);
          Value kwint = castIndexToInt64(b, loc, kw);
          Value kwf = b.create<arith::SIToFPOp>(loc, elementType, kwint);
          // divide value in window by window size
          //TODO: instead separate this computation into another generic
          Value out2 = b.create<arith::DivFOp>(loc, out1, kwf);
          // accumulate out2 to res = args[1]
          Value out3 = b.create<arith::AddFOp>(loc, res, out2);
          b.create<linalg::YieldOp>(loc, out3);
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
