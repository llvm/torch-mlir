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
    void populateAdaptivePoolingPatternsAndLegality(TypeConverter & typeConverter,
                                            RewritePatternSet & patterns,
                                            ConversionTarget & target);

    // location and context
    Location loc = op->getLoc();
    MLIRContext *context = op.getContext();


    const TypeConverter *typeconverter = getTypeConverter();

    // input tensor and output shape
    Value input = adaptor.getSelf();
    Value outputShape = adaptor.getOutputSize();
    SmallVector<Value> outShapeVector;
    getListConstructElements(outputShape,outShapeVector);
    outShapeVector = getTypeConvertedValues(rewriter,loc,typeconverter,outShapeVector);
    Value Hout = outShapeVector[0];

    // get rank of input
    int64_t rank = input.getType().cast<RankedTensorType>().getRank();

    // input operand should be NCH (i.e. rank 3)
    if (rank != 3) {
        return rewriter.notifyMatchFailure(op,"only supports input type NCH");
    }

    Type elementType =
    input.getType().cast<RankedTensorType>().getElementType(); 

    Value N = getDimOp(rewriter, loc, input, 0);
    Value C = getDimOp(rewriter, loc, input, 1);
    Value Hin = getDimOp(rewriter, loc, input, 2);

    //Value indexZero = rewriter.create<arith::ConstantOp>(loc, Hin.getType(),0 );
    //Value indexOne = rewriter.create<arith::ConstantOp>(loc, Hin.getType(),1 );
    
    //Max Kernel Size
    Value Kmax = rewriter.create<arith::CeilDivSIOp>(loc, Hin, Hout);
    //make a useless fucking tensor
    Type dummyType = rewriter.getI1Type();
    Value Kiter = rewriter.create<tensor::EmptyOp>(loc, RankedTensorType::get({ShapedType::kDynamic}, dummyType), Kmax);
    Value BuffVal = rewriter.create<arith::ConstantOp>(loc, elementType, rewriter.getFloatAttr(elementType,0));
    input.dump(); 

    auto BuffInput = rewriter.create<tensor::PadOp>(loc, input.getType(), input,  ArrayRef<int64_t>({0,0,0}), ArrayRef<int64_t>({0,0,1}), ValueRange{}, ValueRange{});
    {
    SmallVector<Type> blockArgTypes(rank, rewriter.getIndexType());
    SmallVector<Location> blockArgLocs(rank, loc);
    OpBuilder::InsertionGuard guard(rewriter); 
    rewriter.createBlock(&BuffInput.getRegion(), BuffInput.getRegion().end(), blockArgTypes, blockArgLocs); 
    rewriter.create<tensor::YieldOp>(loc, BuffVal);
    }
    //Value BuffInputResult = BuffInput.getResult();
    BuffInput.dump();
    

    return failure();
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
