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
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// TODO: Dropout should probably be handled in DecomposeComplexOps instead of
// here.
class ConvertAtenDropoutOp : public OpConversionPattern<AtenDropoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDropoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    bool train;
    if (!matchPattern(op.train(), m_TorchConstantBool(&train)))
      return rewriter.notifyMatchFailure(op,
                                         "Expected train to be constant bool.");

    if (train)
      return failure();
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.input());
    return success();
  }
};
} // namespace


namespace {
class ConvertAtenUniformOp : public OpConversionPattern<AtenUniformOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUniformOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value self = adaptor.self();
    Value from = adaptor.from();
    Value to = adaptor.to();
    Value generator = adaptor.generator();
    RankedTensorType resultType = self.getType().cast<RankedTensorType>();
    Type elemTy = resultType.getElementType();

    if (!elemTy.isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "This op only support float type");

    if (!generator.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to ben None because only global default "
              "generator is supported");

    // Build the core formula of LCG Algorithm that makes use of element index:
    // For output matrix with rank N:
    // temp1 = (cast(I64, index(D.0)) + seed) * multiplier + incrementStep
    // ...
    // tempN = (cast(I64, index(D.(N))) + tempN-1) * multiplier + incr
    // Refer to https://reviews.llvm.org/D101364.
    // The value of multiplier and incrementStep are referenced from
    // https://en.wikipedia.org/wiki/Linear_congruential_generator for 2^64.
    Value multiplier = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(6364136223846793005));
    Value incrementStep = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(1442695040888963407));
    // Tn = (index + Tn-1) * multiplier + incrementStep
    auto getNextTemp = [&](OpBuilder &b, Value index, Value temp) {
      Value castIndex =
          b.create<arith::IndexCastOp>(loc, b.getI64Type(), index);
      Value add = b.create<arith::AddIOp>(loc, castIndex, temp);
      Value mult = b.create<arith::MulIOp>(loc, add, multiplier);
      return b.create<arith::AddIOp>(loc, mult, incrementStep);
    };

    // Get initial seed, min and max used by `linalg.generic` compute payload.
    Value initialSeed = rewriter.create<TorchConversion::GetNextSeedOp>(loc);
    Value min = convertScalarToDtype(rewriter, loc, from, elemTy);
    Value max = convertScalarToDtype(rewriter, loc, to, elemTy);

    // Construct the `linalg.generic` op.
    auto resultRank = resultType.getRank();
    SmallVector<AffineMap, 1> indexingMaps(
        1, rewriter.getMultiDimIdentityMap(resultRank));
    SmallVector<StringRef> iteratorTypes(resultRank,
                                         getParallelIteratorTypeName());
    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, self);
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
    Value uniformRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), /*inputs=*/ValueRange{},
                /*outputs=*/initTensor, indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value temp = initialSeed;
                  for (int i = 0; i < resultRank; i++) {
                    Value index = b.create<linalg::IndexOp>(loc, i);
                    temp = getNextTemp(b, index, temp);
                  }
                  // scale = (max - min) * const(F64,  5.4210108E-20)
                  // which is derived from rand(min,max) =
                  // rand()/(RAND_MAX/(max-min)) where RAND_MAX = 2^64 - 1
                  Value epsilon = b.create<arith::ConstantOp>(
                      loc, b.getFloatAttr(min.getType(), 5.4210108E-20));
                  Value range = b.create<arith::SubFOp>(loc, max, min);
                  Value scale = b.create<arith::MulFOp>(loc, range, epsilon);

                  // res = cast(F64, tempN) * scale + min
                  Value updateFloat =
                      b.create<arith::UIToFPOp>(loc, elemTy, temp);
                  Value updateScaled =
                      b.create<arith::MulFOp>(loc, updateFloat, scale);
                  Value res = b.create<arith::AddFOp>(loc, updateScaled, min);
                  b.create<linalg::YieldOp>(loc, res);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, uniformRes);
    return success();
  }
};
} // namespace


void mlir::torch::torch_to_linalg::populateRandomPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenDropoutOp>();
  patterns.add<ConvertAtenDropoutOp>(typeConverter, context);
  target.addIllegalOp<AtenUniformOp>();
  patterns.add<ConvertAtenUniformOp>(typeConverter, context);
}
