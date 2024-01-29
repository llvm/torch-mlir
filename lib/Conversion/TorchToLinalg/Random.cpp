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
    if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train)))
      return rewriter.notifyMatchFailure(op,
                                         "Expected train to be constant bool.");

    if (train)
      return failure();
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                adaptor.getInput());
    return success();
  }
};
} // namespace

static Value toLinearIndex(OpBuilder &b, Location loc,
                           ArrayRef<Value> indicesIntValues,
                           ArrayRef<Value> shapeIntValues) {
  assert(indicesIntValues.size() == shapeIntValues.size() &&
         "Expected `indices` and `shape` to have the same size");
  Value result =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(b.getI64Type()));
  for (auto [index, stride] : llvm::zip(indicesIntValues, shapeIntValues)) {
    assert(index.getType().isa<mlir::IntegerType>() &&
           stride.getType().isa<mlir::IntegerType>() &&
           "Input arrays to `toLinearIndex` must only contain values of type "
           "`mlir::IntegerType`");
    Value mul = b.create<arith::MulIOp>(loc, result, stride);
    result = b.create<arith::AddIOp>(loc, mul, index);
  }
  return result;
}

// Squares64 Algorithm for generating 64-bit random numbers.
// See: https://arxiv.org/abs/2004.06278
static Value randomUniformUInt(OpBuilder &b, Location loc, Value ctr,
                               Value key) {
  auto mul = [&](Value lhs, Value rhs) -> Value {
    return b.create<arith::MulIOp>(loc, lhs, rhs);
  };
  auto add = [&](Value lhs, Value rhs) -> Value {
    return b.create<arith::AddIOp>(loc, lhs, rhs);
  };
  Value cst32 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(32));
  auto shiftRight32 = [&](Value val) -> Value {
    return b.create<arith::ShRUIOp>(loc, val, cst32);
  };
  auto swapLoHi = [&](Value val) -> Value {
    Value leftShift = b.create<arith::ShLIOp>(loc, val, cst32);
    Value rightShift = shiftRight32(val);
    return b.create<arith::OrIOp>(loc, leftShift, rightShift);
  };
  auto bitwiseXOr = [&](Value lhs, Value rhs) -> Value {
    return b.create<arith::XOrIOp>(loc, lhs, rhs);
  };

  Value t, x, y, z;
  x = mul(ctr, key);
  y = x;
  z = add(y, key);
  x = add(mul(x, x), y);
  x = swapLoHi(x);
  x = add(mul(x, x), z);
  x = swapLoHi(x);
  x = add(mul(x, x), y);
  x = swapLoHi(x);
  t = x = add(mul(x, x), z);
  x = swapLoHi(x);
  return bitwiseXOr(t, shiftRight32(add(mul(x, x), y)));
}

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
    Value self = adaptor.getSelf();
    Value from = adaptor.getFrom();
    Value to = adaptor.getTo();
    Value generator = adaptor.getGenerator();
    RankedTensorType resultType = self.getType().cast<RankedTensorType>();
    Type elemTy = resultType.getElementType();

    if (!elemTy.isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "This op only support float type");

    if (!generator.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");
    // Get key, min and max used by `linalg.generic` compute payload.
    Value key = rewriter.create<TorchConversion::GetNextSeedOp>(loc);
    Value min = convertScalarToDtype(rewriter, loc, from, elemTy);
    Value max = convertScalarToDtype(rewriter, loc, to, elemTy);

    // Construct the `linalg.generic` op.
    auto resultRank = resultType.getRank();
    SmallVector<AffineMap, 1> indexingMaps(
        1, rewriter.getMultiDimIdentityMap(resultRank));
    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);
    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, self);
    SmallVector<Value> sizesIntValues =
        castIndexVectorToInt64Vector(rewriter, loc, sizes);
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
    Value uniformRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), /*inputs=*/ValueRange{},
                /*outputs=*/initTensor, indexingMaps, iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  SmallVector<Value> indicesIntValues;
                  for (int i = 0; i < resultRank; i++) {
                    indicesIntValues.push_back(castIndexToInt64(
                        b, loc, b.create<linalg::IndexOp>(loc, i)));
                  }

                  Value linearIndex =
                      toLinearIndex(b, loc, indicesIntValues, sizesIntValues);
                  Value randomVal = randomUniformUInt(b, loc, linearIndex, key);

                  // scale = (max - min) * const(F64,  5.4210108E-20)
                  // which is derived from rand(min,max) =
                  // rand()/(RAND_MAX/(max-min)) where RAND_MAX = 2^64 - 1
                  Value epsilon = b.create<arith::ConstantOp>(
                      loc, b.getFloatAttr(min.getType(), 5.4210108E-20));
                  Value range = b.create<arith::SubFOp>(loc, max, min);
                  Value scale = b.create<arith::MulFOp>(loc, range, epsilon);

                  // res = cast(F64, tempN) * scale + min
                  Value updateFloat =
                      b.create<arith::UIToFPOp>(loc, elemTy, randomVal);
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
