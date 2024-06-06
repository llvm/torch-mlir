//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
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
    auto resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
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
    assert(isa<mlir::IntegerType>(index.getType()) &&
           isa<mlir::IntegerType>(stride.getType()) &&
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

// generate uniform random Float64
static Value randomUniformF64(OpBuilder &b, Location loc, Value ctr, Value key,
                              Value min, Value max) {
  Value randomVal = randomUniformUInt(b, loc, ctr, key);
  // scale = (max - min) * const(F64,  5.4210108E-20)
  // which is derived from rand(min,max) =
  // rand()/(RAND_MAX/(max-min)) where RAND_MAX = 2^64 - 1
  Value epsilon = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(b.getF64Type(), 5.4210108E-20));
  Value range = b.create<arith::SubFOp>(loc, max, min);
  Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
  // res = cast(F64, tempN) * scale + min
  Value updateFloat = b.create<arith::UIToFPOp>(loc, b.getF64Type(), randomVal);
  Value updateScaled = b.create<arith::MulFOp>(loc, updateFloat, scale);
  Value uniformSample = b.create<arith::AddFOp>(loc, updateScaled, min);

  return uniformSample;
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
    RankedTensorType resultType = cast<RankedTensorType>(self.getType());
    Type elemTy = resultType.getElementType();
    Type f64Ty = rewriter.getF64Type();

    if (!isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "This op only support float type");

    if (!isa<Torch::NoneType>(generator.getType()))
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");
    // Get key, min and max used by `linalg.generic` compute payload.
    Value key = rewriter.create<TorchConversion::GetNextSeedOp>(loc);
    Value min = convertScalarToDtype(rewriter, loc, from, f64Ty);
    Value max = convertScalarToDtype(rewriter, loc, to, f64Ty);

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

                  Value res =
                      randomUniformF64(b, loc, linearIndex, key, min, max);
                  Value truncRes = res;
                  if (isa<Float16Type, Float32Type>(elemTy))
                    truncRes = b.create<arith::TruncFOp>(loc, elemTy, res);
                  b.create<linalg::YieldOp>(loc, truncRes);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, uniformRes);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMultinomialOp : public OpConversionPattern<AtenMultinomialOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMultinomialOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value self = adaptor.getSelf();
    Value numSamples = adaptor.getNumSamples();
    Value generator = adaptor.getGenerator();
    RankedTensorType selfType = cast<RankedTensorType>(self.getType());
    Type elemTy = selfType.getElementType();
    Type f64Ty = rewriter.getF64Type();
    Type i64Ty = rewriter.getI64Type();
    Type indexTy = rewriter.getIndexType();
    int64_t inputRank = selfType.getRank();
    bool bReplacement;

    if (!isa<mlir::FloatType>(elemTy))
      return rewriter.notifyMatchFailure(op, "This op only support float type");

    if (!mlir::isa<Torch::NoneType>(generator.getType()))
      return rewriter.notifyMatchFailure(
          op, "The generator has to be None because only global default "
              "generator is supported");

    if (!matchPattern(op.getReplacement(), m_TorchConstantBool(&bReplacement)))
      return rewriter.notifyMatchFailure(
          op, "Unsupported: replacement must be a boolean value");

    if (!bReplacement)
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented: replacement = False");

    if (!mlir::isa<mlir::IntegerType>(numSamples.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported: num_samples must be an integer value");
    }

    if (!(inputRank == 1 || inputRank == 2)) {
      return rewriter.notifyMatchFailure(
          op, "torch.multinomial accepts only rank 1 or 2 tensors as weights");
    }

    Value cstZero = rewriter.create<arith::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(0));
    Value cstOne = rewriter.create<arith::ConstantOp>(
        loc, i64Ty, rewriter.getI64IntegerAttr(1));
    Value zeroIndex =
        rewriter.create<arith::IndexCastOp>(loc, indexTy, cstZero);
    Value oneIndex = rewriter.create<arith::IndexCastOp>(loc, indexTy, cstOne);
    Value numSamples_index =
        rewriter.create<arith::IndexCastOp>(loc, indexTy, numSamples);

    Value numDistributions;
    Value numCategoriesIndex;
    ValueRange resultShape;
    if (inputRank == 1) {
      numDistributions = rewriter.create<arith::ConstantOp>(
          loc, i64Ty, rewriter.getI64IntegerAttr(1));
      numCategoriesIndex =
          rewriter.create<tensor::DimOp>(loc, indexTy, self, zeroIndex);
      resultShape = ValueRange{numSamples_index};
    } else {
      Value numDistIndex =
          rewriter.create<tensor::DimOp>(loc, indexTy, self, zeroIndex);
      numCategoriesIndex =
          rewriter.create<tensor::DimOp>(loc, indexTy, self, oneIndex);
      numDistributions =
          rewriter.create<arith::IndexCastOp>(loc, i64Ty, numDistIndex);
      resultShape = ValueRange{numDistIndex, numSamples_index};
    }

    Value numCategories =
        rewriter.create<arith::IndexCastOp>(loc, i64Ty, numCategoriesIndex);
    Value resultTensor = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(resultShape), i64Ty);

    // sum weights for normalization
    torch_to_linalg::ReductionOpInfo opInfo;
    if (inputRank == 1) {
      opInfo = {false, self, {0}};
    } else {
      opInfo = {false, self, {1}};
    }
    Value initSum = rewriter.create<arith::ConstantOp>(
        loc, f64Ty, rewriter.getF64FloatAttr(0.0));
    auto sumBody = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      Value input = payloadArgs[0];
      Value result = payloadArgs[1];
      Value nextSum = b.create<arith::AddFOp>(loc, input, result);
      b.create<linalg::YieldOp>(loc, nextSum);
    };
    Value sumWeights = torch_to_linalg::createReductionLinalgGeneric(
        rewriter, loc, opInfo, initSum, sumBody);

    // Get multinomial samples for each weight vector
    auto multinomialComputation = [&](OpBuilder &b, Location loc, Value j,
                                      ValueRange args) {
      Value jIndex = b.create<arith::IndexCastOp>(loc, indexTy, j);

      Value sum;
      if (inputRank == 1) {
        sum = b.create<tensor::ExtractOp>(loc, sumWeights, ValueRange{});
      } else {
        sum = b.create<tensor::ExtractOp>(loc, sumWeights, ValueRange{jIndex});
      }

      // compute cdf in loop
      Value initCdf = b.create<tensor::EmptyOp>(
          loc, getAsOpFoldResult(ValueRange{numCategoriesIndex}), elemTy);
      Value cdf =
          b.create<scf::ForOp>(
               loc, cstZero, numCategories, cstOne, ValueRange{initCdf},
               [&](OpBuilder &b, Location loc, Value i, ValueRange vals) {
                 Value distribution = vals[0];
                 // if (i > 0)
                 auto comparisonPredicate = arith::CmpIPredicateAttr::get(
                     b.getContext(), arith::CmpIPredicate::sgt);
                 Value condition = b.create<arith::CmpIOp>(
                     loc, comparisonPredicate, i, cstZero);
                 Value iIndex = b.create<arith::IndexCastOp>(loc, indexTy, i);
                 // curr_cum = i > 0 ? prob[i] + prob[i-1] : prob[i]
                 ValueRange ind;
                 if (inputRank == 1) {
                   ind = ValueRange{iIndex};
                 } else {
                   ind = ValueRange{jIndex, iIndex};
                 }
                 Value currWeight = b.create<tensor::ExtractOp>(loc, self, ind);
                 Value currMass = b.create<arith::DivFOp>(loc, currWeight, sum);
                 Value currCum =
                     b.create<scf::IfOp>(
                          loc, condition,
                          [&](OpBuilder &b, Location loc) {
                            Value prevI =
                                b.create<arith::SubIOp>(loc, i, cstOne);
                            Value prevIndex = b.create<arith::IndexCastOp>(
                                loc, indexTy, prevI);
                            Value prevMass = b.create<tensor::ExtractOp>(
                                loc, distribution, ValueRange{prevIndex});
                            Value currSum = b.create<arith::AddFOp>(
                                loc, currMass, prevMass);
                            b.create<scf::YieldOp>(loc, ValueRange(currSum));
                          },
                          [&](OpBuilder &b, Location loc) {
                            b.create<scf::YieldOp>(loc, ValueRange{currMass});
                          })
                         .getResult(0);

                 Value updatedCdf = b.create<tensor::InsertOp>(
                     loc, currCum, distribution, ValueRange(iIndex));
                 b.create<scf::YieldOp>(loc, ValueRange(updatedCdf));
               })
              .getResult(0);

      // Get key, min and max used by RNG.
      Value key = b.create<TorchConversion::GetNextSeedOp>(loc);
      Value min = b.create<arith::ConstantOp>(loc, f64Ty,
                                              rewriter.getF64FloatAttr(0.0));
      Value max = b.create<arith::ConstantOp>(loc, f64Ty,
                                              rewriter.getF64FloatAttr(1.0));

      // iterate and sample class indices
      Value result = args[0];
      Value finalResult =
          rewriter
              .create<scf::ForOp>(
                  loc, cstZero, numSamples, cstOne, ValueRange{result},
                  [&](OpBuilder &b, Location loc, Value i, ValueRange args) {
                    // Sample random float
                    Value uniformSample =
                        randomUniformF64(b, loc, i, key, min, max);

                    // binary search in cdf to find our sample
                    Value left = b.create<arith::ConstantOp>(
                        loc, i64Ty, b.getI64IntegerAttr(0));
                    Value right = numCategories;

                    auto checkCondition = [&](OpBuilder &b, Location loc,
                                              ValueRange vals) {
                      Value left = vals[0];
                      Value right = vals[1];

                      // while (right > left)
                      auto comparisonPredicate = arith::CmpIPredicateAttr::get(
                          b.getContext(), arith::CmpIPredicate::sgt);
                      Value loopCondition = b.create<arith::CmpIOp>(
                          loc, comparisonPredicate, right, left);
                      b.create<scf::ConditionOp>(loc, loopCondition, vals);
                    };

                    ValueRange whileResults =
                        b.create<scf::WhileOp>(
                             loc, TypeRange{i64Ty, i64Ty},
                             ValueRange{left, right}, checkCondition,
                             [&](OpBuilder &b, Location loc, ValueRange vals) {
                               Value left = vals[0];
                               Value right = vals[1];

                               Value two = b.create<arith::ConstantOp>(
                                   loc, i64Ty, b.getI64IntegerAttr(2));
                               Value diff =
                                   b.create<arith::SubIOp>(loc, right, left);
                               Value diffMid =
                                   b.create<arith::DivSIOp>(loc, diff, two);
                               Value midPointer =
                                   b.create<arith::AddIOp>(loc, left, diffMid);
                               Type indexTy = b.getIndexType();
                               Value midIndex = b.create<arith::IndexCastOp>(
                                   loc, indexTy, midPointer);

                               // branch and update search indices
                               auto thenBlock = [&](OpBuilder &b,
                                                    Location loc) {
                                 // left += 1
                                 Value one = b.create<arith::ConstantOp>(
                                     loc, i64Ty, b.getI64IntegerAttr(1));
                                 Value newLeft =
                                     b.create<arith::AddIOp>(loc, left, one);

                                 b.create<scf::YieldOp>(
                                     loc, ValueRange{newLeft, right});
                               };
                               auto elseBlock = [&](OpBuilder &b,
                                                    Location loc) {
                                 // right = mid
                                 b.create<scf::YieldOp>(
                                     loc, ValueRange{left, midPointer});
                               };

                               Value cumProb = b.create<tensor::ExtractOp>(
                                   loc, cdf, ValueRange{midIndex});
                               auto cmpPredicate =
                                   arith::CmpFPredicateAttr::get(
                                       b.getContext(),
                                       arith::CmpFPredicate::OLT);
                               Value branchCondition = b.create<arith::CmpFOp>(
                                   loc, cmpPredicate, cumProb, uniformSample);
                               ValueRange branchResults =
                                   b.create<scf::IfOp>(loc, branchCondition,
                                                       thenBlock, elseBlock)
                                       .getResults();
                               Value newLeft = branchResults[0];
                               Value newRight = branchResults[1];

                               b.create<scf::YieldOp>(
                                   loc, ValueRange{newLeft, newRight});
                             })
                            .getResults();

                    // sample_idx = left_pointer
                    Value samplePointer = whileResults[0];
                    Value iIndex =
                        b.create<arith::IndexCastOp>(loc, indexTy, i);

                    Value prevResult = args[0];
                    Value newResult;
                    if (inputRank == 1) {
                      // result[i] = sample_idx
                      newResult = b.create<tensor::InsertOp>(
                          loc, samplePointer, prevResult, ValueRange{iIndex});
                    } else {
                      // result[j][i] = sample_idx
                      newResult = b.create<tensor::InsertOp>(
                          loc, samplePointer, prevResult,
                          ValueRange{jIndex, iIndex});
                    }

                    b.create<scf::YieldOp>(loc, ValueRange{newResult});
                  })
              .getResult(0);

      b.create<scf::YieldOp>(loc, ValueRange{finalResult});
    };

    Value finalResultTensor =
        rewriter
            .create<scf::ForOp>(loc, cstZero, numDistributions, cstOne,
                                ValueRange{resultTensor},
                                multinomialComputation)
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                finalResultTensor);

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
  target.addIllegalOp<AtenMultinomialOp>();
  patterns.add<ConvertAtenMultinomialOp>(typeConverter, context);
}
