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
      arith::ConstantOp::create(b, loc, b.getZeroAttr(b.getI64Type()));
  for (auto [index, stride] : llvm::zip(indicesIntValues, shapeIntValues)) {
    assert(isa<mlir::IntegerType>(index.getType()) &&
           isa<mlir::IntegerType>(stride.getType()) &&
           "Input arrays to `toLinearIndex` must only contain values of type "
           "`mlir::IntegerType`");
    Value mul = arith::MulIOp::create(b, loc, result, stride);
    result = arith::AddIOp::create(b, loc, mul, index);
  }
  return result;
}

// Squares64 Algorithm for generating 64-bit random numbers.
// See: https://arxiv.org/abs/2004.06278
static Value randomUniformUInt(OpBuilder &b, Location loc, Value ctr,
                               Value key) {
  auto mul = [&](Value lhs, Value rhs) -> Value {
    return arith::MulIOp::create(b, loc, lhs, rhs);
  };
  auto add = [&](Value lhs, Value rhs) -> Value {
    return arith::AddIOp::create(b, loc, lhs, rhs);
  };
  Value cst32 = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(32));
  auto shiftRight32 = [&](Value val) -> Value {
    return arith::ShRUIOp::create(b, loc, val, cst32);
  };
  auto swapLoHi = [&](Value val) -> Value {
    Value leftShift = arith::ShLIOp::create(b, loc, val, cst32);
    Value rightShift = shiftRight32(val);
    return arith::OrIOp::create(b, loc, leftShift, rightShift);
  };
  auto bitwiseXOr = [&](Value lhs, Value rhs) -> Value {
    return arith::XOrIOp::create(b, loc, lhs, rhs);
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
  Value epsilon = arith::ConstantOp::create(
      b, loc, b.getFloatAttr(b.getF64Type(), 5.4210108E-20));
  Value range = arith::SubFOp::create(b, loc, max, min);
  Value scale = arith::MulFOp::create(b, loc, range, epsilon);
  // res = cast(F64, tempN) * scale + min
  Value updateFloat =
      arith::UIToFPOp::create(b, loc, b.getF64Type(), randomVal);
  Value updateScaled = arith::MulFOp::create(b, loc, updateFloat, scale);
  Value uniformSample = arith::AddFOp::create(b, loc, updateScaled, min);

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
    Value key = TorchConversion::GetNextSeedOp::create(rewriter, loc);
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
    Value initTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(sizes), elemTy);
    Value uniformRes =
        linalg::GenericOp::create(
            rewriter, loc, initTensor.getType(), /*inputs=*/ValueRange{},
            /*outputs=*/initTensor, indexingMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              SmallVector<Value> indicesIntValues;
              for (int i = 0; i < resultRank; i++) {
                indicesIntValues.push_back(castIndexToInt64(
                    b, loc, linalg::IndexOp::create(b, loc, i)));
              }

              Value linearIndex =
                  toLinearIndex(b, loc, indicesIntValues, sizesIntValues);

              Value res = randomUniformF64(b, loc, linearIndex, key, min, max);
              Value truncRes = res;
              if (isa<BFloat16Type, Float16Type, Float32Type>(elemTy))
                truncRes = arith::TruncFOp::create(b, loc, elemTy, res);
              linalg::YieldOp::create(b, loc, truncRes);
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

    Value cstZero = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                              rewriter.getI64IntegerAttr(0));
    Value cstOne = arith::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(1));
    Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value numSamplesIndex =
        arith::IndexCastOp::create(rewriter, loc, indexTy, numSamples);

    Value numDistributions;
    Value numCategoriesIndex;
    ValueRange resultShape;
    if (inputRank == 1) {
      numDistributions = cstOne;
      numCategoriesIndex =
          tensor::DimOp::create(rewriter, loc, indexTy, self, zeroIndex);
      resultShape = ValueRange{numSamplesIndex};
    } else {
      Value numDistIndex =
          tensor::DimOp::create(rewriter, loc, indexTy, self, zeroIndex);
      numCategoriesIndex =
          tensor::DimOp::create(rewriter, loc, indexTy, self, oneIndex);
      numDistributions =
          arith::IndexCastOp::create(rewriter, loc, i64Ty, numDistIndex);
      resultShape = ValueRange{numDistIndex, numSamplesIndex};
    }

    Value numCategories =
        arith::IndexCastOp::create(rewriter, loc, i64Ty, numCategoriesIndex);
    Value resultTensor = tensor::EmptyOp::create(
        rewriter, loc, getAsOpFoldResult(resultShape), i64Ty);

    // sum weights for normalization
    torch_to_linalg::ReductionOpInfo opInfo;
    if (inputRank == 1)
      opInfo = {false, self, {0}};
    else
      opInfo = {false, self, {1}};

    Value initSum = arith::ConstantOp::create(rewriter, loc, f64Ty,
                                              rewriter.getF64FloatAttr(0.0));
    int64_t srcWidth = cast<mlir::FloatType>(elemTy).getWidth();
    if (srcWidth > 64)
      op->emitWarning("Op bitwidth will be truncated from " +
                      std::to_string(srcWidth) + " bits to 64 bits.");
    auto sumBody = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      Value input = payloadArgs[0];
      if (srcWidth < 64)
        input = arith::ExtFOp::create(b, loc, f64Ty, input);
      if (srcWidth > 64)
        input = arith::TruncFOp::create(b, loc, f64Ty, input);
      Value result = payloadArgs[1];
      Value nextSum = arith::AddFOp::create(b, loc, input, result);
      linalg::YieldOp::create(b, loc, nextSum);
    };
    Value sumWeights = torch_to_linalg::createReductionLinalgGeneric(
        rewriter, loc, opInfo, initSum, sumBody);

    // Get multinomial samples for each weight vector
    auto multinomialComputation = [&](OpBuilder &b, Location loc, Value j,
                                      ValueRange args) {
      Value jIndex = arith::IndexCastOp::create(b, loc, indexTy, j);

      Value sum;
      if (inputRank == 1) {
        sum = tensor::ExtractOp::create(b, loc, sumWeights, ValueRange{});
      } else {
        sum = tensor::ExtractOp::create(b, loc, sumWeights, ValueRange{jIndex});
      }

      // compute cdf in loop
      Value initCdf = tensor::EmptyOp::create(
          b, loc, getAsOpFoldResult(ValueRange{numCategoriesIndex}), f64Ty);
      Value cdf =
          scf::ForOp::create(
              b, loc, cstZero, numCategories, cstOne, ValueRange{initCdf},
              [&](OpBuilder &b, Location loc, Value i, ValueRange vals) {
                Value distribution = vals[0];
                // if (i > 0)
                auto comparisonPredicate = arith::CmpIPredicateAttr::get(
                    b.getContext(), arith::CmpIPredicate::sgt);
                Value condition = arith::CmpIOp::create(
                    b, loc, comparisonPredicate, i, cstZero);
                Value iIndex = arith::IndexCastOp::create(b, loc, indexTy, i);
                // curr_cum = i > 0 ? prob[i] + prob[i-1] : prob[i]
                ValueRange ind;
                if (inputRank == 1) {
                  ind = ValueRange{iIndex};
                } else {
                  ind = ValueRange{jIndex, iIndex};
                }
                Value currWeight = tensor::ExtractOp::create(b, loc, self, ind);
                if (srcWidth < 64)
                  currWeight = arith::ExtFOp::create(b, loc, f64Ty, currWeight);
                if (srcWidth > 64)
                  currWeight =
                      arith::TruncFOp::create(b, loc, f64Ty, currWeight);
                Value currMass = arith::DivFOp::create(b, loc, currWeight, sum);
                Value currCum =
                    scf::IfOp::create(
                        b, loc, condition,
                        [&](OpBuilder &b, Location loc) {
                          Value prevI =
                              arith::SubIOp::create(b, loc, i, cstOne);
                          Value prevIndex = arith::IndexCastOp::create(
                              b, loc, indexTy, prevI);
                          Value prevMass = tensor::ExtractOp::create(
                              b, loc, distribution, ValueRange{prevIndex});
                          Value currSum =
                              arith::AddFOp::create(b, loc, currMass, prevMass);
                          scf::YieldOp::create(b, loc, ValueRange(currSum));
                        },
                        [&](OpBuilder &b, Location loc) {
                          scf::YieldOp::create(b, loc, ValueRange{currMass});
                        })
                        .getResult(0);

                Value updatedCdf = tensor::InsertOp::create(
                    b, loc, currCum, distribution, ValueRange(iIndex));
                scf::YieldOp::create(b, loc, ValueRange(updatedCdf));
              })
              .getResult(0);

      /*
       * Above we've computed the CDF for the unnormalized distribution given to
       * us by the user. In order to actually sample from this distribution we
       * do the following below: 1) Sample a random floating point value, r in
       * [0,1), from a uniform distribution. 2) Perform a binary search in the
       * cdf to find the first bin in the CDF where cdf[i] < r. This guarantees
       * a random sample from the provided distribution with the appropriate
       * probabilities.
       *
       * This logic is pulled straight from PyTorch's Multinomial Kernel:
       * https://github.com/pytorch/pytorch/blob/e4623de4cf6097ff399aa9eb0cef44b44ca76da4/aten/src/ATen/native/cpu/MultinomialKernel.cpp#L23
       * */

      // Get key, min and max used by RNG.
      Value key = TorchConversion::GetNextSeedOp::create(b, loc);
      Value min = arith::ConstantOp::create(b, loc, f64Ty,
                                            rewriter.getF64FloatAttr(0.0));
      Value max = arith::ConstantOp::create(b, loc, f64Ty,
                                            rewriter.getF64FloatAttr(1.0));

      // iterate and sample class indices
      Value result = args[0];
      Value finalResult =
          scf::ForOp::create(
              rewriter, loc, cstZero, numSamples, cstOne, ValueRange{result},
              [&](OpBuilder &b, Location loc, Value i, ValueRange args) {
                // Sample random float
                Value uniformSample =
                    randomUniformF64(b, loc, i, key, min, max);

                // binary search in cdf to find our sample
                Value left = arith::ConstantOp::create(b, loc, i64Ty,
                                                       b.getI64IntegerAttr(0));
                Value right = numCategories;

                auto checkCondition = [&](OpBuilder &b, Location loc,
                                          ValueRange vals) {
                  Value left = vals[0];
                  Value right = vals[1];

                  // while (right > left)
                  auto comparisonPredicate = arith::CmpIPredicateAttr::get(
                      b.getContext(), arith::CmpIPredicate::sgt);
                  Value loopCondition = arith::CmpIOp::create(
                      b, loc, comparisonPredicate, right, left);
                  scf::ConditionOp::create(b, loc, loopCondition, vals);
                };

                ValueRange whileResults =
                    scf::WhileOp::create(
                        b, loc, TypeRange{i64Ty, i64Ty},
                        ValueRange{left, right}, checkCondition,
                        [&](OpBuilder &b, Location loc, ValueRange vals) {
                          Value left = vals[0];
                          Value right = vals[1];

                          Value two = arith::ConstantOp::create(
                              b, loc, i64Ty, b.getI64IntegerAttr(2));
                          Value diff =
                              arith::SubIOp::create(b, loc, right, left);
                          Value diffMid =
                              arith::DivSIOp::create(b, loc, diff, two);
                          Value midPointer =
                              arith::AddIOp::create(b, loc, left, diffMid);
                          Type indexTy = b.getIndexType();
                          Value midIndex = arith::IndexCastOp::create(
                              b, loc, indexTy, midPointer);

                          // branch and update search indices
                          auto thenBlock = [&](OpBuilder &b, Location loc) {
                            // left = mid + 1
                            Value newLeft = arith::AddIOp::create(
                                b, loc, midPointer, cstOne);

                            scf::YieldOp::create(b, loc,
                                                 ValueRange{newLeft, right});
                          };
                          auto elseBlock = [&](OpBuilder &b, Location loc) {
                            // right = mid
                            scf::YieldOp::create(b, loc,
                                                 ValueRange{left, midPointer});
                          };

                          Value cumProb = tensor::ExtractOp::create(
                              b, loc, cdf, ValueRange{midIndex});
                          auto cmpPredicate = arith::CmpFPredicateAttr::get(
                              b.getContext(), arith::CmpFPredicate::OLT);
                          Value branchCondition = arith::CmpFOp::create(
                              b, loc, cmpPredicate, cumProb, uniformSample);
                          ValueRange branchResults =
                              scf::IfOp::create(b, loc, branchCondition,
                                                thenBlock, elseBlock)
                                  .getResults();
                          Value newLeft = branchResults[0];
                          Value newRight = branchResults[1];

                          scf::YieldOp::create(b, loc,
                                               ValueRange{newLeft, newRight});
                        })
                        .getResults();

                // sample_idx = left_pointer
                Value samplePointer = whileResults[0];
                Value iIndex = arith::IndexCastOp::create(b, loc, indexTy, i);

                Value prevResult = args[0];
                Value newResult;
                if (inputRank == 1) {
                  // result[i] = sample_idx
                  newResult = tensor::InsertOp::create(
                      b, loc, samplePointer, prevResult, ValueRange{iIndex});
                } else {
                  // result[j][i] = sample_idx
                  newResult = tensor::InsertOp::create(
                      b, loc, samplePointer, prevResult,
                      ValueRange{jIndex, iIndex});
                }

                scf::YieldOp::create(b, loc, ValueRange{newResult});
              })
              .getResult(0);

      scf::YieldOp::create(b, loc, ValueRange{finalResult});
    };

    Value finalResultTensor =
        scf::ForOp::create(rewriter, loc, cstZero, numDistributions, cstOne,
                           ValueRange{resultTensor}, multinomialComputation)
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
