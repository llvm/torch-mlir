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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#aten ops), which is in the 100s.
//
// Most of these patterns consist of:
// 1. Checking that the operand/result types and other static properties are
//    good-enough to create a valid linalg op (such as operands being of
//    ranks/dtypes acceptable to the linalg op).
// 2. Creating dynamic error guards, usually checking a predicate on the
//    compatibility of operand shapes.
// 3. Creating init tensors for the computation op. Usually this involves
//    reifying IR for a shape transfer function based on the operand shapes.
// 4. Creating a named linalg op to replace the original op.
//
// TODO: Use linalg OpDSL to autogenerate at least 1)/2)/3) such
// that these patterns become mostly mechanical associations of
// "aten.foo -> linalg.foo".

static LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                                 PatternRewriter &rewriter) {
  // Check the value tensor is ranked as expected by Linalg.
  // TODO: Remove this check but use a separate verification pass to verify the
  // invariants expected by later passes.
  auto isValidLinalgType = [](Type type) {
    auto tensor = type.dyn_cast<ValueTensorType>();
    return !tensor ||
           tensor.toBuiltinTensor().dyn_cast_or_null<RankedTensorType>();
  };

  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

static LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op,
                                  Value v) {
  Type type = v.getType();
  if (type.isa<OptionalType>() || type.isa<Torch::NoneType>() ||
      type.isa<mlir::NoneType>())
    return rewriter.notifyMatchFailure(op, "unimplemented None type arg");
  return success();
}

// Hack to deal with the Torch list type arguments which is not supported end
// to end. Constant values can be be extracted directly and non constant
// list values are not supported.
// TODO: loose this constraint when properly support list type
static bool isConstantIntListMatching(Value value,
                                      SmallVectorImpl<int64_t> &expects) {
  SmallVector<int64_t> intValues;
  if (!matchPattern(value, m_TorchConstantIntList(intValues)))
    return false;

  if (intValues.size() != expects.size())
    return false;

  for (auto it : llvm::zip(intValues, expects)) {
    if (std::get<0>(it) != std::get<1>(it))
      return false;
  }
  return true;
}

static Value castIntToIndex(OpBuilder &b, Location loc, Value v) {
  assert(v.getType().isa<IntegerType>() && "must be called with integer type");
  return b.create<IndexCastOp>(loc, b.getIndexType(), v);
}

static Value castIndexToInt(OpBuilder &b, Location loc, Value idx) {
  assert(idx.getType().isa<IndexType>() && "must be called with integer type");
  return b.create<IndexCastOp>(loc, b.getI64Type(), idx);
}

static Value getDimOp(OpBuilder &b, Location loc, Value v, int dimension) {
  return b.create<tensor::DimOp>(loc, v, dimension);
}

static void checkDimEqualHelper(OpBuilder &b, Location loc, Value lhsDim,
                                Value rhsDim) {
  Type lhsType = lhsDim.getType();
  Type rhsType = rhsDim.getType();
  auto checkIntOrIndex = [](Type type) {
    assert(type.isa<IntegerType>() ||
           type.isa<IndexType>() && "must be either integer or index type");
  };
  checkIntOrIndex(lhsType);
  checkIntOrIndex(rhsType);
  Value lhsDimInt = lhsType.isIndex() ? castIndexToInt(b, loc, lhsDim) : lhsDim;
  Value rhsDimInt = rhsType.isIndex() ? castIndexToInt(b, loc, rhsDim) : rhsDim;
  Value contractingDimEqual =
      b.create<CmpIOp>(loc, CmpIPredicate::eq, lhsDimInt, rhsDimInt);
  b.create<AssertOp>(loc, contractingDimEqual,
                     b.getStringAttr("mismatching contracting dimension"));
}

static SmallVector<Value> getTensorSizesUntilDim(OpBuilder &b, Location loc,
                                                 Value tensor, int dim) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  assert(dim < type.getRank() &&
         "The given dim must be smaller than tensor rank");
  (void)type;
  SmallVector<Value> sizes;
  for (int i = 0; i <= dim; i++)
    sizes.push_back(getDimOp(b, loc, tensor, i));
  return sizes;
}

static SmallVector<Value> getTensorSizes(OpBuilder &b, Location loc,
                                         Value tensor) {
  RankedTensorType type = tensor.getType().cast<RankedTensorType>();
  return getTensorSizesUntilDim(b, loc, tensor, type.getRank() - 1);
}

static Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                                  Type elemTy) {
  Value initTensor = b.create<linalg::InitTensorOp>(loc, sizes, elemTy);
  RankedTensorType type = initTensor.getType().cast<RankedTensorType>();
  Value c0 = b.create<ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
}

// Helper function to caculate the output tensor dims for convolution-like ops.
// Along each dim:
// dim_out =
//  floor((dim_in + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1
static Value getOutputDimForConvOps(OpBuilder &b, Location loc, Value in,
                                    Value paddingInt, Value dilationInt,
                                    Value kernelSizeInt, Value strideInt) {
  Value c1 = b.create<ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value c2 = b.create<ConstantOp>(loc, b.getI64IntegerAttr(2));

  Value doublePadding = b.create<MulIOp>(loc, paddingInt, c2);
  // in + 2 * padding
  Value inAddDoublePadding =
      b.create<AddIOp>(loc, castIndexToInt(b, loc, in), doublePadding);

  // dilation * (kernelSize - 1)
  Value kernelSizeSub1 = b.create<SubIOp>(loc, kernelSizeInt, c1);
  Value dilationTimesKernelSize =
      b.create<MulIOp>(loc, dilationInt, kernelSizeSub1);

  Value temp =
      b.create<SubIOp>(loc, inAddDoublePadding, dilationTimesKernelSize);
  Value dividend = b.create<SubIOp>(loc, temp, c1);
  Value division = b.create<SignedFloorDivIOp>(loc, dividend, strideInt);
  Value out = b.create<AddIOp>(loc, division, c1);
  return castIntToIndex(b, loc, out);
}

static SmallVector<Value>
getAsConstantIntValues(OpBuilder &b, Location loc,
                       SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<ConstantOp>(loc, b.getIntegerAttr(b.getI64Type(), val));
  }));
}

static SmallVector<Value>
getAsConstantIndexValues(OpBuilder &b, Location loc,
                         SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<ConstantOp>(loc, b.getIndexAttr(val));
  }));
}

static SmallVector<OpFoldResult>
getAsOpFoldResult(OpBuilder &b, Location loc, SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(
      ints, [&](int64_t val) -> OpFoldResult { return b.getIndexAttr(val); }));
}

// This is a temporary solution to deal with types that are not fully supported
// like list, dict. For those container tyes, this helper can be used to
// convert their elements to valid target type.
// TODO: remove this when list gets full support.
static SmallVector<Value> getTypeConvertedValues(OpBuilder &b, Location loc,
                                                 TypeConverter *converter,
                                                 SmallVectorImpl<Value> &vs) {
  return llvm::to_vector<4>(llvm::map_range(vs, [&](Value v) {
    return converter->materializeTargetConversion(
        b, loc, converter->convertType(v.getType()), v);
  }));
}

// Helper function to get the padding tensor given the padding int values.
// It's assumed that the padding on the low end and high end are the same.
static Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                             SmallVectorImpl<int64_t> &paddingInts) {
  assert(input.getType().isa<RankedTensorType>() &&
         "input must be RankedTensorType");
  Location loc = op->getLoc();
  Value c0 = b.create<ConstantOp>(
      loc,
      b.getZeroAttr(input.getType().cast<RankedTensorType>().getElementType()));
  SmallVector<OpFoldResult> paddings = getAsOpFoldResult(b, loc, paddingInts);
  Type ranked4DTensorType = linalg::PadTensorOp::inferResultType(
      input.getType().cast<RankedTensorType>(), paddingInts, paddingInts);
  Value paddedInput = linalg::PadTensorOp::createPadScalarOp(
      ranked4DTensorType, input, c0, /*low=*/paddings, /*high=*/paddings,
      /*packing=*/false, loc, b);
  return paddedInput;
}

static bool getListConstructElements(Value v, SmallVectorImpl<Value> &elems) {
  auto listConstruct = v.getDefiningOp<PrimListConstructOp>();
  if (!listConstruct)
    return false;
  elems = llvm::to_vector<4>(listConstruct.elements());
  return true;
}

namespace {
class ConvertAtenAdaptiveAvgPool2dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    AtenAdaptiveAvgPool2dOp::Adaptor adaptor(operands);
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
    Value c0 =
        rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
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
                                Value result =
                                    rewriter.create<AddFOp>(loc, sum, input);
                                b.create<linalg::YieldOp>(loc, result);
                              })
                          .getResult(0);

    // Calculate H*W so that avg can be got from sum / (H*W)
    Value H = getDimOp(rewriter, loc, input, 2);
    Value W = getDimOp(rewriter, loc, input, 3);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<IndexCastOp>(loc, IntegerType::get(context, 64),
                                          v);
    };
    Value HtimesW =
        rewriter.create<MulIOp>(loc, castIndexToInt(H), castIndexToInt(W));
    Value HtimesWf = rewriter.create<SIToFPOp>(loc, elementType, HtimesW);

    Value c1Index = rewriter.create<mlir::ConstantIndexOp>(loc, /*value=*/1);
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
                  Value avg = b.create<DivFOp>(loc, args[0], HtimesWf);
                  b.create<linalg::YieldOp>(loc, avg);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, avgPool2d);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenConv2dOp : public OpConversionPattern<AtenConv2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConv2dOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    AtenConv2dOp::Adaptor adaptor(operands);
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value groups = adaptor.groups();

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<IndexCastOp>(loc, intType, v);
    };

    Value N = getDimOp(rewriter, loc, input, 0);
    Value Hin = getDimOp(rewriter, loc, input, 2);
    Value Win = getDimOp(rewriter, loc, input, 3);
    Value F = getDimOp(rewriter, loc, weight, 0);
    Value weightH = getDimOp(rewriter, loc, weight, 2);
    Value weightW = getDimOp(rewriter, loc, weight, 3);

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t> paddingInts;
    if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    if (!op.bias().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "only support None bias");

    Value c1 = rewriter.create<ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value groupEqual1 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, groups, c1);
    rewriter.create<AssertOp>(loc, groupEqual1,
                              rewriter.getStringAttr("expect groups to be 1"));

    // Pad the input tensor according to padding.
    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput =
        getPaddedTensor(op, rewriter, input, paddingIncludingNC);

    SmallVector<Value> paddingIntValues =
        getAsConstantIntValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    Value Hout = getOutputDimForConvOps(
        rewriter, loc, Hin, paddingIntValues[0], dilationIntValues[0],
        castIndexToInt(weightH), strideIntValues[0]);
    Value Wout = getOutputDimForConvOps(
        rewriter, loc, Win, paddingIntValues[1], dilationIntValues[1],
        castIndexToInt(weightW), strideIntValues[1]);

    Value c0float = rewriter.create<ConstantOp>(
        loc,
        FloatAttr::get(
            input.getType().cast<RankedTensorType>().getElementType(), 0.0));
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, F, Hout, Wout}, elementType);
    Value initTensor0 =
        rewriter.create<linalg::FillOp>(loc, c0float, initTensor).getResult(0);

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
    Value conv2d =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, initTensor0.getType(), ValueRange{paddedInput, weight},
                initTensor0, stridesAttr, dilationAttr)
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv2d);
    return success();
  }
};
} // namespace

// Normalization formula:
//   ((input - mean) / sqrt(var + eps)) * weight + bias
static Value createLinalgPayloadCalculationForNormOps(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean, Value var,
    Value eps, Value weight, Value bias) {
  Value inputSubMean = b.create<SubFOp>(loc, input, mean);
  // The eps is always f64.
  Value truncatedEps = b.create<FPTruncOp>(loc, elemTy, eps);
  Value varPlusEps = b.create<AddFOp>(loc, var, truncatedEps);
  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
  Value temp = b.create<MulFOp>(loc, inputSubMean, rSTD);
  Value timesWeight = b.create<MulFOp>(loc, temp, weight);
  Value plusBias = b.create<AddFOp>(loc, timesWeight, bias);
  return plusBias;
}

namespace {
class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AtenBatchNormOp::Adaptor adaptor(operands);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    Value runningMean = adaptor.running_mean();
    Value runningVar = adaptor.running_var();
    Value training = adaptor.training();
    Value eps = adaptor.eps();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias.
    if (failed(checkNotNone(rewriter, op, weight)) ||
        failed(checkNotNone(rewriter, op, bias)) ||
        failed(checkNotNone(rewriter, op, runningMean)) ||
        failed(checkNotNone(rewriter, op, runningVar)))
      return failure();

    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    auto runningMeanType = runningMean.getType().cast<RankedTensorType>();
    auto runningVarType = runningVar.getType().cast<RankedTensorType>();

    auto inputRank = inputType.getRank();
    if (inputRank <= 2)
      return rewriter.notifyMatchFailure(
          op, "input should have rank larger than 2");

    if (weightType.getRank() != 1 || biasType.getRank() != 1 ||
        runningMeanType.getRank() != 1 || runningVarType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expect weight, bias, running_mean and running_var to be rank 1");
    }

    // TODO: Add support for training.
    auto constFalse = rewriter.create<ConstantOp>(
        loc, IntegerAttr::get(IntegerType::get(context, 1), 0));
    auto trainingFalse =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, training, constFalse);
    rewriter.create<AssertOp>(
        loc, trainingFalse,
        rewriter.getStringAttr("training is not supported for now"));

    // num_features – C from an expected input of size (N,C,D,H,W ...)
    Value numFeatures = rewriter.create<tensor::DimOp>(loc, input, 1);
    auto contractingDim0EqualsNumFeatures = [&](Value v) {
      auto dim0 = rewriter.create<tensor::DimOp>(loc, v, 0);
      auto dim0Equal =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, numFeatures, dim0);
      rewriter.create<AssertOp>(
          loc, dim0Equal,
          rewriter.getStringAttr(
              "expect the size of dim 0 equal to the number of features"));
    };
    contractingDim0EqualsNumFeatures(weight);
    contractingDim0EqualsNumFeatures(bias);
    contractingDim0EqualsNumFeatures(runningMean);
    contractingDim0EqualsNumFeatures(runningVar);

    auto indexingMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, rewriter.getAffineDimExpr(1), context);
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputRank), // input
        indexingMap,                                // weight
        indexingMap,                                // bias
        indexingMap,                                // runningMean
        indexingMap,                                // runningVar
        rewriter.getMultiDimIdentityMap(inputRank), // output
    };
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
    Value batchNorm =
        rewriter
            .create<linalg::GenericOp>(
                loc, input.getType(),
                ValueRange{input, weight, bias, runningMean, runningVar}, input,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], weight = args[1], bias = args[2],
                        mean = args[3], var = args[4];
                  Value result = createLinalgPayloadCalculationForNormOps(
                      b, loc, var.getType(), input, mean, var, eps, weight,
                      bias);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, batchNorm);
    return success();
  }
};
} // namespace

// For layernorm, the mean and standard-deviation are calculated separately over
// the last certain number dimensions which have to be of the shape specified by
// normalized_shape.
//
// The shapes of different parts are as the following:
// +-------------------+--------------------+
// |  meanAndVarShape  |   normalizedShape  |
// +-------------------+---------------------
// <------------+ inputShape +-------------->

// There are the following steps:
// Step 1. Check if all the arguments meet the requirements.
// Step 2. Common parts to be used for getting mean and var.
//         This includes elements count, affineMap and iteratorTypes.
// Step 3. Get mean.
// Step 4. Get var.
// Step 5. Get layernorm.
namespace {
class ConvertAtenLayerNormOp : public OpConversionPattern<AtenLayerNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLayerNormOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AtenLayerNormOp::Adaptor adaptor(operands);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    Value eps = adaptor.eps();
    Value normalizedShape = op.normalized_shape();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias.
    if (failed(checkNotNone(rewriter, op, weight)) ||
        failed(checkNotNone(rewriter, op, bias)))
      return failure();

    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    Type elemTy = inputType.getElementType();

    // Step 1. Check if all the arguments meet the requirements.
    SmallVector<Value> normalizedShapeSizesTorchInt;
    if (!getListConstructElements(normalizedShape,
                                  normalizedShapeSizesTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "Unimplemented normalized_shape not"
                                         "constructed from ListConstruct");
    }
    SmallVector<Value> normalizedShapeSizesInt = getTypeConvertedValues(
        rewriter, loc, getTypeConverter(), normalizedShapeSizesTorchInt);
    int64_t normalizedShapeRank = normalizedShapeSizesInt.size();
    if (weightType.getRank() != normalizedShapeRank ||
        biasType.getRank() != normalizedShapeRank ||
        inputRank < normalizedShapeRank || normalizedShapeRank < 1)
      return rewriter.notifyMatchFailure(op, "Input or weight or bias shape or"
                                             "normalized shape not compatible");

    // Check all the dimensions match the normalized_shape
    int64_t meanAndVarShapeRank = inputRank - normalizedShapeSizesInt.size();
    for (auto en : enumerate((normalizedShapeSizesInt))) {
      auto index = en.index();
      auto inputDim =
          getDimOp(rewriter, loc, input, index + meanAndVarShapeRank);
      auto weightDim = getDimOp(rewriter, loc, weight, index);
      auto biasDim = getDimOp(rewriter, loc, bias, index);

      auto expectedSize = en.value();
      checkDimEqualHelper(rewriter, loc, inputDim, expectedSize);
      checkDimEqualHelper(rewriter, loc, weightDim, expectedSize);
      checkDimEqualHelper(rewriter, loc, biasDim, expectedSize);
    }

    // Get iterator types for input shape.
    SmallVector<StringRef> normalizedShapeIteratorTypes(
        normalizedShapeRank, getReductionIteratorTypeName());
    SmallVector<StringRef> meanAndVarIterationTypes(
        meanAndVarShapeRank, getParallelIteratorTypeName());
    SmallVector<StringRef> inputShapeIteratorTypes = meanAndVarIterationTypes;
    inputShapeIteratorTypes.append(normalizedShapeIteratorTypes);

    // Step 2. Common parts to be used for getting mean and var.

    // Get sizes and affineMaps needed for mean and var.
    AffineMap inputShapeAffineMap = rewriter.getMultiDimIdentityMap(inputRank);
    SmallVector<AffineExpr> meanAndVarShapeExprs;
    for (int i = 0; i < meanAndVarShapeRank; i++)
      meanAndVarShapeExprs.push_back(mlir::getAffineDimExpr(i, context));
    auto meanAndVarShapeAffineMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, meanAndVarShapeExprs, context);
    SmallVector<Value> meanAndVarShapeSizes =
        getTensorSizesUntilDim(rewriter, loc, input, meanAndVarShapeRank - 1);

    // Get number of elements to be used for calculating mean and var.
    Value elemCnts = normalizedShapeSizesInt[0];
    for (int i = 1; i < normalizedShapeRank; i++) {
      elemCnts =
          rewriter.create<MulIOp>(loc, elemCnts, normalizedShapeSizesInt[i]);
    }
    Value elemCntsFloat = rewriter.create<SIToFPOp>(loc, elemTy, elemCnts);

    // Helper to calculate mean and var.
    auto genMeanOrVarCalculation = [&](Value sumOrSquareSum) {
      SmallVector<AffineMap> indexingMaps(
          2, rewriter.getMultiDimIdentityMap(meanAndVarShapeRank));
      Value initShapeTensor = rewriter.create<linalg::InitTensorOp>(
          loc, meanAndVarShapeSizes, elemTy);
      return rewriter
          .create<linalg::GenericOp>(
              loc, initShapeTensor.getType(), sumOrSquareSum, initShapeTensor,
              /*indexingMaps=*/indexingMaps,
              /*iteratorTypes=*/meanAndVarIterationTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value sumOrSqureSum = args[0];
                Value result =
                    b.create<DivFOp>(loc, sumOrSqureSum, elemCntsFloat);
                b.create<linalg::YieldOp>(loc, result);
              })
          .getResult(0);
    };

    // Step 3. Get mean.

    // Get sum to be used for calculating mean.
    SmallVector<AffineMap, 2> sumIndexingMaps = {
        inputShapeAffineMap,      // input
        meanAndVarShapeAffineMap, // output
    };
    auto initSumTensor =
        createZeroInitTensor(rewriter, loc, meanAndVarShapeSizes, elemTy);
    Value sum = rewriter
                    .create<linalg::GenericOp>(
                        loc, initSumTensor.getType(), input, initSumTensor,
                        /*indexingMaps=*/sumIndexingMaps,
                        /*iteratorTypes=*/inputShapeIteratorTypes,
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value input = args[0], sum = args[1];
                          Value result =
                              rewriter.create<AddFOp>(loc, sum, input);
                          b.create<linalg::YieldOp>(loc, result);
                        })
                    .getResult(0);
    Value mean = genMeanOrVarCalculation(sum);

    // Step 4. Get var.

    // Calculate squareSum for the layer.
    SmallVector<AffineMap> squareSumIndexingMaps{
        inputShapeAffineMap,
        meanAndVarShapeAffineMap,
        meanAndVarShapeAffineMap,
    };
    auto initSquareSumTensor =
        createZeroInitTensor(rewriter, loc, meanAndVarShapeSizes, elemTy);
    Value squareSum =
        rewriter
            .create<linalg::GenericOp>(
                loc, initSquareSumTensor.getType(), ValueRange{input, mean},
                initSquareSumTensor,
                /*indexingMaps=*/squareSumIndexingMaps,
                /*iteratorTypes=*/inputShapeIteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], mean = args[1], squareSum = args[2];
                  Value sub = rewriter.create<SubFOp>(loc, input, mean);
                  Value square = rewriter.create<MulFOp>(loc, sub, sub);
                  Value result =
                      rewriter.create<AddFOp>(loc, squareSum, square);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    Value var = genMeanOrVarCalculation(squareSum);

    // Step 5. Get layernorm.

    // Get affineMap for normalized shape.
    SmallVector<AffineExpr> normalizedShapeExprs;
    for (int i = meanAndVarShapeRank; i < inputRank; i++)
      normalizedShapeExprs.push_back(mlir::getAffineDimExpr(i, context));
    auto normalizedShapeAffineMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, normalizedShapeExprs, context);

    auto inputSizes = getTensorSizes(rewriter, loc, input);
    Value initLayerNormTensor =
        rewriter.create<linalg::InitTensorOp>(loc, inputSizes, elemTy);
    SmallVector<AffineMap> indexingMaps(1, inputShapeAffineMap);
    indexingMaps.resize(3, meanAndVarShapeAffineMap);
    indexingMaps.resize(5, normalizedShapeAffineMap);
    indexingMaps.push_back(inputShapeAffineMap);
    SmallVector<StringRef> layerNormIterationTypes(
        inputRank, getParallelIteratorTypeName());
    Value layerNorm =
        rewriter
            .create<linalg::GenericOp>(
                loc, initLayerNormTensor.getType(),
                ValueRange{input, mean, var, weight, bias}, initLayerNormTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/layerNormIterationTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], mean = args[1], var = args[2],
                        weight = args[3], bias = args[4];
                  Value result = createLinalgPayloadCalculationForNormOps(
                      b, loc, elemTy, input, mean, var, eps, weight, bias);
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, layerNorm);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMmOp : public OpConversionPattern<AtenMmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = operands[0];
    Value rhs = operands[1];

    // A user can write an errorneous program where `aten.mm` is in fact called
    // with operands of invalid rank or dtype. We cannot convert to linalg in
    // this case or we will get a verifier error, which corresponds to breaking
    // of *internal* compiler invariants, and for a user manifests as a compiler
    // crash in the worst case (such as we try to canonicalize/fold/print the
    // invalid op before the verifier gets to see it -- also release builds of a
    // mature copmiler usually have the verifier turned off for compile time
    // reasons).
    //
    // The compiler cannot crash even if the user wrote an erroneous program!
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    if (lhs.getType().cast<RankedTensorType>().getRank() != 2 ||
        rhs.getType().cast<RankedTensorType>().getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    Value lhsDim0 = rewriter.create<tensor::DimOp>(loc, lhs, 0);
    Value lhsDim1 = rewriter.create<tensor::DimOp>(loc, lhs, 1);
    Value rhsDim0 = rewriter.create<tensor::DimOp>(loc, rhs, 0);
    Value rhsDim1 = rewriter.create<tensor::DimOp>(loc, rhs, 1);
    Value contractingDimEqual =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, lhsDim1, rhsDim0);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{lhsDim0, rhsDim1}, elementType);
    Value c0 =
        rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                                 ValueRange{lhs, rhs}, zeroFill)
                       .getResult(0);
    // When constructed with just dynamic sizes, InitTensorOp will have a result
    // type which has all `?`'s for dimensions, which might not be the result
    // type of `op`. The constraints on later linalg ops means that the result
    // of the MatmulOp will have this type too. So cast it to the desired type
    // so that in the end we have the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenBmmOp : public OpConversionPattern<AtenBmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBmmOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = operands[0];
    Value rhs = operands[1];
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    if (lhsType.getRank() != 3 || rhsType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.bmm to be rank 3");
    }
    if (!lhsType.getElementType().isa<mlir::FloatType>() ||
        lhsType.getElementType() != rhsType.getElementType())
      return op.emitError(
          "unimplemented: non floating point operands or operands of "
          "different types");

    Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
    Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
    Value lhsDim2 = getDimOp(rewriter, loc, lhs, 2);
    Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
    Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
    Value rhsDim2 = getDimOp(rewriter, loc, rhs, 2);

    // Check the batch numbers are equal.
    checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

    // Check the matrixs shapes are valid for mulplication.
    checkDimEqualHelper(rewriter, loc, lhsDim2, rhsDim1);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor0 = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, lhsDim1, rhsDim2}, elementType);

    Value bmm =
        rewriter
            .create<linalg::BatchMatmulOp>(loc, initTensor0.getType(),
                                           ValueRange{lhs, rhs}, initTensor0)
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, bmm);
    return success();
  }
};
} // namespace

namespace {
// See comments at in convertMmOp and the heading for this section for general
// considerations. This function needs to be auto-generated.
class ConvertAtenLinearOp : public OpConversionPattern<AtenLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLinearOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    AtenLinearOp::Adaptor adaptor(operands);
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    // TODO: Handle the case of bias being None (bias is optional).
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    // Only handle the case of rank 2 `input` for now.
    // TODO: Insert the appropriate reshape to collapse any leading dimensions.
    if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
        biasType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op,
          "expected both input and weight to be rank 2 and bias to be rank 1");
    }
    // TODO: Handle type promotion. What are ATen's promotion rules?
    if (inputType.getElementType() != weightType.getElementType() ||
        inputType.getElementType() != biasType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");
    }

    // TODO: We can handle a static size 1 here at some complexity cost, but the
    // dynamic case is not representable in linalg. We don't handle either for
    // now. Biases are generally statically shaped for most models (since for
    // inference they are constants, and for training they don't change shape
    // typically), so this is not too constraining.
    auto biasSize = bias.getType().cast<RankedTensorType>().getShape()[0];
    if (biasSize == 1 || biasSize == ShapedType::kDynamicSize)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size-1 broadcasting for aten::LinearOp");

    Value inputDim0 = getDimOp(rewriter, loc, input, 0);
    Value inputDim1 = getDimOp(rewriter, loc, input, 1);
    Value weightDim0 = getDimOp(rewriter, loc, weight, 0);
    Value weightDim1 = getDimOp(rewriter, loc, weight, 1);
    Value biasDim0 = getDimOp(rewriter, loc, bias, 0);
    Value contractingDimEqual =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, inputDim1, weightDim1);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for aten.linear"));
    // Here we take advantage of ruling out the size-1 case above.
    // In the static-size-1 case, we will not emit this check at all.
    Value biasSizeCorrect =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, weightDim0, biasDim0);
    rewriter.create<AssertOp>(
        loc, biasSizeCorrect,
        rewriter.getStringAttr("mismatching bias size for aten.linear"));

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{inputDim0, weightDim0}, inputType.getElementType());
    SmallVector<AffineMap> broadcastIndexingMaps = {
        AffineMap::get(
            /*dimCount=*/2, /*symbolCount=*/0, rewriter.getAffineDimExpr(1)),
        rewriter.getMultiDimIdentityMap(2)};
    SmallVector<StringRef> iteratorTypes(2, "parallel");
    Value broadcasted =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), bias, initTensor,
                /*indexingMaps=*/broadcastIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    // We need a matmul with dimension ordering (N, K) * (M, K), so transpose
    // the weights to fit into linalg::MatmulOp which is (N, K) * (K, M).
    // TODO: This whole aten.linear lowering should eventually be generated from
    // a single linalg ODS generator statement. Both the bias and matmul part.
    SmallVector<AffineMap> transposeIndexingMaps = {
        AffineMap::get(
            /*dimCount=*/2, /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(1), rewriter.getAffineDimExpr(0)},
            context),
        rewriter.getMultiDimIdentityMap(2)};
    Value transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{weightDim1, weightDim0}, weightType.getElementType());
    Value transposedWeights =
        rewriter
            .create<linalg::GenericOp>(
                loc, transposedWeightInitTensor.getType(), weight,
                transposedWeightInitTensor,
                /*indexingMaps=*/transposeIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(
                           loc, broadcasted.getType(),
                           ValueRange{input, transposedWeights}, broadcasted)
                       .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
    return success();
  }
};
} // namespace

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, ValueRange payloadArgs, Operation *op,
    ArrayRef<Value> operands) {
  if (isa<AtenTanhOp>(op))
    return b.create<math::TanhOp>(loc, payloadArgs[0]);
  if (isa<AtenSigmoidOp>(op)) {
    Type elementType = payloadArgs[0].getType();
    auto one = b.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
    auto negate = b.create<NegFOp>(loc, payloadArgs[0]);
    auto exp = b.create<math::ExpOp>(loc, negate);
    auto added = b.create<AddFOp>(loc, exp, one);
    return b.create<DivFOp>(loc, one, added);
  }
  if (auto relu = dyn_cast<AtenReluOp>(op)) {
    if (!relu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      relu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        b.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    Value pred =
        b.create<CmpFOp>(loc, CmpFPredicate::UGT, payloadArgs[0], constZero);
    return b.create<SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  if (auto add = dyn_cast<AtenAddTensorOp>(op)) {
    AtenAddTensorOp::Adaptor adaptor(operands);
    if (add.alpha().getType().isa<Torch::FloatType>()) {
      add.emitError("unimplemented: !torch.float 'alpha'");
      return nullptr;
    }
    if (!add.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      add.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value alphaFloat = b.create<mlir::SIToFPOp>(loc, payloadArgs[0].getType(),
                                                adaptor.alpha());
    Value scaled = b.create<mlir::MulFOp>(loc, payloadArgs[1], alphaFloat);
    return b.create<mlir::AddFOp>(loc, payloadArgs[0], scaled);
  }
  if (auto sub = dyn_cast<AtenSubTensorOp>(op)) {
    AtenSubTensorOp::Adaptor adaptor(operands);
    if (sub.alpha().getType().isa<Torch::FloatType>()) {
      sub.emitError("unimplemented: !torch.float 'alpha'");
      return nullptr;
    }
    if (!sub.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      sub.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value alphaFloat = b.create<mlir::SIToFPOp>(loc, payloadArgs[0].getType(),
                                                adaptor.alpha());
    Value scaled = b.create<mlir::MulFOp>(loc, payloadArgs[1], alphaFloat);

    return b.create<mlir::SubFOp>(loc, payloadArgs[0], scaled);
  }
  if (auto mul = dyn_cast<AtenMulTensorOp>(op)) {
    if (!mul.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      mul.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<mlir::MulFOp>(loc, payloadArgs[0], payloadArgs[1]);
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    if (!div.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      div.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<DivFOp>(loc, payloadArgs[0], payloadArgs[1]);
  }
  if (auto lerp = dyn_cast<AtenLerpTensorOp>(op)) {
    if (!lerp.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      lerp.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    AtenLerpTensorOp::Adaptor adaptor(payloadArgs);
    auto start = adaptor.self();
    auto end = adaptor.end();
    auto weight = adaptor.weight();
    auto delta = b.create<SubFOp>(loc, end, start);
    auto weightedDelta = b.create<MulFOp>(loc, delta, weight);
    return b.create<AddFOp>(loc, start, weightedDelta);
  }
  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForElementwiseOp");
  return nullptr;
}

static Value createLinalgNeutralElementForReduceOp(OpBuilder &b, Location loc,
                                                   Operation *op,
                                                   Type elementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op) &&
      elementType.isa<mlir::FloatType>())
    return b.create<mlir::ConstantOp>(loc, b.getFloatAttr(elementType, 0.0));

  op->emitError("unimplemented lowering in "
                "createLinalgNeutralElementForReduceOp");
  return nullptr;
}

static Value createLinalgPayloadCalculationForReduceOp(
    OpBuilder &b, Location loc, ValueRange payloadArgs, Operation *op,
    ArrayRef<Value> operands, Type elementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op) &&
      elementType.isa<mlir::FloatType>())
    return b.create<AddFOp>(loc, payloadArgs);

  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForReduceOp");
  return nullptr;
}

namespace {

// Converts an elementwise op.
// This specifically includes:
// - converting elementwise ops of any tensor arity
// - converting elementwise ops with any number of scalar captures (such as a
//   scalar alpha to torch.aten.Add)
// - broadcasting of static size-1 dimensions
//
// Currently, we adopt the behavior that "size 1" broadcasting is a runtime
// error if it happens dynamically.
//
// Looking forward a bit, eventually, it probably makes sense to have
// a "linalg.generic-like" op for modeling a fused subgraph of numpy-broadcasted
// operands. Modeling elementwise ops that way is potentially useful to allow a
// more centralized reasoning about multiversioning. However a cost model will
// be needed for "pre-fusing" elementwise ops that way, as it can potentially be
// a pessimization. A mild extension of this pattern should work for such a
// general op.
struct ConvertElementwiseOp : ConversionPattern {
  ConvertElementwiseOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenTanhOp, AtenReluOp, AtenAddTensorOp, AtenMulTensorOp,
             AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp>(
            op))
      return rewriter.notifyMatchFailure(op, "not a supported elementwise op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
        operands, [](Value v) { return v.getType().isa<RankedTensorType>(); }));
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    auto resultRank = resultType.getRank();

    auto c1 = rewriter.create<mlir::ConstantIndexOp>(loc, /*value=*/1);
    // The overall error handling strategy here is best viewed by thinking about
    // what happens for a single result dimension. This loop not structured that
    // way because it is hard to create the affine maps for each operand unless
    // we structure the loop to iterate over tensor operands as the outer loop
    // instead of inner loop. This pseudocode gives better intuition:
    // ```
    // for each result dimension:
    //   for each tensor operand:
    //     if it doesn't even have high enough rank relative to the result:
    //       continue
    //     if it is a static size-1 along this result dimension:
    //       continue
    //     if this is the first tensor operand that didn't continue above:
    //       take its dimension size as the size of the non-broadcasted
    //       traversal along this dimension (this may include a dynamic size-1,
    //       **non-broadcasted** traversal!)
    //     emit error check "if the size does not match the non-broadcasted
    //     traversal size along this dimension, error"
    // ```
    // Initialize the resultShape to all 1's, as a fallback in case
    // all sizes along that result dimension are statically 1.
    SmallVector<Value> resultShape(resultRank, c1);
    SmallVector<AffineMap> indexingMaps;
    for (Value tensorOperand : tensorOperands) {
      SmallVector<AffineExpr> exprs;
      auto type = tensorOperand.getType().cast<RankedTensorType>();
      for (auto size : llvm::enumerate(type.getShape())) {
        // If the size is statically known to be 1, we don't want any
        // error guards to be spuriously emitted, since we are specifically
        // allowing size-1 broadcasts in this case, as they correspond to a
        // constant-0 indexing map.
        if (size.value() == 1) {
          exprs.push_back(rewriter.getAffineConstantExpr(0));
          continue;
        }

        // The rank of this operand might be smaller than the overall rank of
        // the broadcast. Add an offset to correlate it to the correct
        // dimension of the result.
        auto resultDim = size.index() + (resultRank - type.getRank());

        // The generated linalg op will now be iterating along the full size
        // of this dimension. Record that fact.
        exprs.push_back(rewriter.getAffineDimExpr(resultDim));

        // Now, we need to ensure that such iteration is not going to trigger
        // undefined behavior, by doing appropriate checks against the current
        // dimension size.
        auto currentDimSize =
            rewriter.create<tensor::DimOp>(loc, tensorOperand, size.index());

        // If the result size of this dimension has so far only hit the
        // statically-known-to-be-1 case above (i.e., we have not yet assigned a
        // new Value to `resultShape[resultDim]`), then we have no other dynamic
        // values to check against, and merely need to record the current
        // dimension size.
        if (resultShape[resultDim] == c1) {
          resultShape[resultDim] = currentDimSize;
          continue;
        }

        // We prohibit the size-1 dynamic broadcasting scenario, so just check
        // for exact equality with the running result size.
        // This is the check which protects against the undefined behavior of
        // the generated linalg op in the case of iterating two operands with
        // dimensions sizes that are expected to match.
        auto equalToRunning = rewriter.create<CmpIOp>(
            loc, CmpIPredicate::eq, resultShape[resultDim], currentDimSize);
        rewriter.create<AssertOp>(loc, equalToRunning,
                                  "mismatched size for broadcast");
      }
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, getContext()));
    }

    SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
    // Add the indexing map for the outs init tensor.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultShape, resultType.getElementType());
    bool hadErrorCreatingPayload = false;
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/initTensor.getType(),
        /*inputs=*/tensorOperands,
        /*outputs=*/initTensor,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
          Value result = createLinalgPayloadCalculationForElementwiseOp(
              b, loc, payloadArgs, op, operands);
          if (!result) {
            hadErrorCreatingPayload = true;
            return;
          }
          b.create<linalg::YieldOp>(loc, result);
        });
    if (hadErrorCreatingPayload)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                generic.getResult(0));
    return success();
  }
};
} // namespace

namespace {
struct ConvertReductionOp : ConversionPattern {
  ConvertReductionOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  // This function is in charge of all the rewriting that will take
  // place in `matchAndRewrite`. In particular, it converts
  // the reduce operation into an `linalg.generic` operation
  // to reduce the input tensor along the dimensions specified in
  // `dimeSet`.
  LogicalResult
  createReductionLinalgGeneric(Operation *op, ArrayRef<Value> operands,
                               const DenseSet<int64_t> &dimSet, bool keepDim,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto tensorOperand = operands[0];
    auto inputType = tensorOperand.getType().cast<RankedTensorType>();
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

    // Get the result shape by obtaining the size of each
    // dimension in the input tensor that is not getting reduced.
    // If `keepDim` is true, the rank of the output tensor
    // is kept the same as the rank of the input tensor, and the
    // reduced dimensions are set to have size 1.
    auto c1 = rewriter.create<mlir::ConstantIndexOp>(loc, /*value=*/1);
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < inputType.getRank(); i++) {
      auto currentDimSize =
          rewriter.create<tensor::DimOp>(loc, tensorOperand, i);
      if (!dimSet.contains(i))
        resultShape.push_back(currentDimSize);
      else if (keepDim)
        resultShape.push_back(c1);
    }

    // Create the affine expressions that will be used to
    // iterate over the input and output tensors.
    // Here we also set the type of iterator: parallel or reduction.
    SmallVector<AffineExpr> exprs;
    SmallVector<StringRef> iteratorTypes;
    SmallVector<AffineExpr> resultExprs;
    for (auto size : llvm::enumerate(inputType.getShape())) {
      exprs.push_back(rewriter.getAffineDimExpr(size.index()));

      if (dimSet.contains(size.index())) {
        iteratorTypes.push_back(getReductionIteratorTypeName());
        // If `keepDim`, create affine map to the first element
        // in the current dimension.
        if (keepDim)
          resultExprs.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        iteratorTypes.push_back(getParallelIteratorTypeName());
        resultExprs.push_back(rewriter.getAffineDimExpr(size.index()));
      }
    }

    auto indexingMaps = AffineMap::inferFromExprList({exprs, resultExprs});
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultShape, resultType.getElementType());
    Value initValue = createLinalgNeutralElementForReduceOp(
        rewriter, loc, op, resultType.getElementType());
    Value accumulator =
        rewriter.create<linalg::FillOp>(loc, initValue, initTensor)
            .getResult(0);
    bool hadErrorCreatingPayload = false;
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/accumulator.getType(),
        /*inputs=*/tensorOperand,
        /*outputs=*/accumulator,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
          Value result = createLinalgPayloadCalculationForReduceOp(
              b, loc, payloadArgs, op, operands, resultType.getElementType());
          if (!result) {
            hadErrorCreatingPayload = true;
            return;
          }
          b.create<linalg::YieldOp>(loc, result);
        });

    if (hadErrorCreatingPayload)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                generic.getResult(0));
    return success();
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // Every reduce operation must set a value for the `dimSet` and
    // `keepDim` in accordance with their specification.
    DenseSet<int64_t> dimSet;
    bool keepDim = false;
    if (isa<AtenSumOp>(op)) {
      auto tensorOperand = operands[0];
      auto inputType = tensorOperand.getType().cast<RankedTensorType>();

      // `AtenSumOp` reduces along all the dimensiosn of the input tensor.
      for (int64_t i = 0; i < inputType.getRank(); i++)
        dimSet.insert(i);
    } else if (auto sumDimIntListOp = dyn_cast<AtenSumDimIntListOp>(op)) {
      auto tensorOperand = operands[0];
      auto inputType = tensorOperand.getType().cast<RankedTensorType>();

      if (!matchPattern(sumDimIntListOp.keepdim(),
                        m_TorchConstantBool(&keepDim)))
        return failure();

      SmallVector<int64_t> dimList;
      if (!matchPattern(sumDimIntListOp.dim(), m_TorchConstantIntList(dimList)))
        return failure();
      for (auto dim : dimList) {
        // Torch allows for negative values in dimSet to go in reverse
        // order in the dimensions of the input tensor.
        dim = dim >= 0 ? dim : dim + inputType.getRank();
        // Drop invalid dimensions
        if (dim < inputType.getRank())
          dimSet.insert(dim);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "not a supported reduce op");
    }

    return createReductionLinalgGeneric(op, operands, dimSet, keepDim,
                                        rewriter);
  }
};
} // namespace

namespace {
class ConvertAtenMaxPool2dOp : public OpConversionPattern<AtenMaxPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    AtenMaxPool2dOp::Adaptor adaptor(operands);
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

    Value falseValue = rewriter.create<ConstantOp>(
        loc, IntegerAttr::get(rewriter.getIntegerType(1), 0));
    Value ceilModeFalse =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, ceilMode, falseValue);
    rewriter.create<AssertOp>(
        loc, ceilModeFalse,
        rewriter.getStringAttr("only ceil_mode false is supported"));

    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput = getPaddedTensor(op, rewriter, self, paddingIncludingNC);

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

    Value Hout = getOutputDimForConvOps(
        rewriter, loc, H, paddingIntValues[0], dilationIntValues[0],
        kernelSizeIntValues[0], strideIntValues[0]);
    Value Wout = getOutputDimForConvOps(
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
    Value initValue = rewriter.create<ConstantOp>(loc, initialAttr);
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
class ConvertAtenFlattenUsingIntsOp
    : public OpConversionPattern<AtenFlattenUsingIntsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenFlattenUsingIntsOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t startDim;
    if (!matchPattern(op.start_dim(), m_TorchConstantInt(&startDim)))
      return rewriter.notifyMatchFailure(op, "start_dim must be constant");
    int64_t endDim;
    if (!matchPattern(op.end_dim(), m_TorchConstantInt(&endDim)))
      return rewriter.notifyMatchFailure(op, "start_dim must be constant");
    auto type = operands[0].getType().cast<RankedTensorType>();
    auto inputRank = type.getRank();
    auto resultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    if (startDim < 0)
      startDim += inputRank;
    if (endDim < 0)
      endDim += inputRank;

    if (inputRank == 0) {
      SmallVector<ReassociationIndices> reassociation;
      if (!(startDim >= -1 && startDim <= 0 && endDim >= -1 && endDim <= 0))
        return rewriter.notifyMatchFailure(
            op, "start_dim and end_dim must be in [-1, 0] when inputRank is 0");
      rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
          op, resultType, operands[0], reassociation);
      return success();
    }

    if (startDim < 0 || startDim >= inputRank || endDim < 0 ||
        endDim >= inputRank || startDim > endDim)
      return rewriter.notifyMatchFailure(
          op, "statically invalid flattening dim range");

    SmallVector<ReassociationIndices> reassociation(resultType.getRank());
    int j = 0;
    for (auto i : llvm::seq<int64_t>(0, inputRank)) {
      reassociation[j].push_back(i);
      if (i < startDim || i >= endDim)
        j++;
    }
    Value collapsedTensor = rewriter.create<linalg::TensorCollapseShapeOp>(
        op->getLoc(), operands[0], reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                collapsedTensor);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenUnsqueezeOp : public OpConversionPattern<AtenUnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUnsqueezeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    auto inputRank = operands[0].getType().cast<RankedTensorType>().getRank();
    if (dim < 0)
      dim += inputRank + 1;
    if (!(0 <= dim && dim <= inputRank))
      return rewriter.notifyMatchFailure(op, "statically invalid");

    SmallVector<ReassociationIndices> reassociationMap(inputRank);
    // From the perspective of the reassociation map, the situation of
    // unsqueezing before or after the last dimension is symmetrical.
    // Normalize it to the "before" case.
    // The 0 case is special here, since there is no last dimension to insert
    // before -- we simply rely on the loop below iterating 0 times.
    if (dim == inputRank && inputRank != 0)
      dim = inputRank - 1;
    bool alreadyCrossedExpandedDim = false;
    for (int i = 0; i != inputRank; i++) {
      if (alreadyCrossedExpandedDim) {
        reassociationMap[i].push_back(i + 1);
      } else {
        reassociationMap[i].push_back(i);
        if (i == dim) {
          reassociationMap[i].push_back(i + 1);
          alreadyCrossedExpandedDim = true;
        }
      }
    }
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
        op, resultType, operands[0], reassociationMap);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenTransposeIntOp
    : public OpConversionPattern<AtenTransposeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTransposeIntOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    AtenTransposeIntOp::Adaptor adaptor(operands);

    int64_t dim0;
    if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0)))
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    int64_t dim1;
    if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");

    auto inVector = adaptor.self();
    auto inType = inVector.getType().cast<RankedTensorType>();
    auto inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    auto elementType = inType.getElementType();

    if (dim0 < 0)
      dim0 += inputRank + 1;
    if (dim0 < 0 || dim0 >= inputRank)
      return rewriter.notifyMatchFailure(op, "dim0 out of range");
    if (dim1 < 0)
      dim1 += inputRank + 1;
    if (dim1 < 0 || dim1 >= inputRank)
      return rewriter.notifyMatchFailure(op, "dim1 out of range");

    auto loc = op.getLoc();

    SmallVector<Value> outputDims;
    for (auto i = 0; i < inputRank; i++)
      outputDims.push_back(getDimOp(rewriter, loc, adaptor.self(), i));
    std::swap(outputDims[dim0], outputDims[dim1]);

    Value outVector =
        rewriter.create<linalg::InitTensorOp>(loc, outputDims, elementType);
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (auto i = 0; i < inputRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (auto i = 0; i < inputRank; i++) {
      if (i == dim0) {
        swapExprs.push_back(idExprs[dim1]);
      } else if (i == dim1) {
        swapExprs.push_back(idExprs[dim0]);
      } else {
        swapExprs.push_back(idExprs[i]);
      }
    }

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(inputRank, 0, idExprs, op.getContext()),
        AffineMap::get(inputRank, 0, swapExprs, op.getContext())};
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
    auto transpose = rewriter
                         .create<linalg::GenericOp>(
                             loc, outVector.getType(), inVector, outVector,
                             indexingMaps, iteratorTypes,
                             [](OpBuilder &b, Location loc, ValueRange args) {
                               b.create<linalg::YieldOp>(loc, args[0]);
                             })
                         .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, transpose);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCatOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();
    AtenCatOp::Adaptor adaptor(operands);

    Value dimValue = op.dim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");

    // Collect all the tensors to be concatenated.
    auto tensorList = op.tensors();
    auto listConstruct = tensorList.getDefiningOp<PrimListConstructOp>();
    if (!listConstruct)
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    auto tensors = llvm::to_vector<4>(
        llvm::map_range(listConstruct.elements(), [&](Value tensor) -> Value {
          return typeConverter->materializeTargetConversion(
              rewriter, loc, getTypeConverter()->convertType(tensor.getType()),
              tensor);
        }));

    RankedTensorType newResultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    int rank = newResultType.getRank();
    SmallVector<Value> offsets, sizes, strides;
    sizes.reserve(rank);
    strides.resize(rank, rewriter.create<ConstantIndexOp>(loc, 1));
    offsets.resize(rank, rewriter.create<ConstantIndexOp>(loc, 0));

    for (int i = 0; i < rank; ++i)
      sizes.push_back(rewriter.create<tensor::DimOp>(loc, tensors[0], i));

    // Calculate the size of the `dim` result dimension by adding the dim size
    // of each tensor together.
    Value resultDimSize = sizes[dim];
    Value dimIndex = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(),
                                                  adaptor.dim());
    for (auto tensor : makeArrayRef(tensors).drop_front()) {
      auto size = rewriter.create<tensor::DimOp>(loc, tensor, dimIndex);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;

    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, sizes, newResultType.getElementType());
    for (auto tensor : tensors) {
      sizes[dim] = rewriter.create<tensor::DimOp>(loc, tensor, dimIndex);
      result = rewriter.create<tensor::InsertSliceOp>(loc, tensor, result,
                                                      offsets, sizes, strides);
      offsets[dim] = rewriter.create<AddIOp>(loc, offsets[dim], sizes[dim]);
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenGatherOp : public OpConversionPattern<AtenGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGatherOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    AtenGatherOp::Adaptor adaptor(operands);

    Value dimValue = op.dim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");

    Value indices = adaptor.index();
    Value self = adaptor.self();
    RankedTensorType newResultTy =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    int64_t rank = newResultTy.getRank();

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, indices);
    Value result = createZeroInitTensor(rewriter, loc, sizes,
                                        newResultTy.getElementType());

    SmallVector<AffineMap, 2> affineMaps(2,
                                         rewriter.getMultiDimIdentityMap(rank));
    SmallVector<StringRef> iteratorTypes(rank, getParallelIteratorTypeName());
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, newResultTy, indices, result, affineMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto indexValue = args[0];
          Value indexOfDim = rewriter.create<IndexCastOp>(
              loc, rewriter.getIndexType(), indexValue);
          SmallVector<Value> indices;
          for (int i = 0; i < rank; i++) {
            indices.push_back(i == dim
                                  ? indexOfDim
                                  : rewriter.create<linalg::IndexOp>(loc, i));
          }
          Value extract =
              rewriter.create<tensor::ExtractOp>(loc, self, indices);
          rewriter.create<linalg::YieldOp>(loc, extract);
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToLinalg
    : public ConvertTorchToLinalgBase<ConvertTorchToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<tensor::TensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenMmOp>();
    patterns.add<ConvertAtenMmOp>(typeConverter, context);
    target.addIllegalOp<AtenBmmOp>();
    patterns.add<ConvertAtenBmmOp>(typeConverter, context);
    target.addIllegalOp<AtenLinearOp>();
    patterns.add<ConvertAtenLinearOp>(typeConverter, context);
    target.addIllegalOp<AtenBatchNormOp>();
    patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
    target.addIllegalOp<AtenTanhOp, AtenReluOp, AtenAddTensorOp,
                        AtenMulTensorOp, AtenDivTensorOp, AtenSubTensorOp,
                        AtenLerpTensorOp, AtenSigmoidOp>();
    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    target.addIllegalOp<AtenUnsqueezeOp>();
    patterns.add<ConvertAtenUnsqueezeOp>(typeConverter, context);
    target.addIllegalOp<AtenConv2dOp>();
    patterns.add<ConvertAtenConv2dOp>(typeConverter, context);
    target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
    patterns.add<ConvertAtenAdaptiveAvgPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenFlattenUsingIntsOp>();
    patterns.add<ConvertAtenFlattenUsingIntsOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenSumOp>();
    patterns.add<ConvertReductionOp>(typeConverter, context);
    target.addIllegalOp<AtenTransposeIntOp>();
    patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenCatOp>();
    patterns.add<ConvertAtenCatOp>(typeConverter, context);
    target.addIllegalOp<AtenGatherOp>();
    patterns.add<ConvertAtenGatherOp>(typeConverter, context);
    target.addIllegalOp<AtenLayerNormOp>();
    patterns.add<ConvertAtenLayerNormOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
