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
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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

// Generate IR: dim = dim >= 0 ? dim : dim + inputRank
static Value toPositiveDimDynamic(OpBuilder &b, Location loc, Value dim,
                                  Value inputRank) {
  assert(dim.getType().isa<IntegerType>() &&
         "dim arg of toPositiveDim must be integer type");
  Value dimAddInputRank = b.create<arith::AddIOp>(loc, dim, inputRank);
  Value cst0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(inputRank.getType()));
  Value predDimGEZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, dim, cst0);
  Value dimInt = b.create<SelectOp>(loc, predDimGEZero, dim, dimAddInputRank);
  return dimInt;
}

// Generate IR: assert(dim >= 0 && dim < inputRank)
static void assertIsValidDim(OpBuilder &b, Location loc, Value dim,
                             Value inputRank) {
  assert(dim.getType().isa<IntegerType>() &&
         "dim arg of assertIsValidDim must be integer type");
  Value cst0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(inputRank.getType()));
  Value predGEZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, dim, cst0);
  b.create<AssertOp>(loc, predGEZero,
                     b.getStringAttr("dim must be greater or equal to zero"));
  Value predLTInputRank =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, dim, inputRank);
  b.create<AssertOp>(loc, predLTInputRank,
                     b.getStringAttr("dim must be smaller than inputRank"));
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
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

static Value castIndexToInt(OpBuilder &b, Location loc, Value idx) {
  assert(idx.getType().isa<IndexType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
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
  Value contractingDimEqual = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lhsDimInt, rhsDimInt);
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
  Value c0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
}

// Helper function to caculate the output tensor dims for convolution-like ops.
// Along each dim:
// dim_out =
//  floor((dim_in + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1
static Value getOutputDimForConvOps(OpBuilder &b, Location loc, Value in,
                                    Value paddingInt, Value dilationInt,
                                    Value kernelSizeInt, Value strideInt) {
  Value c1 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value c2 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(2));

  Value doublePadding = b.create<arith::MulIOp>(loc, paddingInt, c2);
  // in + 2 * padding
  Value inAddDoublePadding =
      b.create<arith::AddIOp>(loc, castIndexToInt(b, loc, in), doublePadding);

  // dilation * (kernelSize - 1)
  Value kernelSizeSub1 = b.create<arith::SubIOp>(loc, kernelSizeInt, c1);
  Value dilationTimesKernelSize =
      b.create<arith::MulIOp>(loc, dilationInt, kernelSizeSub1);

  Value temp =
      b.create<arith::SubIOp>(loc, inAddDoublePadding, dilationTimesKernelSize);
  Value dividend = b.create<arith::SubIOp>(loc, temp, c1);
  Value division = b.create<arith::FloorDivSIOp>(loc, dividend, strideInt);
  Value out = b.create<arith::AddIOp>(loc, division, c1);
  return castIntToIndex(b, loc, out);
}

static SmallVector<Value>
getAsConstantIntValues(OpBuilder &b, Location loc,
                       SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc,
                                       b.getIntegerAttr(b.getI64Type(), val));
  }));
}

static SmallVector<Value>
getAsConstantIndexValues(OpBuilder &b, Location loc,
                         SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(val));
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
  Value c0 = b.create<arith::ConstantOp>(
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

static Value buildNormalCdf(OpBuilder &b, Location &loc, Value x, Value mean,
                            Value sigma) {
  Type elementType = x.getType();
  Value xMinusMean = b.create<arith::SubFOp>(loc, x, mean);
  Value two = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 2));
  Value sqrt2 = b.create<math::SqrtOp>(loc, two);
  Value erfArg = b.create<arith::DivFOp>(loc, xMinusMean, sqrt2);
  Value erf = b.create<math::ErfOp>(loc, erfArg);
  Value one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
  Value erfPlus1 = b.create<arith::AddFOp>(loc, one, erf);
  Value oneHalf =
      b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.5));
  Value normalCdf = b.create<arith::MulFOp>(loc, oneHalf, erfPlus1);
  return normalCdf;
}

static Value buildUnitNormalCdf(OpBuilder &b, Location &loc, Value x) {
  Type elementType = x.getType();
  Value zero = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0));
  Value one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
  return buildNormalCdf(b, loc, x, zero, one);
}

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

namespace {
class ConvertAtenConv2dOp : public OpConversionPattern<AtenConv2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value groups = adaptor.groups();

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, intType, v);
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

    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value groupEqual1 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, groups, c1);
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

    Value c0float = rewriter.create<arith::ConstantOp>(
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
  Value inputSubMean = b.create<arith::SubFOp>(loc, input, mean);
  // The eps is always f64.
  Value truncatedEps = b.create<arith::TruncFOp>(loc, elemTy, eps);
  Value varPlusEps = b.create<arith::AddFOp>(loc, var, truncatedEps);
  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
  Value temp = b.create<arith::MulFOp>(loc, inputSubMean, rSTD);
  Value timesWeight = b.create<arith::MulFOp>(loc, temp, weight);
  Value plusBias = b.create<arith::AddFOp>(loc, timesWeight, bias);
  return plusBias;
}

static void createLinalgPayloadCalculationForGatherOps(
    OpBuilder &b, Location loc, Value input, int64_t inputRank, Value index,
    int64_t dim, int64_t outputRank) {
  SmallVector<Value> indices;
  for (int i = 0; i < inputRank; i++) {
    if (i == dim) {
      indices.push_back(castIntToIndex(b, loc, index));
    } else {
      // `outputRank` might be larger than `inputRank`. The `linalg::IndexOp`
      // takes in the dimension of the output. Add `inputDimOffset` to
      // related to the correct dimension of the output for dimension larger
      // than the given `dim`.
      int64_t inputDimOffset = i < dim ? 0 : outputRank - inputRank;
      indices.push_back(b.create<linalg::IndexOp>(loc, i + inputDimOffset));
    }
  }

  // Assert index < input.sizes[dim]
  Value indexLTInputDim = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, index,
      castIndexToInt(b, loc, getDimOp(b, loc, input, dim)));
  b.create<AssertOp>(loc, indexLTInputDim,
                     b.getStringAttr("index must be smaller than dim size"));

  // Assert index >= 0
  Value cst0 = b.create<arith::ConstantOp>(loc, b.getZeroAttr(index.getType()));
  Value indexGEThanZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, cst0);
  b.create<AssertOp>(loc, indexGEThanZero,
                     b.getStringAttr("index must be larger or equal to 0"));

  Value extract = b.create<tensor::ExtractOp>(loc, input, indices);
  b.create<linalg::YieldOp>(loc, extract);
}

namespace {
class ConvertAtenBatchNormOp : public OpConversionPattern<AtenBatchNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBatchNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    auto constFalse = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(IntegerType::get(context, 1), 0));
    auto trainingFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, training, constFalse);
    rewriter.create<AssertOp>(
        loc, trainingFalse,
        rewriter.getStringAttr("training is not supported for now"));

    // num_features – C from an expected input of size (N,C,D,H,W ...)
    Value numFeatures = rewriter.create<tensor::DimOp>(loc, input, 1);
    auto contractingDim0EqualsNumFeatures = [&](Value v) {
      auto dim0 = rewriter.create<tensor::DimOp>(loc, v, 0);
      auto dim0Equal = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, numFeatures, dim0);
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
  matchAndRewrite(AtenLayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
      elemCnts = rewriter.create<arith::MulIOp>(loc, elemCnts,
                                                normalizedShapeSizesInt[i]);
    }
    Value elemCntsFloat =
        rewriter.create<arith::SIToFPOp>(loc, elemTy, elemCnts);

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
                    b.create<arith::DivFOp>(loc, sumOrSqureSum, elemCntsFloat);
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
                              rewriter.create<arith::AddFOp>(loc, sum, input);
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
                  Value sub = rewriter.create<arith::SubFOp>(loc, input, mean);
                  Value square = rewriter.create<arith::MulFOp>(loc, sub, sub);
                  Value result =
                      rewriter.create<arith::AddFOp>(loc, squareSum, square);
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
  matchAndRewrite(AtenMmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.mat2();

    // A user can write an errorneous program where `aten.mm` is in fact called
    // with operands of invalid rank or dtype. We cannot convert to linalg in
    // this case or we will get a verifier error, which corresponds to breaking
    // of *internal* compiler invariants, and for a user manifests as a compiler
    // crash in the worst case (such as we try to canonicalize/fold/print the
    // invalid op before the verifier gets to see it -- also release builds of a
    // mature compiler usually have the verifier turned off for compile time
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
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhsDim1, rhsDim0);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{lhsDim0, rhsDim1}, elementType);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
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
class ConvertAtenMatmulOp : public OpConversionPattern<AtenMatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.other();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    unsigned lhsRank = lhs.getType().cast<RankedTensorType>().getRank();
    unsigned rhsRank = rhs.getType().cast<RankedTensorType>().getRank();

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();

    // The different cases of torch_matmul op is mentioned here:
    // https://pytorch.org/docs/stable/generated/torch.matmul.html

    // First Case: Dot Product.
    if (lhsRank == 1 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);

      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value zeroTensor = createZeroInitTensor(rewriter, loc, {}, elementType);
      Value dotProd =
          rewriter
              .create<linalg::DotOp>(loc, zeroTensor.getType(),
                                     ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, dotProd);
      return success();
    }

    // Second Case: Vec-Mat Multiplication.
    if (lhsRank == 1 && rhsRank == 2) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{rhsDim1}, elementType);
      Value matmul =
          rewriter
              .create<linalg::VecmatOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Third Case: Matrix-Vec Multiplication.
    if (lhsRank == 2 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{lhsDim0}, elementType);
      Value matmul =
          rewriter
              .create<linalg::MatvecOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
              .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fourth Case: Batch-Matrix Multiplication.
    // TODO: Broadcasting of batch dimension is remaining.
    if (lhsRank >= 3 && rhsRank >= 3 && lhsRank == rhsRank) {

      unsigned batchRank = lhsRank - 2;
      SmallVector<Value, 4> resultShape;

      SmallVector<AffineExpr> lhsExpr;
      SmallVector<AffineExpr> rhsExpr;
      SmallVector<AffineExpr> outExpr;
      SmallVector<StringRef> iteratorTypes;

      // Since broadcasting is a TODO, check whether the lhs and rhs batch
      // dimension match.
      for (unsigned i = 0; i < batchRank; i++) {
        Value lhsBatch = getDimOp(rewriter, loc, lhs, i);
        Value rhsBatch = getDimOp(rewriter, loc, rhs, i);
        resultShape.push_back(lhsBatch);
        lhsExpr.push_back(rewriter.getAffineDimExpr(i));
        rhsExpr.push_back(rewriter.getAffineDimExpr(i));
        outExpr.push_back(rewriter.getAffineDimExpr(i));
        iteratorTypes.push_back(getParallelIteratorTypeName());
        checkDimEqualHelper(rewriter, loc, lhsBatch, rhsBatch);
      }

      Value lhsDim0 = getDimOp(rewriter, loc, lhs, batchRank);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, batchRank + 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, batchRank);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, batchRank + 1);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      // Push the final matrix dimension.
      resultShape.insert(resultShape.end(), {lhsDim0, rhsDim1});

      lhsExpr.insert(lhsExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 1)});
      rhsExpr.insert(rhsExpr.end(), {rewriter.getAffineDimExpr(batchRank + 1),
                                     rewriter.getAffineDimExpr(batchRank + 2)});
      outExpr.insert(outExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 2)});

      Value initTensor0 =
          createZeroInitTensor(rewriter, loc, resultShape, elementType);

      auto indexingMaps =
          AffineMap::inferFromExprList({lhsExpr, rhsExpr, outExpr});
      iteratorTypes.insert(iteratorTypes.end(),
                           {"parallel", "reduction", "parallel"});

      Value finalRes =
          rewriter
              .create<linalg::GenericOp>(
                  loc, newResultType, ValueRange{lhs, rhs}, initTensor0,
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value l = args[0], r = args[1], res = args[2];
                    Value mul = b.create<arith::MulFOp>(loc, l, r);
                    Value add = b.create<arith::AddFOp>(loc, mul, res);
                    b.create<linalg::YieldOp>(loc, add);
                  })
              .getResult(0);

      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, finalRes);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class ConvertAtenBmmOp : public OpConversionPattern<AtenBmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.mat2();
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

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
  matchAndRewrite(AtenLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, inputDim1, weightDim1);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for aten.linear"));
    // Here we take advantage of ruling out the size-1 case above.
    // In the static-size-1 case, we will not emit this check at all.
    Value biasSizeCorrect = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, weightDim0, biasDim0);
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

// Convert a scalar value to the target type. The scalar value can be an element
// from a tensor or a scalar in the pytorch dialect. Both the scalar and dtype
// should be converted builtin types.
static Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar,
                                  Type dtype) {
  Type scalarType = scalar.getType();
  if (scalarType == dtype)
    return scalar;

  // TODO: For the byte(ui8) or char(i8) case, we need the unconverted dtype to
  // be able to know if we need signed or unsigned conversion.
  auto isByteOrChar = [](Type type) {
    if (auto integerTy = type.dyn_cast<mlir::IntegerType>()) {
      return integerTy.getWidth() == 8;
    }
    return false;
  };

  if (isByteOrChar(scalarType) || isByteOrChar(dtype) ||
      scalarType.isSignlessInteger(1) || dtype.isSignlessInteger(1)) {
    // TODO: Handle bool type.
    mlir::emitError(loc)
        << "unsupported byte, char or bool type for convertScalarToDtype "
        << scalarType << "(scalar type) -> " << dtype << "(dtype)";
    return nullptr;
  }

  if (auto dtypeFloat = dtype.dyn_cast<mlir::FloatType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>()) {
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, scalar, dtype);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return b.create<arith::ExtFOp>(loc, scalar, dtype);
    }
    assert(scalarType.isa<mlir::IntegerType>());
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::SIToFPOp>(loc, scalar, dtype);
  }

  if (auto dtypeInteger = dtype.dyn_cast<mlir::IntegerType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>())
      return b.create<arith::FPToSIOp>(loc, scalar, dtype);
    assert(scalarType.isa<mlir::IntegerType>());
    auto scalarInteger = scalarType.cast<mlir::IntegerType>();
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return b.create<arith::TruncIOp>(loc, scalar, dtype);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::ExtSIOp>(loc, scalar, dtype);
  }

  llvm_unreachable("convertScalarToDtype should handle all the types");
}

static Value createLinalgPayloadCalculationForElementwiseOp(
    OpBuilder &b, Location loc, TypeConverter *converter,
    ValueRange payloadArgs, Operation *op, ArrayRef<Value> operands) {
  if (isa<AtenTanhOp>(op))
    return b.create<math::TanhOp>(loc, payloadArgs[0]);
  if (isa<AtenExpOp>(op))
    return b.create<math::ExpOp>(loc, payloadArgs[0]);
  if (isa<AtenFloorOp>(op))
    return b.create<math::FloorOp>(loc, payloadArgs[0]);
  if (isa<AtenLogOp>(op))
    return b.create<math::LogOp>(loc, payloadArgs[0]);
  if (isa<AtenSqrtOp>(op))
    return b.create<math::SqrtOp>(loc, payloadArgs[0]);
  if (isa<AtenRsqrtOp>(op))
    return b.create<math::RsqrtOp>(loc, payloadArgs[0]);
  if (isa<AtenLog2Op>(op))
    return b.create<math::Log2Op>(loc, payloadArgs[0]);
  if (isa<AtenSigmoidOp>(op)) {
    Type elementType = payloadArgs[0].getType();
    auto one = b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1));
    auto negate = b.create<arith::NegFOp>(loc, payloadArgs[0]);
    auto exp = b.create<math::ExpOp>(loc, negate);
    auto added = b.create<arith::AddFOp>(loc, exp, one);
    return b.create<arith::DivFOp>(loc, one, added);
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
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    return b.create<SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  if (auto gelu = dyn_cast<AtenGeluOp>(op)) {
    if (!gelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      gelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value cdf = buildUnitNormalCdf(b, loc, payloadArgs[0]);
    return b.create<arith::MulFOp>(loc, payloadArgs[0], cdf);
  }
  if (auto geluBackward = dyn_cast<AtenGeluBackwardOp>(op)) {
    if (!geluBackward.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      geluBackward.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[1].getType();
    Value cstAlpha0 = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 1.12837916709551257390));
    Value cstAlpha1 = b.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.70710678118654752440));
    Value oneHalf =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.5));
    Value kAlpha = b.create<arith::MulFOp>(loc, cstAlpha0, cstAlpha1);
    Value kAlphaHalf = b.create<arith::MulFOp>(loc, kAlpha, oneHalf);
    Value negOneHalf =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, -0.5));
    Value inputSquared =
        b.create<arith::MulFOp>(loc, payloadArgs[1], payloadArgs[1]);
    Value negHalfInputSquared =
        b.create<arith::MulFOp>(loc, inputSquared, negOneHalf);
    Value dinput = b.create<math::ExpOp>(loc, negHalfInputSquared);
    Value cdf = buildUnitNormalCdf(b, loc, payloadArgs[1]);
    Value dinputInput = b.create<arith::MulFOp>(loc, dinput, payloadArgs[1]);
    Value dinputInputAlpha =
        b.create<arith::MulFOp>(loc, dinputInput, kAlphaHalf);
    Value cdfExt = b.create<arith::AddFOp>(loc, dinputInputAlpha, cdf);
    return b.create<arith::MulFOp>(loc, payloadArgs[0], cdfExt);
  }
  if (auto add = dyn_cast<AtenAddTensorOp>(op)) {
    AtenAddTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(add.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, adaptor.alpha(), dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::AddFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::AddIOp>(loc, lhs, scaled);
    }
  }
  if (auto sub = dyn_cast<AtenSubTensorOp>(op)) {
    AtenSubTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(sub.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, adaptor.alpha(), dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value scaled = b.create<arith::MulFOp>(loc, rhs, alpha);
      return b.create<arith::SubFOp>(loc, lhs, scaled);
    } else {
      Value scaled = b.create<arith::MulIOp>(loc, rhs, alpha);
      return b.create<arith::SubIOp>(loc, lhs, scaled);
    }
  }
  if (auto mul = dyn_cast<AtenMulTensorOp>(op)) {
    if (!mul.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      mul.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<arith::MulFOp>(loc, payloadArgs[0], payloadArgs[1]);
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    if (!div.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      div.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    return b.create<arith::DivFOp>(loc, payloadArgs[0], payloadArgs[1]);
  }
  if (auto pow = dyn_cast<AtenPowTensorScalarOp>(op)) {
    if (!pow.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      pow.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type dtype = pow.self().getType().cast<ValueTensorType>().getDtype();
    Value expPromoted = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<math::PowFOp>(loc, payloadArgs[0], expPromoted);
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
    auto delta = b.create<arith::SubFOp>(loc, end, start);
    auto weightedDelta = b.create<arith::MulFOp>(loc, delta, weight);
    return b.create<arith::AddFOp>(loc, start, weightedDelta);
  }
  if (auto minimum = dyn_cast<AtenMinimumOp>(op)) {
    if (!minimum.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      minimum.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                         payloadArgs[0], payloadArgs[1]);
    return b.create<SelectOp>(loc, pred, payloadArgs[0], payloadArgs[1]);
  }
  if (auto maximum = dyn_cast<AtenMaximumOp>(op)) {
    if (!maximum.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      maximum.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], payloadArgs[1]);
    return b.create<SelectOp>(loc, pred, payloadArgs[0], payloadArgs[1]);
  }
  if (auto clamp = dyn_cast<AtenClampOp>(op)) {
    Type dtype = converter->convertType(clamp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      clamp.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    AtenClampOp::Adaptor adaptor(operands);
    auto min = adaptor.min();
    auto max = adaptor.max();
    if (min.getType().isa<Torch::OptionalType>() ||
        max.getType().isa<Torch::OptionalType>()) {
      clamp.emitError("unimplemented: runtime optional type");
      return nullptr;
    }
    auto result = payloadArgs[0];
    if (!min.getType().isa<Torch::NoneType>()) {
      auto minPromoted = convertScalarToDtype(b, loc, min, dtype);
      auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                          result, minPromoted);
      result = b.create<SelectOp>(loc, pred, minPromoted, result);
    }
    if (!max.getType().isa<Torch::NoneType>()) {
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                          result, maxPromoted);
      result = b.create<SelectOp>(loc, pred, maxPromoted, result);
    }
    return result;
  }
  if (auto rsub = dyn_cast<AtenRsubScalarOp>(op)) {
    Type dtype = converter->convertType(rsub.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      rsub.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = payloadArgs[0];
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    Value mult = b.create<arith::MulFOp>(loc, self, alpha);
    return b.create<arith::SubFOp>(loc, other, mult);
  }
  if (auto mulScalar = dyn_cast<AtenMulScalarOp>(op)) {
    Type dtype = converter->convertType(mulScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      mulScalar.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = payloadArgs[0];
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<arith::MulFOp>(loc, self, other);
  }
  if (auto atenToDtype = dyn_cast<AtenToDtypeOp>(op)) {
    Value input = payloadArgs[0];
    Type dtype = converter->convertType(atenToDtype.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value result = convertScalarToDtype(b, loc, input, dtype);
    return result;
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
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(elementType, 0.0));

  op->emitError("unimplemented lowering in "
                "createLinalgNeutralElementForReduceOp");
  return nullptr;
}

static Value createLinalgPayloadCalculationForReduceOp(
    OpBuilder &b, Location loc, ValueRange payloadArgs, Operation *op,
    ArrayRef<Value> operands, Type elementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op) &&
      elementType.isa<mlir::FloatType>())
    return b.create<arith::AddFOp>(loc, payloadArgs);
  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForReduceOp");
  return nullptr;
}

namespace {
// Aten argmax lowering represents the ArgMax op as an linalg.indexed_generic
// op, producing two output buffers.
//
// The first output buffer contains the index of the found maximum value. It is
// initialized to 0 and is resulting integer type.
//
// The second output buffer contains the maximum value found. It is initialized
// to the minimum representable value of the input element type. After being
// populated by indexed_generic, this buffer is disgarded as only the index is
// requested.
//
// The indexed_generic op updates both the maximum value and index if the
// current value exceeds the running max.
class ConvertAtenArgmaxOp : public OpConversionPattern<AtenArgmaxOp> {
public:
  using OpConversionPattern<AtenArgmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenArgmaxOp argmaxOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = argmaxOp.getLoc();
    Value input = adaptor.self();
    RankedTensorType resultType =
        getTypeConverter()
            ->convertType(argmaxOp.getResult().getType())
            .cast<RankedTensorType>();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type outElementType = resultType.getElementType();
    if (!outElementType.isa<IntegerType>())
      return rewriter.notifyMatchFailure(
          argmaxOp,
          "aten.arg_max to linalg.* requires integer-like result type");

    bool keepDim = false;
    if (!matchPattern(argmaxOp.keepdim(), m_TorchConstantBool(&keepDim)))
      return failure();

    int64_t dim;
    if (!matchPattern(argmaxOp.dim(), m_TorchConstantInt(&dim))) {
      if (!argmaxOp.dim().getType().isa<Torch::NoneType>())
        return rewriter.notifyMatchFailure(
            argmaxOp,
            "aten.arg_max to linalg.* requires int or NoneType value for Dim");
      // For pytorch, if the value of Dim is None, argmax
      // returns the index of the max value of the flattened input tensor,
      // so here we flatten the input tensor.
      SmallVector<ReassociationIndices> reassociation(1);
      for (auto i : llvm::seq<int64_t>(0, inputType.getRank()))
        reassociation[0].push_back(i);
      input = rewriter.create<linalg::TensorCollapseShapeOp>(
          argmaxOp->getLoc(), input, reassociation);
      // Becomes 0 for flattened tensor.
      dim = 0;
      // Recast to fix shape.
      inputType = input.getType().cast<RankedTensorType>();
    }
    Type inElementType = inputType.getElementType();
    if (!inElementType.isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          argmaxOp,
          "aten.arg_max to linalg.* requires Float input element type");
    }

    // Constant op to account for the reduction along dim.
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < inputType.getRank(); i++) {
      if (dim != i) {
        auto currentDimSize = rewriter.create<tensor::DimOp>(loc, input, i);
        resultShape.push_back(currentDimSize);
      } else if (keepDim)
        resultShape.push_back(c1);
    }
    // First fill the output buffer for the index.
    Value filledTensorIdx =
        createZeroInitTensor(rewriter, loc, resultShape, outElementType);

    // Second fill the output buffer for the running max.
    Value initTensorMax =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, inElementType)
            .result();

    FloatAttr fillValueMaxAttr = rewriter.getFloatAttr(
        inElementType,
        APFloat::getLargest(
            inElementType.cast<mlir::FloatType>().getFloatSemantics(), true));

    Value fillValueMax =
        rewriter.create<arith::ConstantOp>(loc, fillValueMaxAttr);
    Value filledTensorMax =
        rewriter.create<linalg::FillOp>(loc, fillValueMax, initTensorMax)
            .result();

    // Create the affine expressions that will be used to
    // iterate over the input and output tensors.
    // Here we also set the type of iterator: parallel or reduction.
    SmallVector<AffineExpr> exprs;
    SmallVector<StringRef> iteratorTypes;
    SmallVector<AffineExpr> resultExprs;
    for (auto size : llvm::enumerate(inputType.getShape())) {
      exprs.push_back(rewriter.getAffineDimExpr(size.index()));

      if (unsigned(dim) == size.index()) {
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
    auto maps = AffineMap::inferFromExprList({exprs, resultExprs, resultExprs});
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        ArrayRef<Type>({filledTensorIdx.getType(), filledTensorMax.getType()}),
        input, ValueRange({filledTensorIdx, filledTensorMax}), maps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value newValue = blockArgs[0];
          Value oldIndex = blockArgs[1];
          Value oldValue = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, dim));

          Value predicate;
          if (inElementType.isa<mlir::FloatType>())
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          auto resultMax = rewriter.create<mlir::SelectOp>(nestedLoc, predicate,
                                                           newValue, oldValue);
          auto resultIndex = rewriter.create<mlir::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultIndex, resultMax}));
        });

    // This cast is required to fix the shape in the case of keepDim=True
    rewriter.replaceOpWithNewOp<tensor::CastOp>(argmaxOp, resultType,
                                                linalgOp.getResult(0));
    return success();
  }
};
} // namespace
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
    if (!isa<AtenTanhOp, AtenReluOp, AtenGeluOp, AtenGeluBackwardOp,
             AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp, AtenSubTensorOp,
             AtenLerpTensorOp, AtenSigmoidOp, AtenExpOp, AtenMinimumOp,
             AtenMaximumOp, AtenToDtypeOp, AtenClampOp, AtenRsubScalarOp,
             AtenMulScalarOp, AtenLogOp, AtenSqrtOp, AtenFloorOp,
             AtenPowTensorScalarOp, AtenLog2Op, AtenRsqrtOp>(op))
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

    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
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
        auto equalToRunning = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, resultShape[resultDim],
            currentDimSize);
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
              b, loc, getTypeConverter(), payloadArgs, op, operands);
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
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
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
class ConvertAtenFlattenUsingIntsOp
    : public OpConversionPattern<AtenFlattenUsingIntsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t startDim;
    if (!matchPattern(op.start_dim(), m_TorchConstantInt(&startDim)))
      return rewriter.notifyMatchFailure(op, "start_dim must be constant");
    int64_t endDim;
    if (!matchPattern(op.end_dim(), m_TorchConstantInt(&endDim)))
      return rewriter.notifyMatchFailure(op, "end_dim must be constant");
    auto type = adaptor.self().getType().cast<RankedTensorType>();
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
          op, resultType, adaptor.self(), reassociation);
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
        op->getLoc(), adaptor.self(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                collapsedTensor);
    return success();
  }
};
} // namespace

namespace {
/// The `ConvertAtenViewOp` conversion pattern converts `aten.View` op to
/// `linalg.TensorExpandShape` op only when one or multiple static dimensions
/// are expanded. All the other cases of `aten.View` op need to be handled.
/// TODO: Handle all the other cases of `aten.View` op.
class ConvertAtenViewOp : public OpConversionPattern<AtenViewOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.self();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();
    // When we only have expansion of dimensions in `aten.View`, the output
    // tensor rank will be strictly greater than the input tensor rank.
    // TODO: Handle the cases of `aten.View` op where,
    // 1. One or multiple dimensions are collapsed.
    // 2. Few dimensions are expanded and few other dimensions are collapsed.
    if (inputRank >= resultRank) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: operand tensor rank should be strictly less than "
              "the desired output rank");
    }

    // Extract the desired output size as a list of integers. This list should
    // have been created using the operation `torch.prim.ListConstruct`.
    SmallVector<Value> expectedSizeTorchInt;
    if (!getListConstructElements(op.size(), expectedSizeTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: the desired size is "
                                         "not constructed from ListConstruct");
    }
    SmallVector<Value> expectedSize = getTypeConvertedValues(
        rewriter, loc, typeConverter, expectedSizeTorchInt);
    if (resultRank != (int64_t)expectedSize.size()) {
      return rewriter.notifyMatchFailure(
          op, "desired size list length mismatches with the result type rank");
    }

    // Check if the `aten.View` can be legalized to `linalg.TensorExpandShape`.
    // It only handles the case of static dimension expansion. If the dimension
    // is dynamic, it must not be expanded/splitted.
    // TODO: Handle the case of dynamic dimension expansion.
    SmallVector<ReassociationIndices> reassociation(inputRank);
    SmallVector<int64_t> resultShape;
    int64_t j = 0;
    for (auto i : llvm::seq<int64_t>(0, inputRank)) {
      if (inputType.isDynamicDim(i)) {
        Value dim = getDimOp(rewriter, loc, input, i);
        if (j >= resultRank) {
          return rewriter.notifyMatchFailure(
              op, "desired size is not compatible with the input tensor size");
        }
        checkDimEqualHelper(rewriter, loc, dim, expectedSize[j]);
        reassociation[i].push_back(j++);
        resultShape.push_back(kUnknownSize);
      } else {
        int64_t expandedDim = inputType.getDimSize(i);
        int64_t outputDim;
        // A do-while loop is used here to handle the cases where the input
        // tensor has a dimension of size 1.
        do {
          if (j >= resultRank ||
              !matchPattern(expectedSizeTorchInt[j],
                            m_TorchConstantInt(&outputDim)) ||
              expandedDim % outputDim != 0) {
            return rewriter.notifyMatchFailure(
                op, "total number of elements mismatch in the expansion");
          }
          reassociation[i].push_back(j++);
          resultShape.push_back(outputDim);
          expandedDim /= outputDim;
        } while (expandedDim != 1);
      }
    }
    // Make sure that the splitted dimensions have the same number of elements
    // as the dimension got splitted from.
    if (j != resultRank)
      return rewriter.notifyMatchFailure(
          op, "desired size is not compatible with the input tensor size");

    Type expandType =
        RankedTensorType::get(resultShape, resultType.getElementType());
    Value expandOp = rewriter.create<linalg::TensorExpandShapeOp>(
        loc, expandType, adaptor.self(), reassociation);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, expandOp);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenUnsqueezeOp : public OpConversionPattern<AtenUnsqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenUnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    auto inputRank =
        adaptor.self().getType().cast<RankedTensorType>().getRank();
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
        op, resultType, adaptor.self(), reassociationMap);
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
  matchAndRewrite(AtenTransposeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

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

    dim0 = toPositiveDim(dim0, inputRank);
    if (!isValidDim(dim0, inputRank))
      return rewriter.notifyMatchFailure(op, "dim0 out of range");
    dim1 = toPositiveDim(dim1, inputRank);
    if (!isValidDim(dim1, inputRank))
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
      if (i == dim0)
        swapExprs.push_back(idExprs[dim1]);
      else if (i == dim1)
        swapExprs.push_back(idExprs[dim0]);
      else
        swapExprs.push_back(idExprs[i]);
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
class ConvertAtenPermuteOp : public OpConversionPattern<AtenPermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    SmallVector<int64_t> dimensions;
    if (!matchPattern(op.dims(), m_TorchConstantIntList(dimensions)))
      return rewriter.notifyMatchFailure(op, "all dimensions must be constant");

    Value inVector = adaptor.self();
    auto inType = inVector.getType().cast<RankedTensorType>();
    int64_t inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    Type elementType = inType.getElementType();

    // Check if the dimensions are a valid constants.
    int64_t numDimensions = dimensions.size();
    if (inputRank != numDimensions)
      return rewriter.notifyMatchFailure(
          op, "size of `dims` must be equal to the rank of the input");
    for (unsigned i = 0; i < numDimensions; i++) {
      if (dimensions[i] < 0)
        dimensions[i] = toPositiveDim(dimensions[i], inputRank);
      if (!isValidDim(dimensions[i], inputRank))
        return rewriter.notifyMatchFailure(op, "dimension out of range");
    }

    Location loc = op.getLoc();

    SmallVector<Value> outputDims;
    for (unsigned i = 0; i < inputRank; i++)
      outputDims.push_back(getDimOp(rewriter, loc, inVector, dimensions[i]));

    Value outVector =
        rewriter.create<linalg::InitTensorOp>(loc, outputDims, elementType);
    SmallVector<AffineExpr> idExprs;
    SmallVector<AffineExpr> swapExprs;
    for (unsigned i = 0; i < inputRank; i++)
      idExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    for (unsigned i = 0; i < inputRank; i++)
      swapExprs.push_back(idExprs[dimensions[i]]);

    SmallVector<AffineMap> indexingMaps =
        AffineMap::inferFromExprList({idExprs, swapExprs});
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
  matchAndRewrite(AtenCatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    Value dimValue = op.dim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");

    // Collect all the tensors to be concatenated.
    auto tensorList = op.tensors();
    SmallVector<Value> tensorsTorchType;
    if (!getListConstructElements(tensorList, tensorsTorchType))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    auto tensors =
        getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int rank = newResultType.getRank();
    SmallVector<Value> offsets, sizes, strides;
    sizes.reserve(rank);
    strides.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 1));
    offsets.resize(rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));

    for (int i = 0; i < rank; ++i)
      sizes.push_back(rewriter.create<tensor::DimOp>(loc, tensors[0], i));

    // Calculate the size of the `dim` result dimension by adding the dim size
    // of each tensor together.
    Value resultDimSize = sizes[dim];
    Value dimIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), adaptor.dim());
    for (auto tensor : makeArrayRef(tensors).drop_front()) {
      auto size = rewriter.create<tensor::DimOp>(loc, tensor, dimIndex);
      resultDimSize = rewriter.create<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;

    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, sizes, newResultType.getElementType());
    for (auto tensor : tensors) {
      sizes[dim] = rewriter.create<tensor::DimOp>(loc, tensor, dimIndex);
      result = rewriter.create<tensor::InsertSliceOp>(loc, tensor, result,
                                                      offsets, sizes, strides);
      offsets[dim] =
          rewriter.create<arith::AddIOp>(loc, offsets[dim], sizes[dim]);
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
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();

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
          auto index = args[0];
          createLinalgPayloadCalculationForGatherOps(b, loc, self, rank, index,
                                                     dim, rank);
        });
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenEmbeddingOp : public OpConversionPattern<AtenEmbeddingOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenEmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value weight = adaptor.weight();
    Value indices = adaptor.indices();
    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();

    auto weightTy = weight.getType().cast<RankedTensorType>();
    if (weightTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "weight must be rank 2");
    Value embeddingDim = getDimOp(rewriter, loc, weight, 1);
    Type elemTy = weightTy.getElementType();

    SmallVector<Value> sizes = getTensorSizes(rewriter, loc, indices);
    sizes.push_back(embeddingDim);
    int64_t resultRank = sizes.size();

    auto indicesTy = weight.getType().cast<RankedTensorType>();
    int64_t indicesRank = indicesTy.getRank();
    SmallVector<AffineExpr> indicesExprs;
    for (int i = 0; i < indicesRank; i++)
      indicesExprs.push_back(rewriter.getAffineDimExpr(i));
    auto indicesAffineMap = AffineMap::get(
        /*dimCount=*/resultRank,
        /*symbolCount=*/0, indicesExprs, op->getContext());
    SmallVector<AffineMap, 2> indexingMaps = {
        indicesAffineMap,
        rewriter.getMultiDimIdentityMap(resultRank),
    };
    SmallVector<StringRef> iteratorTypes(sizes.size(),
                                         getParallelIteratorTypeName());
    Value initTensor =
        rewriter.create<linalg::InitTensorOp>(loc, sizes, elemTy);
    Value embeddingResult =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), indices, initTensor,
                /*indexingMaps=*/indexingMaps, /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = args[0];
                  createLinalgPayloadCalculationForGatherOps(
                      b, loc, weight, weightTy.getRank(), index, /*dim=*/0,
                      resultRank);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
                                                embeddingResult);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSizeIntOp : public OpConversionPattern<AtenSizeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSizeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    Value dim = adaptor.dim();
    auto type = self.getType().cast<RankedTensorType>();
    Value inputRank = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(type.getRank()));
    Value dimPositive = toPositiveDimDynamic(rewriter, loc, dim, inputRank);
    assertIsValidDim(rewriter, loc, dimPositive, inputRank);
    Value size = rewriter.create<tensor::DimOp>(
        loc, adaptor.self(), castIntToIndex(rewriter, loc, dimPositive));
    rewriter.replaceOp(op, castIndexToInt(rewriter, loc, size));
    return success();
  }
};
} // namespace

// Casts a 0d integer tensor to elemental type.
namespace {
class ConvertAtenIntTensorOp : public OpConversionPattern<AtenIntTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIntTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value intTensor = adaptor.a();
    auto tensorType = intTensor.getType().cast<RankedTensorType>();

    if (tensorType.getRank() != 0)
      return rewriter.notifyMatchFailure(
          op, "invalid rank: the rank of the input tensor must be 0");

    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, intTensor);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value self = adaptor.self();
    auto selfType = self.getType().cast<RankedTensorType>();
    ArrayRef<int64_t> selfShape = selfType.getShape();
    Type elementType = selfType.getElementType();
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();

    SmallVector<Value> inShape, outShape;
    if (!getListConstructElements(adaptor.size(), inShape)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the size list is not from list construct");
    }
    SmallVector<Value> inShapeConverted =
        getTypeConvertedValues(rewriter, loc, getTypeConverter(), inShape);
    if (inShape.size() < selfShape.size())
      return rewriter.notifyMatchFailure(
          op, "invalid shape: must not be smaller than rank of tensor");
    size_t diff = inShape.size() - selfShape.size();

    // Create affine map and shapes for tensor initialization.
    SmallVector<AffineExpr> outExpr;
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
    for (size_t i = 0; i < inShape.size(); i++) {
      Value shapeValue = inShapeConverted[i];
      size_t j = i - diff;
      if (i < diff) {
        Value isValid = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, shapeValue, zero);
        rewriter.create<AssertOp>(
            loc, isValid,
            rewriter.getStringAttr(
                "negative values not allowed in new dimensions"));
        outShape.push_back(castIntToIndex(rewriter, loc, shapeValue));
        continue;
      }
      if (selfShape[j] == 1) {
        // Broadcast singleton dimension
        Value one =
            rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
        Value isNegative = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, shapeValue, zero);
        Value select = rewriter.create<SelectOp>(
            loc, isNegative, one, castIntToIndex(rewriter, loc, shapeValue));
        outShape.push_back(select);
        outExpr.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      // Non-broadcast case
      Value dim = getDimOp(rewriter, loc, self, j);
      Value isNegative = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, shapeValue, zero);
      Value isEqual = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, castIndexToInt(rewriter, loc, dim),
          shapeValue);
      Value isValid = rewriter.create<arith::OrIOp>(loc, isNegative, isEqual);
      rewriter.create<AssertOp>(
          loc, isValid,
          rewriter.getStringAttr(
              "only broadcasting singleton dimensions supported"));
      outShape.push_back(dim);
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }

    Value outTensor =
        rewriter.create<linalg::InitTensorOp>(loc, outShape, elementType);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(inShape.size(), 0, outExpr, context),
        rewriter.getMultiDimIdentityMap(inShape.size())};
    SmallVector<StringRef> iteratorTypes(inShape.size(), "parallel");
    Value result = rewriter
                       .create<linalg::GenericOp>(
                           loc, outTensor.getType(), self, outTensor,
                           indexingMaps, iteratorTypes,
                           [](OpBuilder &b, Location loc, ValueRange args) {
                             b.create<linalg::YieldOp>(loc, args[0]);
                           })
                       .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, result);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenContiguousOp : public OpConversionPattern<AtenContiguousOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenContiguousOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    rewriter.replaceOp(op, adaptor.self());
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenOnesOp : public OpConversionPattern<AtenOnesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOnesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();

    // We ignore device, but add simple asserts for unimplemented kwargs
    if (!op.layout().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op,
                                         "only default layout is supported");

    bool pinMemory = false;
    if (!op.pin_memory().getType().isa<Torch::NoneType>() &&
        !matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory))) {
      return rewriter.notifyMatchFailure(
          op, "pin_memory must be constant bool or None");
    }
    if (pinMemory)
      return rewriter.notifyMatchFailure(op, "memory pinning not supported");

    SmallVector<Value> size, sizeIndex;
    if (!getListConstructElements(op.size(), size)) {
      return rewriter.notifyMatchFailure(
          op, "size must be created by ListConstruct");
    }
    size = getTypeConvertedValues(rewriter, loc, getTypeConverter(), size);
    for (size_t i = 0; i < size.size(); i++)
      sizeIndex.push_back(castIntToIndex(rewriter, loc, size[i]));

    RankedTensorType newResultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    Type outElementType = newResultType.getElementType();

    Value one = rewriter.create<arith::ConstantOp>(
        loc, outElementType,
        (outElementType.isa<mlir::FloatType>()
             ? rewriter.getFloatAttr(outElementType, 1).cast<mlir::Attribute>()
             : rewriter.getIntegerAttr(outElementType, 1)
                   .cast<mlir::Attribute>()));
    Value outTensor = rewriter
                          .create<linalg::InitTensorOp>(
                              loc, sizeIndex, newResultType.getElementType())
                          .getResult();
    Value fillOp =
        rewriter.create<linalg::FillOp>(loc, one, outTensor).getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, fillOp);

    return success();
  }
};
} // namespace

namespace {
class ConvertPrimNumToTensorScalarOp
    : public OpConversionPattern<PrimNumToTensorScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimNumToTensorScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value a = adaptor.a();
    Value outTensor =
        rewriter.create<linalg::InitTensorOp>(loc, ValueRange{}, a.getType())
            ->getResult(0);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, a, outTensor);

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
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenMmOp>();
    patterns.add<ConvertAtenMmOp>(typeConverter, context);
    target.addIllegalOp<AtenMatmulOp>();
    patterns.add<ConvertAtenMatmulOp>(typeConverter, context);
    target.addIllegalOp<AtenBmmOp>();
    patterns.add<ConvertAtenBmmOp>(typeConverter, context);
    target.addIllegalOp<AtenLinearOp>();
    patterns.add<ConvertAtenLinearOp>(typeConverter, context);
    target.addIllegalOp<AtenBatchNormOp>();
    patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
    target.addIllegalOp<
        AtenTanhOp, AtenReluOp, AtenGeluOp, AtenGeluBackwardOp, AtenAddTensorOp,
        AtenMulTensorOp, AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp,
        AtenSigmoidOp, AtenMinimumOp, AtenMaximumOp, AtenToDtypeOp, AtenClampOp,
        AtenRsubScalarOp, AtenLogOp, AtenSqrtOp, AtenFloorOp,
        AtenPowTensorScalarOp, AtenLog2Op, AtenRsqrtOp>();
    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    target.addIllegalOp<AtenUnsqueezeOp>();
    patterns.add<ConvertAtenUnsqueezeOp>(typeConverter, context);
    target.addIllegalOp<AtenConv2dOp>();
    patterns.add<ConvertAtenConv2dOp>(typeConverter, context);
    target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
    patterns.add<ConvertAtenAdaptiveAvgPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenFlattenUsingIntsOp>();
    patterns.add<ConvertAtenFlattenUsingIntsOp>(typeConverter, context);
    target.addIllegalOp<AtenViewOp>();
    patterns.add<ConvertAtenViewOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenSumOp>();
    patterns.add<ConvertReductionOp>(typeConverter, context);
    target.addIllegalOp<AtenTransposeIntOp>();
    patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenPermuteOp>();
    patterns.add<ConvertAtenPermuteOp>(typeConverter, context);
    target.addIllegalOp<AtenCatOp>();
    patterns.add<ConvertAtenCatOp>(typeConverter, context);
    target.addIllegalOp<AtenGatherOp>();
    patterns.add<ConvertAtenGatherOp>(typeConverter, context);
    target.addIllegalOp<AtenLayerNormOp>();
    patterns.add<ConvertAtenLayerNormOp>(typeConverter, context);
    target.addIllegalOp<AtenBroadcastToOp>();
    patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);
    target.addIllegalOp<AtenArgmaxOp>();
    patterns.add<ConvertAtenArgmaxOp>(typeConverter, context);
    target.addIllegalOp<AtenSizeIntOp>();
    patterns.add<ConvertAtenSizeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenEmbeddingOp>();
    patterns.add<ConvertAtenEmbeddingOp>(typeConverter, context);
    target.addIllegalOp<AtenOnesOp>();
    patterns.add<ConvertAtenOnesOp>(typeConverter, context);
    target.addIllegalOp<AtenContiguousOp>();
    patterns.add<ConvertAtenContiguousOp>(typeConverter, context);
    target.addIllegalOp<AtenIntTensorOp>();
    patterns.add<ConvertAtenIntTensorOp>(typeConverter, context);
    target.addIllegalOp<PrimNumToTensorScalarOp>();
    patterns.add<ConvertPrimNumToTensorScalarOp>(typeConverter, context);

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
