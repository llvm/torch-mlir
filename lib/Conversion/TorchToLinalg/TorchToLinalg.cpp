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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;
using namespace mlir::torch::torch_upstream; // For ScalarType and type

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
  Value dimInt =
      b.create<arith::SelectOp>(loc, predDimGEZero, dim, dimAddInputRank);
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
  b.create<cf::AssertOp>(
      loc, predGEZero, b.getStringAttr("dim must be greater or equal to zero"));
  Value predLTInputRank =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, dim, inputRank);
  b.create<cf::AssertOp>(loc, predLTInputRank,
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

static Value getDimOp(OpBuilder &b, Location loc, Value v, int dim) {
  return b.createOrFold<tensor::DimOp>(loc, v, dim);
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
  b.create<cf::AssertOp>(loc, contractingDimEqual,
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

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
static Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                              Type elemTy, Value initElem) {
  Value initTensor = b.create<linalg::InitTensorOp>(loc, sizes, elemTy);
  return b.create<linalg::FillOp>(loc, initElem, initTensor).getResult(0);
}
// Creates a constant of type `elemType` with value `val`.
static Value getConstant(OpBuilder &b, Location loc, int64_t val,
                         Type elemType) {
  Attribute attr = {};
  if (elemType.isa<mlir::FloatType>())
    attr = b.getFloatAttr(elemType, val);
  if (elemType.isa<mlir::IndexType>())
    attr = b.getIndexAttr(val);
  if (elemType.isa<mlir::IntegerType>())
    attr = b.getIntegerAttr(
        elemType, APInt(elemType.cast<IntegerType>().getWidth(), val));
  if (!attr)
    return nullptr;
  return b.create<arith::ConstantOp>(loc, elemType, attr);
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
static Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                             SmallVectorImpl<int64_t> &lowPaddingInts,
                             SmallVectorImpl<int64_t> &highPaddingInts,
                             Value pad) {
  Location loc = op->getLoc();
  Type rankedTensorType = tensor::PadOp::inferResultType(
      input.getType().cast<RankedTensorType>(), lowPaddingInts,
      highPaddingInts);
  SmallVector<OpFoldResult> lowPaddings =
      getAsOpFoldResult(b, loc, lowPaddingInts);
  SmallVector<OpFoldResult> highPaddings =
      getAsOpFoldResult(b, loc, highPaddingInts);
  Value paddedInput = tensor::createPadScalarOp(
      rankedTensorType, input, pad, /*low=*/lowPaddings, /*high=*/highPaddings,
      /*packing=*/false, loc, b);
  return paddedInput;
}

// Helper function to get the padding tensor given the padding int values.
// It's assumed that the padding on the low end and high end are the same,
// and that zero padding is required.
static Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                             SmallVectorImpl<int64_t> &paddingInts) {
  assert(input.getType().isa<RankedTensorType>() &&
         "input must be RankedTensorType");
  Location loc = op->getLoc();
  Value c0 = b.create<arith::ConstantOp>(
      loc,
      b.getZeroAttr(input.getType().cast<RankedTensorType>().getElementType()));
  return getPaddedTensor(op, b, input, paddingInts, paddingInts, c0);
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
    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value groupEqual1 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, groups, c1);
    rewriter.create<cf::AssertOp>(
        loc, groupEqual1, rewriter.getStringAttr("expect groups to be 1"));

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

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, F, Hout, Wout}, elementType);

    Value bias = adaptor.bias();
    Value biasInitTensor;
    if (bias.getType().isa<Torch::NoneType>()) {
      Value c0float = rewriter.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, 0.0));
      biasInitTensor = rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
                           .getResult(0);
    } else {
      auto biasType = bias.getType().cast<RankedTensorType>();
      if (biasType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "expect bias to be rank 1");
      if (elementType != biasType.getElementType())
        return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");

      auto resultRank = initTensor.getType().cast<RankedTensorType>().getRank();
      SmallVector<AffineMap> indexingMaps = {
          // bias is used to initialize the channels - dimension 1 of output
          AffineMap::get(/*dimCount=*/resultRank, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(1), context),
          rewriter.getMultiDimIdentityMap(resultRank)};
      SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
      biasInitTensor = rewriter
                           .create<linalg::GenericOp>(
                               loc, initTensor.getType(), bias, initTensor,
                               indexingMaps, iteratorTypes,
                               [](OpBuilder &b, Location loc, ValueRange args) {
                                 b.create<linalg::YieldOp>(loc, args[0]);
                               })
                           .getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
    Value conv2d =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, biasInitTensor.getType(), ValueRange{paddedInput, weight},
                biasInitTensor, stridesAttr, dilationAttr)
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
  b.create<cf::AssertOp>(
      loc, indexLTInputDim,
      b.getStringAttr("index must be smaller than dim size"));

  // Assert index >= 0
  Value cst0 = b.create<arith::ConstantOp>(loc, b.getZeroAttr(index.getType()));
  Value indexGEThanZero =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, cst0);
  b.create<cf::AssertOp>(loc, indexGEThanZero,
                         b.getStringAttr("index must be larger or equal to 0"));

  Value extract = b.create<tensor::ExtractOp>(loc, input, indices);
  b.create<linalg::YieldOp>(loc, extract);
}

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
class ConvertAtenNativeLayerNormOp
    : public OpConversionPattern<AtenNativeLayerNormOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNativeLayerNormOp op, OpAdaptor adaptor,
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
    Type layerNormResultType = getTypeConverter()->convertType(op.getType(0));
    Type meanResultType = getTypeConverter()->convertType(op.getType(1));
    Type varResultType = getTypeConverter()->convertType(op.getType(2));
    Value layerNorm_ =
        rewriter.create<tensor::CastOp>(loc, layerNormResultType, layerNorm);
    Value mean_ = rewriter.create<tensor::CastOp>(loc, meanResultType, mean);
    Value var_ = rewriter.create<tensor::CastOp>(loc, varResultType, var);
    rewriter.replaceOp(op, {layerNorm_, mean_, var_});
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
    rewriter.create<cf::AssertOp>(
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
                  loc, initTensor0.getType(), ValueRange{lhs, rhs}, initTensor0,
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

// Given `input`, `target`, `nll_loss_forward` is given by:
//   for i in range(0, len(target)):
//     indi = target[i];
//     nll_loss_forward[i] = -(input[i][indi]);
// TODO: `weight` and `reduction` operands are still to be taken care of.
namespace {
class ConvertAtenNllLossForwardOp
    : public OpConversionPattern<AtenNllLossForwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNllLossForwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value input = adaptor.self();
    Value target = adaptor.target();
    Value weight = adaptor.weight();

    int64_t reduce_dim;
    if (!matchPattern(op.reduction(), m_TorchConstantInt(&reduce_dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    // TODO: Handle reduction.
    if (reduce_dim != 0)
      return rewriter.notifyMatchFailure(
          op, "reduction along dimensions is not supported.");

    // TODO: Incorporate the weight argument.
    if (!weight.getType().isa<mlir::torch::Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.ignore_index();
    Value ignoreIndexVal = castIntToIndex(rewriter, loc, ignoreIndex);

    unsigned inputRank = input.getType().cast<RankedTensorType>().getRank();
    unsigned targetRank = target.getType().cast<RankedTensorType>().getRank();

    // TODO: Cases with targetRank != 1 where `Mean` reduction is required.
    if (inputRank != 2 || targetRank != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected  input and target to be rank 2 and 1 respectively");
    }
    RankedTensorType resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

    Type elementType = resultType.getElementType();

    Value targetDim = getDimOp(rewriter, loc, target, 0);
    Value initTensor0 =
        createZeroInitTensor(rewriter, loc, {targetDim}, elementType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    SmallVector<AffineExpr> targetExpr;
    targetExpr.push_back(rewriter.getAffineDimExpr(0));
    SmallVector<StringRef> iteratorTypes{getParallelIteratorTypeName()};
    auto indexingMaps = AffineMap::inferFromExprList({targetExpr, targetExpr});
    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor0.getType(), ValueRange{target}, initTensor0,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value indTarget = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), args[0]);
                  Value indI = rewriter.create<linalg::IndexOp>(loc, 0);

                  // The final result is given by:
                  // final_res = (indI == ignoreIndexVal) ? 0 :
                  // input[indI][IndTarget]
                  Value cmpEq = rewriter.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, indI, ignoreIndexVal);
                  Value result = rewriter.create<tensor::ExtractOp>(
                      loc, input, ValueRange{indI, indTarget});
                  Value negate =
                      rewriter.create<arith::NegFOp>(loc, elementType, result);
                  Value selectFinal = rewriter.create<arith::SelectOp>(
                      loc, cmpEq, zeroVal, negate);
                  b.create<linalg::YieldOp>(loc, selectFinal);
                })
            .getResult(0);

    // TODO: Update the second result tensor.
    Value weightUpdated =
        createZeroInitTensor(rewriter, loc, {}, elementType);
    rewriter.replaceOp(op, {finalRes, weightUpdated});
    return success();
  }
};
} // namespace

// Given `grad_output`, `input`, `target`, `nll_loss_backward` is given by:
//   for i in range(0, len(input[0])):
//      for j in range(0, len(input[1])):
//          nll_loss_backward[i][j] = (j == target[i]) ? -grad_output[i] : 0
// TODO: `weight` and `reduction` operands are still to be taken care of.
namespace {
class ConvertAtenNllLossBackwardOp
    : public OpConversionPattern<AtenNllLossBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNllLossBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value input = adaptor.self();
    Value target = adaptor.target();
    Value weight = adaptor.weight();
    Value gradOutput = adaptor.grad_output();

    int64_t reduction;
    if (!matchPattern(op.reduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");

    // TODO: Handle reduction.
    if (reduction != Reduction::None)
      return rewriter.notifyMatchFailure(
          op, "reduction along dimensions is not supported.");

    // TODO: Incorporate the weight argument.
    if (!weight.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    Value ignoreIndex = adaptor.ignore_index();
    Value ignoreIndexVal = castIntToIndex(rewriter, loc, ignoreIndex);

    unsigned inputRank = input.getType().cast<RankedTensorType>().getRank();
    unsigned targetRank = target.getType().cast<RankedTensorType>().getRank();

    // TODO: Cases with targetRank != 1 where `Mean` or `Sum` reduction is
    // required.
    if (inputRank != 2 || targetRank != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected  input and target to be rank 2 and 1 respectively");
    }
    RankedTensorType resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();

    Type elementType = resultType.getElementType();

    // Given there is no reduction `grad_input` size is equal to `input` size.
    auto outputSize = getTensorSizes(rewriter, loc, input);
    Value initTensor0 =
        createZeroInitTensor(rewriter, loc, outputSize, elementType);
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));

    SmallVector<AffineExpr> targetExpr{rewriter.getAffineDimExpr(0)};
    SmallVector<AffineExpr> resultExpr{rewriter.getAffineDimExpr(0),
                                       rewriter.getAffineDimExpr(1)};
    SmallVector<StringRef> iteratorTypes{getParallelIteratorTypeName(),
                                         getParallelIteratorTypeName()};
    auto indexingMaps =
        AffineMap::inferFromExprList({targetExpr, targetExpr, resultExpr});
    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor0.getType(), ValueRange{target, gradOutput},
                initTensor0,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value indTarget = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), args[0]);
                  Value indJ = rewriter.create<linalg::IndexOp>(loc, 1);

                  // The final result is given by:
                  // grad_input[i][j] = (j == target[i]) ? -grad_output[i] : 0
                  Value cmpEq = rewriter.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::eq, indJ, indTarget);

                  // The target index shouldn't be equal to `ignoreIndex`.
                  Value cmpNe = rewriter.create<arith::CmpIOp>(
                      loc, arith::CmpIPredicate::ne, ignoreIndexVal, indTarget);
                  Value finalPredicate =
                      rewriter.create<arith::AndIOp>(loc, cmpEq, cmpNe);
                  Value negate =
                      rewriter.create<arith::NegFOp>(loc, elementType, args[1]);
                  Value selectFinal = rewriter.create<arith::SelectOp>(
                      loc, finalPredicate, negate, zeroVal);
                  b.create<linalg::YieldOp>(loc, selectFinal);
                })
            .getResult(0);

    rewriter.replaceOp(op, finalRes);
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

    if (inputType.getRank() != 2 && inputType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected  input to be rank 2 or rank 3");
    }

    // Only handle the case of rank 2 `weight` for now.
    // TODO: Insert the appropriate reshape to collapse any leading dimensions.
    if (weightType.getRank() != 2 || biasType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected weight to be rank 2 and bias to be rank 1");
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

    Value batchDim = nullptr;
    int restDim = 0;
    if (inputType.getRank() == 3) {
      batchDim = getDimOp(rewriter, loc, input, 0);
      restDim = 1;
    }

    Value inputDim0 = getDimOp(rewriter, loc, input, restDim + 0);
    Value inputDim1 = getDimOp(rewriter, loc, input, restDim + 1);
    Value weightDim0 = getDimOp(rewriter, loc, weight, 0);
    Value weightDim1 = getDimOp(rewriter, loc, weight, 1);
    Value biasDim0 = getDimOp(rewriter, loc, bias, 0);
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, inputDim1, weightDim1);
    rewriter.create<cf::AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for aten.linear"));
    // Here we take advantage of ruling out the size-1 case above.
    // In the static-size-1 case, we will not emit this check at all.
    Value biasSizeCorrect = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, weightDim0, biasDim0);
    rewriter.create<cf::AssertOp>(
        loc, biasSizeCorrect,
        rewriter.getStringAttr("mismatching bias size for aten.linear"));

    Value initTensor;
    SmallVector<AffineMap> broadcastIndexingMaps;
    Value transposedWeightInitTensor;
    if (inputType.getRank() > 2) {
      initTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{batchDim, inputDim0, weightDim0},
          inputType.getElementType());
      transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{batchDim, weightDim1, weightDim0},
          weightType.getElementType());
      broadcastIndexingMaps = {
          AffineMap::get(
              /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
              {rewriter.getAffineDimExpr(1 + restDim)}, context),
          rewriter.getMultiDimIdentityMap(inputType.getRank())};
    } else {
      initTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{inputDim0, weightDim0},
          inputType.getElementType());
      transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{weightDim1, weightDim0}, weightType.getElementType());
      broadcastIndexingMaps = {
          AffineMap::get(
              /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
              {rewriter.getAffineDimExpr(1)}, context),
          rewriter.getMultiDimIdentityMap(inputType.getRank())};
    }

    SmallVector<StringRef> iteratorTypes(inputType.getRank(), "parallel");
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
            /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(1 + restDim),
             rewriter.getAffineDimExpr(0 + restDim)},
            context),
        rewriter.getMultiDimIdentityMap(inputType.getRank())};
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
    Value matmul;
    if (batchDim)
      matmul = rewriter
                   .create<linalg::BatchMatmulOp>(
                       loc, broadcasted.getType(),
                       ValueRange{input, transposedWeights}, broadcasted)
                   .getResult(0);
    else
      matmul = rewriter
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
      dtype.isSignlessInteger(1)) {
    // TODO: Handle to-boolean conversion(from-boolean conversion is handled).
    mlir::emitError(loc)
        << "unsupported byte, char or bool type for convertScalarToDtype "
        << scalarType << "(scalar type) -> " << dtype << "(dtype)";
    return nullptr;
  }

  if (auto dtypeFloat = dtype.dyn_cast<mlir::FloatType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>()) {
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, dtype, scalar);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return b.create<arith::ExtFOp>(loc, dtype, scalar);
    }
    assert(scalarType.isa<mlir::IntegerType>());
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::UIToFPOp>(loc, dtype, scalar);
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::SIToFPOp>(loc, dtype, scalar);
  }

  if (auto dtypeInteger = dtype.dyn_cast<mlir::IntegerType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>())
      return b.create<arith::FPToSIOp>(loc, dtype, scalar);
    assert(scalarType.isa<mlir::IntegerType>());
    auto scalarInteger = scalarType.cast<mlir::IntegerType>();
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return b.create<arith::TruncIOp>(loc, dtype, scalar);
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::ExtUIOp>(loc, dtype, scalar);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::ExtSIOp>(loc, dtype, scalar);
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
  if (isa<AtenCeilOp>(op))
    return b.create<math::CeilOp>(loc, payloadArgs[0]);
  if (isa<AtenLogOp>(op))
    return b.create<math::LogOp>(loc, payloadArgs[0]);
  if (isa<AtenSqrtOp>(op))
    return b.create<math::SqrtOp>(loc, payloadArgs[0]);
  if (isa<AtenRsqrtOp>(op))
    return b.create<math::RsqrtOp>(loc, payloadArgs[0]);
  if (auto clone = dyn_cast<AtenCloneOp>(op)) {
    if (!clone.memory_format().getType().isa<Torch::NoneType>()) {
      clone.emitError("unimplemented: only default memory format is supported");
      return nullptr;
    }
    return payloadArgs[0];
  }
  if (auto bitwiseAndTensor = dyn_cast<AtenBitwiseAndTensorOp>(op)) {
    if (bitwiseAndTensor.getType()
            .cast<ValueTensorType>()
            .getDtype()
            .isa<mlir::FloatType>()) {
      bitwiseAndTensor.emitError(
          "Bitwise_And does not support floating point dtype");
      return nullptr;
    }
    Type dtype = converter->convertType(bitwiseAndTensor.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::AndIOp>(loc, lhs, rhs);
  }
  if (isa<AtenLog2Op>(op))
    return b.create<math::Log2Op>(loc, payloadArgs[0]);
  if (isa<AtenAbsOp>(op))
    return b.create<math::AbsOp>(loc, payloadArgs[0]);
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
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    return b.create<arith::SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  if (auto lrelu = dyn_cast<AtenLeakyReluOp>(op)) {
    if (!lrelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      lrelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    Value positivePart =
        b.create<arith::SelectOp>(loc, pred, payloadArgs[0], constZero);
    Value negativePart =
        b.create<arith::SelectOp>(loc, pred, constZero, payloadArgs[0]);
    Value scale = convertScalarToDtype(b, loc, operands[1], elementType);
    Value scaledNegativePart = b.create<arith::MulFOp>(loc, negativePart, scale);
    return b.create<arith::AddFOp>(loc, positivePart, scaledNegativePart);
  }
  if (auto gelu = dyn_cast<AtenGeluOp>(op)) {
    if (!gelu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      gelu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    // TODO: Take approximation into account.
    std::string approximate;
    if (!matchPattern(gelu.approximate(), m_TorchConstantStr(approximate)) ||
        approximate != "none")
      return nullptr;
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
    // TODO: Take approximation into account.
    std::string approximate;
    if (!matchPattern(geluBackward.approximate(),
                      m_TorchConstantStr(approximate)) ||
        approximate != "none")
      return nullptr;
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
  if (auto subScalar = dyn_cast<AtenSubScalarOp>(op)) {
    Type dtype = converter->convertType(subScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::SubFOp>(loc, self, mult);
    } else if (dtype.isa<mlir::IntegerType>()) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::SubIOp>(loc, self, mult);
    }
    subScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto addScalar = dyn_cast<AtenAddScalarOp>(op)) {
    Type dtype = converter->convertType(addScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value self = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    Value alpha = convertScalarToDtype(b, loc, operands[2], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      Value mult = b.create<arith::MulFOp>(loc, other, alpha);
      return b.create<arith::AddFOp>(loc, self, mult);
    } else if (dtype.isa<mlir::IntegerType>()) {
      Value mult = b.create<arith::MulIOp>(loc, other, alpha);
      return b.create<arith::AddIOp>(loc, self, mult);
    }
    addScalar.emitError("unimplemented: dtype other than float and integer "
                        "types are not supported.");
    return nullptr;
  }
  if (auto mul = dyn_cast<AtenMulTensorOp>(op)) {
    AtenMulTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(mul.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    if (dtype.isa<mlir::FloatType>()) {
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    } else {
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    }
  }
  if (auto gtTensor = dyn_cast<AtenGtTensorOp>(op)) {
    AtenGtTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      gtTensor.emitError("unimplemented: different lhs and rhs dtype");
      return nullptr;
    }

    Type elementalType =
        gtTensor.self().getType().cast<BaseTensorType>().getDtype();

    if (elementalType.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                     payloadArgs[0], payloadArgs[1]);
    if (IntegerType intType = elementalType.dyn_cast<mlir::IntegerType>()) {
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                       payloadArgs[0], payloadArgs[1]);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       payloadArgs[0], payloadArgs[1]);
    }
    gtTensor.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }
  if (auto eqTensor = dyn_cast<AtenEqTensorOp>(op)) {
    AtenEqTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      eqTensor.emitError("unimplemented: lhs and rhs dtype must be same");
      return nullptr;
    }

    Type elementalType =
        eqTensor.self().getType().cast<BaseTensorType>().getDtype();

    if (elementalType.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                     payloadArgs[0], payloadArgs[1]);
    if (elementalType.isa<mlir::IntegerType>()) {
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                     payloadArgs[0], payloadArgs[1]);
    }
    eqTensor.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }
  if (auto ltTensor = dyn_cast<AtenLtTensorOp>(op)) {
    AtenLtTensorOp::Adaptor adaptor(operands);
    Type lhsDtype = payloadArgs[0].getType();
    Type rhsDtype = payloadArgs[1].getType();

    // TODO: Type promotion in case of different `lhsDtype` and `rhsDtype` needs
    // to be handled.
    if (lhsDtype != rhsDtype) {
      ltTensor.emitError("unimplemented: lhs and rhs dtype must be same");
      return nullptr;
    }

    Type elementalType =
        ltTensor.self().getType().cast<BaseTensorType>().getDtype();

    if (elementalType.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                     payloadArgs[0], payloadArgs[1]);
    if (IntegerType intType = elementalType.dyn_cast<mlir::IntegerType>()) {
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                       payloadArgs[0], payloadArgs[1]);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       payloadArgs[0], payloadArgs[1]);
    }
    ltTensor.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }
  if (auto div = dyn_cast<AtenDivTensorOp>(op)) {
    AtenDivTensorOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(div.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>())
      div.emitError("unimplemented: non-floating point dtype");
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    return b.create<arith::DivFOp>(loc, lhs, rhs);
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

  if (auto gtScalar = dyn_cast<AtenGtScalarOp>(op)) {
    Type dtype = gtScalar.self().getType().cast<BaseTensorType>().getDtype();

    // TODO: `gtTensor` and `gtScalar` share similar code and can be called from
    // one static function.
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor args from integer to float.
        gtScalar.emitError(
            "unimplemented: type promotion from tensor to scalar.");
        return nullptr;
      }

      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                       payloadArgs[0], otherPromoted);
    }
    gtScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto geScalar = dyn_cast<AtenGeScalarOp>(op)) {
    Type dtype = geScalar.self().getType().cast<BaseTensorType>().getDtype();

    // TODO: The `AtenGeScalarOp` and `AtenGtScalarOp` share a lot of code that
    // can be refactored.
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGE,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor args from integer to float.
        geScalar.emitError(
            "unimplemented: type promotion from tensor to scalar.");
        return nullptr;
      }

      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                       payloadArgs[0], otherPromoted);
    }
    geScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto eqScalar = dyn_cast<AtenEqScalarOp>(op)) {
    Type dtype = eqScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ,
                                     payloadArgs[0], otherPromoted);
    if (dtype.isa<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        eqScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                     payloadArgs[0], otherPromoted);
    }
    eqScalar.emitError("unimplemented: dtype isn't supported");
    return nullptr;
  }

  if (auto ltScalar = dyn_cast<AtenLtScalarOp>(op)) {
    Type dtype = ltScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO:  Both tensor and scalar variants of `aten.gt` and `aten.lt` share a
    // lot of code that can be refactored.
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULT,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        ltScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                       payloadArgs[0], otherPromoted);
    }
    ltScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto leScalar = dyn_cast<AtenLeScalarOp>(op)) {
    Type dtype = leScalar.self().getType().cast<BaseTensorType>().getDtype();
    Value otherPromoted =
        convertScalarToDtype(b, loc, operands[1], payloadArgs[0].getType());

    // TODO: The `AtenLeScalarOp` and `AtenLtScalarOp` share a lot of code that
    // can be refactored.
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE,
                                     payloadArgs[0], otherPromoted);
    if (IntegerType intType = dtype.dyn_cast<mlir::IntegerType>()) {
      if (!operands[1].getType().isa<mlir::IntegerType>()) {
        // TODO: Promote tensor operand from integer to float.
        leScalar.emitError(
            "unimplemented: type promotion from tensor to scalar");
        return nullptr;
      }
      if (intType.isUnsigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                       payloadArgs[0], otherPromoted);
      if (intType.isSigned())
        return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                       payloadArgs[0], otherPromoted);
    }
    leScalar.emitError("unimplemented: dtype isn't supported.");
    return nullptr;
  }

  if (auto whereSelf = dyn_cast<AtenWhereSelfOp>(op)) {
    Type dtype = converter->convertType(whereSelf.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value rhs = convertScalarToDtype(b, loc, payloadArgs[2], dtype);
    return b.create<arith::SelectOp>(loc, payloadArgs[0], lhs, rhs);
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
    return b.create<arith::SelectOp>(loc, pred, payloadArgs[0], payloadArgs[1]);
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
    return b.create<arith::SelectOp>(loc, pred, payloadArgs[0], payloadArgs[1]);
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
      result = b.create<arith::SelectOp>(loc, pred, minPromoted, result);
    }
    if (!max.getType().isa<Torch::NoneType>()) {
      auto maxPromoted = convertScalarToDtype(b, loc, max, dtype);
      auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                          result, maxPromoted);
      result = b.create<arith::SelectOp>(loc, pred, maxPromoted, result);
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
    Value lhs = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value rhs = convertScalarToDtype(b, loc, operands[1], dtype);
    if (dtype.isa<mlir::FloatType>())
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    if (dtype.isa<mlir::IntegerType>())
      return b.create<arith::MulIOp>(loc, lhs, rhs);
    mulScalar.emitError("unimplemented: Only integer/float dtype supported");
    return nullptr;
  }
  if (auto atenToDtype = dyn_cast<AtenToDtypeOp>(op)) {
    Value input = payloadArgs[0];
    Type dtype = converter->convertType(atenToDtype.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    Value result = convertScalarToDtype(b, loc, input, dtype);
    return result;
  }
  if (auto divScalar = dyn_cast<AtenDivScalarOp>(op)) {
    Type dtype = converter->convertType(divScalar.getType())
                     .cast<RankedTensorType>()
                     .getElementType();
    if (!dtype.isa<mlir::FloatType>()) {
      divScalar.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Value self = payloadArgs[0];
    Value other = convertScalarToDtype(b, loc, operands[1], dtype);
    return b.create<arith::DivFOp>(loc, self, other);
  }
  if (auto reciprocal = dyn_cast<AtenReciprocalOp>(op)) {
    if (!reciprocal.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      reciprocal.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }

    Type elementType = payloadArgs[0].getType();
    // assert(element != 0)
    auto zero =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    auto pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                        payloadArgs[0], zero);
    b.create<cf::AssertOp>(
        loc, pred, b.getStringAttr("unimplemented: tensor with zero element"));

    auto one =
        b.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 1.0));
    return b.create<arith::DivFOp>(loc, one, payloadArgs[0]);
  }
  if (auto thresholdOp = dyn_cast<AtenThresholdOp>(op)) {
    // The approach used here is as follows:
    //        result = self <= threshold ? value : self
    AtenThresholdOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(thresholdOp.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value self = payloadArgs[0];
    Value threshold = convertScalarToDtype(b, loc, adaptor.threshold(), dtype);
    Value value = convertScalarToDtype(b, loc, adaptor.value(), dtype);

    Value predicate;
    if (dtype.isa<mlir::FloatType>())
      predicate = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, self,
                                          threshold);
    else
      predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, self,
                                          threshold);
    return b.create<arith::SelectOp>(loc, predicate, value, self);
  }
  if (auto thresholdBackward = dyn_cast<AtenThresholdBackwardOp>(op)) {
    // The approach used here is as follows:
    //        result = self <= threshold ? 0 : grad
    AtenThresholdBackwardOp::Adaptor adaptor(operands);
    Type dtype = converter->convertType(thresholdBackward.getType())
                     .cast<RankedTensorType>()
                     .getElementType();

    Value grad = convertScalarToDtype(b, loc, payloadArgs[0], dtype);
    Value self = convertScalarToDtype(b, loc, payloadArgs[1], dtype);
    Value threshold = convertScalarToDtype(b, loc, adaptor.threshold(), dtype);
    Value constantZero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(dtype));

    Value predicate;
    if (dtype.isa<mlir::FloatType>())
      predicate = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ULE, self,
                                          threshold);
    else
      predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, self,
                                          threshold);
    return b.create<arith::SelectOp>(loc, predicate, constantZero, grad);
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
  if (isa<AtenMaxOp>(op) && elementType.isa<mlir::FloatType>())
    return b.create<arith::ConstantOp>(
        loc, b.getFloatAttr(
                 elementType,
                 APFloat::getLargest(
                     elementType.cast<mlir::FloatType>().getFloatSemantics(),
                     /*Negative=*/true)));

  op->emitError("unimplemented lowering in "
                "createLinalgNeutralElementForReduceOp");
  return nullptr;
}

static Value createLinalgPayloadCalculationForReduceOp(
    OpBuilder &b, Location loc, ValueRange payloadArgs, Operation *op,
    ArrayRef<Value> operands, Type resultElementType) {
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op) &&
      resultElementType.isa<mlir::FloatType>()) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    return b.create<arith::AddFOp>(loc, self, result);
  } else if (isa<AtenMaxOp>(op) && resultElementType.isa<mlir::FloatType>()) {
    Value self =
        convertScalarToDtype(b, loc, payloadArgs[0], resultElementType);
    Value result = payloadArgs[1];
    return b.create<arith::MaxFOp>(loc, self, result);
  }
  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForReduceOp");
  return nullptr;
}

namespace {
// Aten maxdim lowering represents the MaxDim op as an linalg.indexed_generic
// op, producing two output buffers.
//
// The first output buffer contains the maximum value found. It is initialized
// to the minimum representable value of the input element type.
//
// The second output buffer contains the index of the found maximum value. It is
// initialized to 0 and is resulting integer type.
//
// The indexed_generic op updates both the maximum value and index if the
// current value exceeds the running max.
class ConvertAtenMaxDimOp : public OpConversionPattern<AtenMaxDimOp> {
public:
  using OpConversionPattern<AtenMaxDimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenMaxDimOp maxDimOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = maxDimOp.getLoc();
    Value input = adaptor.self();
    RankedTensorType valResultType =
        getTypeConverter()
            ->convertType(maxDimOp.getResult(0).getType())
            .cast<RankedTensorType>();
    RankedTensorType idxResultType =
        getTypeConverter()
            ->convertType(maxDimOp.getResult(1).getType())
            .cast<RankedTensorType>();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type idxElementType = idxResultType.getElementType();
    if (!idxElementType.isa<IntegerType>())
      return rewriter.notifyMatchFailure(
          maxDimOp,
          "aten.max_dim to linalg.* requires integer-like result type");

    bool keepDim = false;
    if (!matchPattern(maxDimOp.keepdim(), m_TorchConstantBool(&keepDim)))
      return failure();

    int64_t dim;
    if (!matchPattern(maxDimOp.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          maxDimOp, "aten.max_dim to linalg.* requires int value for Dim");
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank()))
      return rewriter.notifyMatchFailure(maxDimOp, "dim is not a valid dim");

    Type inElementType = inputType.getElementType();
    if (!inElementType.isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          maxDimOp,
          "aten.max_dim to linalg.* requires Float input element type");
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
        createZeroInitTensor(rewriter, loc, resultShape, idxElementType);

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
        ArrayRef<Type>({filledTensorMax.getType(), filledTensorIdx.getType()}),
        input, ValueRange({filledTensorMax, filledTensorIdx}), maps,
        iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value newValue = blockArgs[0];
          Value oldValue = blockArgs[1];
          Value oldIndex = blockArgs[2];

          Value newIndex = rewriter.create<arith::IndexCastOp>(
              nestedLoc, oldIndex.getType(),
              rewriter.create<linalg::IndexOp>(loc, dim));

          Value predicate;
          if (inElementType.isa<mlir::FloatType>())
            predicate = rewriter.create<arith::CmpFOp>(
                nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);
          auto resultMax = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newValue, oldValue);
          auto resultIndex = rewriter.create<arith::SelectOp>(
              nestedLoc, predicate, newIndex, oldIndex);
          nestedBuilder.create<linalg::YieldOp>(
              nestedLoc, ValueRange({resultMax, resultIndex}));
        });

    // This cast is required to fix the shape in the case of keepDim=True
    Value maxValuesCast = rewriter.create<tensor::CastOp>(
        loc, valResultType, linalgOp.getResult(0));
    Value maxIdxCast = rewriter.create<tensor::CastOp>(loc, idxResultType,
                                                       linalgOp.getResult(1));
    rewriter.replaceOp(maxDimOp, {maxValuesCast, maxIdxCast});
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
    if (!isa<AtenTanhOp, AtenReluOp, AtenLeakyReluOp, AtenGeluOp,
             AtenGeluBackwardOp, AtenAddTensorOp, AtenMulTensorOp,
             AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp, AtenSigmoidOp,
             AtenExpOp, AtenMinimumOp, AtenMaximumOp, AtenToDtypeOp,
             AtenClampOp, AtenRsubScalarOp, AtenMulScalarOp, AtenLogOp,
             AtenSqrtOp, AtenFloorOp, AtenPowTensorScalarOp, AtenLog2Op,
             AtenRsqrtOp, AtenDivScalarOp, AtenAbsOp, AtenReciprocalOp,
             AtenBitwiseAndTensorOp, AtenGtScalarOp, AtenGeScalarOp,
             AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp, AtenWhereSelfOp,
             AtenCeilOp, AtenGtTensorOp, AtenEqTensorOp, AtenLtTensorOp,
             AtenSubScalarOp, AtenAddScalarOp, AtenThresholdOp,
             AtenThresholdBackwardOp, AtenCloneOp>(op))
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
            getDimOp(rewriter, loc, tensorOperand, size.index());

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
        rewriter.create<cf::AssertOp>(loc, equalToRunning,
                                      "mismatched size for broadcast");
      }
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, getContext()));
    }

    SmallVector<StringRef> iteratorTypes(resultRank,
                                         getParallelIteratorTypeName());
    // Add the indexing map for the outs init tensor.
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, getAsOpFoldResult(resultShape), resultType.getElementType());
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
    if (isa<AtenSumOp>(op) || isa<AtenMaxOp>(op)) {
      auto tensorOperand = operands[0];
      auto inputType = tensorOperand.getType().cast<RankedTensorType>();

      // `AtenSumOp` and `AtenMaxOp` reduces along all the dimensions of the
      // input tensor.
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
    rewriter.create<cf::AssertOp>(
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
class ConvertAtenConstantPadNdOp
    : public OpConversionPattern<AtenConstantPadNdOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConstantPadNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    auto type = self.getType().cast<RankedTensorType>();
    int64_t rank = type.getRank();

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t> padInts;
    if (!matchPattern(op.pad(), m_TorchConstantIntList(padInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad ranges");
    uint64_t padRank = padInts.size() / 2;
    if (padRank * 2 != padInts.size())
      return rewriter.notifyMatchFailure(op, "pad range size is not even");
    if (rank < 0 || padRank > (uint64_t)rank)
      return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

    // Initialize low/high paddings with the dims that should not be padded.
    SmallVector<int64_t, 4> lowPadding(/*Size=*/rank - padRank, /*Value=*/0);
    SmallVector<int64_t, 4> highPadding(/*Size=*/rank - padRank, /*Value=*/0);
    // Add the requested padding - note op.pad() is highest dim first ordered
    // pairs of low,high.
    for (uint64_t i = padRank; i > 0; --i) {
      lowPadding.push_back(padInts[i * 2 - 2]);
      highPadding.push_back(padInts[i * 2 - 1]);
    }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<RankedTensorType>().getElementType();
    Value castedValue =
        convertScalarToDtype(rewriter, loc, adaptor.value(), elementType);
    Value paddedInput = getPaddedTensor(op, rewriter, self, lowPadding,
                                        highPadding, castedValue);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, paddedInput);
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
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
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
    Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
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
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t inputRank = inputType.getRank();
    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();
    // Currently, we only handle the expanding OR collapsing cases, we do not
    // handle expanding And collapsing happening at the same time or cases where
    // it's neither collapsing nor expanding like view of [2,3] for 3x2 tensor.
    // TODO: For the expanding And collapsing case, we will need to identify
    // which dimensions are collapsing and which are expanding and do it in two
    // steps.
    // TODO: For neither collapsing nor expanding, we could find a intermediate
    // shape to collapse and then expanded to the target shape. Like [2,3] =>
    // [6] => [3, 2].
    if (inputRank == resultRank)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the view op is neither expanding nor collapsing");

    if (resultRank == 0)
      return rewriter.notifyMatchFailure(op,
                                         "result shape of rank 0 is invalid");

    // TODO: add support for case inputRank 0 expanded to size 1
    if (inputRank == 0)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input rank 0 is not supported");

    bool isCollapse = inputRank > resultRank ? true : false;
    int64_t collapsedRank = isCollapse ? resultRank : inputRank;
    int64_t expandedRank = isCollapse ? inputRank : resultRank;

    // Extract the desired output size as a list of integers. This list should
    // have been created using the operation `torch.prim.ListConstruct`.
    SmallVector<Value> outputSizeTorchInt;
    if (!getListConstructElements(op.size(), outputSizeTorchInt)) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: the target size is "
                                         "not constructed from ListConstruct");
    }
    SmallVector<Value> outputSizeInt = getTypeConvertedValues(
        rewriter, loc, typeConverter, outputSizeTorchInt);
    if (resultRank != (int64_t)outputSizeInt.size()) {
      return rewriter.notifyMatchFailure(
          op, "desired size list length mismatches with the result type rank");
    }
    SmallVector<Value> inputSizeTorchInt = getTensorSizes(rewriter, loc, input);
    ArrayRef<Value> expandedShapeTorchInt =
        llvm::makeArrayRef(isCollapse ? inputSizeTorchInt : outputSizeInt);
    ArrayRef<Value> collapsedShapeTorchInt =
        llvm::makeArrayRef(isCollapse ? outputSizeInt : inputSizeTorchInt);

    // Iterate through the view op size list to do the following:
    //
    // 1. Combine output size list and input tensor type info to get the most
    // static outputShape.
    //
    // 2. Fill in the reassociation for size list item where the output dim size
    // is got from `torch.aten.size.int(inputTensor, inputDim)`. We naively
    // assume this means the corresponding dimension is not expanded or
    // collapsed. Note this may technically not always be true.
    // TODO: think of a way better way to at least detect when this assumption
    // is violated.
    SmallVector<int64_t> outputShape(resultRank, kUnknownSize);
    SmallVector<ReassociationIndices> reassociation(collapsedRank);
    for (auto en : llvm::enumerate(outputSizeTorchInt)) {
      int64_t inputDim;
      int64_t outputDim = en.index();
      // Match torch.aten.size.int(inputTensor, inputDim) with constant inputDim
      if (matchPattern(en.value(),
                       m_TorchTensorSizeInt(op.self(), &inputDim))) {
        auto collapsedDim = isCollapse ? outputDim : inputDim;
        auto expandedDim = isCollapse ? inputDim : outputDim;
        reassociation[collapsedDim].push_back(expandedDim);
        if (!inputType.isDynamicDim(inputDim)) {
          outputShape[outputDim] = inputShape[inputDim];
          continue;
        }
      }

      int64_t size;
      if (matchPattern(en.value(), m_TorchConstantInt(&size)))
        outputShape[outputDim] = size;
    }

    SmallVector<int64_t> collapsedShape =
        isCollapse ? outputShape : llvm::to_vector(inputShape);
    SmallVector<int64_t> expandedShape =
        isCollapse ? llvm::to_vector(inputShape) : outputShape;

    // The while loop does the following:
    // 1. Fill in the reassociation indices for dimensions that are expanded.
    // Check the interval dimensions between two unchanged dims in the
    // collapsedShape. If the interval is size 1, associate all the dims
    // in the expandedShape shape until the next unchanged dim. If the interval
    // is larger than size 1, figure out the associations with assumptions that
    // dynamic dimensions are not splitted.
    // 2. Set collapsedShape and expandedShape following the requirements by
    // tensor.expand_shape verification code:
    //    a. As long as one or more of the related dimensions in the expanded
    //    shape is dynamic the collapsed dimension is dynamic.
    //    b. If all of the related dimensions are static, the collapsed
    //    dimension must be static. In other words, if a collapsed dimension is
    //    dynamic, at least one of the related dimensions need to be dynamic.
    int64_t collapsedDim = 0, expandedDim = 0;
    while (collapsedDim < collapsedRank && expandedDim < expandedRank) {
      // Not empty means the associations has been filled in and the dimension
      // is unchanged.
      if (!reassociation[collapsedDim].empty()) {
        if (expandedDim != reassociation[collapsedDim][0])
          return op.emitOpError("Unsupported: expanded dims are off from the "
                                "expected dim got from reassociation");
        collapsedDim++;
        expandedDim++;
        continue;
      }

      // Collect the dims that are collapsed until hitting the next dim that's
      // unchanged.
      SmallVector<int64_t> collapsedDims;
      while (collapsedDim < collapsedRank &&
             reassociation[collapsedDim].empty()) {
        collapsedDims.push_back(collapsedDim);
        collapsedDim++;
      }
      // the next reassociation is for a dim that's unchanged.
      int64_t expandedDimNext = collapsedDim != collapsedRank
                                    ? reassociation[collapsedDim][0]
                                    : expandedRank;
      if (collapsedDims.size() == 1) {
        int64_t collapsedDimSize = 1;
        int64_t collapsedDim = collapsedDims[0];
        for (auto i : llvm::seq<int64_t>(expandedDim, expandedDimNext)) {
          reassociation[collapsedDim].push_back(i);
          if (collapsedDimSize == kUnknownSize)
            continue;

          int64_t expandedDimSize = expandedShape[i];
          if (expandedDimSize == kUnknownSize) {
            collapsedDimSize = kUnknownSize;
            continue;
          }
          collapsedDimSize *= expandedShape[i];
        }
        // To meet both requirements from tensor.expand_shape verification code.
        collapsedShape[collapsedDim] = collapsedDimSize;
        expandedDim = expandedDimNext;
        continue;
      }

      // collpasedDims are expanded to [expandedDim, expandedDimNext)
      if (expandedDimNext - expandedDim < (int64_t)collapsedDims.size())
        op.emitError("unimplemented: mixed of expanding and collapsing "
                     "operations for view");
      for (auto collapsedDim : collapsedDims) {
        if (collapsedShape[collapsedDim] == kUnknownSize) {
          if (expandedDim >= expandedDimNext) {
            return rewriter.notifyMatchFailure(
                op,
                "desired size is not compatible with the input tensor size");
          }
          checkDimEqualHelper(rewriter, loc,
                              collapsedShapeTorchInt[collapsedDim],
                              expandedShapeTorchInt[expandedDim]);
          // To meet the second requirement from tensor.expand_shape
          // verification code.
          expandedShape[expandedDim] = kUnknownSize;
          reassociation[collapsedDim].push_back(expandedDim++);
        } else {
          int64_t remainingSizeToExpand = collapsedShape[collapsedDim];
          // A do-while loop is used here to handle the cases where the
          // collapsed shape tensor has a dimension of size 1.
          do {
            int64_t expandedDimSize = expandedShape[expandedDim];
            if (expandedDim >= expandedDimNext ||
                expandedShape[expandedDim] == kUnknownSize ||
                remainingSizeToExpand % expandedDimSize != 0) {
              return rewriter.notifyMatchFailure(
                  op, "total number of elements mismatch in the expansion");
            }
            reassociation[collapsedDim].push_back(expandedDim++);
            remainingSizeToExpand /= expandedDimSize;
          } while (remainingSizeToExpand != 1);
        }
      }
    }

    if (collapsedDim != collapsedRank || expandedDim != expandedRank)
      return rewriter.notifyMatchFailure(op, "view shape is not supported");
    Type adjustedResultType =
        RankedTensorType::get(isCollapse ? collapsedShape : expandedShape,
                              resultType.getElementType());
    Type adjustedInputType =
        RankedTensorType::get(isCollapse ? expandedShape : collapsedShape,
                              resultType.getElementType());
    Value castedInput =
        rewriter.create<tensor::CastOp>(loc, adjustedInputType, input);
    Value result =
        isCollapse
            ? rewriter
                  .create<tensor::CollapseShapeOp>(loc, adjustedResultType,
                                                   castedInput, reassociation)
                  .result()
            : rewriter
                  .create<tensor::ExpandShapeOp>(loc, adjustedResultType,
                                                 castedInput, reassociation)
                  .result();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenConvolutionOverrideableOp : public OpConversionPattern<AtenConvolutionOverrideableOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConvolutionOverrideableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO:
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSqueezeOp : public OpConversionPattern<AtenSqueezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSqueezeOp op, OpAdaptor adaptor,
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

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    // In case the operand tensor type is statically shaped with all dimensions
    // being unit extent, it will be collapsed to a 0-D tensor.
    if (resultRank == 0) {
      SmallVector<ReassociationIndices> reassociation;
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, input, reassociation);
      return success();
    }

    // All the static size-1 dimensions at the beginning(going from higher to
    // lower dimensions) will be collapsed into the first dynamic or first non
    // size-1 static dimension. All the other static size-1 dimensions will be
    // collapsed into its previous dynamic or non size-1 static dimension.
    SmallVector<ReassociationIndices> reassociation(resultRank);
    bool isSqueezed = false;
    int64_t headOnesCount = 0;
    while (headOnesCount < inputRank &&
           inputType.getDimSize(headOnesCount) == 1) {
      isSqueezed = true;
      reassociation[0].push_back(headOnesCount++);
    }

    // TODO: Add support for size-1 dynamic dimensions.
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    int64_t j = -1;
    for (auto i : llvm::seq<int64_t>(headOnesCount, inputRank)) {
      if (inputType.isDynamicDim(i)) {
        // Make sure that size-1 dynamic dimension does not exist.
        Value dimSize = getDimOp(rewriter, loc, input, i);
        Value dimSizeNotOne = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, dimSize, one);
        rewriter.create<cf::AssertOp>(
            loc, dimSizeNotOne,
            rewriter.getStringAttr(
                "unimplemented: size 1 dynamic dimension is not supported"));
        ++j;
      } else if (inputType.getDimSize(i) != 1) {
        ++j;
      } else {
        // `isSqueezed` checks if the operand tensor type contains at least one
        // unit dimension.
        isSqueezed = true;
      }
      if (j == resultRank)
        break;
      reassociation[j].push_back(i);
    }

    // Make sure that result type rank is compatible with the squeezed size.
    if (j != resultRank - 1)
      return rewriter.notifyMatchFailure(
          op, "expected output size mismatches with the result type rank");

    if (isSqueezed) {
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          op, resultType, input, reassociation);

    } else {
      // If the operand tensor type does not have any unit dimension,
      // `aten.squeeze` will behave as an identity operation.
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
    }
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenSqueezeDimOp : public OpConversionPattern<AtenSqueezeDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSqueezeDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Value input = adaptor.self();
    auto inputType = input.getType().cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();

    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(
          op, "zero input rank should have been handled by the folder");
    }

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be constant");
    dim = toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

    // TODO: Handle the case where the dim(th) dimension is dynamic.
    if (inputType.isDynamicDim(dim)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dim(th) dimension is not expected to be dynamic");
    }

    TypeConverter *typeConverter = getTypeConverter();
    auto resultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int64_t resultRank = resultType.getRank();

    // If the dim(th) dimension of operand tensor type is not statically unit,
    // `aten.squeeze` will behave as an identity operation.
    if (inputType.getDimSize(dim) != 1) {
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, input);
      return success();
    }

    SmallVector<ReassociationIndices> reassociationMap(resultRank);
    bool alreadyCrossedSqueezedDim = false;
    for (int i = 0; i != resultRank; i++) {
      if (alreadyCrossedSqueezedDim) {
        reassociationMap[i].push_back(i + 1);
      } else {
        reassociationMap[i].push_back(i);
        if (dim != 0 && i != dim - 1)
          continue;

        alreadyCrossedSqueezedDim = true;
        if (dim == 0)
          reassociationMap[0].push_back(1);
        if (i == dim - 1)
          reassociationMap[i].push_back(dim);
      }
    }
    // Note: In case the operand tensor type is of unit rank and is statically
    // shaped with unit dimension, the `reassociationMap` will be empty and the
    // input will be collapsed to a 0-D tensor.
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, resultType, input,
                                                         reassociationMap);
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
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
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
class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    TypeConverter *typeConverter = getTypeConverter();

    auto input = adaptor.self();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    int64_t dim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
      return op->emitError("unimplemented: dim is not constant");

    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
    Value dimSize = inputShape[dim];

    auto adjustStartOrEnd = [&](Value startOrEndTorchType,
                                Value startOrEndBuiltin, Value valueForNone) {
      if (startOrEndTorchType.getType().isa<Torch::NoneType>())
        return valueForNone;
      auto dimSizeAsInt = castIndexToInt(rewriter, loc, dimSize);
      Value startOrEndToPositive =
          toPositiveDimDynamic(rewriter, loc, startOrEndBuiltin, dimSizeAsInt);
      // startOrEnd < 0 ? 0 : startOrEnd
      Value cst0 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(dimSizeAsInt.getType()));
      Value predDimSltZero = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, startOrEndToPositive, cst0);
      Value startOrEndAtLeastZero = rewriter.create<arith::SelectOp>(
          loc, predDimSltZero, cst0, startOrEndToPositive);
      // startOrEnd > dimSizeAsInt ? dimSizeAsInt : startOrEnd
      Value startOrEndSgtDimSize = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, startOrEndAtLeastZero, dimSizeAsInt);
      Value startOrEndBoundedByDimSize = rewriter.create<arith::SelectOp>(
          loc, startOrEndSgtDimSize, dimSizeAsInt, startOrEndAtLeastZero);

      return castIntToIndex(rewriter, loc, startOrEndBoundedByDimSize);
    };

    if (op.start().getType().isa<OptionalType>() ||
        op.end().getType().isa<OptionalType>())
      return rewriter.notifyMatchFailure(op, "unimplemented optional type arg");
    Value start = adjustStartOrEnd(op.start(), adaptor.start(), zero);
    Value end = adjustStartOrEnd(op.end(), adaptor.end(), dimSize);

    // end >= start ? end : start
    Value endSgeStart = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, end, start);
    end = rewriter.create<arith::SelectOp>(loc, endSgeStart, end, start);

    int64_t step;
    if (!matchPattern(op.step(), m_TorchConstantInt(&step))) {
      if (!op.step().getType().isa<Torch::NoneType>())
        return op->emitError("unimplemented: step is not constant");
      step = 1;
    }

    // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
    Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);
    Value len = rewriter.create<arith::SubIOp>(loc, end, start);
    Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
    resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, one);
    resultSize =
        rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);

    SmallVector<Value> resultShape = getTensorSizes(rewriter, loc, input);
    resultShape[dim] = resultSize;

    SmallVector<Value> offsets(inputType.getRank(), zero);
    SmallVector<Value> strides(inputType.getRank(), one);
    offsets[dim] = start;
    strides[dim] = rewriter.create<arith::MulIOp>(loc, strides[dim], stepIndex);

    Value result = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, offsets, resultShape, strides);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
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
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(loc, tensors[0], i));

    // Calculate the size of the `dim` result dimension by adding the dim size
    // of each tensor together.
    Value resultDimSize = sizes[dim];

    Value dimIndex = rewriter.createOrFold<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(dim));
    for (auto tensor : makeArrayRef(tensors).drop_front()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, tensor, dimIndex);
      resultDimSize =
          rewriter.createOrFold<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;

    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op)
        return v;
      return op.getValue();
    };

    Value result = rewriter.create<linalg::InitTensorOp>(
        loc, sizes, newResultType.getElementType());
    for (auto tensor : tensors) {
      SmallVector<Value> sizes = getTensorSizes(rewriter, loc, tensor);
      result = rewriter.createOrFold<tensor::InsertSliceOp>(
          loc, tensor, result,
          llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
      offsets[dim] =
          rewriter.createOrFold<arith::AddIOp>(loc, offsets[dim], sizes[dim]);
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
    auto genericOp = rewriter
                         .create<linalg::GenericOp>(
                             loc, result.getType(), indices, result, affineMaps,
                             iteratorTypes,
                             [&](OpBuilder &b, Location loc, ValueRange args) {
                               auto index = args[0];
                               createLinalgPayloadCalculationForGatherOps(
                                   b, loc, self, rank, index, dim, rank);
                             })
                         .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultTy, genericOp);
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

namespace {
// Casts a tensor of exactly one element to an elemental type.
template <typename OpTy>
class ConvertAtenTensorToScalarLikeOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.a();
    SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
    int64_t inputRank = inputSizes.size();

    // The `input` tensor must contain exactly one element, i.e., either the
    // `input` is a zero rank tensor or all the dimensions of the `input` tensor
    // are unit.
    Value constantOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    for (int64_t i = 0; i < inputRank; i++)
      checkDimEqualHelper(rewriter, loc, inputSizes[i], constantOne);

    // Extract the only element from the `input` tensor.
    Value constantZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    SmallVector<Value> indices(inputRank, constantZero);
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, input, indices);
    return success();
  }
};
} // namespace

namespace {
class ConvertPseudoAtenFillScalarOp
    : public OpConversionPattern<PseudoAtenFillScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PseudoAtenFillScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    Value initVal = adaptor.value();
    auto tensorType = self.getType().cast<RankedTensorType>();
    RankedTensorType resultType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

    Value initValCasted = convertScalarToDtype(rewriter, loc, initVal,
                                               tensorType.getElementType());
    Value result =
        createInitTensor(rewriter, loc, getTensorSizes(rewriter, loc, self),
                         tensorType.getElementType(), initValCasted);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
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
        rewriter.create<cf::AssertOp>(
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
        Value select = rewriter.create<arith::SelectOp>(
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
      rewriter.create<cf::AssertOp>(
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
struct ConvertAtenScalarToTensorLike : ConversionPattern {
  ConvertAtenScalarToTensorLike(TypeConverter &typeConverter,
                                MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenTensorIntOp, AtenTensorFloatOp>(op))
      return rewriter.notifyMatchFailure(
          op, "not a supported Scalar to Tensor like op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value elemVal, dtype, device, requires_grad;
    if (AtenTensorIntOp tensorIntOp = dyn_cast<AtenTensorIntOp>(op)) {
      AtenTensorIntOp::Adaptor adaptor(operands);
      elemVal = adaptor.t();
      dtype = tensorIntOp.dtype();
      device = tensorIntOp.device();
      requires_grad = tensorIntOp.requires_grad();
    }
    if (AtenTensorFloatOp tensorFloatOp = dyn_cast<AtenTensorFloatOp>(op)) {
      AtenTensorFloatOp::Adaptor adaptor(operands);
      elemVal = adaptor.t();
      dtype = tensorFloatOp.dtype();
      device = tensorFloatOp.device();
      requires_grad = tensorFloatOp.requires_grad();
    }
    // TODO: Dtype conversion.
    if (!dtype.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "Unimplemented non-None dtype");

    // TODO: Device information.
    if (!device.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None device information");

    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type outElementType = resultType.getElementType();
    Value elemValProm =
        convertScalarToDtype(rewriter, loc, elemVal, outElementType);
    Value zeroDTensor =
        createInitTensor(rewriter, loc, {}, outElementType, elemValProm);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, zeroDTensor);
    return success();
  }
};
} // namespace

namespace {
// Converts constant tensor allocation like ops.
template <typename OpTy, int fillVal>
class ConvertConstantTensorAllocOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support for layout, pin_memory features.
    // Only `none` layout is supported.
    if (!op.layout().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only default layout is supported");

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.size(), resultSizeTorchInt)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size must be constructed using ListConstruct");
    }
    resultSize = getTypeConvertedValues(rewriter, loc, typeConverter,
                                        resultSizeTorchInt);
    for (auto size : resultSize)
      resultSizeIndex.push_back(castIntToIndex(rewriter, loc, size));

    auto resultType =
        typeConverter->convertType(op.getType()).template cast<RankedTensorType>();
    Type outElemType = resultType.getElementType();

    // Create an uninitialized tensor of `resultSize` shape and fill it with
    // value `fillVal`.
    Value constVal = getConstant(rewriter, loc, fillVal, outElemType);
    Value outputTensor =
        createInitTensor(rewriter, loc, resultSizeIndex, outElemType, constVal);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, outputTensor);
    return success();
  }
};
} // namespace

namespace {
// Converts `aten.empty` to `linalg.init_tensor` op.
class ConvertAtenEmptyMemoryFormatOp
    : public OpConversionPattern<AtenEmptyMemoryFormatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenEmptyMemoryFormatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support for layout, pin_memory and memory_format features.
    // Only `none` layout is supported.
    if (!op.layout().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only default layout is supported");

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");

    // Only `none` memory_format is supported.
    if (!op.memory_format().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only default memory format is supported");

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
    if (!getListConstructElements(op.size(), resultSizeTorchInt)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size must be constructed using ListConstruct");
    }
    resultSize = getTypeConvertedValues(rewriter, loc, typeConverter,
                                        resultSizeTorchInt);
    for (auto size : resultSize)
      resultSizeIndex.push_back(castIntToIndex(rewriter, loc, size));

    auto resultType = typeConverter->convertType(op.getType())
                          .template cast<RankedTensorType>();
    // Create an uninitialized tensor of `resultSize` shape.
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultSizeIndex, resultType.getElementType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, initTensor);
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

namespace {
class ConvertAtenNumelOp : public OpConversionPattern<AtenNumelOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNumelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value self = adaptor.self();
    SmallVector<Value> sizes(getTensorSizes(rewriter, loc, self));
    Value productResult =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    for (size_t i = 0; i < sizes.size(); i++)
      productResult =
          rewriter.create<arith::MulIOp>(loc, productResult, sizes[i]);
    rewriter.replaceOp(op, castIndexToInt(rewriter, loc, productResult));
    return success();
  }
};
} // namespace

namespace {
// Let's say we have an input tensor: initialized with some random values of
// size [4, 5, 6]. An index tensor (always 1-d): [0, 2] of size [2], and an
// integer argument dim = 1. The size of the output tensor will be [4, 2, 6].
// The approach is as follows:
//
// for i in range(input.size[0])
//    for j in range(index.size[0])
//       for k in range(input.size[2])
//          indexValue = index[j]
//          output[i,j,k] = input[i,indexValue,k]

class ConvertAtenIndexSelectOp : public OpConversionPattern<AtenIndexSelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.self();
    Value indices = adaptor.index();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    unsigned inputRank = inputType.getRank();

    int64_t dimInt;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&dimInt)))
      return op->emitError("unimplemented: dim is not constant");

    SmallVector<Value> resultShape = getTensorSizes(rewriter, loc, input);
    resultShape[dimInt] = getTensorSizes(rewriter, loc, indices)[0];
    Value initTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, elementType);

    SmallVector<AffineExpr> resultExpr;
    AffineExpr indicesExpr = rewriter.getAffineDimExpr(dimInt);
    SmallVector<StringRef> iteratorTypes;

    for (unsigned i = 0; i < inputRank; i++) {
      resultExpr.push_back(rewriter.getAffineDimExpr(i));
      iteratorTypes.push_back(getParallelIteratorTypeName());
    }

    auto indexingMaps = AffineMap::inferFromExprList({indicesExpr, resultExpr});

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), ValueRange{indices}, initTensor,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value index = rewriter.create<arith::IndexCastOp>(
                      loc, rewriter.getIndexType(), args[0]);
                  SmallVector<Value> indexTarget;
                  for (unsigned i = 0; i < inputRank; i++)
                    indexTarget.push_back(b.create<linalg::IndexOp>(loc, i));
                  indexTarget[dimInt] = index;
                  Value extractedElement =
                      b.create<tensor::ExtractOp>(loc, input, indexTarget);
                  b.create<linalg::YieldOp>(loc, extractedElement);
                })
            .getResult(0);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

namespace {
// Let's say the result of the `aten.arange.start_step` is `output` which is a
// 1-d output tensor. The approach used for generating the output tensor is as
// follows:
//    for i in range(ceil((end-start)/step))
//          output[i] = start + (i * step)
class ConvertAtenArangeStartStepOp
    : public OpConversionPattern<AtenArangeStartStepOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenArangeStartStepOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Add support for layout, pin_memory features.
    // Only `none` layout is supported.
    if (!op.layout().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only default layout is supported");

    // The pin_memory should be either `False` or `none`.
    bool pinMemory;
    if (!op.pin_memory().getType().isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: pin_memory must be either None or false");
    }

    Location loc = op.getLoc();
    TypeConverter *typeConverter = this->getTypeConverter();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();
    Type dtype = resultType.getElementType();
    Value start = convertScalarToDtype(rewriter, loc, adaptor.start(), dtype);
    Value end = convertScalarToDtype(rewriter, loc, adaptor.end(), dtype);
    Value step = convertScalarToDtype(rewriter, loc, adaptor.step(), dtype);

    // The result will always be a 1-d tensor.
    // The size of the result is calculated as follows:
    //          ceil((end - start)/step)
    Value resultShape;
    if (dtype.isa<mlir::IntegerType>()) {
      Value subOut = rewriter.create<arith::SubIOp>(loc, end, start);
      resultShape = rewriter.create<arith::CeilDivSIOp>(loc, subOut, step);
    } else {
      Value subOut = rewriter.create<arith::SubFOp>(loc, end, start);
      Value divOut = rewriter.create<arith::DivFOp>(loc, subOut, step);
      Value ceilOut = rewriter.create<math::CeilOp>(loc, divOut);
      resultShape =
          rewriter.create<arith::FPToUIOp>(loc, rewriter.getI64Type(), ceilOut);
    }
    resultShape = castIntToIndex(rewriter, loc, resultShape);

    Value resultTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resultShape, dtype);

    StringRef iteratorType = getParallelIteratorTypeName();
    AffineMap indexingMap =
        AffineMap::getMultiDimIdentityMap(1, op->getContext());

    Value finalRes =
        rewriter
            .create<linalg::GenericOp>(
                loc, /*resultTensorTypes=*/resultTensor.getType(),
                /*inputs=*/ValueRange({}),
                /*outputs=*/resultTensor,
                /*indexingMaps=*/indexingMap,
                /*iteratorTypes=*/iteratorType,
                [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
                  Value index = b.create<linalg::IndexOp>(loc, 0);
                  index = castIndexToInt(b, loc, index);
                  index = convertScalarToDtype(b, loc, index, dtype);
                  Value mulOut, result;
                  if (dtype.isa<mlir::FloatType>()) {
                    mulOut = b.create<arith::MulFOp>(loc, step, index);
                    result = b.create<arith::AddFOp>(loc, start, mulOut);
                  } else {
                    mulOut = b.create<arith::MulIOp>(loc, step, index);
                    result = b.create<arith::AddIOp>(loc, start, mulOut);
                  }
                  b.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenIndexTensorOp : public OpConversionPattern<AtenIndexTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenIndexTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.self();
    Value indices = op.indices();
    SmallVector<Value> indicesTuple;
    if (!getListConstructElements(indices, indicesTuple)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the indices list is not from a list construct");
    }

    SmallVector<Value> indicesVal =
        getTypeConvertedValues(rewriter, loc, getTypeConverter(), indicesTuple);

    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type elementType = resultType.getElementType();
    unsigned inputRank = inputType.getRank();
    unsigned numIndexTensors = indicesTuple.size();
    SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);

    // Case 1 : When numIndexTensors == 1 and `input` is a 1-d tensor.
    // TODO: generalize the implementation for other cases.
    if (numIndexTensors == 1 && inputRank == 1) {
      if (failed(checkNotNone(rewriter, op, indicesVal[0])))
        return rewriter.notifyMatchFailure(op, "unimplemented None type arg");
      unsigned resultRank =
          indicesVal[0].getType().cast<RankedTensorType>().getRank();
      SmallVector<Value> resultShape;
      SmallVector<AffineExpr> indicesExpr, resultExpr;
      SmallVector<StringRef> iteratorTypes;
      for (unsigned i = 0; i < resultRank; i++)
        resultShape.push_back(getDimOp(rewriter, loc, indicesVal[0], i));
      Value initTensor =
          rewriter.create<linalg::InitTensorOp>(loc, resultShape, elementType);
      for (unsigned i = 0; i < resultRank; i++) {
        indicesExpr.push_back(rewriter.getAffineDimExpr(i));
        resultExpr.push_back(rewriter.getAffineDimExpr(i));
        iteratorTypes.push_back(getParallelIteratorTypeName());
      }
      auto indexingMaps =
          AffineMap::inferFromExprList({indicesExpr, resultExpr});

      Value finalRes =
          rewriter
              .create<linalg::GenericOp>(
                  loc, initTensor.getType(), ValueRange{indicesVal[0]},
                  initTensor,
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value indexTarget = castIntToIndex(b, loc, args[0]);
                    Value extractedElement =
                        b.create<tensor::ExtractOp>(loc, input, indexTarget);
                    b.create<linalg::YieldOp>(loc, extractedElement);
                  })
              .getResult(0);

      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, finalRes);
      return success();
    } else
      return rewriter.notifyMatchFailure(
          op, "unimplemented: support for this set of inputs not present");
  }
};
} // namespace

namespace {
class ConvertPseudoAtenUniformOp
    : public OpConversionPattern<PseudoAtenUniformOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PseudoAtenUniformOp op, OpAdaptor adaptor,
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
    Value initialSeed = rewriter.create<GetNextSeedOp>(loc);
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
        rewriter.create<linalg::InitTensorOp>(loc, sizes, elemTy);
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
    registry.insert<cf::ControlFlowDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           cf::ControlFlowDialect, math::MathDialect,
                           tensor::TensorDialect, arith::ArithmeticDialect>();
    target.addLegalOp<GetNextSeedOp>();

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
    target.addIllegalOp<
        AtenTanhOp, AtenReluOp, AtenLeakyReluOp, AtenGeluOp, AtenGeluBackwardOp,
        AtenAddTensorOp, AtenMulTensorOp, AtenDivTensorOp, AtenSubTensorOp,
        AtenLerpTensorOp, AtenSigmoidOp, AtenMinimumOp, AtenMaximumOp,
        AtenToDtypeOp, AtenClampOp, AtenRsubScalarOp, AtenLogOp, AtenSqrtOp,
        AtenFloorOp, AtenCeilOp, AtenPowTensorScalarOp, AtenLog2Op, AtenRsqrtOp,
        AtenAbsOp, AtenReciprocalOp, AtenBitwiseAndTensorOp, AtenGtScalarOp,
        AtenGeScalarOp, AtenEqScalarOp, AtenLtScalarOp, AtenLeScalarOp,
        AtenWhereSelfOp, AtenGtTensorOp, AtenEqTensorOp, AtenLtTensorOp,
        AtenThresholdOp, AtenThresholdBackwardOp, AtenCloneOp>();
    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    target.addIllegalOp<AtenSqueezeOp>();
    patterns.add<ConvertAtenSqueezeOp>(typeConverter, context);
    target.addIllegalOp<AtenSqueezeDimOp>();
    patterns.add<ConvertAtenSqueezeDimOp>(typeConverter, context);
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
    target.addIllegalOp<AtenConvolutionOverrideableOp>();
    patterns.add<ConvertAtenConvolutionOverrideableOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);
    target.addIllegalOp<AtenConstantPadNdOp>();
    patterns.add<ConvertAtenConstantPadNdOp>(typeConverter, context);
    target.addIllegalOp<AtenSumOp>();
    target.addIllegalOp<AtenSumDimIntListOp>();
    target.addIllegalOp<AtenMaxOp>();
    patterns.add<ConvertReductionOp>(typeConverter, context);
    target.addIllegalOp<AtenTransposeIntOp>();
    patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenPermuteOp>();
    patterns.add<ConvertAtenPermuteOp>(typeConverter, context);
    target.addIllegalOp<AtenCatOp>();
    patterns.add<ConvertAtenCatOp>(typeConverter, context);
    target.addIllegalOp<AtenGatherOp>();
    patterns.add<ConvertAtenGatherOp>(typeConverter, context);
    target.addIllegalOp<AtenNativeLayerNormOp>();
    patterns.add<ConvertAtenNativeLayerNormOp>(typeConverter, context);
    target.addIllegalOp<AtenBroadcastToOp>();
    patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxDimOp>();
    patterns.add<ConvertAtenMaxDimOp>(typeConverter, context);
    target.addIllegalOp<AtenSizeIntOp>();
    patterns.add<ConvertAtenSizeIntOp>(typeConverter, context);
    target.addIllegalOp<AtenEmbeddingOp>();
    patterns.add<ConvertAtenEmbeddingOp>(typeConverter, context);
    target.addIllegalOp<AtenEmptyMemoryFormatOp>();
    patterns.add<ConvertAtenEmptyMemoryFormatOp>(typeConverter, context);
    target.addIllegalOp<AtenZerosOp, AtenOnesOp>();
    patterns.add<ConvertConstantTensorAllocOp<AtenZerosOp, 0>>(typeConverter,
                                                               context);
    patterns.add<ConvertConstantTensorAllocOp<AtenOnesOp, 1>>(typeConverter,
                                                              context);
    target.addIllegalOp<AtenContiguousOp>();
    patterns.add<ConvertAtenContiguousOp>(typeConverter, context);
    target.addIllegalOp<AtenIntTensorOp, AtenFloatTensorOp, AtenBoolTensorOp>();
    patterns.add<ConvertAtenTensorToScalarLikeOp<AtenIntTensorOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenTensorToScalarLikeOp<AtenFloatTensorOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenTensorToScalarLikeOp<AtenBoolTensorOp>>(
        typeConverter, context);
    target.addIllegalOp<PrimNumToTensorScalarOp>();
    patterns.add<ConvertPrimNumToTensorScalarOp>(typeConverter, context);
    target.addIllegalOp<AtenDropoutOp>();
    patterns.add<ConvertAtenDropoutOp>(typeConverter, context);
    target.addIllegalOp<PseudoAtenFillScalarOp>();
    patterns.add<ConvertPseudoAtenFillScalarOp>(typeConverter, context);
    target.addIllegalOp<AtenNumelOp>();
    patterns.add<ConvertAtenNumelOp>(typeConverter, context);
    target.addIllegalOp<AtenSliceTensorOp>();
    patterns.add<ConvertAtenSliceTensorOp>(typeConverter, context);
    target.addIllegalOp<AtenNllLossForwardOp>();
    patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
    target.addIllegalOp<AtenNllLossBackwardOp>();
    patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
    target.addIllegalOp<AtenIndexSelectOp>();
    patterns.add<ConvertAtenIndexSelectOp>(typeConverter, context);
    patterns.add<ConvertAtenScalarToTensorLike>(typeConverter, context);
    target.addIllegalOp<AtenTensorIntOp, AtenTensorFloatOp>();
    patterns.add<ConvertAtenArangeStartStepOp>(typeConverter, context);
    target.addIllegalOp<AtenArangeStartStepOp>();
    patterns.add<ConvertAtenIndexTensorOp>(typeConverter, context);
    target.addIllegalOp<AtenIndexTensorOp>();
    patterns.add<ConvertPseudoAtenUniformOp>(typeConverter, context);
    target.addIllegalOp<PseudoAtenUniformOp>();

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
