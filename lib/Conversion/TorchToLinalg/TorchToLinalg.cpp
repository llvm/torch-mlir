//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"
#include "npcomp/Dialect/Torch/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

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

// Hack to deal with the Torch list type arguments which is not supported end
// to end. Constant values can be be extracted directly and non constant
// list values are not supported.
// TODO: loose this constraint when properly support list type
static bool isConstantIntListMatching(Value &value,
                                      llvm::SmallVectorImpl<int64_t> &expects) {
  llvm::SmallVector<int64_t> intValues;
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

namespace {
class ConvertAtenAdaptiveAvgPool2dOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOp op, llvm::ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    AtenAdaptiveAvgPool2dOp::Adaptor adaptor(operands);
    Value outputSize = adaptor.output_size();
    Value input = adaptor.self(); /* in form of N*C*H*W */
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type elementType = inputType.getElementType();
    if (!elementType.isa<mlir::FloatType>())
      op.emitError("unimplemented: non-floating point type");

    auto inputRank = inputType.getRank();
    if (inputRank != 4)
      return rewriter.notifyMatchFailure(op, "input should be rank 4");

    SmallVector<int64_t, 2> expects{1, 1};
    if (!isConstantIntListMatching(outputSize, expects))
      return rewriter.notifyMatchFailure(
          op, "only support output_size with H and W both equal to constant 1");

    auto getDimOp = [&](Value v, int dimension) {
      return rewriter.create<tensor::DimOp>(loc, v, dimension);
    };
    Value N = getDimOp(input, 0);
    Value C = getDimOp(input, 1);
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
    Value H = getDimOp(input, 2);
    Value W = getDimOp(input, 3);
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
  matchAndRewrite(AtenConv2dOp op, llvm::ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    AtenConv2dOp::Adaptor adaptor(operands);
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value padding = adaptor.padding();
    Value stride = adaptor.stride();
    Value dilation = adaptor.dilation();
    Value groups = adaptor.groups();

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      op.emitError("unimplemented: non-floating point type");
    Type intType = IntegerType::get(context, 64);
    Type indexType = IndexType::get(context);

    auto getDimOp = [&](Value v, int dimension) {
      return rewriter.create<tensor::DimOp>(loc, v, dimension);
    };

    auto castIntToIndex = [&](Value v) {
      return rewriter.create<IndexCastOp>(loc, indexType, v);
    };

    auto castIndexToInt = [&](Value v) {
      return rewriter.create<IndexCastOp>(loc, intType, v);
    };

    Value N = getDimOp(input, 0);
    Value Hin = getDimOp(input, 2);
    Value Win = getDimOp(input, 3);
    Value F = getDimOp(weight, 0);
    Value weightH = getDimOp(weight, 2);
    Value weightW = getDimOp(weight, 3);

    Value c1 = rewriter.create<ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value c2 = rewriter.create<ConstantOp>(loc, IntegerAttr::get(intType, 2));

    llvm::SmallVector<int64_t> paddingIntValues;
    if (!matchPattern(padding, m_TorchConstantIntList(paddingIntValues))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    SmallVector<int64_t, 2> expects{1, 1};
    if (!isConstantIntListMatching(stride, expects))
      return rewriter.notifyMatchFailure(op, "only support stride [1, 1]");
    if (!isConstantIntListMatching(dilation, expects))
      return rewriter.notifyMatchFailure(op, "only support dilation [1, 1]");
    // Unit strides and dilations.
    auto linalgStrides = rewriter.getI64VectorAttr({1, 1});
    auto linalgDilations = rewriter.getI64VectorAttr({1, 1});

    if (!op.bias().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "only support None bias");
    Value groupEqual1 =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, groups, c1);
    rewriter.create<AssertOp>(loc, groupEqual1,
                              rewriter.getStringAttr("expect groups to be 1"));

    // Pad the input tensor according to padding.
    Value paddingH = rewriter.create<ConstantOp>(
        loc, intType,
        IntegerAttr::get(IntegerType::get(context, 64), paddingIntValues[0]));
    Value paddingW = rewriter.create<ConstantOp>(
        loc, intType,
        IntegerAttr::get(IntegerType::get(context, 64), paddingIntValues[1]));
    Value paddingHIndexType = castIntToIndex(paddingH);
    Value paddingWIndexType = castIntToIndex(paddingW);
    auto c0IndexAttr = rewriter.getIndexAttr(0);
    Value c0float =
        rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
    SmallVector<OpFoldResult, 4> paddings = {
        c0IndexAttr, c0IndexAttr, paddingHIndexType, paddingWIndexType};

    Type ranked4DTensorType =
        RankedTensorType::get({-1, -1, -1, -1}, elementType);
    Value paddedInput = linalg::PadTensorOp::createPadScalarOp(
        ranked4DTensorType, input, c0float, /*low=*/paddings, /*high=*/paddings,
        loc, rewriter);

    // Caculate the output tensor dims.
    // Along each dim: dim_out = dim_in + 2*padding - weightSize + 1
    auto getOutputDim = [&](Value in, Value paddingIntType, Value weightSize) {
      Value doublePadding = rewriter.create<MulIOp>(loc, paddingIntType, c2);
      // in + 2*paddingIntType
      Value inAddDoublePadding =
          rewriter.create<AddIOp>(loc, castIndexToInt(in), doublePadding);
      Value weightSizeIntType = castIndexToInt(weightSize);
      Value temp =
          rewriter.create<SubIOp>(loc, inAddDoublePadding, weightSizeIntType);
      Value out = rewriter.create<AddIOp>(loc, temp, c1);
      return castIntToIndex(out);
    };
    Value Hout = getOutputDim(Hin, paddingH, weightH);
    Value Wout = getOutputDim(Win, paddingW, weightW);

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, F, Hout, Wout}, elementType);
    Value initTensor0 =
        rewriter.create<linalg::FillOp>(loc, c0float, initTensor).getResult(0);

    Value conv2d =
        rewriter
            .create<linalg::Conv2DNchwOp>(
                loc, ranked4DTensorType, ValueRange{paddedInput, weight},
                ValueRange{initTensor0}, linalgStrides, linalgDilations)
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv2d);
    return success();
  }
};
} // namespace

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

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias, running_mean, running_var.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
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
                  // ((input - mean) / sqrt(var + eps)) * weight + bias
                  Value inputSubMean = b.create<SubFOp>(loc, input, mean);
                  // The eps is always f64.
                  Value truncatedEps =
                      b.create<FPTruncOp>(loc, var.getType(), eps);
                  Value varPlusEps = b.create<AddFOp>(loc, var, truncatedEps);
                  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
                  Value temp = b.create<MulFOp>(loc, inputSubMean, rSTD);
                  Value timesWeight = b.create<MulFOp>(loc, temp, weight);
                  Value plusBias = b.create<AddFOp>(loc, timesWeight, bias);
                  b.create<linalg::YieldOp>(loc, plusBias);
                })
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, batchNorm);
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

    auto getDimOp = [&](Value v, int dimension) {
      return rewriter.create<tensor::DimOp>(loc, v, dimension);
    };
    Value inputDim0 = getDimOp(input, 0);
    Value inputDim1 = getDimOp(input, 1);
    Value weightDim0 = getDimOp(weight, 0);
    Value weightDim1 = getDimOp(weight, 1);
    Value biasDim0 = getDimOp(bias, 0);
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
             AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp>(op))
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
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenMmOp>();
    patterns.add<ConvertAtenMmOp>(typeConverter, context);
    target.addIllegalOp<AtenLinearOp>();
    patterns.add<ConvertAtenLinearOp>(typeConverter, context);
    target.addIllegalOp<AtenBatchNormOp>();
    patterns.add<ConvertAtenBatchNormOp>(typeConverter, context);
    target
        .addIllegalOp<AtenTanhOp, AtenReluOp, AtenAddTensorOp, AtenMulTensorOp,
                      AtenDivTensorOp, AtenSubTensorOp, AtenLerpTensorOp>();
    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    target.addIllegalOp<AtenUnsqueezeOp>();
    patterns.add<ConvertAtenUnsqueezeOp>(typeConverter, context);
    target.addIllegalOp<AtenConv2dOp>();
    patterns.add<ConvertAtenConv2dOp>(typeConverter, context);
    target.addIllegalOp<AtenAdaptiveAvgPool2dOp>();
    patterns.add<ConvertAtenAdaptiveAvgPool2dOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::createConvertTorchToLinalgPass() {
  return std::make_unique<ConvertTorchToLinalg>();
}
