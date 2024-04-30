//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <numeric>
#include <type_traits>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

namespace {

template <typename T>
static Value getConstantLike(OpBuilder &b, Location loc, T constant,
                             Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  auto getAttr = [&]() -> Attribute {
    if (isa<mlir::IntegerType>(ty))
      return b.getIntegerAttr(ty, constant);
    if (isa<mlir::FloatType>(ty))
      return b.getFloatAttr(ty, constant);
    if (auto complexTy = dyn_cast<mlir::ComplexType>(ty))
      return complex::NumberAttr::get(complexTy, constant, 0);
    llvm_unreachable("unhandled element type");
  };
  return b.create<mlir::chlo::ConstantLikeOp>(loc, cast<TypedAttr>(getAttr()),
                                              val);
}

Value getConstantLike(OpBuilder &b, Location loc, const APFloat &constant,
                      Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  return b.create<mlir::chlo::ConstantLikeOp>(loc, b.getFloatAttr(ty, constant),
                                              val);
}

} // namespace

LogicalResult broadcastRanks(PatternRewriter &rewriter, Operation *op,
                             mlir::Value &self, mlir::Value &other,
                             size_t dimSizeIndexBits) {
  auto selfTy = dyn_cast<RankedTensorType>(self.getType());
  auto otherTy = dyn_cast<RankedTensorType>(other.getType());
  auto selfRank = selfTy.getRank();
  auto otherRank = otherTy.getRank();
  if (selfRank == 0 || otherRank == 0)
    return success();
  if (selfRank > otherRank) {
    auto unsqueezeDims =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, selfRank - otherRank));
    auto unsqueezeInfo = hlo::unsqueezeTensor(rewriter, op, other,
                                              unsqueezeDims, dimSizeIndexBits);
    if (failed(unsqueezeInfo))
      return failure();
    other = *unsqueezeInfo;
  } else if (otherRank > selfRank) {
    auto unsqueezeDims =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, otherRank - selfRank));
    auto unsqueezeInfo = hlo::unsqueezeTensor(rewriter, op, self, unsqueezeDims,
                                              dimSizeIndexBits);
    if (failed(unsqueezeInfo))
      return failure();
    self = *unsqueezeInfo;
  }
  return success();
}

bool skipMultiplyAlpha(Value alphaValue) {
  double doubleValue;
  auto isFloat = matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(alphaValue, m_TorchConstantInt(&intValue));

  return ((isFloat && doubleValue == 1.0) || (isInt && intValue == 1.0));
}

static FailureOr<Value> getMaxValueOfDtype(Operation *op, Type elementType,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementType);
  if (isa<mlir::FloatType>(elementType)) {
    auto constAttr = SplatElementsAttr::get(
        constType,
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
                        /*negative=*/false));
    return rewriter
        .create<stablehlo::ConstantOp>(op->getLoc(), constType, constAttr)
        .getResult();
  }
  if (isa<mlir::IntegerType>(elementType)) {
    auto integerType = cast<mlir::IntegerType>(elementType);
    DenseElementsAttr constAttr;
    if (integerType.isUnsigned()) {
      constAttr = SplatElementsAttr::get(
          constType, APInt::getMaxValue(integerType.getWidth()));
    } else {
      constAttr = SplatElementsAttr::get(
          constType, APInt::getSignedMaxValue(integerType.getWidth()));
    }
    return rewriter
        .create<stablehlo::ConstantOp>(op->getLoc(), constType, constAttr)
        .getResult();
  }
  return failure();
}

static FailureOr<Value> getMinValueOfDtype(Operation *op, Type elementType,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementType);
  if (isa<mlir::FloatType>(elementType)) {
    auto constAttr = SplatElementsAttr::get(
        constType,
        APFloat::getInf(cast<mlir::FloatType>(elementType).getFloatSemantics(),
                        /*negative=*/true));
    return rewriter
        .create<stablehlo::ConstantOp>(op->getLoc(), constType, constAttr)
        .getResult();
  }
  if (isa<mlir::IntegerType>(elementType)) {
    auto integerType = cast<mlir::IntegerType>(elementType);
    DenseElementsAttr constAttr;
    if (integerType.isUnsigned()) {
      constAttr = SplatElementsAttr::get(
          constType, APInt::getMinValue(integerType.getWidth()));
    } else {
      constAttr = SplatElementsAttr::get(
          constType, APInt::getSignedMinValue(integerType.getWidth()));
    }
    return rewriter
        .create<stablehlo::ConstantOp>(op->getLoc(), constType, constAttr)
        .getResult();
  }
  return failure();
}

// These legalizations are for unary ops.
namespace {
template <typename AtenOpT, typename StablehloOpT>
class ConvertAtenUnaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfType = cast<TensorType>(self.getType());
    if (!selfType) {
      return op.emitError("only Tensor types supported in StableHLO");
    }
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();
    self = hlo::promoteType(rewriter, op.getLoc(), self, outType);
    rewriter.replaceOpWithNewOp<StablehloOpT>(op, outType, self);
    return success();
  }
};
} // namespace

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
namespace {
template <typename AtenOpT, typename StablehloOpT>
class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());

    if (!selfTy)
      return op.emitError("only Tensor types supported in StableHLO");

    if (isa<mlir::FloatType>(selfTy.getElementType())) {
      rewriter.replaceOpWithNewOp<StablehloOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          self);
      return success();
    } else {
      return op.emitError(
          "only floating-point datatype legalization supported");
    }
  }
};
} // namespace

// These legalizations are for unary ops with promoting to floating point
// datatypes.
namespace {
template <typename AtenOpT, typename StablehloOpT>
class ConvertAtenUnaryPromoteToFPOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());
    if (!selfTy)
      return op.emitError("only Tensor types supported in StableHLO");
    auto resultTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<TensorType>();

    if (isa<mlir::FloatType>(resultTy.getElementType())) {
      Value src = hlo::promoteType(rewriter, op.getLoc(), self, resultTy);
      rewriter.replaceOpWithNewOp<StablehloOpT>(op, resultTy, src);
      return success();
    } else {
      return op.emitError(
          "only result to be floating-point datatype legalization supported");
    }
  }
};
} // namespace

// aten.ones & aten.zeros
// Ref: Error checking based on the Torch to TOSA lowering
namespace {
template <typename AtenOpT, int fillVal>
class ConvertAtenConstPatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType)
      return op.emitError("only Tensor types supported in StableHLO");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "only floating-point or integer datatype legalization supported");

    SmallVector<int64_t> shape;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(shape))) {
      return op.emitError("shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        hlo::getConstTensor<int32_t>(rewriter, op, values, shape).value();

    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, outType, constOp);
    return success();
  }
};

} // namespace

namespace {
// Casts a tensor of exactly one element to an elemental type.
// Many codes borrowed from
// `lib/Conversion/TorchToLinalg/TensorScalarInterop.cpp`
template <typename AtenOpT>
class ConvertAtenTensorToScalarLikeOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getA().getType());
    if (!inputType)

      op.emitError("only Tensor types supported in StableHLO");
    Location loc = op.getLoc();
    Value input = adaptor.getA();
    SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
    int64_t inputRank = inputSizes.size();
    Type inputDtype = cast<BaseTensorType>(op.getA().getType()).getDtype();

    Value constantOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    for (int64_t i = 0; i < inputRank; i++)
      checkDimEqualHelper(rewriter, loc, inputSizes[i], constantOne);

    Value constantZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    SmallVector<Value> indices(inputRank, constantZero);
    Value result = rewriter.create<tensor::ExtractOp>(loc, input, indices);
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.replaceOp(op, convertScalarToDtype(rewriter, loc, result,
                                                resultType, inputDtype));
    return success();
  }
};
} // namespace

// The binary broadcast patterns
namespace {
template <typename AtenOpT, typename ChloOpT>
class ConvertAtenBinaryBroadcastOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = cast<TensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return op.emitError("only Tensor types supported");

    auto outTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                     ->convertType(op.getType())
                     .template cast<TensorType>();

    lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outTy);
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outTy);

    rewriter.replaceOpWithNewOp<ChloOpT>(op, outTy, lhs, rhs,
                                         /*broadcast_attr*/ nullptr);
    return success();
  }
};
} // namespace

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
namespace {
template <typename AtenOpT, typename ChloOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    RankedTensorType lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    RankedTensorType rhsType = dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsType)
      return op.emitError("only Tensor types supported in StableHLO");

    TensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(op.getType())
                             .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (!rhsType) {
      rhs = hlo::scalarToStablehloTensor(rewriter, op, adaptor.getOther(),
                                         outElemTy);
      if (isa<AtenRsubScalarOp>(op)) {
        std::swap(lhs, rhs);
      }
    }

    lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);

    if (!skipMultiplyAlpha(op.getAlpha())) {
      Value alpha = hlo::scalarToStablehloTensor(rewriter, op,
                                                 adaptor.getAlpha(), outElemTy);
      DenseI64ArrayAttr bcastDimensions;
      rhs = rewriter.create<chlo::BroadcastMulOp>(op->getLoc(), rhs, alpha,
                                                  bcastDimensions);
    }

    DenseI64ArrayAttr bcastDimensions;
    rewriter.replaceOpWithNewOp<ChloOpT>(op, outType, lhs, rhs,
                                         bcastDimensions);
    return success();
  }
};
} // namespace

// Binary op legalizations for Mul/Div variants.
namespace {
template <typename AtenOpT, typename ChloOpT>
class ConvertAtenMulDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    TensorType rhsType = dyn_cast<TensorType>(rhs.getType());

    if (!lhsType)
      return op.emitError("only Tensor types supported in StableHLO");

    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhs = lhs;
    } else if (!rhsType) {
      rhs = hlo::scalarToStablehloTensor(rewriter, op, adaptor.getOther(),
                                         outElemTy);
    }
    DenseI64ArrayAttr bcastDimensions;
    lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);
    auto loc = op.getLoc();
    Value result =
        rewriter.create<ChloOpT>(loc, outType, lhs, rhs, bcastDimensions);

    if (!std::is_same<AtenDivTensorModeOp, AtenOpT>() &&
        !std::is_same<AtenDivScalarModeOp, AtenOpT>()) {
      rewriter.replaceOp(op, result);
      return success();
    }

    auto tensorOp = dyn_cast<AtenDivTensorModeOp>(op.getOperation());
    auto opRoundingMode =
        tensorOp
            ? tensorOp.getRoundingMode()
            : cast<AtenDivScalarModeOp>(op.getOperation()).getRoundingMode();

    std::string roundingMode;
    if (!matchPattern(opRoundingMode, m_TorchConstantStr(roundingMode))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant str rounding mode");
    }

    // if trunc and int, do nothing
    if (roundingMode == "trunc" && isa<mlir::FloatType>(outElemTy)) {
      // "trunc" - rounds the results of the division towards zero. Equivalent
      // to C-style integer division.
      auto sign = rewriter.create<stablehlo::SignOp>(loc, result);
      auto abs = rewriter.create<stablehlo::AbsOp>(loc, result);
      auto floor = rewriter.create<stablehlo::FloorOp>(loc, abs);
      result = rewriter.create<stablehlo::MulOp>(loc, sign, floor).getResult();
    }
    if (roundingMode == "floor") {
      // "floor" - rounds the results of the division down. Equivalent to
      // floor division in Python (the // operator)
      if (isa<mlir::FloatType>(outElemTy))
        result = rewriter.create<stablehlo::FloorOp>(loc, result).getResult();
      else if (!outElemTy.isUnsignedInteger()) {
        TensorType defaultIntToFloatType =
            outType.cloneWith(outType.getShape(), rewriter.getF64Type());
        lhs =
            hlo::promoteType(rewriter, op.getLoc(), lhs, defaultIntToFloatType);
        rhs =
            hlo::promoteType(rewriter, op.getLoc(), rhs, defaultIntToFloatType);
        result = rewriter.create<ChloOpT>(loc, defaultIntToFloatType, lhs, rhs,
                                          bcastDimensions);
        result = rewriter.create<stablehlo::FloorOp>(loc, result).getResult();
        result = hlo::promoteType(rewriter, op.getLoc(), result, outType);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// Binary op legalizations for comparator ops.
namespace {
template <typename AtenOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    Value rhs = adaptor.getOther();
    RankedTensorType lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsTy = dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsTy)
      return op.emitError("only Tensor types supported in StableHLO");

    RankedTensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                                   ->convertType(op.getType())
                                   .template cast<RankedTensorType>();

    Type lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (!rhsTy) {
      rhs = hlo::scalarToStablehloTensor(rewriter, op, adaptor.getOther(),
                                         lhsElemTy);
    }

    // TODO: what is the PyTorch default type promotion?
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, lhsTy);

    chlo::ComparisonTypeAttr compareTypeAttr;
    chlo::ComparisonDirectionAttr compareDirectionAttr;

    if (isa<mlir::FloatType>(lhsElemTy)) {
      compareTypeAttr = chlo::ComparisonTypeAttr::get(
          op->getContext(), chlo::ComparisonType::FLOAT);
    } else if (isa<mlir::IntegerType>(lhsElemTy)) {
      compareTypeAttr = chlo::ComparisonTypeAttr::get(
          op->getContext(), chlo::ComparisonType::SIGNED);
    }

    if (std::is_same<AtenOpT, AtenLtTensorOp>() ||
        std::is_same<AtenOpT, AtenLtScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::LT);
    } else if (std::is_same<AtenOpT, AtenGtTensorOp>() ||
               std::is_same<AtenOpT, AtenGtScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::GT);
    } else if (std::is_same<AtenOpT, AtenGeTensorOp>() ||
               std::is_same<AtenOpT, AtenGeScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::GE);
    } else if (std::is_same<AtenOpT, AtenEqTensorOp>() ||
               std::is_same<AtenOpT, AtenEqScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::EQ);
    } else if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
               std::is_same<AtenOpT, AtenNeScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::NE);
    } else if (std::is_same<AtenOpT, AtenLtTensorOp>() ||
               std::is_same<AtenOpT, AtenLtScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::LT);
    } else if (std::is_same<AtenOpT, AtenLeTensorOp>() ||
               std::is_same<AtenOpT, AtenLeScalarOp>()) {
      compareDirectionAttr = chlo::ComparisonDirectionAttr::get(
          op->getContext(), chlo::ComparisonDirection::LE);
    } else {
      return op.emitError("operator haven't been supported");
    }
    DenseI64ArrayAttr bcastDimensions;
    rewriter.replaceOpWithNewOp<chlo::BroadcastCompareOp>(
        op, outType, lhs, rhs, bcastDimensions, compareDirectionAttr,
        compareTypeAttr);
    return success();
  }
};
} // namespace

// Binary op legalizations for Logical And/Or/Xor.
namespace {
template <typename AtenOpT, typename ChloOpT>
class ConvertAtenLogicalBinaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    Value rhs = adaptor.getOther();

    RankedTensorType lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    RankedTensorType rhsTy = dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsTy)
      return op.emitError("lhs must be a ranked tensor type");

    TensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(op.getType())
                             .template cast<TensorType>();
    Type outElemTy = outType.getElementType();
    lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
    if (!rhsTy) {
      rhs = hlo::scalarToStablehloTensor(rewriter, op, rhs, outElemTy);
    }
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);

    DenseI64ArrayAttr bcastDimensions;
    rewriter.replaceOpWithNewOp<ChloOpT>(op, outType, lhs, rhs,
                                         bcastDimensions);
    return success();
  }
};
} // namespace

// AtenTransposeIntOp
namespace {
class ConvertAtenTransposeIntOp
    : public OpConversionPattern<AtenTransposeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTransposeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    int64_t dim0;
    if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0))) {
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    }
    int64_t dim1;
    if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1))) {
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");
    }

    auto inType = cast<RankedTensorType>(self.getType());
    auto inputRank = inType.getRank();
    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    dim0 = toPositiveDim(dim0, inputRank);
    if (!isValidDim(dim0, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim0 out of range");
    }
    dim1 = toPositiveDim(dim1, inputRank);
    if (!isValidDim(dim1, inputRank)) {
      return rewriter.notifyMatchFailure(op, "dim1 out of range");
    }

    SmallVector<int64_t> permValues(inputRank);
    std::iota(std::begin(permValues), std::end(permValues), 0);
    std::swap(permValues[dim0], permValues[dim1]);
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, outType, self,
                                                        permValues);
    return success();
  }
};
} // namespace

// AtenToDtypeOp
template <>
LogicalResult ConvertAtenOp<AtenToDtypeOp>::matchAndRewrite(
    AtenToDtypeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, outType, self);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSizeIntOp>::matchAndRewrite(
    AtenSizeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return op.emitError("only tensor types are currently supported");

  Value dim;
  int64_t dimInt;
  if (matchPattern(op.getDim(), m_TorchConstantInt(&dimInt))) {
    dimInt = toPositiveDim(dimInt, selfType.getRank());
    if (!isValidDim(dimInt, selfType.getRank()))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    dim = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dimInt);
  } else {
    Value inputRank = rewriter.create<arith::ConstantOp>(
        op.getLoc(), rewriter.getI64IntegerAttr(selfType.getRank()));
    dim = toPositiveDimDynamic(rewriter, op.getLoc(), adaptor.getDim(),
                               inputRank);
    dim = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                              rewriter.getIndexType(), dim);
  }

  auto dimSize = rewriter.create<tensor::DimOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.getSelf(), dim);

  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      op, getTypeConverter()->convertType(op.getType()), dimSize);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenWhereSelfOp>::matchAndRewrite(
    AtenWhereSelfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  Value cond = adaptor.getCondition();
  Value other = adaptor.getOther();

  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  // promote self and other types
  self = hlo::promoteType(rewriter, op.getLoc(), self, outType);
  other = hlo::promoteType(rewriter, op.getLoc(), other, outType);

  if (failed(
          broadcastRanks(rewriter, op, self, cond, options.dimSizeIndexBits)))
    return op.emitError("failed broadcast self and condition ranks");

  if (failed(
          broadcastRanks(rewriter, op, other, cond, options.dimSizeIndexBits)))
    return op.emitError("failed broadcast other and condition ranks");

  rewriter.replaceOpWithNewOp<chlo::BroadcastSelectOp>(
      op, getTypeConverter()->convertType(op.getType()),
      ArrayRef<Value>{cond, self, other});
  return success();
}

// AtenBroadcastToOp
template <>
LogicalResult ConvertAtenOp<AtenBroadcastToOp>::matchAndRewrite(
    AtenBroadcastToOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  auto outType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(0).getType()));

  if (options.enableStaticShape && selfTy.hasStaticShape()) {
    Value bcastOp = hlo::promoteAndBroadcast(rewriter, self, outType);
    rewriter.replaceOp(op, bcastOp);
    return success();
  }

  SmallVector<Value> shape;
  if (!(getListConstructElements(adaptor.getSize(), shape))) {
    return op->emitError("desired shape must be a list of scalar");
  }
  SmallVector<Value> bcastShapeVec;
  int64_t totalRank = shape.size();
  int64_t selfRank = selfTy.getRank();
  int64_t leadingRank = totalRank - selfRank;

  for (int64_t i = 0; i < totalRank; ++i) {
    Value dValue = shape[i];
    Value newD;
    int64_t dInt;
    if (i >= leadingRank && matchPattern(dValue, m_TorchConstantInt(&dInt)) &&
        dInt == -1) {
      newD = rewriter.create<mlir::tensor::DimOp>(op->getLoc(), self,
                                                  i - leadingRank);
    } else {
      dValue = rewriter.create<torch::TorchConversion::ToI64Op>(op->getLoc(),
                                                                dValue);
      newD = rewriter.create<mlir::arith::IndexCastOp>(
          op->getLoc(), rewriter.getIndexType(), dValue);
    }
    bcastShapeVec.push_back(newD);
  }

  if (options.dimSizeIndexBits == 32) {
    for (auto &dsize : bcastShapeVec) {
      auto dsizeI64 = rewriter.create<mlir::arith::IndexCastOp>(
          op->getLoc(), rewriter.getI64Type(), dsize);
      dsize = rewriter.create<arith::TruncIOp>(op->getLoc(),
                                               rewriter.getI32Type(), dsizeI64);
    }
  }

  if (bcastShapeVec.size() == 0) {
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outType, self);
  } else {
    Value bcastShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), ValueRange{bcastShapeVec});
    auto dimensionNumbers =
        llvm::to_vector<4>(llvm::seq<int64_t>(leadingRank, totalRank));
    rewriter.replaceOpWithNewOp<stablehlo::DynamicBroadcastInDimOp>(
        op, outType, self, bcastShapeTensor,
        rewriter.getDenseI64ArrayAttr(dimensionNumbers));
  }
  return success();
}

// AtenPermuteOp
template <>
LogicalResult ConvertAtenOp<AtenPermuteOp>::matchAndRewrite(
    AtenPermuteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  // Not a ranked tensor type
  auto inType = dyn_cast<RankedTensorType>(self.getType());
  auto outType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(0).getType()));
  if (!inType)
    return op.emitError("only ranked tensor types with static shapes are "
                        "currently supported");

  SmallVector<int64_t> permValues;
  if (!matchPattern(adaptor.getDims(), m_TorchListOfConstantInts(permValues)))
    return rewriter.notifyMatchFailure(
        op, "only constant dimensions are currently supported");

  int64_t inRank = inType.getRank();
  for (auto &d : permValues) {
    d = toPositiveDim(d, inRank);
    if (!isValidDim(d, inRank))
      return op.emitError("not all dims are valid");
  }

  rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, outType, self,
                                                      permValues);
  return success();
}

// ValueTensorLiteralOp
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType resultType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(0).getType()));

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  // TODO: what about unsigned integer?
  if (auto elements = dyn_cast<DenseIntElementsAttr>(op.getValueAttr())) {
    Type builtinTensorElemTy = resultType.getElementType();
    unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();

    DenseElementsAttr valueAttr =
        elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
          return APInt(bitWidth, v.getSExtValue());
        });
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, resultType,
                                                       valueAttr);
    return success();
  }

  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, resultType,
                                                     adaptor.getValue());
  return success();
}

// AtenTensorIntOp
template <>
LogicalResult ConvertAtenOp<AtenTensorIntOp>::matchAndRewrite(
    AtenTensorIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType resultType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(0).getType()));
  Type outElementType = resultType.getElementType();
  Value innerValue = adaptor.getT();
  Value stablehloTensor =
      hlo::scalarToStablehloTensor(rewriter, op, innerValue, outElementType);
  rewriter.replaceOp(op, stablehloTensor);
  return success();
}

// AtenReciprocalOp
// Reciprocal(x) = Div(1, x)
template <>
LogicalResult ConvertAtenOp<AtenReciprocalOp>::matchAndRewrite(
    AtenReciprocalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto outTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!isa<mlir::FloatType>(inputTy.getElementType())) {
    return op.emitError("only floating-point datatype legalization supported "
                        "for AtenReciprocalOp");
  }

  Value oneTensor = getConstantLike(rewriter, op->getLoc(), 1, input);
  rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, outTy, oneTensor, input);
  return success();
}

// AtenPowTensorScalarOp
template <>
LogicalResult ConvertAtenOp<AtenPowTensorScalarOp>::matchAndRewrite(
    AtenPowTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  auto lhsType = dyn_cast<TensorType>(lhs.getType());
  Value rhs = adaptor.getExponent();
  TensorType rhsType = dyn_cast<TensorType>(rhs.getType());

  if (!lhsType)
    return op.emitError("only Tensor types supported in StableHLO");

  auto outType = OpConversionPattern<AtenPowTensorScalarOp>::getTypeConverter()
                     ->convertType(op.getType())
                     .template cast<TensorType>();

  Type outElemTy = outType.getElementType();
  if (!outElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }

  if (!rhsType) {
    rhs = hlo::scalarToStablehloTensor(rewriter, op, rhs, outElemTy);
  }
  DenseI64ArrayAttr bcastDimensions;
  lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
  rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);
  auto loc = op.getLoc();
  Value result = rewriter.create<chlo::BroadcastPowOp>(loc, outType, lhs, rhs,
                                                       bcastDimensions);

  rewriter.replaceOp(op, result);
  return success();
}

// AtenPowScalarOp
template <>
LogicalResult ConvertAtenOp<AtenPowScalarOp>::matchAndRewrite(
    AtenPowScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  auto lhsType = dyn_cast<TensorType>(lhs.getType());
  Value rhs = adaptor.getExponent();
  auto rhsType = dyn_cast<TensorType>(rhs.getType());

  if (!rhsType)
    return op.emitError("only Tensor types supported in StableHLO");

  auto outType = cast<TensorType>(
      OpConversionPattern<AtenPowScalarOp>::getTypeConverter()->convertType(
          op.getType()));

  Type outElemTy = outType.getElementType();
  if (!outElemTy.isIntOrFloat()) {
    return op.emitError(
        "only floating-point or integer datatype legalization supported");
  }

  if (!lhsType) {
    lhs = hlo::scalarToStablehloTensor(rewriter, op, lhs, outElemTy);
  }
  DenseI64ArrayAttr bcastDimensions;
  lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
  rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);
  auto loc = op.getLoc();
  Value result = rewriter.create<chlo::BroadcastPowOp>(loc, outType, lhs, rhs,
                                                       bcastDimensions);

  rewriter.replaceOp(op, result);
  return success();
}

// PrimNumToTensorScalarOp
template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType outputType = cast<RankedTensorType>(
      getTypeConverter()->convertType(op->getResult(0).getType()));
  auto outputElemType = outputType.getElementType();
  Value stablehloTensor = hlo::scalarToStablehloTensor(
      rewriter, op, adaptor.getA(), outputElemType);
  rewriter.replaceOp(op, stablehloTensor);
  return success();
}

// AtenScalarImplicitOp
template <>
LogicalResult ConvertAtenOp<AtenScalarImplicitOp>::matchAndRewrite(
    AtenScalarImplicitOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Type inputDtype = cast<BaseTensorType>(op.getA().getType()).getDtype();
  Type resultType =
      this->getTypeConverter()->convertType(op->getResult(0).getType());
  auto result = rewriter.create<tensor::ExtractOp>(loc, adaptor.getA());

  rewriter.replaceOp(
      op, convertScalarToDtype(rewriter, loc, result, resultType, inputDtype));
  return success();
}

// AtenContiguousOp
// Ref: TosaToTosa.cpp for implementation details
template <>
LogicalResult ConvertAtenOp<AtenContiguousOp>::matchAndRewrite(
    AtenContiguousOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return op.emitError("only tensor types are currently supported");

  // FIXME: memory_format is not handled.

  rewriter.replaceOp(op, adaptor.getSelf());

  return success();
}

// AtenReluOp
// Relu(x) = Max(0, x)
template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  auto lhsTy = cast<RankedTensorType>(lhs.getType());
  auto lhsElemTy = lhsTy.getElementType();

  if (!isa<mlir::FloatType>(lhsElemTy)) {
    return op->emitError("only float tensor in relu op is supported");
  }

  Value zeroTensor;
  zeroTensor = getConstantLike(
      rewriter, op->getLoc(),
      APFloat::getZero(cast<mlir::FloatType>(lhsElemTy).getFloatSemantics(),
                       false),
      lhs);
  rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, lhs, zeroTensor);
  return success();
}

// Convert a Aten::GELU to HLO
// Gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.getSelf();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy) {
    return op.emitError("only ranked tensor type is supported.");
  }

  Value one = getConstantLike(rewriter, loc, 1.0, input);
  Value two = getConstantLike(rewriter, loc, 2.0, input);
  Value half = getConstantLike(rewriter, loc, 0.5, input);
  auto rsqrtTwo = rewriter.create<mlir::stablehlo::RsqrtOp>(loc, two);
  auto erfElement = rewriter.create<stablehlo::MulOp>(loc, input, rsqrtTwo);
  auto erf = rewriter.create<mlir::chlo::ErfOp>(loc, erfElement);
  auto erfAdd = rewriter.create<stablehlo::AddOp>(loc, erf, one);
  auto halfMul = rewriter.create<stablehlo::MulOp>(loc, erfAdd, half);
  rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, input, halfMul);
  return success();
}

// AtenLog2Op
template <>
LogicalResult ConvertAtenOp<AtenLog2Op>::matchAndRewrite(
    AtenLog2Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy) {
    return op.emitError("only ranked tensor type is supported.");
  }
  auto outTy = cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  input = hlo::promoteType(rewriter, op.getLoc(), input, outTy);

  auto two = getConstantLike(rewriter, op.getLoc(), 2.0, input);
  auto log2Op = rewriter.create<stablehlo::LogOp>(op.getLoc(), two);
  auto logInputOp = rewriter.create<stablehlo::LogOp>(op.getLoc(), input);

  rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, outTy, logInputOp, log2Op);
  return success();
}

// AtenLog10Op
template <>
LogicalResult ConvertAtenOp<AtenLog10Op>::matchAndRewrite(
    AtenLog10Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy) {
    return op.emitError("only ranked tensor type is supported.");
  }

  auto outTy = cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  input = hlo::promoteType(rewriter, op.getLoc(), input, outTy);

  auto ten = getConstantLike(rewriter, op.getLoc(), 10.0, input);
  auto log10Op = rewriter.create<stablehlo::LogOp>(op.getLoc(), ten);
  auto logInputOp = rewriter.create<stablehlo::LogOp>(op.getLoc(), input);

  rewriter.replaceOpWithNewOp<stablehlo::DivOp>(op, outTy, logInputOp, log10Op);
  return success();
}

// AtenErfOp
template <>
LogicalResult ConvertAtenOp<AtenErfOp>::matchAndRewrite(
    AtenErfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputType = cast<TensorType>(input.getType());
  if (!isa<mlir::FloatType>(inputType.getElementType())) {
    return rewriter.notifyMatchFailure(op, "only float tensor is supported");
  }
  rewriter.replaceOpWithNewOp<chlo::ErfOp>(
      op, getTypeConverter()->convertType(op.getType()), input);
  return success();
}

// AtenBatchNormOp
template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  auto inputTy = cast<RankedTensorType>(input.getType());
  Value weight = adaptor.getWeight();
  Value bias = adaptor.getBias();
  Value runningMean = adaptor.getRunningMean();
  Value runningVar = adaptor.getRunningVar();
  // momentum is ignored
  Value momentum = adaptor.getMomentum();
  (void)momentum;

  // handle feature index, see torch's BatchNorm1d, BatchNorm2d, BatchNorm3d,
  // all of NC, NCL, NCHW, NCDHW's feature index is 1.
  int64_t feature_index = 1;

  if (!isa<mlir::FloatType>(inputTy.getElementType())) {
    return op.emitError("only input tensor of float type is supported");
  }
  auto inputElemTy = cast<mlir::FloatType>(inputTy.getElementType());

  Value channelDim =
      rewriter.create<tensor::DimOp>(op->getLoc(), input, feature_index);

  if (options.dimSizeIndexBits == 32) {
    auto channelDimI64 = rewriter.create<mlir::arith::IndexCastOp>(
        op->getLoc(), rewriter.getI64Type(), channelDim);
    channelDim = rewriter.create<arith::TruncIOp>(
        op->getLoc(), rewriter.getI32Type(), channelDimI64);
  }

  Value channelShape = rewriter.create<tensor::FromElementsOp>(
      op->getLoc(), ValueRange{channelDim});
  if (failed(checkNotNone(rewriter, op, weight))) {
    weight = hlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 1),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, bias))) {
    bias = hlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 0),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, runningVar))) {
    runningVar = hlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 1),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, runningMean))) {
    runningMean = hlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 0),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }

  auto weightTy = cast<RankedTensorType>(weight.getType());
  auto biasTy = cast<RankedTensorType>(bias.getType());
  auto runningMeanTy = cast<RankedTensorType>(runningMean.getType());
  auto runningVarTy = cast<RankedTensorType>(runningVar.getType());

  if (weightTy.getRank() != 1 || biasTy.getRank() != 1 ||
      runningMeanTy.getRank() != 1 || runningVarTy.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "expect weight, bias, running_mean and running_var to be rank 1");
  }
  if (!isa<mlir::FloatType>(weightTy.getElementType()) ||
      !isa<mlir::FloatType>(biasTy.getElementType()) ||
      !isa<mlir::FloatType>(runningMeanTy.getElementType()) ||
      !isa<mlir::FloatType>(runningVarTy.getElementType())) {
    return op.emitError("only float weight/bias/runningMean/runningVar tensor "
                        "of float type is supported");
  }

  double eps = 0.0;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op, "non-float(double) eps unsupported");
  }
  bool training = false;
  if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training))) {
    return rewriter.notifyMatchFailure(op, "non-bool training unsupported");
  }
  // TODO: handle cudnnEnabled parameter. Here, we just ignore it!
  bool cudnnEnabled = false;
  if (!matchPattern(op.getCudnnEnabled(), m_TorchConstantBool(&cudnnEnabled))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-bool cudnn_enabled unsupported");
  }
  if (training) {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    Type batchMeanOrVarTy =
        RankedTensorType::get(weightTy.getShape(), inputTy.getElementType());

    Value output;
    // supported mixed types, like input type is fp16 and weight type is fp32.
    if (inputTy.getElementType() != weightTy.getElementType()) {
      RankedTensorType convertedType = inputTy;
      if (cast<FloatType>(weightTy.getElementType()).getWidth() >
          cast<FloatType>(inputTy.getElementType()).getWidth()) {
        convertedType = RankedTensorType::get(inputTy.getShape(),
                                              weightTy.getElementType());
      }
      input = hlo::promoteType(rewriter, op.getLoc(), input, convertedType);
      weight = hlo::promoteType(rewriter, op.getLoc(), weight, convertedType);
      bias = hlo::promoteType(rewriter, op.getLoc(), bias, convertedType);
      auto batchNormTrainingResult =
          rewriter.create<stablehlo::BatchNormTrainingOp>(
              op.getLoc(), outputTy, batchMeanOrVarTy, batchMeanOrVarTy, input,
              weight, bias, rewriter.getF32FloatAttr(eps),
              rewriter.getI64IntegerAttr(feature_index));
      output = hlo::promoteType(rewriter, op.getLoc(),
                                batchNormTrainingResult.getResult(0),
                                cast<TensorType>(outputTy));
    } else {
      auto batchNormTrainingResult =
          rewriter.create<stablehlo::BatchNormTrainingOp>(
              op.getLoc(), outputTy, batchMeanOrVarTy, batchMeanOrVarTy, input,
              weight, bias, rewriter.getF32FloatAttr(eps),
              rewriter.getI64IntegerAttr(feature_index));
      output = batchNormTrainingResult.getResult(0);
    }
    rewriter.replaceOp(op, output);
    return success();
  } else {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    SmallVector<int64_t, 4> castShape{inputTy.getShape().begin(),
                                      inputTy.getShape().end()};
    castShape[1] = weightTy.getShape()[0];
    auto castTy = RankedTensorType::get(castShape, inputTy.getElementType());
    // Feature counts must match among operands of
    // stablehlo::BatchNormInferenceOp.
    Value inputCasted =
        rewriter.create<tensor::CastOp>(op.getLoc(), castTy, input);

    Value output;
    // supported mixed types, like input type is fp16 and weight type is fp32.
    if (inputTy.getElementType() != weightTy.getElementType()) {
      RankedTensorType convertedType = inputTy;
      if (cast<FloatType>(weightTy.getElementType()).getWidth() >
          cast<FloatType>(inputTy.getElementType()).getWidth()) {
        convertedType = RankedTensorType::get(inputTy.getShape(),
                                              weightTy.getElementType());
      }
      input =
          hlo::promoteType(rewriter, op.getLoc(), inputCasted, convertedType);
      weight = hlo::promoteType(rewriter, op.getLoc(), weight, convertedType);
      bias = hlo::promoteType(rewriter, op.getLoc(), bias, convertedType);
      runningMean =
          hlo::promoteType(rewriter, op.getLoc(), runningMean, convertedType);
      runningVar =
          hlo::promoteType(rewriter, op.getLoc(), runningVar, convertedType);
      Value bnResult = rewriter.create<stablehlo::BatchNormInferenceOp>(
          op.getLoc(), convertedType, input, weight, bias, runningMean,
          runningVar, rewriter.getF32FloatAttr(eps),
          rewriter.getI64IntegerAttr(feature_index));
      output = hlo::promoteType(rewriter, op.getLoc(), bnResult,
                                cast<TensorType>(outputTy));
    } else {
      output = rewriter.create<stablehlo::BatchNormInferenceOp>(
          op.getLoc(), inputCasted.getType(), inputCasted, weight, bias,
          runningMean, runningVar,
          // 'epsilon' must satisfy constraint: 32-bit float attribute.
          rewriter.getF32FloatAttr(eps),
          rewriter.getI64IntegerAttr(feature_index));
    }
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, outputTy, output);
    return success();
  }
}

// AtenNativeLayerNormOp
template <>
LogicalResult ConvertAtenOp<AtenNativeLayerNormOp>::matchAndRewrite(
    AtenNativeLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getInput();
  auto inputTy = cast<RankedTensorType>(input.getType());
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  Value weight = adaptor.getWeight();
  Value bias = adaptor.getBias();

  if (!inputTy.hasStaticShape()) {
    return op->emitError("dynamic shaped input is not supported");
  }

  SmallVector<int64_t> normalizedShape;
  if (!matchPattern(op.getNormalizedShape(),
                    m_TorchListOfConstantInts(normalizedShape))) {
    return rewriter.notifyMatchFailure(
        op, "normalized_shape must be a list of const int");
  }
  double eps = 0;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op,
                                       "non const float eps is unsupported");
  }
  if (failed(checkNotNone(rewriter, op, weight)) ||
      failed(checkNotNone(rewriter, op, bias))) {
    return op->emitError("none weight or bias is unsupported");
  }
  auto weightTy = cast<RankedTensorType>(weight.getType());
  auto biasTy = cast<RankedTensorType>(bias.getType());

  if (!isa<mlir::FloatType>(inputTy.getElementType()) ||
      !isa<mlir::FloatType>(biasTy.getElementType()) ||
      !isa<mlir::FloatType>(weightTy.getElementType())) {
    return op->emitError("currently only float data type are supported");
  }
  int64_t normalizedShapeRank = normalizedShape.size();
  if (weightTy.getRank() != normalizedShapeRank ||
      biasTy.getRank() != normalizedShapeRank ||
      inputRank < normalizedShapeRank || normalizedShapeRank < 1) {
    return rewriter.notifyMatchFailure(op, "input or weight or bias shape or"
                                           "normalized shape not compatible");
  }
  for (int64_t i = 1; i <= normalizedShapeRank; i++) {
    if (inputShape[inputRank - i] != normalizedShape[normalizedShapeRank - i] ||
        weightTy.getShape()[normalizedShapeRank - i] !=
            normalizedShape[normalizedShapeRank - i] ||
        biasTy.getShape()[normalizedShapeRank - i] !=
            normalizedShape[normalizedShapeRank - i]) {
      return op.emitError("mismatching contracting dimension");
    }
  }

  // Flatten dims to fit batch_norm operation.
  int64_t numFeatureDimSize = 1;
  int64_t numEmbeddingDimSize = 1;
  for (int64_t i = 0; i < inputRank - normalizedShapeRank; i++) {
    numFeatureDimSize *= inputShape[i];
  }
  for (int64_t i = 0; i < normalizedShapeRank; i++) {
    numEmbeddingDimSize *= normalizedShape[i];
  }
  SmallVector<int64_t> inputFlattenShape{1, numFeatureDimSize,
                                         numEmbeddingDimSize};
  SmallVector<int64_t> meanOrVarStablehloOutShape{numFeatureDimSize};

  auto stablehloBatchNormOutTy =
      RankedTensorType::get(inputFlattenShape, inputTy.getElementType());
  auto stablehloBathNormOutMeanOrVarTy = RankedTensorType::get(
      meanOrVarStablehloOutShape, inputTy.getElementType());

  // Reshape input
  auto stablehloInput = rewriter.create<stablehlo::DynamicReshapeOp>(
      op->getLoc(), stablehloBatchNormOutTy, input,
      hlo::getConstTensor(rewriter, op, llvm::ArrayRef(inputFlattenShape),
                          {static_cast<int64_t>(inputFlattenShape.size())})
          .value());

  // Generate "scale" and "offset" Value for stablehlo.BatchNormTrainingOp.
  SmallVector<APFloat> zeroConstVec(
      numFeatureDimSize, APFloat::getZero(inputTy.getElementType()
                                              .cast<mlir::FloatType>()
                                              .getFloatSemantics()));
  SmallVector<APFloat> oneConstVec(
      numFeatureDimSize,
      APFloat(
          cast<mlir::FloatType>(inputTy.getElementType()).getFloatSemantics(),
          1));
  auto oneOrZeroConstType =
      RankedTensorType::get({numFeatureDimSize}, inputTy.getElementType());

  Value scale = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), oneOrZeroConstType,
      DenseElementsAttr::get(oneOrZeroConstType, oneConstVec));
  Value offset = rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), oneOrZeroConstType,
      DenseElementsAttr::get(oneOrZeroConstType, zeroConstVec));
  auto batchNormTrainingResult =
      rewriter.create<stablehlo::BatchNormTrainingOp>(
          op->getLoc(), stablehloBatchNormOutTy,
          stablehloBathNormOutMeanOrVarTy, stablehloBathNormOutMeanOrVarTy,
          stablehloInput, scale, offset, rewriter.getF32FloatAttr(eps),
          rewriter.getI64IntegerAttr(1));

  // Reshape back
  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(0)));
  auto outputMeanOrVarTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType(1)));

  auto output = rewriter.create<stablehlo::DynamicReshapeOp>(
      op->getLoc(), outputTy, batchNormTrainingResult.getResult(0),
      hlo::getConstTensor(rewriter, op, outputTy.getShape(),
                          {static_cast<int64_t>(outputTy.getShape().size())})
          .value());
  auto mean = rewriter.create<stablehlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(1),
      hlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .value());
  auto var = rewriter.create<stablehlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(2),
      hlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .value());

  // Apply affine transform: output x weight + bias [element-wise]
  auto bcastedWeight = hlo::promoteAndBroadcast(rewriter, weight, outputTy);
  auto bcastedBias = hlo::promoteAndBroadcast(rewriter, bias, outputTy);
  auto outputMulWeight =
      rewriter.create<stablehlo::MulOp>(op->getLoc(), output, bcastedWeight);
  auto finalOuput = rewriter.create<stablehlo::AddOp>(
      op->getLoc(), outputMulWeight, bcastedBias);
  rewriter.replaceOp(op, {finalOuput, mean, var});
  return success();
}

// AtenCatOp
template <>
LogicalResult ConvertAtenOp<AtenCatOp>::matchAndRewrite(
    AtenCatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op,
                                       "only constant dim param is supported");
  }
  dim = toPositiveDim(dim, outType.getRank());
  if (!isValidDim(dim, outType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<Value> torchTensors;
  if (!getListConstructElements(op.getTensors(), torchTensors)) {
    return rewriter.notifyMatchFailure(
        op, "input should comes from a PrimListConstructOp");
  }
  SmallVector<Value> builtinTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), torchTensors);

  // Promote type
  for (auto &v : builtinTensors) {
    v = hlo::promoteType(rewriter, op->getLoc(), v, outType);
  }

  rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
      op, outType, ValueRange(builtinTensors), dim);
  return success();
}

// AtenNumelOp
template <>
LogicalResult ConvertAtenOp<AtenNumelOp>::matchAndRewrite(
    AtenNumelOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();
  auto selfTy = dyn_cast<RankedTensorType>(self.getType());
  size_t rank = selfTy.getRank();

  Type intType = rewriter.getIntegerType(options.dimSizeIndexBits);
  auto loc = op->getLoc();
  Value numel = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));
  for (size_t d = 0; d < rank; ++d) {
    Value dimSize = rewriter.create<arith::IndexCastOp>(
        loc, intType, rewriter.create<tensor::DimOp>(loc, self, d));
    numel = rewriter.create<arith::MulIOp>(loc, numel, dimSize);
  }

  auto outTy = getTypeConverter()->convertType(op.getType());
  if (outTy != numel.getType()) {
    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, outTy, numel);
  } else {
    rewriter.replaceOp(op, numel);
  }
  return success();
}

// AtenClampOp
template <>
LogicalResult ConvertAtenOp<AtenClampOp>::matchAndRewrite(
    AtenClampOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputElemType = inputType.getElementType();
  Value minValue = adaptor.getMin();
  Value maxValue = adaptor.getMax();
  if (failed(checkNotNone(rewriter, op, minValue)) &&
      failed(checkNotNone(rewriter, op, maxValue))) {
    return rewriter.notifyMatchFailure(
        op, "this op should be folded as its `min` and `max` both are none");
  } else if (failed(checkNotNone(rewriter, op, minValue))) {
    maxValue =
        hlo::scalarToStablehloTensor(rewriter, op, maxValue, inputElemType);
    auto minInfo = getMinValueOfDtype(op, inputElemType, rewriter);
    if (failed(minInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to generate min value of dtype");
    }
    minValue = *minInfo;
  } else if (failed(checkNotNone(rewriter, op, maxValue))) {
    minValue =
        hlo::scalarToStablehloTensor(rewriter, op, minValue, inputElemType);
    auto maxInfo = getMaxValueOfDtype(op, inputElemType, rewriter);
    if (failed(maxInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to generate max value of dtype");
    }
    maxValue = *maxInfo;
  } else {
    minValue =
        hlo::scalarToStablehloTensor(rewriter, op, minValue, inputElemType);
    maxValue =
        hlo::scalarToStablehloTensor(rewriter, op, maxValue, inputElemType);
  }
  rewriter.replaceOpWithNewOp<stablehlo::ClampOp>(op, minValue, input,
                                                  maxValue);
  return success();
}

// AtenClampTensorOp
template <>
LogicalResult ConvertAtenOp<AtenClampTensorOp>::matchAndRewrite(
    AtenClampTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.getSelf();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputElemType = inputType.getElementType();
  Value minValue = adaptor.getMin();
  Value maxValue = adaptor.getMax();
  auto minIsNotNone = checkNotNone(rewriter, op, minValue);
  auto maxIsNotNone = checkNotNone(rewriter, op, maxValue);
  if (failed(minIsNotNone) && failed(maxIsNotNone)) {
    return rewriter.notifyMatchFailure(
        op, "this op should be folded as its `min` and `max` both are none");
  } else if (failed(minIsNotNone)) {
    auto minInfo = getMinValueOfDtype(op, inputElemType, rewriter);
    if (failed(minInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to generate min value of dtype");
    }
    minValue = *minInfo;
  } else if (failed(maxIsNotNone)) {
    auto maxInfo = getMaxValueOfDtype(op, inputElemType, rewriter);
    if (failed(maxInfo)) {
      return rewriter.notifyMatchFailure(
          op, "failed to generate max value of dtype");
    }
    maxValue = *maxInfo;
  }
  if (inputType.hasStaticShape()) {
    minValue = hlo::promoteAndBroadcast(rewriter, minValue, inputType);
    maxValue = hlo::promoteAndBroadcast(rewriter, maxValue, inputType);
  }
  rewriter.replaceOpWithNewOp<stablehlo::ClampOp>(op, minValue, input,
                                                  maxValue);
  return success();
}

// AtenArangeStartStepOp
// aten.arange.start_step = range(ceil((end-start)/step)) * step + start.
template <>
LogicalResult ConvertAtenOp<AtenArangeStartStepOp>::matchAndRewrite(
    AtenArangeStartStepOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op->getLoc();

  // Get element type of resultType as dtype
  auto outType = this->getTypeConverter()
                     ->convertType(op.getType())
                     .cast<RankedTensorType>();
  auto dtype = outType.getElementType();
  if (!isa<mlir::IntegerType>(dtype) && !isa<mlir::FloatType>(dtype)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: only int or float dtype supported");
  }

  Value start =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getStart(), dtype);
  Value end =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getEnd(), dtype);
  Value step =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getStep(), dtype);

  // Get length of the 1-d output tensor
  Value subOut = rewriter.create<stablehlo::SubtractOp>(loc, end, start);
  // promote div to f64
  Type divType = RankedTensorType::get({}, rewriter.getF64Type());
  Value divOut = rewriter.create<stablehlo::DivOp>(
      loc, rewriter.create<stablehlo::ConvertOp>(loc, divType, subOut),
      rewriter.create<stablehlo::ConvertOp>(loc, divType, step));
  // ceil to i64
  Value resultLength = rewriter.create<stablehlo::ConvertOp>(
      loc, RankedTensorType::get({}, rewriter.getI64Type()),
      rewriter.create<stablehlo::CeilOp>(loc, divOut));
  resultLength = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get({1}, rewriter.getI64Type()), resultLength);

  Value window =
      rewriter.create<stablehlo::DynamicIotaOp>(loc, outType, resultLength, 0);
  DenseI64ArrayAttr broadcastDimensions;
  Value mulOut = rewriter.create<chlo::BroadcastMulOp>(loc, window, step,
                                                       broadcastDimensions);
  rewriter.replaceOpWithNewOp<chlo::BroadcastAddOp>(op, mulOut, start,
                                                    broadcastDimensions);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenConstantPadNdOp>::matchAndRewrite(
    AtenConstantPadNdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = self.getType().cast<RankedTensorType>();
  auto selfElemTy = selfTy.getElementType();
  int64_t rank = selfTy.getRank();

  SmallVector<int64_t> padInts;
  if (!matchPattern(op.getPad(), m_TorchListOfConstantInts(padInts)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant int pad ranges");
  uint64_t padRank = padInts.size() / 2;
  if (padRank * 2 != padInts.size())
    return rewriter.notifyMatchFailure(op, "pad range size is not even");
  if (rank < 0 || padRank > (uint64_t)rank)
    return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

  // Initialize low/high paddings with 0 for all the dims.
  SmallVector<int64_t> lowPadding(/*Size=*/rank, /*Value=*/0);
  SmallVector<int64_t> highPadding(/*Size=*/rank, /*Value=*/0);
  // Add the requested padding - note op.pad() is highest dim first ordered
  // pairs of low,high.
  // Add the requested padding - note op.pad() is highest dim first ordered
  // pairs of low,high.
  for (uint64_t i = 0; i < padRank; ++i) {
    lowPadding[rank - i - 1] = padInts[i * 2];
    highPadding[rank - i - 1] = padInts[i * 2 + 1];
  }

  Value constantValue = hlo::scalarToStablehloTensor(
      rewriter, op, adaptor.getValue(), selfElemTy);

  SmallVector<int64_t> interiorPadding(rank, 0);
  rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
      op, self, constantValue, lowPadding, highPadding, interiorPadding);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenGeluBackwardOp>::matchAndRewrite(
    AtenGeluBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.getSelf();
  auto outType =
      cast<TensorType>(this->getTypeConverter()->convertType(op.getType()));
  if (!outType) {
    return op.emitError("only tensor type is supported");
  }
  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return rewriter.notifyMatchFailure(op, "Unsupported value of approximate");
  }
  // Create constant value
  Value kAlpha = getConstantLike(rewriter, loc, 0.70710678118654752440, input);
  Value cstAlpha0 =
      getConstantLike(rewriter, loc, 1.12837916709551257390, input);
  Value half = getConstantLike(rewriter, loc, .5, input);
  Value one = getConstantLike(rewriter, loc, 1.0, input);
  Value negHalf = getConstantLike(rewriter, loc, -0.5, input);

  // Compute
  Value kBeta0 =
      rewriter.create<stablehlo::MulOp>(loc, outType, kAlpha, cstAlpha0);
  Value kBeta = rewriter.create<stablehlo::MulOp>(loc, outType, kBeta0, half);
  Value erfArg = rewriter.create<stablehlo::MulOp>(loc, outType, kAlpha,
                                                   adaptor.getSelf());
  Value erf = rewriter.create<mlir::chlo::ErfOp>(loc, outType, erfArg);
  Value erfAdd = rewriter.create<stablehlo::AddOp>(loc, outType, erf, one);
  Value cdf = rewriter.create<stablehlo::MulOp>(loc, outType, erfAdd, half);
  Value inputSquared = rewriter.create<stablehlo::MulOp>(
      loc, outType, adaptor.getSelf(), adaptor.getSelf());
  Value negHalfInputSquared =
      rewriter.create<stablehlo::MulOp>(loc, outType, inputSquared, negHalf);
  Value expRes =
      rewriter.create<stablehlo::ExpOp>(loc, outType, negHalfInputSquared);
  Value pdf = rewriter.create<stablehlo::MulOp>(loc, outType, kBeta, expRes);
  Value pdfTimesInput =
      rewriter.create<stablehlo::MulOp>(loc, outType, pdf, adaptor.getSelf());
  Value pdfTimesInputAddCdf =
      rewriter.create<stablehlo::AddOp>(loc, outType, pdfTimesInput, cdf);
  rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
      op, outType, adaptor.getGradOutput(), pdfTimesInputAddCdf);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPowTensorTensorOp>::matchAndRewrite(
    AtenPowTensorTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  auto lhsTy = cast<TensorType>(lhs.getType());
  Value rhs = adaptor.getExponent();
  auto rhsTy = cast<TensorType>(rhs.getType());

  if (!lhsTy || !rhsTy)
    return op.emitError("only Tensor types supported");

  auto outTy =
      cast<TensorType>(this->getTypeConverter()->convertType(op.getType()));

  lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outTy);
  rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outTy);

  rewriter.replaceOpWithNewOp<chlo::BroadcastPowOp>(op, outTy, lhs, rhs,
                                                    /*broadcast_attr*/ nullptr);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUniformOp>::matchAndRewrite(
    AtenUniformOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  Value generator = adaptor.getGenerator();
  Location loc = op.getLoc();

  if (!isa<Torch::NoneType>(generator.getType()))
    return rewriter.notifyMatchFailure(
        op, "The generator has to be None because only global default "
            "generator is supported");

  auto elements = cast<RankedTensorType>(self.getType()).getShape();
  if (llvm::any_of(elements,
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return rewriter.notifyMatchFailure(op, "Dynamic shape support TBD");
  auto shape_tensor = rewriter.create<stablehlo::ConstantOp>(
      loc, rewriter.getI64TensorAttr(elements));
  auto outTy = getTypeConverter()->convertType(op.getType());
  auto outElemTy = cast<RankedTensorType>(outTy).getElementType();
  Value from =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getFrom(), outElemTy);
  Value to =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getTo(), outElemTy);
  rewriter.replaceOpWithNewOp<stablehlo::RngOp>(
      op, outTy, from, to, shape_tensor, stablehlo::RngDistribution::UNIFORM);
  return success();
}

// Converts `aten.empty.memory_format` to `tensor.empty` op.
template <>
LogicalResult ConvertAtenOp<AtenEmptyMemoryFormatOp>::matchAndRewrite(
    AtenEmptyMemoryFormatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // TODO: Add support pin_memory and memory_format features.
  // At this point all tensors should have value semantics, and hence the
  // `layout` check can be ignored.

  // The pin_memory should be either `False` or `none`.
  bool pinMemory;
  if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: pin_memory must be either None or false");

  // Only `none`, `contiguous` and `preserve` memory_format is supported.
  if (!isa<Torch::NoneType>(op.getMemoryFormat().getType())) {
    int64_t memoryFormat;
    if (!matchPattern(op.getMemoryFormat(), m_TorchConstantInt(&memoryFormat)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the memory format should be specified in "
              "an integer constant");
    if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
        memoryFormat != torch_upstream::MemoryFormat::Preserve)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only none, contiguous and preserve "
              "memory_format is supported");
  }

  if (!isa<Torch::NoneType>(op.getDevice().getType())) {
    std::string device;
    if (!matchPattern(op.getDevice(), m_TorchConstantDevice(device)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: device must be a constant str");
  }

  // TODO: Add support for non-strided layout.
  // torch.layout is by default strided i.e. 0.
  if (!isa<Torch::NoneType>(op.getLayout().getType())) {
    int64_t tensorLayout;
    if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: layout must be a constant");
    else if (tensorLayout != torch_upstream::Layout::Strided)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: layout is expected to be strided");
  }

  Location loc = op.getLoc();
  const TypeConverter *typeConverter = this->getTypeConverter();
  SmallVector<Value> resultSizeTorchInt, resultSize, resultSizeIndex;
  if (!getListConstructElements(op.getSize(), resultSizeTorchInt)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: size must be constructed using ListConstruct");
  }
  resultSize =
      getTypeConvertedValues(rewriter, loc, typeConverter, resultSizeTorchInt);
  for (auto size : resultSize)
    resultSizeIndex.push_back(castIntToIndex(rewriter, loc, size));

  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  Type resultElementType;
  if (isa<Torch::NoneType>(op.getDtype().getType())) {
    resultElementType = resultType.getElementType();
  } else {
    int64_t dtypeInt;
    if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dtype must be a constant integer or none");
    FailureOr<Type> maybeResultElementType =
        torch_to_stablehlo::getBackendTypeForScalarType(
            op->getContext(), (torch_upstream::ScalarType)dtypeInt);
    if (failed(maybeResultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "unable to convert `dtypeInt` to builtin type");
    }
    resultElementType = *maybeResultElementType;
  }

  // Create an uninitialized tensor of `resultSize` shape.
  Value initTensor = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(resultSizeIndex), resultElementType);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, initTensor);
  return success();
}

// RuntimeAssertOp
namespace {
class ConvertRuntimeAssertOp : public OpConversionPattern<RuntimeAssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeAssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool condition;
    if (!matchPattern(op.getCondition(), m_TorchConstantBool(&condition))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: condition must be a constant");
    }
    if (!condition) {
      return op->emitError("condition must be true");
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// AtenFillScalarOp
template <>
LogicalResult ConvertAtenOp<AtenFillScalarOp>::matchAndRewrite(
    AtenFillScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  auto dtype = outType.getElementType();
  Value scalarTensor =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getValue(), dtype);
  Value shapeTensor =
      rewriter.create<shape::ShapeOfOp>(op->getLoc(), adaptor.getSelf());
  Value bcastScalar = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
      op->getLoc(), outType, scalarTensor, shapeTensor,
      rewriter.getDenseI64ArrayAttr({}));
  rewriter.replaceOp(op, bcastScalar);
  return success();
}

// AtenFlipOp
template <>
LogicalResult ConvertAtenOp<AtenFlipOp>::matchAndRewrite(
    AtenFlipOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto outType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  SmallVector<int64_t> dims;
  if (!matchPattern(op.getDims(), m_TorchListOfConstantInts(dims))) {
    return rewriter.notifyMatchFailure(op, "dims must be a list of const int");
  }
  for (unsigned i = 0, e = dims.size(); i < e; i++) {
    dims[i] = toPositiveDim(dims[i], outType.getRank());
    if (!isValidDim(dims[i], outType.getRank())) {
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    }
  }

  rewriter.replaceOpWithNewOp<stablehlo::ReverseOp>(op, outType, self, dims);
  return success();
}

// AtenRemainderTensorOp
template <>
LogicalResult ConvertAtenOp<AtenRemainderTensorOp>::matchAndRewrite(
    AtenRemainderTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  Value rhs = adaptor.getOther();

  auto resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  lhs = hlo::promoteType(rewriter, op->getLoc(), lhs, resultType);
  rhs = hlo::promoteType(rewriter, op->getLoc(), rhs, resultType);
  rewriter.replaceOpWithNewOp<stablehlo::RemOp>(op, lhs, rhs);
  return success();
}

// AtenFmodTensorOp
// torch.fmod(a, b) == a - a.div(b, rounding_mode="trunc") * b
template <>
LogicalResult ConvertAtenOp<AtenFmodTensorOp>::matchAndRewrite(
    AtenFmodTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  Value lhs = adaptor.getSelf();
  Value rhs = adaptor.getOther();

  auto resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  lhs = hlo::promoteType(rewriter, op->getLoc(), lhs, resultType);
  rhs = hlo::promoteType(rewriter, op->getLoc(), rhs, resultType);

  stablehlo::MulOp mul;
  auto div = rewriter.create<stablehlo::DivOp>(loc, lhs, rhs);
  if (isa<mlir::FloatType>(resultType.getElementType())) {
    // rounding mode is trunc
    auto sign = rewriter.create<stablehlo::SignOp>(loc, div);
    auto abs = rewriter.create<stablehlo::AbsOp>(loc, div);
    auto floor = rewriter.create<stablehlo::FloorOp>(loc, abs);
    auto trunc = rewriter.create<stablehlo::MulOp>(loc, sign, floor);
    mul = rewriter.create<stablehlo::MulOp>(loc, trunc, rhs);
  } else {
    mul = rewriter.create<stablehlo::MulOp>(loc, div, rhs);
  }
  rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(op, lhs, mul);
  return success();
}

// AtenBitwiseLeftShiftTensorOp
template <>
LogicalResult ConvertAtenOp<AtenBitwiseLeftShiftTensorOp>::matchAndRewrite(
    AtenBitwiseLeftShiftTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  Value rhs = adaptor.getOther();

  auto resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  rhs = hlo::promoteAndBroadcast(rewriter, rhs, resultType);
  rewriter.replaceOpWithNewOp<stablehlo::ShiftLeftOp>(op, lhs, rhs);
  return success();
}

// AtenBitwiseRightShiftTensorOp
template <>
LogicalResult ConvertAtenOp<AtenBitwiseRightShiftTensorOp>::matchAndRewrite(
    AtenBitwiseRightShiftTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.getSelf();
  Value rhs = adaptor.getOther();

  auto resultType =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  rhs = hlo::promoteAndBroadcast(rewriter, rhs, resultType);
  rewriter.replaceOpWithNewOp<stablehlo::ShiftRightArithmeticOp>(op, lhs, rhs);
  return success();
}

void mlir::torch::torch_to_stablehlo::populateBasicOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenTransposeIntOp>();
  patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);
  target.addIllegalOp<RuntimeAssertOp>();
  patterns.add<ConvertRuntimeAssertOp>(typeConverter, context);

#define INSERT_UNARY_PATTERN(AtenOp, StablehloOp)                              \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryOp<AtenOp, StablehloOp>>(typeConverter, context)
  INSERT_UNARY_PATTERN(AtenNegOp, stablehlo::NegOp);
  INSERT_UNARY_PATTERN(AtenLogicalNotOp, stablehlo::NotOp);
  INSERT_UNARY_PATTERN(AtenBitwiseNotOp, stablehlo::NotOp);
  INSERT_UNARY_PATTERN(AtenAbsOp, stablehlo::AbsOp);
  INSERT_UNARY_PATTERN(AtenExpm1Op, stablehlo::Expm1Op);
#undef INSERT_UNARY_PATTERN

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, StablehloOp)                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, StablehloOp>>(typeConverter,   \
                                                              context)
  INSERT_UNARY_FPONLY_PATTERN(AtenCeilOp, stablehlo::CeilOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenFloorOp, stablehlo::FloorOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenRoundOp, stablehlo::RoundNearestEvenOp);
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenOp, StablehloOp)                \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryPromoteToFPOp<AtenOp, StablehloOp>>(            \
      typeConverter, context)
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenLogOp, stablehlo::LogOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenLog1pOp, stablehlo::Log1pOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenExpOp, stablehlo::ExpOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenSqrtOp, stablehlo::SqrtOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenRsqrtOp, stablehlo::RsqrtOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenSigmoidOp, stablehlo::LogisticOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenTanhOp, stablehlo::TanhOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenSinOp, stablehlo::SineOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenCosOp, stablehlo::CosineOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenTanOp, chlo::TanOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAsinOp, chlo::AsinOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenSinhOp, chlo::SinhOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAcosOp, chlo::AcosOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenCoshOp, chlo::CoshOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAtanOp, chlo::AtanOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAsinhOp, chlo::AsinhOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAcoshOp, chlo::AcoshOp);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenAtanhOp, chlo::AtanhOp);
#undef INSERT_UNARY_PROMOTE_TO_FP_PATTERN

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context)
  INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
  INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_TENSOR_TO_SCALAR_PATTERN(AtenOp)                                \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenTensorToScalarLikeOp<AtenOp>>(typeConverter, context)

  INSERT_TENSOR_TO_SCALAR_PATTERN(AtenIntTensorOp);
  INSERT_TENSOR_TO_SCALAR_PATTERN(AtenFloatTensorOp);
  INSERT_TENSOR_TO_SCALAR_PATTERN(AtenBoolTensorOp);
#undef INSERT_TENSOR_TO_SCALAR_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, ChloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, ChloOp>>(typeConverter, context)
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, chlo::BroadcastAddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, chlo::BroadcastAddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, chlo::BroadcastSubOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, chlo::BroadcastSubOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenRsubScalarOp, chlo::BroadcastSubOp);
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_MULDIV_PATTERN(AtenOp, ChloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMulDivOp<AtenOp, ChloOp>>(typeConverter, context)
  INSERT_BINARY_MULDIV_PATTERN(AtenMulTensorOp, chlo::BroadcastMulOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenMulScalarOp, chlo::BroadcastMulOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivTensorOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivTensorModeOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivScalarOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivScalarModeOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenRemainderScalarOp, chlo::BroadcastRemOp);
#undef INSERT_BINARY_MULDIV_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp>>(typeConverter, context)

  INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGeTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGeScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLeTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLeScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp);
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_LOGICAL_PATTERN(AtenOp, ChloOp)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenLogicalBinaryOp<AtenOp, ChloOp>>(typeConverter,      \
                                                           context)

  INSERT_BINARY_LOGICAL_PATTERN(AtenLogicalOrOp, chlo::BroadcastOrOp);
  INSERT_BINARY_LOGICAL_PATTERN(AtenLogicalAndOp, chlo::BroadcastAndOp);
  INSERT_BINARY_LOGICAL_PATTERN(AtenLogicalXorOp, chlo::BroadcastXorOp);
  INSERT_BINARY_LOGICAL_PATTERN(AtenBitwiseAndScalarOp, chlo::BroadcastAndOp);

#undef INSERT_BINARY_LOGICAL_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)

  INSERT_ATENOP_PATTERN(AtenBroadcastToOp);
  INSERT_ATENOP_PATTERN(AtenPermuteOp);

  INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATENOP_PATTERN(AtenTensorIntOp);
  INSERT_ATENOP_PATTERN(AtenReciprocalOp);
  INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);
  INSERT_ATENOP_PATTERN(AtenPowScalarOp);
  INSERT_ATENOP_PATTERN(PrimNumToTensorScalarOp);
  INSERT_ATENOP_PATTERN(AtenScalarImplicitOp);
  INSERT_ATENOP_PATTERN(AtenContiguousOp);
  INSERT_ATENOP_PATTERN(AtenConstantPadNdOp);

  INSERT_ATENOP_PATTERN(AtenReluOp);
  INSERT_ATENOP_PATTERN(AtenGeluOp);
  INSERT_ATENOP_PATTERN(AtenLog2Op);
  INSERT_ATENOP_PATTERN(AtenLog10Op);
  INSERT_ATENOP_PATTERN(AtenErfOp);
  INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);

  INSERT_ATENOP_PATTERN(AtenCatOp);
  INSERT_ATENOP_PATTERN(AtenClampOp);
  INSERT_ATENOP_PATTERN(AtenClampTensorOp);
  INSERT_ATENOP_PATTERN(AtenArangeStartStepOp);

  INSERT_ATENOP_PATTERN(AtenBatchNormOp);
  INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
  INSERT_ATENOP_PATTERN(AtenNumelOp);
  INSERT_ATENOP_PATTERN(AtenSizeIntOp);
  INSERT_ATENOP_PATTERN(AtenToDtypeOp);
  INSERT_ATENOP_PATTERN(AtenWhereSelfOp);
  INSERT_ATENOP_PATTERN(AtenPowTensorTensorOp);
  INSERT_ATENOP_PATTERN(AtenUniformOp);
  INSERT_ATENOP_PATTERN(AtenEmptyMemoryFormatOp);
  INSERT_ATENOP_PATTERN(AtenFillScalarOp);
  INSERT_ATENOP_PATTERN(AtenFlipOp);
  INSERT_ATENOP_PATTERN(AtenRemainderTensorOp);
  INSERT_ATENOP_PATTERN(AtenFmodTensorOp);
  INSERT_ATENOP_PATTERN(AtenBitwiseLeftShiftTensorOp);
  INSERT_ATENOP_PATTERN(AtenBitwiseRightShiftTensorOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_BINARY_BROADCAST_PATTERN(AtenOp, StablehloOp)                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenBinaryBroadcastOp<AtenOp, StablehloOp>>(             \
      typeConverter, context)
  INSERT_BINARY_BROADCAST_PATTERN(AtenMaximumOp, chlo::BroadcastMaxOp);
  INSERT_BINARY_BROADCAST_PATTERN(AtenMinimumOp, chlo::BroadcastMinOp);
  INSERT_BINARY_BROADCAST_PATTERN(Aten__And__TensorOp, chlo::BroadcastAndOp);
  INSERT_BINARY_BROADCAST_PATTERN(AtenBitwiseAndTensorOp, chlo::BroadcastAndOp);
  INSERT_BINARY_BROADCAST_PATTERN(AtenBitwiseOrTensorOp, chlo::BroadcastOrOp);
  INSERT_BINARY_BROADCAST_PATTERN(AtenBitwiseXorTensorOp, chlo::BroadcastXorOp);
  INSERT_BINARY_BROADCAST_PATTERN(AtenAtan2Op, chlo::BroadcastAtan2Op);
#undef INSERT_BINARY_BROADCAST_PATTERN
}
