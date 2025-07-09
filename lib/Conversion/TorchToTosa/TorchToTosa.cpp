//===----------------------------------------------------------------------===//
////
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cmath>
#include <numeric>
#include <optional>
#include <random>

#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

// These legalizations are for unary ops with promoting input to floating-point
// datatypes only. There is no supported quantized integer mode for these.
template <typename AtenOpT, typename TosaOpT>
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
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto resultTy = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!isa<mlir::FloatType>(resultTy.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "Only floating-point datatype result types are supported");

    // Non floating point inputs are not supported in TOSA so we cast the input
    // to result type
    if (!isa<mlir::FloatType>(selfTy.getElementType()))
      self = tosa::tosaCastTensorToType(rewriter, self, resultTy).value();

    rewriter.replaceOpWithNewOp<TosaOpT>(op, resultTy, self);

    return success();
  }
};

// These unary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenUnaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto self = adaptor.getSelf();

    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    self = tosa::tosaCastTensorToType(rewriter, self, outType).value();

    rewriter.replaceOpWithNewOp<TosaOpT>(op, outType, self);

    return success();
  }
};

// These binary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenBinaryOp : public OpConversionPattern<AtenOpT> {
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
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhs).failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto outTy = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Value binaryOp;

    if constexpr (std::is_same<AtenOpT, AtenBitwiseRightShiftTensorOp>()) {
      // TOSA ArithmeticRightShiftOp has a round parameter.
      binaryOp = rewriter.create<TosaOpT>(op->getLoc(), outTy, lhs, rhs,
                                          /*round=*/false);
    } else if constexpr (std::is_same<TosaOpT, tosa::MaximumOp>() ||
                         std::is_same<TosaOpT, tosa::MinimumOp>()) {
      lhs = tosa::tosaCastTensorToType(rewriter, lhs, outTy).value();
      rhs = tosa::tosaCastTensorToType(rewriter, rhs, outTy).value();
      // Use default NaN Propagation mode "PROPAGATE" for tosa.maximum and
      // tosa.minimum
      binaryOp = rewriter.create<TosaOpT>(
          op->getLoc(), outTy, lhs, rhs,
          /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
    } else {
      binaryOp =
          tosa::createBinaryOpAndCast<TosaOpT>(rewriter, op, outTy, lhs, rhs);
    }

    rewriter.replaceOp(op, binaryOp);
    return success();
  }
};

template <typename T>
static bool isInValidRange(bool isFloat, const double &doubleValue, bool isInt,
                           const int64_t &intValue) {
  if (isFloat) {
    return (doubleValue >=
            static_cast<double>(std::numeric_limits<T>::min())) &&
           (doubleValue <= static_cast<double>(std::numeric_limits<T>::max()));
  } else if (isInt) {
    return (intValue >= static_cast<int64_t>(std::numeric_limits<T>::min())) &&
           (intValue <= static_cast<int64_t>(std::numeric_limits<T>::max()));
  }
  return false;
}

// FIXME: This will eventually go into a Tosa*Utils file.
LogicalResult torchScalarToTosaTensor(ConversionPatternRewriter &rewriter,
                                      Operation *op, Value torchScalarValue,
                                      Value &tosaTensor, Type dtype,
                                      llvm::ArrayRef<int64_t> dshape) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  int64_t numElem = 1;
  for (int64_t dim : dshape)
    numElem *= dim;

  if (isa<mlir::FloatType>(dtype)) {
    tosaTensor =
        tosa::getConstTensor<float>(
            rewriter, op,
            SmallVector<float>(numElem, (isFloat ? doubleValue : intValue)),
            dshape, dtype)
            .value();
  } else if (auto intType = dyn_cast<mlir::IntegerType>(dtype)) {
    auto width = intType.getWidth();
    if (width != 1 && width != 8 && width != 32 && width != 64)
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unsupported integer type: " << intType;
      });

    if (width == 1) {
      if (!isInValidRange<bool>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      bool d = isFloat ? static_cast<bool>(doubleValue)
                       : static_cast<bool>(intValue);
      tosaTensor = tosa::getConstTensor<bool>(
                       rewriter, op, SmallVector<bool>(numElem, d), dshape)
                       .value();
    } else if (width == 8) {
      if (!isInValidRange<int8_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int8_t d = isFloat ? static_cast<int8_t>(doubleValue)
                         : static_cast<int8_t>(intValue);
      tosaTensor = tosa::getConstTensor<int8_t>(
                       rewriter, op, SmallVector<int8_t>(numElem, d), dshape)
                       .value();
    } else if (width == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      tosaTensor = tosa::getConstTensor<int32_t>(
                       rewriter, op, SmallVector<int32_t>(numElem, d), dshape)
                       .value();
    } else if (width == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      tosaTensor = tosa::getConstTensor<int64_t>(
                       rewriter, op, SmallVector<int64_t>(numElem, d), dshape)
                       .value();
    }
  } else {
    return rewriter.notifyMatchFailure(op, "Usupported element type");
  }

  return success();
}

LogicalResult torchAlphaToTosaTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value alphaScalar,
                                     Value &alphaTensor, Type dtype,
                                     bool checkForUnity) {
  if (succeeded(torchScalarToTosaTensor(rewriter, op, alphaScalar, alphaTensor,
                                        dtype, {})))
    return success();

  // `alpha` has not been specified.
  int64_t alphaValue;
  if (!matchPattern(alphaScalar, m_TorchConstantInt(&alphaValue)))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "alpha in TOSA operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return rewriter.notifyMatchFailure(op,
                                       "Unsupported integer value for alpha");

  alphaTensor = tosa::getConstTensor<float>(
                    rewriter, op, {static_cast<float>(alphaValue)}, {}, dtype)
                    .value();

  return success();
}

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // left  : tensor: tensor<i32/i64/f32>
    // right : scalar: i32/i64/f32
    //         tensor: tensor<i32/i64/f32>
    // alpha : scalar: i32/i64/f32
    // output: tensor: tensor<i32/i64/f32>
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsType = dyn_cast<TensorType>(rhs.getType());

    if (!lhsType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    if (auto lhsElemTy = dyn_cast<IntegerType>(lhsType.getElementType())) {
      if (lhsElemTy.getWidth() > 64)
        return rewriter.notifyMatchFailure(
            op, "Integers with widths greater than 64 are not supported");
    }

    // Get output type: tensor<i32/i64/f32>
    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    Type rhsAlphaMulElemType;
    if (isa<mlir::FloatType>(outElemTy)) {
      rhsAlphaMulElemType = outElemTy;
    } else {
      // if output type is 64, input type should also be 32
      rhsAlphaMulElemType = rewriter.getIntegerType(32);
    }

    // if right is scalar, rhgType==None, which need to be manually cast to
    // TensorType else right is tensor, rhsType==tensor<i32/i64/f32>
    Value rhsAsTensor;
    if (!rhsType) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, rhsAlphaMulElemType, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    } else {
      if (rhsType.getElementType() != rhsAlphaMulElemType) {
        // right is tensor, rhsType == tensor<i32/i64/f32>
        // right must be cast to same type as the alpha, so MulOp success
        rhs =
            tosa::tosaCastTensorToType(
                rewriter, rhs,
                RankedTensorType::get(rhsType.getShape(), rhsAlphaMulElemType))
                .value();
        // reinitialize right value type to tensor<i32/f32>
        rhsType = dyn_cast<TensorType>(rhs.getType());
      }
    }
    auto rhsTensor = rhsType ? rhs : rhsAsTensor;
    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhsTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto rhsTensorType = dyn_cast<TensorType>(rhsTensor.getType());

    // Handle scalar value alpha.
    // It should be either f32/i32
    Value alphaTensor;
    if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(),
                                      op.getAlpha(), alphaTensor,
                                      rhsAlphaMulElemType,
                                      /*checkForUnity=*/false))) {
      return rewriter.notifyMatchFailure(
          op, "Currently only scalar constants are supported for "
              "alpha in conversion to TOSA operation");
    }
    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, alphaTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto mulAlphaOp = tosa::createMulOpAndCast(
        rewriter, op, rhsTensorType, rhsTensor, alphaTensor, /*shift=*/0);

    if (outElemTy.isInteger(64)) {
      // Tosa doesn't support 64-bit elementwise addition and subtraction.
      // if outElemTy tensor<i64>, mulTensor must be tensor<i32>,
      //    left value could be tensor<f32/i32/i64> type, cast left value to
      //    tensor<i32> type
      auto addOrSubi64Op = tosa::createBinaryOpAndCast<TosaOpT>(
          rewriter, op,
          RankedTensorType::get(outType.getShape(), rhsAlphaMulElemType), lhs,
          mulAlphaOp);

      // cast tensor<i32> back to tensor<i64>
      auto result =
          tosa::tosaCastTensorToType(rewriter, addOrSubi64Op, outType).value();
      rewriter.replaceOp(op, result);

      return success();
    }

    auto binaryOp = tosa::createBinaryOpAndCast<TosaOpT>(rewriter, op, outType,
                                                         lhs, mulAlphaOp);
    rewriter.replaceOp(op, binaryOp.getResult());
    return success();
  }
}; // namespace

// Binary op legalizations for comparator ops.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = dyn_cast<TensorType>(rhs.getType());

    if (!lhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // For bitwise operators, only integer datatype legalization is supported
    constexpr bool isBitwiseOp =
        std::is_same<AtenOpT, AtenBitwiseAndTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseAndScalarOp>() ||
        std::is_same<AtenOpT, AtenBitwiseOrTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseXorTensorOp>();
    if (isa<mlir::FloatType>(lhsElemTy) && isBitwiseOp) {
      return rewriter.notifyMatchFailure(op,
                                         "For bitwise operators, only integer "
                                         "datatype legalization is supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, rhs.getType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    }

    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhsTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto rhsTensorTy = dyn_cast<TensorType>(rhsTensor.getType());
    auto rhsElemTy = rhsTensorTy.getElementType();

    // There is no Lesser operator in TOSA.
    constexpr auto swapLhsRhs = (std::is_same<AtenOpT, AtenLtTensorOp>() ||
                                 std::is_same<AtenOpT, AtenLtScalarOp>() ||
                                 std::is_same<AtenOpT, AtenLeTensorOp>() ||
                                 std::is_same<AtenOpT, AtenLeScalarOp>());

    // Promote lhs and rhs dtypes for bitwise operators.
    TensorType resultTy = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (isBitwiseOp) {
      lhs = tosa::tosaCastTensorToType(rewriter, lhs, resultTy).value();
      rhsTensor =
          tosa::tosaCastTensorToType(rewriter, rhsTensor, resultTy).value();
    }

    // Support different types comparisons
    auto isLhsElemFloat = isa<mlir::FloatType>(lhsElemTy);
    auto isRhsElemFloat = isa<mlir::FloatType>(rhsElemTy);

    if (lhsElemTy != rhsElemTy && !isBitwiseOp) {
      if (isLhsElemFloat && !isRhsElemFloat) {
        rhsTensor =
            tosa::tosaCastTensorToType(rewriter, rhsTensor, lhsTy).value();
      } else if (!isLhsElemFloat && isRhsElemFloat) {
        lhs = tosa::tosaCastTensorToType(rewriter, lhs, rhsTensorTy).value();
      } else if (isLhsElemFloat && isRhsElemFloat) {
        auto lhsElemFloatTy = dyn_cast<mlir::FloatType>(lhsElemTy);
        auto rhsElemFloatTy = dyn_cast<mlir::FloatType>(rhsElemTy);
        if (lhsElemFloatTy.getWidth() > rhsElemFloatTy.getWidth()) {
          rhsTensor =
              tosa::tosaCastTensorToType(rewriter, rhsTensor, lhsTy).value();
        } else {
          lhs = tosa::tosaCastTensorToType(rewriter, lhs, rhsTensorTy).value();
        }
      } else {
        auto lhsElemIntTy = dyn_cast<mlir::IntegerType>(lhsElemTy);
        auto rhsElemIntTy = dyn_cast<mlir::IntegerType>(rhsElemTy);
        if (lhsElemIntTy.getWidth() > rhsElemIntTy.getWidth()) {
          rhsTensor =
              tosa::tosaCastTensorToType(rewriter, rhsTensor, lhsTy).value();
        } else {
          lhs = tosa::tosaCastTensorToType(rewriter, lhs, rhsTensorTy).value();
        }
      }
    }

    auto resultOp = rewriter.create<TosaOpT>(op.getLoc(), resultTy,
                                             (swapLhsRhs ? rhsTensor : lhs),
                                             (swapLhsRhs ? lhs : rhsTensor));

    // There is no NE operator in TOSA.
    if constexpr (std::is_same<AtenOpT, AtenNeTensorOp>() ||
                  std::is_same<AtenOpT, AtenNeScalarOp>()) {
      rewriter.replaceOpWithNewOp<tosa::LogicalNotOp>(op, resultTy,
                                                      resultOp.getResult());
    } else {
      rewriter.replaceOp(op, resultOp.getResult());
    }

    return success();
  }
};

// Binary op legalizations for Mul variants.
template <typename AtenOpT>
class ConvertAtenMulOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());

    if (!lhsType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value rhsTensor;
    if constexpr (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      Value rhsAsTensor;
      Value rhs = adaptor.getOther();
      auto rhsType = dyn_cast<TensorType>(rhs.getType());
      if (!rhsType) {
        if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                           rhsAsTensor, outElemTy, {}))) {
          return rewriter.notifyMatchFailure(
              op, "Currently only scalar constants are supported for "
                  "conversion in TOSA operation");
        }
      }
      rhsTensor = rhsType ? rhs : rhsAsTensor;
    }

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhsTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    if (isa<mlir::FloatType>(outElemTy) || isa<mlir::IntegerType>(outElemTy)) {
      auto outType = cast<TensorType>(
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()));

      auto mulOp = tosa::createMulOpAndCast(rewriter, op, outType, lhs,
                                            rhsTensor, /*shift=*/0);
      rewriter.replaceOp(op, mulOp.getResult());
      return success();
    }

    // Quantized multiplication may need to rescale inputs.
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype "
            "legalization currently supported");
  }
};

// Function to perform division with trunc rounding mode (rounding result
// towards zero) for float type inputs.
// This function takes in the division result between lhs and rhs rather
// than takes in the original lhs and rhs tensors as parameters.
std::optional<Value> truncFloatDivWithDivResult(PatternRewriter &rewriter,
                                                Operation *op,
                                                TensorType outType,
                                                Value divResult) {
  // To implement trunc mode for float inputs, multiply the floored abs
  // of the tensor with the elementwise signedness of the tensor.
  // div_result = lhs / rhs
  // trunc_val = floor(abs(div_result)) * sign(div_result)
  auto zero =
      tosa::getConstTensor<float>(rewriter, op, 0, {}, outType.getElementType())
          .value();

  auto one =
      tosa::getConstTensor<float>(rewriter, op, 1, {}, outType.getElementType())
          .value();

  auto minusOne = tosa::getConstTensor<float>(rewriter, op, -1, {},
                                              outType.getElementType())
                      .value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), divResult, one)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), divResult, zero)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), divResult, minusOne)
          .failed())
    return std::nullopt;

  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      op->getLoc(),
      RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1)),
      divResult, zero);

  auto selectOp = rewriter.create<tosa::SelectOp>(op->getLoc(), outType, cond,
                                                  one, minusOne);

  auto absDivResult =
      rewriter.create<tosa::AbsOp>(op->getLoc(), outType, divResult);

  auto flooredAbsDivResult =
      rewriter.create<tosa::FloorOp>(op->getLoc(), outType, absDivResult);

  Value result =
      tosa::createMulOpAndCast(rewriter, op, outType, flooredAbsDivResult,
                               selectOp, /*shift=*/0)
          .getResult();

  return result;
}

// Function to perform division with trunc rounding mode (rounding result
// towards zero) for float type inputs
Value truncFloatDiv(PatternRewriter &rewriter, Operation *op,
                    TensorType outType, Value lhs, Value rhs) {
  rhs = tosa::tosaCastTensorToType(rewriter, rhs, outType).value();

  auto rhsRcp =
      rewriter.create<tosa::ReciprocalOp>(op->getLoc(), rhs.getType(), rhs);

  auto divResult = tosa::createMulOpAndCast(rewriter, op, outType, lhs, rhsRcp,
                                            /*shift=*/0);

  return truncFloatDivWithDivResult(rewriter, op, outType, divResult).value();
}

// Function to perform division with floor rounding mode (rounding result
// down) for integer type inputs.
std::optional<Value> floorIntDiv(PatternRewriter &rewriter, Operation *op,
                                 TensorType outType, Value lhs, Value rhs) {
  // To implement floor mode int input, utilize tosa::IntDivOp (trunc div
  // result) with the following formula elementwise:
  // floor_val = trunc_val - ((trunc_val * rhs != lhs)
  //                                && (sign(lhs) != sign(rhs)))

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhs).failed())
    return std::nullopt;

  // TOSA IntDiv requires inputs to be i32
  auto i32Type =
      RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(32));
  lhs = tosa::tosaCastTensorToType(rewriter, lhs, i32Type).value();
  rhs = tosa::tosaCastTensorToType(rewriter, rhs, i32Type).value();

  auto intDivOp =
      rewriter.create<tosa::IntDivOp>(op->getLoc(), i32Type, lhs, rhs);

  auto zero = tosa::getConstTensor<int32_t>(rewriter, op, 0, {}).value();

  auto one = tosa::getConstTensor<int32_t>(rewriter, op, 1, {}).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, one).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, zero).failed())
    return std::nullopt;

  auto boolType =
      RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1));

  auto lhsMulRhs = tosa::createMulOpAndCast(rewriter, op, i32Type, lhs, rhs,
                                            /*shift=*/0);

  auto lhsRhsDifferentSign =
      rewriter.create<tosa::GreaterOp>(op->getLoc(), boolType, zero, lhsMulRhs);

  auto truncMulRhs = tosa::createMulOpAndCast(rewriter, op, i32Type, intDivOp,
                                              rhs, /*shift=*/0);

  auto truncMulRhsEqualLhs =
      rewriter.create<tosa::EqualOp>(op->getLoc(), boolType, truncMulRhs, lhs);

  auto truncMulRhsNotEqualLhs = rewriter.create<tosa::LogicalNotOp>(
      op->getLoc(), boolType, truncMulRhsEqualLhs);

  auto truncMinusOne =
      rewriter.create<tosa::SubOp>(op->getLoc(), i32Type, intDivOp, one);

  auto cond = rewriter.create<tosa::LogicalAndOp>(
      op->getLoc(), boolType, lhsRhsDifferentSign, truncMulRhsNotEqualLhs);

  auto selectOp = rewriter.create<tosa::SelectOp>(op->getLoc(), i32Type, cond,
                                                  truncMinusOne, intDivOp);

  Value result =
      tosa::tosaCastTensorToType(rewriter, selectOp, outType).value();

  return result;
}

template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = dyn_cast<TensorType>(rhs.getType());

    if (!lhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, lhsElemTy, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, rhsTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    // Get rounding mode for aten.div.Tensor_mode
    std::string roundMode;
    if constexpr (std::is_same<AtenOpT, AtenDivTensorModeOp>() ||
                  std::is_same<AtenOpT, AtenDivScalarModeOp>()) {
      if (!matchPattern(op.getRoundingMode(), m_TorchConstantStr(roundMode)))
        return rewriter.notifyMatchFailure(
            op, "Non-const rounding mode parameter unsupported");
    }

    Value result;
    if (isa<mlir::FloatType>(outType.getElementType())) {
      // The input to the reciprocal is an integer sometimes, and we may need
      // to promote it to a floating point. Per TOSA specification, the input
      // types can only be floating point for tosa::ReciprocalOp.
      rhsTensor =
          tosa::tosaCastTensorToType(rewriter, rhsTensor, outType).value();
      auto rhsRcp = rewriter.create<tosa::ReciprocalOp>(
          op->getLoc(), rhsTensor.getType(), rhsTensor);

      auto divResult = tosa::createMulOpAndCast(rewriter, op, outType, lhs,
                                                rhsRcp, /*shift=*/0);

      // Round result based on rounding mode
      if (roundMode.compare("floor") == 0) {
        // "floor": rounds the results of the division down. Equivalent to
        // floor division in Python (the // operator).
        auto floorOp =
            rewriter.create<tosa::FloorOp>(op->getLoc(), outType, divResult);

        result = floorOp.getResult();
      } else if (roundMode.compare("trunc") == 0) {
        // "trunc": rounds the results of the division towards zero. Equivalent
        // to C-style integer division.
        result = truncFloatDivWithDivResult(rewriter, op, outType, divResult)
                     .value();
      } else {
        // None: No rounding mode
        result = divResult.getResult();
      }
    } else {
      if (roundMode.compare("floor") == 0) {
        // "floor": rounds the results of the division down. Equivalent to floor
        // division in Python (the // operator).
        result = floorIntDiv(rewriter, op, outType, lhs, rhsTensor).value();
      } else {
        // "trunc": rounds the results of the division towards zero. Equivalent
        // to C-style integer division.
        // None: no rounding mode.

        // TOSA IntDiv requires inputs to be i32
        auto i32Type = RankedTensorType::get(outType.getShape(),
                                             rewriter.getIntegerType(32));
        lhs = tosa::tosaCastTensorToType(rewriter, lhs, i32Type).value();
        rhsTensor =
            tosa::tosaCastTensorToType(rewriter, rhsTensor, i32Type).value();

        auto intDivOp = rewriter.create<tosa::IntDivOp>(op->getLoc(), i32Type,
                                                        lhs, rhsTensor);

        result =
            tosa::tosaCastTensorToType(rewriter, intDivOp, outType).value();
      }
    }

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenActivationFunctionOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = dyn_cast<TensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(op, "Only Tensor types supported");

    auto resultTy = dyn_cast<TensorType>(
        this->getTypeConverter()->convertType(op.getType()));

    if (!isa<mlir::FloatType>(resultTy.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "Only floating-point datatype result types are supported");

    // Non floating point inputs are not supported for activation functions
    // (erf, sigmoid, tanh) in TOSA so we cast the input to result type
    if (!isa<mlir::FloatType>(selfTy.getElementType()))
      self = tosa::tosaCastTensorToType(rewriter, self, resultTy).value();

    rewriter.replaceOpWithNewOp<TosaOpT>(op, resultTy, self);

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());

  if (!selfTy) {
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");
  }

  // Rescale self for quantized types. TBD
  if (!isa<mlir::FloatType>(selfTy.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization currently supported");
  }

  // Maps to tosa.clamp
  // Use default NaN Propagation mode "PROPAGATE" for tosa.clamp
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(
      op, getTypeConverter()->convertType(op.getType()), self,
      rewriter.getF32FloatAttr(0.0f),
      rewriter.getF32FloatAttr(std::numeric_limits<float>::max()),
      /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLeakyReluOp>::matchAndRewrite(
    AtenLeakyReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());
  if (!isa<mlir::FloatType>(selfTy.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization currently supported");
  }

  Value alphaScalar = op.getNegativeSlope();
  Value alphaTensor;
  if (failed(torchScalarToTosaTensor(rewriter, op.getOperation(), alphaScalar,
                                     alphaTensor, selfTy.getElementType(), {})))
    return rewriter.notifyMatchFailure(
        op, "Negative slope needs to be a scalar constant for conversion to "
            "TOSA LeakyReLU operation");
  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), alphaTensor, self)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto zero =
      tosa::getConstTensor<float>(rewriter, op, 0, {}, selfTy.getElementType())
          .value();
  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), zero, self).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      op->getLoc(),
      RankedTensorType::get(selfTy.getShape(), rewriter.getIntegerType(1)),
      self, zero);

  auto resultTy =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto mulTensor = tosa::createMulOpAndCast(rewriter, op, resultTy, self,
                                            alphaTensor, /*shift=*/0);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultTy, cond, self,
                                              mulTensor);

  return success();
}

using ReductionConvFunc = std::optional<Value> (*)(PatternRewriter &,
                                                   Operation *,
                                                   RankedTensorType, Value,
                                                   ElementsAttr, bool);

// They all constitute a common form invoking the appropriate
// converion function in TosaLegalizeCommon.cpp
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenReductionOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      ElementsAttr &reduceDimsAttr, bool &keepDims) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented reduce_dims and keep_dims parsing function");
  }

  // Common rewriter for all reduction ops, calls the specific implementation of
  // readReduceDimsAndKeepDims() needed for the op variant.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto outputTy = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (!outputTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor type outputs permitted for reduce_mean");

    auto selfElemTy = selfTy.getElementType();
    if (!selfElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // TOSA ReduceAll and ReduceAny ops only accept bool input
    if constexpr (std::is_same<AtenOpT, AtenAllDimOp>() ||
                  std::is_same<AtenOpT, AtenAnyDimOp>() ||
                  std::is_same<AtenOpT, AtenAllOp>() ||
                  std::is_same<AtenOpT, AtenAnyOp>()) {
      self = tosa::tosaCastTensorToType(
                 rewriter, self,
                 RankedTensorType::get(selfTy.getShape(),
                                       rewriter.getIntegerType(1)))
                 .value();
    }

    // Handle dtype output and bool elem type for ReduceSum and ReduceProd ops
    if constexpr (std::is_same<AtenOpT, AtenSumDimIntListOp>() ||
                  std::is_same<AtenOpT, AtenSumOp>() ||
                  std::is_same<AtenOpT, AtenProdDimIntOp>() ||
                  std::is_same<AtenOpT, AtenProdOp>()) {
      auto dtype = op.getDtype();
      int64_t dtypeInt;
      if (!isa<Torch::NoneType>(dtype.getType())) {
        if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
          return rewriter.notifyMatchFailure(op, "dtype is not a constant int");

        FailureOr<Type> maybeDtypeType = getTypeForScalarType(
            op.getContext(), (torch_upstream::ScalarType)dtypeInt);
        if (failed(maybeDtypeType)) {
          return rewriter.notifyMatchFailure(op, "dtype is undefined");
        } else {
          Type dtypeType = maybeDtypeType.value();

          if (isa<mlir::IntegerType>(dtypeType))
            dtypeType =
                rewriter.getIntegerType(dtypeType.getIntOrFloatBitWidth());

          self = tosa::tosaCastTensorToType(
                     rewriter, self,
                     RankedTensorType::get(selfTy.getShape(), dtypeType))
                     .value();
        }
      } else {
        if (selfElemTy.isInteger(1))
          self = tosa::tosaCastTensorToType(rewriter, self, outputTy).value();
      }
    }

    ElementsAttr reduceDimsAttr;
    bool keepDims;

    if (failed(readReduceDimsAndKeepDims(op, adaptor, rewriter, reduceDimsAttr,
                                         keepDims)))
      return failure();

    std::optional<Value> result =
        ConversionFuncT(rewriter, op, outputTy, self, reduceDimsAttr, keepDims);

    if (!result)
      return failure();

    rewriter.replaceOp(op, {result.value()});

    return success();
  }
};

// This reduction op legalization template handles op variants that have
// explicit reduce_dims dimensions (provided as a list) and keep_dims
// parameters.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenMultipleDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    int64_t inputRank =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();

    SmallVector<int64_t> reduceDims;
    // If dim list is none, all dimensions are reduced
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(reduceDims))) {
      for (int64_t i = 0; i < inputRank; i++)
        reduceDims.push_back(i);
    }

    int64_t N = reduceDims.size();
    for (unsigned i = 0; i < N; i++) {
      reduceDims[i] = toPositiveDim(reduceDims[i], inputRank);
      if (!isValidDim(reduceDims[i], inputRank))
        return rewriter.notifyMatchFailure(op,
                                           "reduce dim is statically invalid");
    }
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce in
// only one explicit dim which is provided as a number (rather than a list), and
// a keep_dims parameter.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenOneDimReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    int64_t reduceDim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t inputRank =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();
    reduceDim = toPositiveDim(reduceDim, inputRank);
    if (!isValidDim(reduceDim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    auto reduceDimsType = RankedTensorType::get({1}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef({reduceDim}));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce all
// dims does not keep dims.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenAllDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
public:
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    auto self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    // Select all dims to reduce
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < selfTy.getRank(); i++)
      reduceDims.push_back(i);
    int64_t N = selfTy.getRank();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));
    keepDims = false;

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenArgmaxOp>::matchAndRewrite(
    AtenArgmaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());

  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA argmax");

  int64_t reduceDim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&reduceDim))) {
    // NoneType indicates reduce on all dims
    reduceDim = -1;
  } else {
    int64_t inputRank = selfTy.getRank();
    reduceDim = toPositiveDim(reduceDim, inputRank);
    if (!isValidDim(reduceDim, inputRank))
      return rewriter.notifyMatchFailure(op,
                                         "reduce dim is statically invalid");
  }

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
    return rewriter.notifyMatchFailure(
        op, "non-const keepdim parameter unsupported");

  auto resultTy = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  auto outputETy = resultTy.getElementType();

  // Create a single instance of tosa.argmax.
  // Multiple dims require chained construct.
  auto buildArgmax = [&](int64_t reduceDim, Value input) -> Value {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    SmallVector<int64_t> outputShapeArr = {};
    int32_t i = 0;

    for (auto &dim : inputShape) {
      if (i++ != reduceDim) {
        outputShapeArr.push_back(dim);
      } else {
        if (keepDim)
          outputShapeArr.push_back(1);
      }
    }

    // Tosa argmax output is i32, while Torch backend mandates i64.
    auto outputReduceTy = RankedTensorType::get(
        makeShapeLLVMCompatible(ArrayRef<int64_t>(outputShapeArr)),
        rewriter.getI32Type());
    auto reduceDimAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), reduceDim);

    // Use default NaN Propagation mode "PROPAGATE" for tosa.argmax
    return rewriter
        .create<tosa::ArgMaxOp>(
            op->getLoc(), getTypeConverter()->convertType(outputReduceTy),
            input, reduceDimAttr,
            /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"))
        .getResult();
  };

  // Convert the final index to i64 for backend finalization, However, i64
  // is not a defined type for tosa.cast, so using arith.extsi instead.
  auto castToInt64 = [&](Value result) -> LogicalResult {
    auto resTy = cast<ShapedType>(result.getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op,
                                         "Argmax: Result is not a shaped type");

    auto resShape = makeShapeTorchCompatible(resTy.getShape());
    auto outTy =
        RankedTensorType::get(makeShapeLLVMCompatible(resShape), outputETy);

    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
        op, getTypeConverter()->convertType(outTy), result);

    return success();
  };

  if (reduceDim == -1) { // reducing on all dims
    Value input = self;
    for (int dim = 0; dim < selfTy.getRank(); dim++) {
      // progressively reduce each 0-th dim
      input = buildArgmax(0, input);
    }
    return castToInt64(input);
  } else {
    return castToInt64(buildArgmax(reduceDim, self));
  }

  return success();
}

template <typename AtenOpT>
class ConvertAtenSqueezeOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented dim/dim-list parsing function");
  }

  // Common rewriter for all squeeze ops, calls the specific implementation of
  // generateSqueezedShape() needed for the op variant.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA argmax");

    SmallVector<int64_t> newOutputShape;
    if (failed(generateSqueezedShape(op, selfTy, rewriter, newOutputShape)))
      return rewriter.notifyMatchFailure(op,
                                         "Squeeze could not compute new shape");

    auto resultTy = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getResult().getType()));
    auto resultElemTy = resultTy.getElementType();

    auto newOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(newOutputShape), resultElemTy);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            newOutputTy),
        self, tosa::getTosaConstShape(rewriter, op->getLoc(), newOutputShape));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            newOutputTy),
        reshapeOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenSqueezeOneDimOp : public ConvertAtenSqueezeOp<AtenOpT> {
  using ConvertAtenSqueezeOp<AtenOpT>::ConvertAtenSqueezeOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const override {
    int64_t squeezeDim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&squeezeDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");

    // Handle negative dim
    if (squeezeDim < 0)
      squeezeDim = squeezeDim + selfTy.getRank();

    auto selfShape = makeShapeTorchCompatible(selfTy.getShape());

    // Only dims statically known to have size=1 are reduced.
    // Dynamic dims are treated as unknowns and will not be squeezed
    // even if dim parameter says it should be.
    uint32_t dimNum = 0;
    for (auto &dim : selfShape) {
      if (dim != 1 || squeezeDim != dimNum)
        squeezedShape.push_back(dim);
      dimNum++;
    }

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenSqueezeAllDimsOp : public ConvertAtenSqueezeOp<AtenOpT> {
  using ConvertAtenSqueezeOp<AtenOpT>::ConvertAtenSqueezeOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const override {
    auto selfShape = makeShapeTorchCompatible(selfTy.getShape());

    // Dims that may dynamically resolve to 1 are not reduced here. Only
    // compile-time resolvable dims are handled here.
    for (auto &dim : selfShape) {
      if (dim != 1)
        squeezedShape.push_back(dim);
    }
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenPowOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType =
        cast<TensorType>(this->getTypeConverter()->convertType(op.getType()));

    if (!isa<mlir::FloatType>(outType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "Only floating-point datatype result types are supported");

    Value selfTensor;
    if constexpr (std::is_same<AtenOpT, AtenPowScalarOp>()) {
      Value selfScalar = op.getSelf();
      if (failed(torchScalarToTosaTensor(rewriter, op, selfScalar, selfTensor,
                                         outType.getElementType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA PowScalar operation");
    } else {
      selfTensor = adaptor.getSelf();
      auto selfTy = cast<RankedTensorType>(selfTensor.getType());

      if (!selfTy)
        return rewriter.notifyMatchFailure(
            op, "Only ranked tensor types supported in TOSA Pow");

      // Non floating point inputs are not supported for tosa.pow so we cast the
      // input to result type
      if (!isa<mlir::FloatType>(selfTy.getElementType()))
        selfTensor =
            tosa::tosaCastTensorToType(rewriter, selfTensor, outType).value();
    }

    Value expTensor;
    if constexpr (std::is_same<AtenOpT, AtenPowTensorScalarOp>()) {
      Value expScalar = op.getExponent();
      if (failed(torchScalarToTosaTensor(rewriter, op, expScalar, expTensor,
                                         outType.getElementType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA Pow operation");
    } else {
      expTensor = adaptor.getExponent();
      auto expTy = cast<RankedTensorType>(expTensor.getType());

      if (!expTy)
        return rewriter.notifyMatchFailure(
            op, "Only ranked tensor types supported in TOSA Pow");

      // Non floating point exponents are not supported for tosa.pow so we cast
      // the exponent to result type
      if (!isa<mlir::FloatType>(expTy.getElementType()))
        expTensor =
            tosa::tosaCastTensorToType(rewriter, expTensor, outType).value();
    }

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), selfTensor, expTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    auto powOp = tosa::createBinaryOpAndCast<tosa::PowOp>(
        rewriter, op, outType, selfTensor, expTensor);
    rewriter.replaceOp(op, powOp.getResult());

    return success();
  }
};

// Perform the basic n-dim matmul operation encompassing the handling of
// broadcasting and dynamic shape propagation.
// All PyTorch ops that leverage matrix multiplication will derive this and
// implement their specialized input processing (e.g transpose), and output
// processing, e.g. GEMM or fully connected bias handling.
template <typename AtenOpT>
class ConvertAtenMatmulBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  // Each variant must implement corresponding parameter parsing options.
  // Maintain separate input read functions for each variant because it is not
  // necessarily true with all variants that the first two operands are the lhs
  // and rhs.
  virtual LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         Value &lhs, Value &rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "Unimplemented matrix multiplication variant input parsing function");
  }
  LogicalResult performMatmul(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &lhs,
                              Value &rhs, Value &output) const {

    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = makeShapeTorchCompatible(lhsTy.getShape());
    auto rhsShape = makeShapeTorchCompatible(rhsTy.getShape());

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "Matmul: input datatypes mismatched");

    // Legalization constructs may offer input shapes but expect output shapes
    // to be inferred, e.g.
    // func @forward(%arg0: !torch.vtensor<[14,19],f32>,
    //               %arg1: !torch.vtensor<[19,28],f32>) ->
    //               !torch.vtensor<[?,?],f32>
    // This is tricky with matmul, since TOSA matmul is on 3D inputs.
    // This means the need to reshape potentially both inputs and outputs,
    // and reshape to unknown shape is undefined.

    auto maxInputRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    // If performing dot product on vectors, the RHS is synthetically transposed
    if (maxInputRank == 1)
      maxInputRank++;

    // Obtaining the rank broadcasted shapes of tensors makes it easier to
    // construct the input and output reshaping logic.
    auto getRankBroadcastedShape = [&](Value tensor,
                                       bool isRHS) -> SmallVector<int64_t> {
      auto tensorTy = cast<TensorType>(tensor.getType());
      auto tensorShape = makeShapeTorchCompatible(tensorTy.getShape());
      auto tensorRank = tensorTy.getRank();

      SmallVector<int64_t> bcastedShape;

      auto bcastDims = maxInputRank - tensorRank;

      if (isRHS && (tensorRank == 1) && bcastDims) {
        // RHS with rank1 is special. It be synthetically transposed to dim[:-2]
        for (int32_t i = 0; i < bcastDims - 1; i++)
          bcastedShape.push_back(1);
        bcastedShape.push_back(tensorShape[0]);
        bcastedShape.push_back(1);
      } else {
        if (bcastDims > 0) { // rank broadcast
          for (uint32_t i = 0; i < bcastDims; i++)
            bcastedShape.push_back(1);
        }
        for (auto &dim : tensorShape)
          bcastedShape.push_back(dim);
      }
      return bcastedShape;
    };

    // Step: Rank broadcast the two inputs.
    auto lhsBroadcastedShape = getRankBroadcastedShape(lhs, false);
    auto lhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(lhsBroadcastedShape), lhsElemTy);
    auto rhsBroadcastedShape = getRankBroadcastedShape(rhs, true);
    auto rhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(rhsBroadcastedShape), rhsElemTy);

    auto rankBroadcastedLhs =
        lhsRank == maxInputRank
            ? lhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      lhsBroadcastedTy),
                  lhs,
                  tosa::getTosaConstShape(rewriter, op->getLoc(),
                                          lhsBroadcastedShape));

    auto rankBroadcastedRhs =
        rhsRank == maxInputRank
            ? rhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      rhsBroadcastedTy),
                  rhs,
                  tosa::getTosaConstShape(rewriter, op->getLoc(),
                                          rhsBroadcastedShape));

    // TOSA matmul is performed on two 3D inputs and generates a 3D output.
    // Lower ranked tensors are dim-1 reshaped up to 3D
    auto reshapeUpTo3DTensor = [&](Value tensor) -> Value {
      auto tensorTy = cast<TensorType>(tensor.getType());
      auto rank = tensorTy.getRank();

      assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
      if (rank == 3)
        return tensor;

      auto shape = makeShapeTorchCompatible(tensorTy.getShape());
      SmallVector<int64_t> newShape({1, 1, 1});

      if (rank == 2) { // batchsize = 1
        newShape[1] = shape[0];
        newShape[2] = shape[1];
      } else { // rank 1
        newShape[2] = shape[0];
      }
      auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                           tensorTy.getElementType());

      return rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newType),
          tensor, tosa::getTosaConstShape(rewriter, op->getLoc(), newShape));
    };

    // Where broadcasting is required in one or more batch dims, the following
    // is done.
    // Where all batch dims are involved in broadcasting:
    // Given A: 3x1x5x6 and B: 1x4x6x7
    // 1. Reshape A to 1x15x6 (squeeze all batchdims into dim1)
    // 2. Transpose B to 6x1x4x7, Reshape to 1x6x28
    // 3. tosa.Matmul 1x15x6 1x6x28 = 1x15x28
    // 4. Reshape out to 3x5x4x7, Transpose to 3x4x5x7
    // Where there are batch dimensions that are broadcast and not, the
    // treatment is to have dim0 correspond to product of all non-broadcast
    // dimsizes:
    // Given A: 4x8x16x32 B: 8x32x17
    // 1. Reshape A to 8x64x32 (squeeze all unbroadcasted dims into dim0,
    // broadcasted dims into dim1)
    // 2. No transpose or reshape of B as its batchdims are not broadcast to.
    // 3. tosa.Matmul 8x64x32 8x32x17 = 8x64x17
    // 4. Reshape to 8x4x16x17, Transpose to 4x8x16x17

    // Check if we need to perform the broadcast on batch dim
    // Not needed if max rank < 3, or if maxrank == 3 and dim[0] matches
    auto needsBatchDimBroadcast = [&]() -> bool {
      if (maxInputRank < 3) {
        return false;
      } else {
        if (maxInputRank == 3 &&
            lhsBroadcastedShape[0] == rhsBroadcastedShape[0]) {
          return false;
        }
        return true;
      }
    };

    auto performBatchDimBroadcast = needsBatchDimBroadcast();

    // Inputs to the tosa.matmul
    Value matmulLhs, matmulRhs;

    using TensorShape_t = struct {
      int64_t dim;
      int64_t shape;
    };

    // Transpose needs to done if transposedDims are not non-monotonically
    // increasing. E.g. [0, 1, 2, 3]: No transpose [1, 0, 2, 3]: Transpose dim0
    // and dim1 The order need not be sequential, since one or more dims may
    // have been removed due to broadcasting.
    auto isTransposeRequired = [](SmallVector<int32_t> transposedDims) -> bool {
      int32_t lastDim = -1;
      for (auto &dim : transposedDims) {
        if (lastDim > dim)
          return true;
        lastDim = dim;
      }
      return false;
    };

    SmallVector<TensorShape_t> batchElems, lhsSqueezedElems, rhsSqueezedElems;

    if (!performBatchDimBroadcast) {
      // Simple with no broadcasting artifacts. Just reshape up to 3D
      matmulLhs = reshapeUpTo3DTensor(rankBroadcastedLhs);
      matmulRhs = reshapeUpTo3DTensor(rankBroadcastedRhs);

    } else {
      // In this case, either or both input matrices involve broadcasting on
      // their batch dimensions. For example:
      // 4x5x6, 1x6x7 -> 4x5x7
      // 4x1x5x6, 1x3x6x7 -> 4x3x5x7
      // Though maxInputRank is necessarily >=3 here, individual matrices may be
      // lower rank.
      // E.g. 3x4x5x6, 6 -> 3x4x5

      // These are the accumulated products of the shape of each dim:
      // 1. common dimensions: upper dimensions (dims other than two rightmost)
      // whose shapes are the same for both LHS and RHS.
      // 2. LHS squeezed dimensions: all dimensions of LHS that involve
      // broadcasting in either direction, plus the LHS[-2] shape
      // 3. RHS squeezed dimensions: all dimensions of RHS that involve
      // broadcasting in either direction, plus the RHS[-1] shape
      int64_t commonValue = 1, lhsSqueezedValue = 1, rhsSqueezedValue = 1;

      // For both LHS and RHS, the dimensions are separated into the common,
      // squeezed and remaining dim. E.g. given
      // LHS = 3x4x5x6
      // RHS = 1x4x6x7
      // common = {{dim=1, shape=4}}
      // lhs squeezed = {{dim=0, shape=3},
      //                 {dim=2, shape=5}}
      // rhs squeezed = {{dim=0, shape=1},
      //                 {dim=2, shape=7}}
      // The matmul dim is LHS[-1] and RHS[-2], i.e. 6.
      // Once this is obtained, LHS and RHS are expressed as:
      // LHS = {common, lhs_squeezed, matmul_dim}
      // RHS = {common, matmul_dim, rhs_squeezed}
      // The matmul is then performed to obtain output:
      // matmul_out = {common, lhs_squeezed, rhs_squeezed}
      // Finally, we reshape to 'unsqueeze' the LHS and RHS parts and transpose
      // them back to their correct positions.

      SmallVector<int64_t> transposedLhsShape;
      SmallVector<int32_t> transposedLhsDims;

      // Step: generate the common dim/shape information
      bool hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (isDynamicDim ||
            lhsBroadcastedShape[dim] == rhsBroadcastedShape[dim]) {
          commonValue *= lhsBroadcastedShape[dim];
          batchElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      commonValue = commonValue < 0 ? kUnknownSize : commonValue;

      // TODO: Handle the case when there are dynamic batch dimensions.
      if (hasDynamicDims)
        commonValue = kUnknownSize;

      // Step: generate the LHS squeezed dim/shape information.
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        if (!isDynamicDim &&
            lhsBroadcastedShape[dim] != rhsBroadcastedShape[dim]) {
          lhsSqueezedValue *= lhsBroadcastedShape[dim];
          lhsSqueezedElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      // including LHS[-2]
      lhsSqueezedElems.push_back(
          {maxInputRank - 2, lhsBroadcastedShape[maxInputRank - 2]});
      lhsSqueezedValue *= lhsBroadcastedShape[maxInputRank - 2];
      lhsSqueezedValue = lhsSqueezedValue < 0 ? kUnknownSize : lhsSqueezedValue;

      // Step: Create the tosa.transpose array. If this array has a
      // non-monotonic series of dims, perform transpose.
      // First the common_elems
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedLhsShape.push_back(batchElems[i].shape);
        transposedLhsDims.push_back(batchElems[i].dim);
      }
      // then the lhs_squeezed elems
      for (uint32_t i = 0; i < lhsSqueezedElems.size(); i++) {
        transposedLhsShape.push_back(lhsSqueezedElems[i].shape);
        transposedLhsDims.push_back(lhsSqueezedElems[i].dim);
      }
      // then the final dim
      transposedLhsDims.push_back(maxInputRank - 1);
      transposedLhsShape.push_back(lhsBroadcastedShape[maxInputRank - 1]);

      bool lhsNeedsTranspose = isTransposeRequired(transposedLhsDims);

      auto lhsReshapeInput = rankBroadcastedLhs;

      if (lhsNeedsTranspose) {
        auto transposedLhsType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedLhsShape), rhsElemTy);

        lhsReshapeInput =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedLhsType),
                    rankBroadcastedLhs,
                    rewriter.getDenseI32ArrayAttr(transposedLhsDims))
                .getResult();
      }

      // LHS = {common, lhs_squeezed, matmul_dim}
      SmallVector<int64_t> newLhsShape(
          {1, 1, lhsBroadcastedShape[maxInputRank - 1]});
      newLhsShape[0] = commonValue;
      newLhsShape[1] = hasDynamicDims ? kUnknownSize : lhsSqueezedValue;

      auto newLhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newLhsShape), lhsElemTy);

      matmulLhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newLhsType),
          lhsReshapeInput,
          tosa::getTosaConstShape(rewriter, op->getLoc(), newLhsShape));

      SmallVector<int64_t> transposedRhsShape;
      SmallVector<int32_t> transposedRhsDims;

      // Step: Create the RHS transpose sequence
      // RHS = {common, matmul_dim, rhs_squeezed}
      // first the common_dims
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedRhsShape.push_back(batchElems[i].shape);
        transposedRhsDims.push_back(batchElems[i].dim);
      }
      // The matmul_dim of RHS
      transposedRhsDims.push_back(maxInputRank - 2);
      transposedRhsShape.push_back(rhsBroadcastedShape[maxInputRank - 2]);
      // finally all the rhs_squeeze dims
      hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(rhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (!isDynamicDim &&
            rhsBroadcastedShape[dim] != lhsBroadcastedShape[dim]) {
          rhsSqueezedElems.push_back({dim, rhsBroadcastedShape[dim]});
          rhsSqueezedValue *= rhsBroadcastedShape[dim];
        }
      }
      rhsSqueezedElems.push_back(
          {maxInputRank - 1, rhsBroadcastedShape[maxInputRank - 1]});
      rhsSqueezedValue *= rhsBroadcastedShape[maxInputRank - 1];
      for (uint32_t i = 0; i < rhsSqueezedElems.size(); i++) {
        transposedRhsShape.push_back(rhsSqueezedElems[i].shape);
        transposedRhsDims.push_back(rhsSqueezedElems[i].dim);
      }

      auto transposedRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);

      if (hasDynamicDims)
        rhsSqueezedValue = kUnknownSize;

      SmallVector<int64_t> newRhsShape(
          {commonValue < 0 ? kUnknownSize : commonValue,
           rhsBroadcastedShape[maxInputRank - 2], rhsSqueezedValue});
      auto newRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newRhsShape), rhsElemTy);

      bool rhsNeedsTranspose = isTransposeRequired(transposedRhsDims);

      auto transposedRhsValue = rankBroadcastedRhs;

      if (rhsNeedsTranspose)
        transposedRhsValue =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedRhsType),
                    rankBroadcastedRhs,
                    rewriter.getDenseI32ArrayAttr(transposedRhsDims))
                .getResult();

      // reshape
      matmulRhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newRhsType),
          transposedRhsValue,
          tosa::getTosaConstShape(rewriter, op->getLoc(), newRhsShape));
    }

    auto matmulLhsShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(matmulLhs.getType()).getShape());
    auto matmulRhsShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(matmulRhs.getType()).getShape());

    // The reshape/transpose should ensure the tosa.matmul always has same
    // batch size for either matrix. If if shapes are dynamic, they'll be
    // appropriately handled.
    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "tosa.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});

    bool isInputElemTyQInt8 = false;
    Type inputElemTy{lhsElemTy};
    if (auto inputQTy =
            dyn_cast<mlir::quant::UniformQuantizedType>(lhsElemTy)) {
      if (inputQTy.getStorageTypeIntegralWidth() == 8)
        isInputElemTyQInt8 = true;
      inputElemTy = inputQTy.getStorageType();
    }

    auto accElemTy = getDefaultAccType(rewriter, inputElemTy);
    auto mmOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(matmulOutputShape), accElemTy);

    Value mmOpResult;
    if (!isInputElemTyQInt8) {
      // LHS and RHS tensors' zero points must be zero for non-int8 types
      Value lhsZp =
          tosa::createZeroPointTensor(rewriter, op->getLoc(), lhsElemTy, 0)
              .value();
      Value rhsZp =
          tosa::createZeroPointTensor(rewriter, op->getLoc(), rhsElemTy, 0)
              .value();
      mmOpResult =
          rewriter
              .create<tosa::MatMulOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      mmOutputTy),
                  matmulLhs, matmulRhs, lhsZp, rhsZp)
              .getResult();
    } else {
      mmOpResult =
          rewriter
              .create<tosa::MatMulOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      mmOutputTy),
                  matmulLhs, matmulRhs)
              .getResult();
    }

    // Perform the reshape to output shape. This is always required unless max
    // input rank=3 and there was no broadcasting, in which case the tosa.matmul
    // output itself is correctly shaped.
    bool performOpReshape = !(maxInputRank == 3 && !performBatchDimBroadcast);

    if (performOpReshape) {
      // Since the output shape may be unknown, we construct it
      // independently and reshape. Otherwise reshape may be expressed for
      // an unknown to-be-inferred output shape. The final tensor.cast
      // reshapes the known shape to the desired output shape.
      auto computeOpShape = [&](SmallVector<int64_t> &reshapedOpShape,
                                SmallVector<int32_t> &transposedOpDims,
                                SmallVector<int64_t> &transposedOpShapes) {
        if (maxInputRank == 1)
          return;

        if (maxInputRank == 2) {
          if (lhsRank == 2)
            reshapedOpShape.push_back(lhsShape[0]);
          if (rhsRank == 2)
            reshapedOpShape.push_back(rhsShape[1]);
          return;
        }

        // Step: Construct the output transpose/reshape information
        // First the common_dims
        for (uint32_t i = 0; i < batchElems.size(); i++) {
          reshapedOpShape.push_back(batchElems[i].shape);
          transposedOpDims.push_back(batchElems[i].dim);
        }

        // Then the LHS squeezed dims
        for (uint32_t i = 0; i < lhsSqueezedElems.size() - 1; i++) {
          // Only dims that don't broadcast - broadcasting ones come from the
          // other input.
          if (lhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(lhsSqueezedElems[i].shape);
            transposedOpDims.push_back(lhsSqueezedElems[i].dim);
          }
        }
        // The last squeezed dim is lhs[-2] which needs to be
        // checked separately for broadcasting
        if (lhsRank > 1) {
          reshapedOpShape.push_back(lhsBroadcastedShape[maxInputRank - 2]);
          transposedOpDims.push_back(maxInputRank - 2);
        }

        // then the RHS squeezed dims except rhs[-1] which is handled like
        // lhs[-2]
        for (uint32_t i = 0; i < rhsSqueezedElems.size() - 1; i++) {
          if (rhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(rhsSqueezedElems[i].shape);
            transposedOpDims.push_back(rhsSqueezedElems[i].dim);
          }
        }
        // rhs[-1]
        if (rhsRank > 1) {
          reshapedOpShape.push_back(rhsBroadcastedShape[maxInputRank - 1]);
          transposedOpDims.push_back(maxInputRank - 1);
        }

        // The transposition order is the inverse of what we actually want,
        // inversing should fix this:
        llvm::SmallVector<int> inverseTransposeDims(transposedOpDims.size());
        for (int i = 0, s = transposedOpDims.size(); i < s; ++i)
          inverseTransposeDims[transposedOpDims[i]] = i;

        transposedOpDims = inverseTransposeDims;

        // Final transposed output shape construction
        for (uint32_t i = 0; i < maxInputRank - 2; i++) {
          if (lhsBroadcastedTy.isDynamicDim(i)) {
            transposedOpShapes.push_back(kUnknownSize);
          } else {
            if (lhsBroadcastedShape[i] == rhsBroadcastedShape[i]) {
              transposedOpShapes.push_back(lhsBroadcastedShape[i]);
            } else {
              transposedOpShapes.push_back(lhsBroadcastedShape[i] == 1
                                               ? rhsBroadcastedShape[i]
                                               : lhsBroadcastedShape[i]);
            }
          }
        }
        if (lhsRank > 1)
          transposedOpShapes.push_back(lhsBroadcastedShape[maxInputRank - 2]);
        if (rhsRank > 1)
          transposedOpShapes.push_back(rhsBroadcastedShape[maxInputRank - 1]);

        return;
      };

      SmallVector<int64_t> reshapedOpShape, transposedOpShape;
      SmallVector<int32_t> transposedOpDims;

      computeOpShape(reshapedOpShape, transposedOpDims, transposedOpShape);

      bool opNeedsTranspose = isTransposeRequired(transposedOpDims);

      // Perform reshape
      auto reshapedOpType = RankedTensorType::get(
          makeShapeLLVMCompatible(reshapedOpShape), accElemTy);
      auto reshapedOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              reshapedOpType),
          mmOpResult,
          tosa::getTosaConstShape(rewriter, op->getLoc(), reshapedOpShape));

      if (opNeedsTranspose) {
        auto transposedOpType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedOpShape), accElemTy);
        output = rewriter
                     .create<tosa::TransposeOp>(
                         op->getLoc(),
                         OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(transposedOpType),
                         reshapedOp.getResult(),
                         rewriter.getDenseI32ArrayAttr(transposedOpDims))
                     .getResult();

      } else {
        output = reshapedOp.getResult();
      }
    } else {
      output = mmOpResult;
    }

    return success();
  }
  // The default version just reads two inputs, computes output and returns it.
  // Other versions may add a bias, apply GEMM-style alpha/beta scaling etc.
  virtual LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    rewriter.replaceOp(
        op,
        {tosa::tosaCastTensorToType(
             rewriter, output,
             cast<RankedTensorType>(
                 OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                     op.getType())))
             .value()});

    return success();
  }
};

// Legalizes the torch.matmul op for general n-dim matmul.
template <typename AtenOpT>
class ConvertAtenMatMulOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.getSelf();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getOther();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    return success();
  }
};

// Implements handling of aten.mm and aten.bmm ops.
template <typename AtenOpT>
class ConvertAtenMmOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {

    lhs = adaptor.getSelf();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getMat2();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (isa<AtenMmOp>(op)) {
      // Mm takes two 2D tensors.
      if (lhsRank != 2 || rhsRank != 2)
        return op.emitError("aten.mm called but matrix rank != 2");
    } else if (isa<AtenBmmOp>(op)) {
      // Bmm takes two 3D tensors.
      if (lhsRank != 3 || rhsRank != 3)
        return op.emitError("aten.bmm called but matrix rank != 3");
    }

    return success();
  }
};

// Implements handling of aten.linear op.
template <typename AtenOpT>
class ConvertAtenLinearOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {

    lhs = adaptor.getInput();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getWeight();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    // Protection against crash due to unguarded code in TOSA->LinAlg.
    // TODO: This should be handled in TOSA->LinAlg instead.
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "aten.Linear needs statically shaped input");

    return success();
  }
  // Override the default rewriter to perform RHS transpose and bias addition as
  // well.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    // The aten.Linear op has a bias tensor that is added to the matmul output.
    auto bias = adaptor.getBias();
    auto biasTy = bias.getType();

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), lhs, bias).failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    // TOSA does not mandate that elementwise op tensors need to be ranked.
    if (!isa<Torch::NoneType>(biasTy) && !isa<TensorType>(biasTy))
      return rewriter.notifyMatchFailure(
          op, "Only tensor types supported in GEMM to TOSA for bias tensor");

    // RHS must have its last two dims transposed prior to matrix
    // multiplication.
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    auto rhsRank = rhsTy.getRank();
    auto rhsShape = makeShapeTorchCompatible(rhsTy.getShape());
    auto rhsElemTy = rhsTy.getElementType();

    // Create a non-const shape array to transpose dims.
    SmallVector<int64_t> transposedRhsShape;
    for (auto &shape : rhsShape)
      transposedRhsShape.push_back(shape);
    SmallVector<int32_t> transposedRhsDims;
    for (int32_t i = 0; i < rhsRank; i++)
      transposedRhsDims.push_back(i);

    // Swap the last two dims.
    std::swap(transposedRhsShape[rhsRank - 1], transposedRhsShape[rhsRank - 2]);
    std::swap(transposedRhsDims[rhsRank - 1], transposedRhsDims[rhsRank - 2]);

    auto transposedRhsType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);
    rhs = rewriter.create<tosa::TransposeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            transposedRhsType),
        rhs, rewriter.getDenseI32ArrayAttr(transposedRhsDims));

    Value matmulOutput;
    if (failed(
            this->performMatmul(op, adaptor, rewriter, lhs, rhs, matmulOutput)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    Value matmulPlusBias = matmulOutput;
    if (!isa<Torch::NoneType>(biasTy)) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias =
          rewriter
              .create<tosa::AddOp>(op->getLoc(), matmulOutput.getType(),
                                   matmulOutput, bias)
              .getResult();
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        cast<RankedTensorType>(
            OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                op.getType())),
        matmulPlusBias);

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenRsubScalarOp>::matchAndRewrite(
    AtenRsubScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();
  auto otherScalar = op.getOther();
  auto alphaScalar = op.getAlpha();

  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Rsub");

  auto resultTy =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto resultElemTy = resultTy.getElementType();

  self = tosa::tosaCastTensorToType(rewriter, self, resultTy).value();

  Value otherTensor, alphaTensor;

  if (failed(torchScalarToTosaTensor(rewriter, op, otherScalar, otherTensor,
                                     resultElemTy, {})))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "conversion in TOSA Rsub operation");

  if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(), alphaScalar,
                                    alphaTensor, resultElemTy,
                                    /*checkForUnity=*/true)))
    return failure();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, otherTensor)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, alphaTensor)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto multTensor = tosa::createMulOpAndCast(rewriter, op, resultTy, self,
                                             alphaTensor, /*shift=*/0);

  rewriter.replaceOpWithNewOp<tosa::SubOp>(op, resultTy, otherTensor,
                                           multTensor);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  bool transposed;
  if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: non-constant value for transposed not supported");
  if (transposed)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: transposed convolution not supported");

  auto input = adaptor.getInput();
  auto weight = adaptor.getWeight();

  auto inputTy = cast<RankedTensorType>(input.getType());
  auto weightTy = cast<RankedTensorType>(weight.getType());
  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!inputTy || !weightTy || !outputTy)
    return rewriter.notifyMatchFailure(
        op, "Input, weight and output to Convolution must be ranked tensors");

  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
  auto weightShape = makeShapeTorchCompatible(weightTy.getShape());
  auto outputElemTy = outputTy.getElementType();

  if (inputTy.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: only 2D convolutions supported");

  if (!weightTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: TOSA only supports static weight");

  // Bias is optional. TOSA mandates a zero tensor here, so construct one if
  // required.
  auto bias = adaptor.getBias();

  if (isa<Torch::NoneType>(bias.getType())) {
    auto bias_result = tosa::getConvBiasForNoneType(op, rewriter, inputElemTy,
                                                    outputElemTy, weightShape);
    if (failed(bias_result))
      return rewriter.notifyMatchFailure(
          op, "Failed to create bias tensor for none type.");
    bias = bias_result.value();
  } else {
    if (!isa<RankedTensorType>(bias.getType()))
      return rewriter.notifyMatchFailure(
          op, "Bias provided but not a ranked tensor");
  }

  Type biasElemTy = cast<RankedTensorType>(bias.getType()).getElementType();

  int64_t groups;
  if (!matchPattern(op.getGroups(), m_TorchConstantInt(&groups))) {
    return rewriter.notifyMatchFailure(op, "non-const group size unsupported");
  } else if (groups != 1 && weightShape[1] != 1) {
    return rewriter.notifyMatchFailure(
        op, "group size must be 1 (convolution) or weight.dim(1) must be 1 "
            "(depthwise convolution)");
  }

  SmallVector<int64_t, 2> stride;
  if (!matchPattern(adaptor.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");

  SmallVector<int64_t, 2> padding_2d;
  if (!matchPattern(adaptor.getPadding(),
                    m_TorchListOfConstantInts(padding_2d)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");
  // TOSA uses 4D padding {top, bottom, left, right} while Torch defines 2D
  // padding {height, width}. The Torch OFM computation uses 2*pad in each
  // spatial direction, implying the same top=bottom=height and left=right=width
  // values for TOSA.
  SmallVector<int64_t> padding(
      {padding_2d[0], padding_2d[0], padding_2d[1], padding_2d[1]});

  SmallVector<int64_t, 2> dilation;
  if (!matchPattern(adaptor.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");

  TypeAttr accType;
  if (failed(tosa::getConvOpsAccType(rewriter, inputTy, weightTy, outputTy,
                                     accType)))
    return rewriter.notifyMatchFailure(
        op, "failed to get accumulator type for convolution ops");

  // TOSA works in NHWC and takes OHWI (conv) / HWIM (depthwise conv) weights.
  // Perform the necessary transformations.
  SmallVector<int32_t> nchwToNhwcDims({0, 2, 3, 1});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
  auto transposedInput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedInputType), input,
              rewriter.getDenseI32ArrayAttr(nchwToNhwcDims))
          .getResult();

  SmallVector<int64_t> transformedWeightShape;
  RankedTensorType transformedWeightType;
  Value transformedWeight;
  int64_t outputCDim;
  if (groups == 1) {
    // full convolution: O(I/G)HW-> OHWI
    transformedWeightShape = {weightShape[0], weightShape[2], weightShape[3],
                              weightShape[1]};
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<tosa::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType), weight,
                rewriter.getDenseI32ArrayAttr(nchwToNhwcDims))
            .getResult();
    outputCDim = transformedWeightShape[0];
  } else if (weightShape[1] == 1) {
    // depthwise convolution: O(I/G)HW-> HWIM)
    // transpose: O(I/G)HW -> HWO(I/G)
    SmallVector<int32_t> transposedDims({2, 3, 0, 1});
    SmallVector<int64_t> transposedWeightShape = {
        weightShape[2], weightShape[3], weightShape[0], weightShape[1]};
    auto transposedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedWeightShape), weightElemTy);
    auto transposedWeight =
        rewriter
            .create<tosa::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transposedWeightType), weight,
                rewriter.getDenseI32ArrayAttr(transposedDims))
            .getResult();

    // reshape: HWO(I/G) -> HWIM
    outputCDim = makeShapeTorchCompatible(outputTy.getShape())[1];
    if (outputCDim == kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "number of output channels must be statically known for "
              "depthwise convolutions");
    }
    transformedWeightShape = {
        transposedWeightShape[0],
        transposedWeightShape[1],
        groups,
        outputCDim / groups,
    };
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<tosa::ReshapeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType),
                transposedWeight,
                tosa::getTosaConstShape(rewriter, op->getLoc(),
                                        transformedWeightShape))
            .getResult();
  } else {
    llvm_unreachable("Unhandled convolution type");
  }

  int64_t outputHDim, outputWDim;
  int64_t inputHDim = inputShape[2];
  int64_t inputWDim = inputShape[3];

  bool isStaticSpatialDims =
      !ShapedType::isDynamic(inputHDim) && !ShapedType::isDynamic(inputWDim);
  if (isStaticSpatialDims) {

    int64_t weightHDim = weightShape[2];
    int64_t weightWDim = weightShape[3];

    // fullDim =
    //    inputDim + padBefore + padAfter - dilation * (weightDim - 1) - 1
    // According to TOSA spec:
    // https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d, fullDim values
    // must be divisible by stride values.
    int64_t fullHDim = inputHDim + padding[0] + padding[1] -
                       dilation[0] * (weightHDim - 1) - 1;
    int64_t remainderHDim = fullHDim % stride[0];
    if (remainderHDim != 0) {
      if (remainderHDim > padding[1]) {
        SmallVector<int64_t> startHSlice(inputTy.getRank(), 0);
        SmallVector<int64_t> sizeHSlice(transposedInputShape);
        // TOSA uses NHWC, so we will slice dim 1 for Height value
        sizeHSlice[1] = inputHDim - (remainderHDim - padding[1]);
        transposedInput = tosa::CreateOpAndInfer<tosa::SliceOp>(
            rewriter, op->getLoc(), UnrankedTensorType::get(inputElemTy),
            transposedInput,
            tosa::getTosaConstShape(rewriter, op->getLoc(), startHSlice),
            tosa::getTosaConstShape(rewriter, op->getLoc(), sizeHSlice));
        fullHDim = fullHDim - padding[1];
        padding[1] = 0;
      } else {
        fullHDim = fullHDim - padding[1];
        padding[1] = padding[1] - remainderHDim;
        fullHDim = fullHDim + padding[1];
      }
    }
    outputHDim = fullHDim / stride[0] + 1;

    int64_t fullWDim = inputWDim + padding[2] + padding[3] -
                       dilation[1] * (weightWDim - 1) - 1;
    int64_t remainderWDim = fullWDim % stride[1];
    if (remainderWDim != 0) {
      if (remainderWDim > padding[3]) {
        SmallVector<int64_t> startWSlice(inputTy.getRank(), 0);
        SmallVector<int64_t> sizeWSlice(
            dyn_cast<RankedTensorType>(transposedInput.getType()).getShape());
        // TOSA uses NHWC, so we will slice dim 2 for Width value
        sizeWSlice[2] = inputWDim - (remainderWDim - padding[3]);
        transposedInput = tosa::CreateOpAndInfer<tosa::SliceOp>(
            rewriter, op->getLoc(), UnrankedTensorType::get(inputElemTy),
            transposedInput,
            tosa::getTosaConstShape(rewriter, op->getLoc(), startWSlice),
            tosa::getTosaConstShape(rewriter, op->getLoc(), sizeWSlice));
        fullHDim = fullHDim - padding[3];
        padding[3] = 0;
      } else {
        fullWDim = fullWDim - padding[3];
        padding[3] = padding[3] - remainderWDim;
        fullWDim = fullWDim + padding[3];
      }
    }
    outputWDim = fullWDim / stride[1] + 1;
  } else {
    outputHDim = kUnknownSize;
    outputWDim = kUnknownSize;
  }

  // Output shape is NHWC, to be transposed back to NCHW. Output elemTy for
  // quantized input is i32, which gets rescaled down to quantized output range.
  SmallVector<int64_t> outputShape = {transposedInputShape[0], outputHDim,
                                      outputWDim, outputCDim};
  auto convOpTy =
      RankedTensorType::get(makeShapeLLVMCompatible(outputShape), biasElemTy);

  // create zero-point tensors for input and weight
  auto zps = tosa::createZPsAsConst(rewriter, input, weight);
  // for i8 input/weight, zero-points are returned as un-initialized
  Value inputZp =
      zps.first
          ? zps.first
          : tosa::createZeroPointTensor(rewriter, op->getLoc(), inputElemTy, 0)
                .value();

  Value weightZp =
      zps.second
          ? zps.second
          : tosa::createZeroPointTensor(rewriter, op->getLoc(), weightElemTy, 0)
                .value();

  Value convOpResult;
  if (groups == 1) {
    // full convolution
    convOpResult =
        rewriter
            .create<tosa::Conv2DOp>(
                op->getLoc(), getTypeConverter()->convertType(convOpTy),
                transposedInput, transformedWeight, bias, inputZp, weightZp,
                rewriter.getDenseI64ArrayAttr(padding),
                rewriter.getDenseI64ArrayAttr(stride),
                rewriter.getDenseI64ArrayAttr(dilation), accType)
            .getResult();
  } else if (weightShape[1] == 1) {
    // depthwise convolution
    convOpResult =
        rewriter
            .create<tosa::DepthwiseConv2DOp>(
                op->getLoc(), getTypeConverter()->convertType(convOpTy),
                transposedInput, transformedWeight, bias, inputZp, weightZp,
                rewriter.getDenseI64ArrayAttr(padding),
                rewriter.getDenseI64ArrayAttr(stride),
                rewriter.getDenseI64ArrayAttr(dilation), accType)
            .getResult();
  } else {
    llvm_unreachable("Unhandled convolution type");
  }

  SmallVector<int32_t> nhwcToNchwDims({0, 3, 1, 2});
  SmallVector<int64_t> transposedOutputShape(
      {outputShape[0], outputShape[3], outputShape[1], outputShape[2]});
  auto transposedOutputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedOutputShape), biasElemTy);
  auto transposedOutput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedOutputType),
              convOpResult, rewriter.getDenseI32ArrayAttr(nhwcToNchwDims))
          .getResult();

  Value rescaledResult = transposedOutput;
  if (isa<quant::QuantizedType>(inputElemTy)) {
    rescaledResult = tosa::buildRescaleOpConvOutput(
        rewriter, op, transposedOutput, inputTy, weightTy, outputTy);
  }

  // cast to outputTy is required if convOpTy is not same as outputTy
  // the difference is not in the shape information, rather the element-type
  // itself
  rewriter.replaceOp(
      op,
      {tosa::tosaCastTensorToType(rewriter, rescaledResult, outputTy).value()});

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenReshapeOp>::matchAndRewrite(
    AtenReshapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();

  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Reshape");

  // Check that at most one dimension is -1
  SmallVector<int64_t> newShape;
  if (!matchPattern(op.getShape(), m_TorchListOfConstantInts(newShape)))
    return rewriter.notifyMatchFailure(
        op, "Only constant shape supported in TOSA Reshape");

  int auto_sz = 0;
  for (auto s : newShape)
    auto_sz += (s == -1 ? 1 : 0);
  if (auto_sz > 1)
    return rewriter.notifyMatchFailure(
        op, "At most one dimension may be specified as -1 to "
            "automatically calculate its size");

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfTy.getElementType());

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), self,
      tosa::getTosaConstShape(rewriter, op->getLoc(), newShape));

  return success();
}

std::optional<Value> computeBatchNorm(Operation *op,
                                      ConversionPatternRewriter &rewriter,
                                      Type outType, Value input, Value variance,
                                      Value eps, Value mean, Value weight,
                                      Value bias) {
  // For PyTorch:
  //   scale  = gamma = weight
  //   offset = beta  = bias
  // Lowering:
  // fused batchnorm = (input-mean) * scale * rsqrt(var+epsilon)) + offset
  //
  // shape_0 = ones(input.rank)
  // shape_0[input.rank-1] = input.shape[input.rank-1]
  // shape_1 = ones(1)
  //
  // bmean  = reshape(mean, shape_0)
  // bscale = reshape(scale, shape_0)
  // boffset= reshape(offset, shape_0)
  // beps   = reshape(epsilon, shape_1)
  //
  // op1 = sub(input, bmean)
  // op2 = add(var, beps)
  // op3 = rsqrt(op2)
  // bvar = reshape(op3, shape_0)
  // op4 = mul(op1, bvar)
  // op5 = mul(op4, bscale)
  // op6 = add(op5, boffset)

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, mean).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, variance)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, eps).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, weight)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, bias).failed())
    return std::nullopt;

  auto op1SubInputMean =
      rewriter.create<tosa::SubOp>(op->getLoc(), outType, input, mean);

  auto op2AddVarEpsilon = rewriter.create<tosa::AddOp>(
      op->getLoc(), variance.getType(), variance, eps);

  auto op3RsqrtOp2 = rewriter.create<tosa::RsqrtOp>(
      op->getLoc(), variance.getType(), op2AddVarEpsilon.getResult());

  auto op4MulOp1Op3 = tosa::createMulOpAndCast(
      rewriter, op, dyn_cast<TensorType>(outType), op1SubInputMean.getResult(),
      op3RsqrtOp2.getResult(), 0);

  auto op5MulOp4Scale =
      tosa::createMulOpAndCast(rewriter, op, dyn_cast<TensorType>(outType),
                               op4MulOp1Op3.getResult(), weight, 0);

  return rewriter
      .create<tosa::AddOp>(op->getLoc(), outType, op5MulOp4Scale.getResult(),
                           bias)
      .getResult();
}

// This lowering is based on the TensorFlow to TOSA lowering.
template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor output
  if (!dyn_cast<RankedTensorType>(adaptor.getInput().getType()))
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  auto outType = getTypeConverter()->convertType(op.getType());

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle training and momentum.
  if (isa<Torch::NoneType>(op.getMomentum().getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for momentum");

  auto meanType = dyn_cast<TensorType>(adaptor.getRunningMean().getType());
  auto varianceType = dyn_cast<TensorType>(adaptor.getRunningVar().getType());
  if (!varianceType || !meanType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  // Normalization ops perform elementwise ops of a single mean/stdev value
  // against the feature map and because input is NCHW, the rank-1 value must be
  // reshaped so it sits on the same dim as 'C'.
  auto reshapeToNormInputDim = [&](Operation *op,
                                   ConversionPatternRewriter &rewriter,
                                   const TypeConverter *converter, Type outType,
                                   const Value toBcast, Value &result) {
    RankedTensorType toBcastType =
        dyn_cast<RankedTensorType>(toBcast.getType());
    if (toBcastType.getRank() > 1)
      return rewriter.notifyMatchFailure(op, "Rank cannot be more than 1");

    RankedTensorType outTensorType = cast<RankedTensorType>(outType);
    SmallVector<int64_t> newShape = {
        makeShapeTorchCompatible(toBcastType.getShape())[0]};
    for (auto i = 2; i < outTensorType.getRank(); ++i)
      newShape.push_back(1);
    auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                         outTensorType.getElementType());

    result = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), newType, toBcast,
        tosa::getTosaConstShape(rewriter, op->getLoc(), newShape));

    return success();
  };

  Value meanVal, varianceVal, weightVal, biasVal;
  assert(meanType.getNumElements() != 0 && varianceType.getNumElements() != 0);
  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getRunningMean(), meanVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape running mean");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getRunningVar(), varianceVal)))
    return rewriter.notifyMatchFailure(op,
                                       "Failed to reshape running variance");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getWeight(), weightVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape weight");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getBias(), biasVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape bias");

  double eps;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
    return rewriter.notifyMatchFailure(op, "eps must be a scalar constant");

  auto epsilonConst = tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                                  {static_cast<float>(eps)}, {},
                                                  meanType.getElementType())
                          .value();

  auto batchNorm =
      computeBatchNorm(op, rewriter, outType, adaptor.getInput(), varianceVal,
                       epsilonConst, meanVal, weightVal, biasVal)
          .value();

  rewriter.replaceOp(op, {batchNorm});

  return success();
}

// This lowering is loosely based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenNativeLayerNormOp>::matchAndRewrite(
    AtenNativeLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // The key difference from BatchNorm is that a specified set of dims
  // (normalized_shape) are chosen to compute the mean and variance from input.
  // Where as in BatchNorm the mean and variance are operands. tosa::ReduceSumOp
  // is used to sum up the these dims for mean and for variance. The results
  // eventually being reshaped for broadcasting.

  // Not a ranked tensor output
  auto input = adaptor.getInput();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());

  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  if (inputType.getRank() > 4)
    return rewriter.notifyMatchFailure(op,
                                       "Only up to 4D tensors are supported");

  auto outType = getTypeConverter()->convertType(op.getType(0));

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle the None cases for the optional parameters.
  auto weight = adaptor.getWeight();
  if (isa<Torch::NoneType>(weight.getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for weight");

  auto bias = adaptor.getBias();
  if (isa<Torch::NoneType>(bias.getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for bias");

  auto weightType = cast<RankedTensorType>(weight.getType());
  auto biasType = cast<RankedTensorType>(bias.getType());
  int64_t inputRank = inputType.getRank();
  Type elemTy = inputType.getElementType();
  SmallVector<int64_t> inputTypeShape(
      makeShapeTorchCompatible(inputType.getShape()));

  // Check if all the arguments meet the requirements.
  SmallVector<int64_t> normalizedShapeSizesInt;
  if (!matchPattern(op.getNormalizedShape(),
                    m_TorchListOfConstantInts(normalizedShapeSizesInt))) {
    return rewriter.notifyMatchFailure(op, "Unimplemented normalized_shape not"
                                           "constructed from ListConstruct");
  }
  int64_t normalizedShapeRank = normalizedShapeSizesInt.size();
  if (weightType.getRank() != normalizedShapeRank ||
      biasType.getRank() != normalizedShapeRank ||
      inputRank < normalizedShapeRank || normalizedShapeRank < 1)
    return rewriter.notifyMatchFailure(op, "Input or weight or bias shape or"
                                           "normalized shape not compatible");

  // Check all the dimensions match the normalized_shape, only static shapes as
  // of now
  int64_t meanAndVarShapeRank = inputRank - normalizedShapeSizesInt.size();
  for (auto en : llvm::enumerate((normalizedShapeSizesInt))) {
    int64_t index = en.index();
    int64_t value = en.value();
    if (inputTypeShape[index + meanAndVarShapeRank] != value ||
        makeShapeTorchCompatible(weightType.getShape())[index] != value ||
        makeShapeTorchCompatible(biasType.getShape())[index] != value)
      return rewriter.notifyMatchFailure(op,
                                         "mismatching contracting dimension");
  }

  // Helper for computing mean and variance.
  auto computeSumAndReshape = [&](Value toReduce, RankedTensorType toReduceType,
                                  Type outType, SmallVector<int64_t> outShape) {
    Value sumDiv = toReduce;
    SmallVector<int64_t> toReduceShape(
        makeShapeTorchCompatible(toReduceType.getShape()));
    for (int64_t i = toReduceShape.size() - 1; i >= meanAndVarShapeRank; i--) {
      toReduceShape[i] = 1;
      sumDiv = rewriter.create<tosa::ReduceSumOp>(
          op.getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(toReduceShape),
                                inputType.getElementType()),
          sumDiv, rewriter.getI32IntegerAttr(i));
    }

    return rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), outType, sumDiv,
        tosa::getTosaConstShape(rewriter, op->getLoc(), outShape));
  };

  // TOSA has integer Div so, compute reciprocal of element count to be used in
  // mul.
  int64_t elemCnt = 1;
  for (auto i : normalizedShapeSizesInt)
    elemCnt *= i;

  auto elemCntConst =
      tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                  {static_cast<float>(elemCnt)}, {1}, elemTy)
          .value();
  Value elemCntRcp = rewriter.create<tosa::ReciprocalOp>(
      op.getLoc(), elemCntConst.getType(), elemCntConst);

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), input, elemCntRcp)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  // Broadcast type and shape for various intermediate values.
  SmallVector<int64_t> bcastOutShape;
  for (auto en : llvm::enumerate(inputTypeShape)) {
    bcastOutShape.push_back(
        static_cast<int64_t>(en.index()) >= meanAndVarShapeRank ? 1
                                                                : en.value());
  }
  auto bcastOutType =
      RankedTensorType::get(makeShapeLLVMCompatible(bcastOutShape), elemTy);

  // Compute mean.
  Value sum =
      computeSumAndReshape(input, inputType, bcastOutType, bcastOutShape);
  Value meanVal = tosa::createMulOpAndCast(rewriter, op, bcastOutType, sum,
                                           elemCntRcp, /*shift=*/0);

  // Compute variance.
  Value squareSumSub =
      rewriter.create<tosa::SubOp>(op.getLoc(), inputType, input, meanVal);
  Value squareSum = tosa::createMulOpAndCast(rewriter, op, inputType,
                                             squareSumSub, squareSumSub, 0);

  Value squareSumReduced =
      computeSumAndReshape(squareSum, inputType, bcastOutType, bcastOutShape);
  Value varianceVal = tosa::createMulOpAndCast(
      rewriter, op, bcastOutType, squareSumReduced, elemCntRcp, /*shift=*/0);

  // Reshape weight and bias.
  SmallVector<int64_t> weightAndBiasBcastShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(inputType.getShape()))) {
    weightAndBiasBcastShape.push_back(
        static_cast<int64_t>(en.index()) < meanAndVarShapeRank ? 1
                                                               : en.value());
  }
  auto weightAndMeanBcastType = RankedTensorType::get(
      makeShapeLLVMCompatible(weightAndBiasBcastShape), elemTy);

  Value weightVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, weight,
      tosa::getTosaConstShape(rewriter, op->getLoc(), weightAndBiasBcastShape));

  Value biasVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, bias,
      tosa::getTosaConstShape(rewriter, op->getLoc(), weightAndBiasBcastShape));

  double eps;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
    return rewriter.notifyMatchFailure(op, "eps must be a scalar constant");
  auto epsilonConst =
      tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                  {static_cast<float>(eps)}, {}, elemTy)
          .value();

  // Compute layer norm.
  auto layerNorm = computeBatchNorm(op, rewriter, outType, input, varianceVal,
                                    epsilonConst, meanVal, weightVal, biasVal)
                       .value();

  rewriter.replaceOp(op, {layerNorm, meanVal, varianceVal});

  return success();
}

// Torch constants are converted to tosa.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  // TODO: what about unsigned integer?
  if (auto elements = dyn_cast<DenseIntElementsAttr>(op.getValueAttr())) {
    if (elements.getElementType().isSignedInteger()) {
      Type builtinTensorElemTy = outputTy.getElementType();
      unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();
      DenseElementsAttr valueAttr =
          elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
            return APInt(bitWidth, v.getSExtValue(), /*isSigned=*/true);
          });
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputTy, valueAttr);
      return success();
    }
  }
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputTy, adaptor.getValue());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenFlattenUsingIntsOp>::matchAndRewrite(
    AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op,
                                       "Only ranked tensor types supported");

  int64_t selfRank = selfType.getRank();

  int64_t start_dim, end_dim;

  if (!matchPattern(op.getStartDim(), m_TorchConstantInt(&start_dim)))
    return rewriter.notifyMatchFailure(op,
                                       "start_dim must be a Scalar constant");
  start_dim = toPositiveDim(start_dim, selfRank);

  if (!matchPattern(op.getEndDim(), m_TorchConstantInt(&end_dim)))
    return rewriter.notifyMatchFailure(op, "end_dim must be a Scalar constant");
  end_dim = toPositiveDim(end_dim, selfRank);

  if (selfRank > 0 && !isValidDim(start_dim, selfRank))
    return rewriter.notifyMatchFailure(op, "start_dim is statically invalid");
  if (selfRank > 0 && !isValidDim(end_dim, selfRank))
    return rewriter.notifyMatchFailure(op, "end_dim is statically invalid");
  if (end_dim < start_dim)
    return rewriter.notifyMatchFailure(op,
                                       "end_dim must be larger than start_dim");

  SmallVector<int64_t> newShape;
  for (auto s :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    int64_t idx = s.index();
    if (idx < start_dim || idx > end_dim) {
      newShape.push_back(s.value());
    } else {
      if (idx == start_dim)
        newShape.push_back(s.value());
      // Only updating when the shapes are static
      else if (s.value() != kUnknownSize && newShape.back() != kUnknownSize)
        newShape.back() *= s.value();
      else
        newShape.back() = kUnknownSize;
    }
  }

  // Handle the Scalar case
  if (newShape.size() == 0)
    newShape.push_back(1);

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfType.getElementType());
  auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), newType, adaptor.getSelf(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), newShape));

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), reshapeOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnflattenIntOp>::matchAndRewrite(
    AtenUnflattenIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op,
        "Only ranked tensor types with static shapes are currently supported");

  int64_t selfRank = selfType.getRank();
  int64_t dim;

  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  SmallVector<int64_t> sizes;
  if (!matchPattern(op.getSizes(), m_TorchListOfConstantInts(sizes)))
    return rewriter.notifyMatchFailure(
        op, "Only constant sizes are currently supported");

  if (selfRank > 0 && !isValidDim(dim, selfRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<int64_t> newShape;
  for (auto s :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    int64_t idx = s.index();
    if (idx < dim || idx > dim) {
      newShape.push_back(s.value());
    } else {
      auto sum = 1;
      for (auto newDims : sizes) {
        newShape.push_back(newDims);
        sum *= newDims;
      }
      if (sum != s.value())
        return rewriter.notifyMatchFailure(op,
                                           "sizes mismatch with original dim");
    }
  }

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfType.getElementType());

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), adaptor.getSelf(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), newShape));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPermuteOp>::matchAndRewrite(
    AtenPermuteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op,
        "Only ranked tensor types with static shapes are currently supported");

  SmallVector<int64_t> dimListInt;
  if (!matchPattern(adaptor.getDims(), m_TorchListOfConstantInts(dimListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dimensions are currently supported");

  int64_t selfRank = selfType.getRank();
  // TODO: If this is already verified on the op then we can drop checking here.
  for (auto &d : dimListInt) {
    d = toPositiveDim(d, selfRank);
    if (!isValidDim(d, selfRank))
      return rewriter.notifyMatchFailure(op, "Not all dims are valid");
  }

  SmallVector<int32_t> dimListInt32;
  for (auto v : dimListInt)
    dimListInt32.push_back(v);

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      rewriter.getDenseI32ArrayAttr(dimListInt32));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLog2Op>::matchAndRewrite(
    AtenLog2Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto outType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  // If input is not a float type then cast it to output type
  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy))
    self = tosa::tosaCastTensorToType(rewriter, self, outType).value();

  // Constant value of ln2.
  SmallVector<int64_t> ln2Shape(selfType.getRank(), 1);
  auto ln2Op = tosa::getConstTensor<float>(rewriter, op, {0.69314718056f},
                                           ln2Shape, outType.getElementType())
                   .value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, ln2Op).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto rcpOp =
      rewriter.create<tosa::ReciprocalOp>(op.getLoc(), ln2Op.getType(), ln2Op);

  auto logOp = rewriter.create<tosa::LogOp>(op.getLoc(), outType, self);
  auto result = tosa::createMulOpAndCast(rewriter, op, outType, logOp, rcpOp,
                                         /*shift=*/0);

  rewriter.replaceOp(op, result.getResult());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenThresholdOp>::matchAndRewrite(
    AtenThresholdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat())
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");

  auto outType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto outElemTy = outType.getElementType();

  SmallVector<int64_t> constTypeShape(selfType.getRank(), 1);
  Value threshold, value;
  if (failed(torchScalarToTosaTensor(rewriter, op, op.getThreshold(), threshold,
                                     selfElemTy, constTypeShape)))
    return rewriter.notifyMatchFailure(
        op, "Only scalar constant is supported for threshold");

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getValue(), value,
                                     outElemTy, constTypeShape)))
    return rewriter.notifyMatchFailure(
        op, "Only scalar constant is supported for value");

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, threshold)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, value).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto cmpOp = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      self, threshold);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cmpOp, self, value);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType) {
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");
  }

  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  // toPositiveDim converts negative dims to the range [0, inputRank). So, -1
  // will be converted to inputRank-1. For `torch.unsqueeze` op, -1 has to be
  // converted to inputRank, and the valid dim range is [0, inputRank + 1).
  dim = toPositiveDim(dim, selfRank + 1);
  if (!isValidDim(dim, selfRank + 1))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<int64_t> outShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    if (static_cast<int64_t>(en.index()) == dim)
      outShape.push_back(1);

    outShape.push_back(en.value());
  }
  if (dim == selfRank)
    outShape.push_back(1);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), outShape));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenContiguousOp>::matchAndRewrite(
    AtenContiguousOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  // FIXME: memory_format is not handled.

  rewriter.replaceOp(op, adaptor.getSelf());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenDropoutOp>::matchAndRewrite(
    AtenDropoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto self = adaptor.getInput();
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  // FIXME: train and p are not handled.

  bool train;
  if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train)))
    return rewriter.notifyMatchFailure(op, "train must be a Scalar constant");

  if (train)
    return rewriter.notifyMatchFailure(op, "train must be false");

  auto resultType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto result = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  rewriter.replaceOp(op, result);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewOp>::matchAndRewrite(
    AtenViewOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> outShape;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(outShape)))
    return rewriter.notifyMatchFailure(op,
                                       "size must consist of Scalar constants");

  // the shape -1 is inferred from other dimensions
  size_t countNegativeShape{0};
  // Check at most one -1 shape
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] < 0) {
      countNegativeShape++;
      if (countNegativeShape > 1)
        return rewriter.notifyMatchFailure(op, "At most one -1 shape");
    }
  }

  auto inputShape = selfType.getShape();
  size_t totalSize = 1;
  for (size_t i = 0; i < inputShape.size(); i++) {
    totalSize *= inputShape[i];
  }

  size_t otherSize = 1;
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] > 0) {
      otherSize *= outShape[i];
    }
  }
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] < 0) {
      outShape[i] = totalSize / otherSize;
      break;
    }
  }

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), outShape));

  return success();
}

static std::optional<Value>
buildUnitNormalCdf(ConversionPatternRewriter &rewriter, Operation *op, Value x,
                   Type dtype) {
  auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}, dtype).value();
  auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}, dtype).value();
  auto oneHalf =
      tosa::getConstTensor<float>(rewriter, op, 0.5, {}, dtype).value();
  // rsqrt of 2
  auto rsqrt2 =
      tosa::getConstTensor<float>(rewriter, op, 0.70710678f, {}, dtype).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), x, zero).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), x, one).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), x, oneHalf).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), x, rsqrt2).failed())
    return std::nullopt;

  auto loc = op->getLoc();

  // buildNormalCdf, mean = zero, sigma = one
  auto outType = dyn_cast<TensorType>(x.getType());
  auto mean = zero;
  Value xMinusMean = rewriter.create<tosa::SubOp>(loc, outType, x, mean);

  Value erfArg =
      tosa::createMulOpAndCast(rewriter, op, outType, xMinusMean, rsqrt2,
                               /*shift=*/0);
  Value erf = rewriter.create<tosa::ErfOp>(loc, outType, erfArg);
  Value erfPlus1 = rewriter.create<tosa::AddOp>(loc, outType, one, erf);

  Value normalCdf = tosa::createMulOpAndCast(rewriter, op, outType, oneHalf,
                                             erfPlus1, /*shift=*/0);
  return normalCdf;
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy)) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");
  }

  auto resultType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  std::string approximate;
  if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate))) {
    return rewriter.notifyMatchFailure(
        op, "Non-const approximate value not supported");
  }

  if (approximate.compare("none") == 0) {
    // GELU(x) = x * CDF(x)
    Value cdf =
        buildUnitNormalCdf(rewriter, op, adaptor.getSelf(), selfElemTy).value();
    cdf = tosa::tosaCastTensorToType(rewriter, cdf, selfType).value();

    auto result = tosa::createMulOpAndCast(rewriter, op, resultType, self, cdf,
                                           /*shift=*/0);

    rewriter.replaceOp(op, result.getResult());
  } else if (approximate.compare("tanh") == 0) {
    // "tanh" approximate
    // GELU(x) = 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    // Formula taken from:
    // https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    auto selfShape = selfType.getShape();
    if (!selfType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only static shape tensor types are currently supported for Tanh "
              "approximation");

    auto numElem = std::accumulate(selfShape.begin(), selfShape.end(), 1,
                                   std::multiplies<int64_t>());

    Value half = tosa::getConstTensor<float>(rewriter, op,
                                             SmallVector<float>(numElem, 0.5f),
                                             selfShape, selfElemTy)
                     .value();
    Value one = tosa::getConstTensor<float>(rewriter, op,
                                            SmallVector<float>(numElem, 1.0f),
                                            selfShape, selfElemTy)
                    .value();
    Value three = tosa::getConstTensor<float>(rewriter, op,
                                              SmallVector<float>(numElem, 3.0f),
                                              selfShape, selfElemTy)
                      .value();

    // 0.044715
    Value magicNumber =
        tosa::getConstTensor<float>(rewriter, op,
                                    SmallVector<float>(numElem, 0.044715f),
                                    selfShape, selfElemTy)
            .value();

    // From <cmath> header: M_2_PI = 2 / pi
    Value twoOverPi =
        tosa::getConstTensor<float>(
            rewriter, op,
            SmallVector<float>(numElem, static_cast<float>(M_2_PI)), selfShape,
            selfElemTy)
            .value();

    // 0.5 * x
    auto halfInput = tosa::createMulOpAndCast(rewriter, op, resultType, half,
                                              self, /*shift=*/0);

    // sqrt(2/pi)
    auto sqrtTwoOverPi =
        rewriter.create<tosa::PowOp>(op->getLoc(), resultType, twoOverPi, half);

    // x^3
    auto inputPowThree =
        rewriter.create<tosa::PowOp>(op->getLoc(), resultType, self, three);

    // 0.044715 * x^3
    auto inputPowThreeMul =
        tosa::createMulOpAndCast(rewriter, op, resultType, magicNumber,
                                 inputPowThree.getResult(), /*shift=*/0);

    // x + 0.044715 * x^3
    auto inputPowThreeMulAdd = rewriter.create<tosa::AddOp>(
        op->getLoc(), resultType, self, inputPowThreeMul.getResult());

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    auto sqrtTwoOverPiMul = tosa::createMulOpAndCast(
        rewriter, op, resultType, sqrtTwoOverPi.getResult(),
        inputPowThreeMulAdd.getResult(), /*shift=*/0);

    // tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    auto tanh = rewriter.create<tosa::TanhOp>(op->getLoc(), resultType,
                                              sqrtTwoOverPiMul.getResult());

    // 1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    auto tanhAdd = rewriter.create<tosa::AddOp>(op->getLoc(), resultType, one,
                                                tanh.getResult());

    auto result = tosa::createMulOpAndCast(rewriter, op, resultType,
                                           halfInput.getResult(),
                                           tanhAdd.getResult(), /*shift=*/0);

    rewriter.replaceOp(op, result.getResult());
  } else {
    return rewriter.notifyMatchFailure(op,
                                       "Unsupported approximation algorithm");
  }

  return success();
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluBackwardOp>::matchAndRewrite(
    AtenGeluBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto self = adaptor.getSelf();
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy)) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");
  }

  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return rewriter.notifyMatchFailure(op, "Unsupported value of approximate");
  }

  auto loc = op->getLoc();

  const float cstAlpha0 = 1.12837916709551257390f;
  const float cstAlpha1 = 0.70710678118654752440f;
  const float oneHalf = 0.5f;
  const float kAlpha = cstAlpha0 * cstAlpha1;

  Value kAlphaHalf = tosa::getConstTensor<float>(rewriter, op, kAlpha * oneHalf,
                                                 {}, selfElemTy)
                         .value();
  Value negOneHalf =
      tosa::getConstTensor<float>(rewriter, op, -0.5f, {}, selfElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, loc, self, kAlphaHalf).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, loc, self, negOneHalf).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  Value inputSquared =
      tosa::createMulOpAndCast(rewriter, op, selfType, self, self, /*shift=*/0);
  Value negHalfInputSquared = tosa::createMulOpAndCast(
      rewriter, op, selfType, inputSquared, negOneHalf, /*shift=*/0);
  Value dinput =
      rewriter.create<tosa::ExpOp>(loc, selfType, negHalfInputSquared);
  Value cdf = buildUnitNormalCdf(rewriter, op, self, selfElemTy).value();
  Value dinputInput = tosa::createMulOpAndCast(rewriter, op, selfType, dinput,
                                               self, /*shift=*/0);
  Value dinputInputAlpha = tosa::createMulOpAndCast(
      rewriter, op, selfType, dinputInput, kAlphaHalf, /*shift=*/0);
  Value cdfExt =
      rewriter.create<tosa::AddOp>(loc, selfType, dinputInputAlpha, cdf);

  auto resultTy =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto result = tosa::createMulOpAndCast(
      rewriter, op, resultTy, adaptor.getGradOutput(), cdfExt, /*shift=*/0);

  rewriter.replaceOp(op, result.getResult());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenHardtanhBackwardOp>::matchAndRewrite(
    AtenHardtanhBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto self = adaptor.getSelf();
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType) {
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");
  }

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  // Integer types with width > 32 are not supported
  auto selfIntType = dyn_cast<IntegerType>(selfElemTy);
  if (selfIntType && selfIntType.getWidth() > 32) {
    return rewriter.notifyMatchFailure(
        op, "Integer types with width greater than 32 are not supported");
  }

  Value gradOutput = adaptor.getGradOutput();
  auto gradOutputType = dyn_cast<TensorType>(gradOutput.getType());

  Type gradOutputElemType = gradOutputType.getElementType();

  if (selfElemTy != gradOutputElemType) {
    return rewriter.notifyMatchFailure(
        op,
        "Input element type should be same as the grad_output element type.");
  }

  SmallVector<int64_t> constTypeShape(selfType.getRank(), 1);
  Value maxVal, minVal;

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getMinVal(), minVal,
                                     selfElemTy, constTypeShape))) {
    return rewriter.notifyMatchFailure(op, "Only scalar constant is supported");
  }

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getMaxVal(), maxVal,
                                     selfElemTy, constTypeShape))) {
    return rewriter.notifyMatchFailure(op, "Only scalar constant is supported");
  }

  Value replace =
      tosa::getConstTensor<float>(rewriter, op, 0, {}, selfElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, minVal)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, maxVal)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, gradOutput)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, replace).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  Type outType = getTypeConverter()->convertType(op.getType());

  Value lesser = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      minVal, self);

  Value greater = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      self, maxVal);

  Value cmp = rewriter.create<tosa::LogicalOrOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      lesser, greater);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cmp, replace,
                                              gradOutput);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenEmbeddingOp>::matchAndRewrite(
    AtenEmbeddingOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value weight = adaptor.getWeight();
  Value indices = adaptor.getIndices();
  RankedTensorType outType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));

  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!indicesType || !isa<IntegerType>(indicesType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Indices must be of integer tensor type");

  auto weightType = cast<RankedTensorType>(weight.getType());
  if (weightType.getRank() != 2)
    return op.emitError("weight must be of rank 2");

  // FIXME: padding_idx, scale_grad_by_freq and sparse are not handled yet.
  int64_t paddingIdx;
  if (!matchPattern(op.getPaddingIdx(), m_TorchConstantInt(&paddingIdx)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant int padding_idx for embedding op");

  bool scaleGradByFreq;
  if (!matchPattern(op.getScaleGradByFreq(),
                    m_TorchConstantBool(&scaleGradByFreq)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant bool scale_grad_by_freq for embedding op");
  if (scaleGradByFreq)
    return rewriter.notifyMatchFailure(
        op,
        "only supports scale_grad_by_freq equals to False for embedding op");

  bool isSparse;
  if (!matchPattern(op.getSparse(), m_TorchConstantBool(&isSparse)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant bool sparse for embedding op");
  if (isSparse)
    return rewriter.notifyMatchFailure(
        op, "only support sparse equals to False for embedding op");

  // For inference:
  //    Weights [num_embeddings, embedding_dim], Indices [X, Y]
  //    Output [X, Y, embedding_dim] = Weights[Indices[x, y]] forall x in X, y
  //    in Y
  //
  //    Condition: num_embeddings > Indices [x, y] forall x in X, y in Y

  // Reshape the weight, since tosa.gather expects a 3D tensor
  auto indicesShape = makeShapeTorchCompatible(indicesType.getShape());
  auto weightShape = makeShapeTorchCompatible(weightType.getShape());

  SmallVector<int64_t> newWeightShape = {1};
  for (auto s : weightShape)
    newWeightShape.push_back(s);

  auto reshapedWeight = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(newWeightShape),
                            weightType.getElementType()),
      weight, tosa::getTosaConstShape(rewriter, op->getLoc(), newWeightShape));

  int64_t numIndices = 1;
  if (indicesType.hasStaticShape()) {
    for (auto s : indicesShape)
      numIndices *= s;
  } else {
    numIndices = kUnknownSize;
  }

  SmallVector<int64_t> newIndicesShape = {1, numIndices};
  auto reshapedIndices = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(newIndicesShape),
                            indicesType.getElementType()),
      indices,
      tosa::getTosaConstShape(rewriter, op->getLoc(), newIndicesShape));

  auto castIndices =
      tosa::tosaCastTensorToType(
          rewriter, reshapedIndices,
          RankedTensorType::get(makeShapeLLVMCompatible(newIndicesShape),
                                rewriter.getIntegerType(32)))
          .value();

  SmallVector<int64_t> intermediateOutShape = {1, numIndices, weightShape[1]};
  auto gatherOp = rewriter.create<tosa::GatherOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(intermediateOutShape),
                            weightType.getElementType()),
      reshapedWeight, castIndices);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, outType, gatherOp,
      tosa::getTosaConstShape(rewriter, op->getLoc(),
                              makeShapeTorchCompatible(outType.getShape())));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTransposeIntOp>::matchAndRewrite(
    AtenTransposeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  // Only statically resolvable values are currently supported
  int64_t dim0, dim1;
  if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0)))
    return rewriter.notifyMatchFailure(op, "dim0 must be a Scalar constant");

  if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
    return rewriter.notifyMatchFailure(op, "dim1 must be a Scalar constant");

  dim0 = toPositiveDim(dim0, selfType.getRank());
  dim1 = toPositiveDim(dim1, selfType.getRank());

  auto selfRank = selfType.getRank();
  if (!isValidDim(dim0, selfRank) || !isValidDim(dim1, selfRank))
    return rewriter.notifyMatchFailure(
        op, "dim0 and dim1 must be less than tensor rank");

  SmallVector<int32_t> transposedDims;
  for (auto i = 0; i < selfType.getRank(); ++i)
    transposedDims.push_back(i);

  transposedDims[dim0] = dim1;
  transposedDims[dim1] = dim0;

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      rewriter.getDenseI32ArrayAttr(transposedDims));

  return success();
}

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenMinMaxDimOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto self = adaptor.getSelf();
    auto selfType = dyn_cast<TensorType>(self.getType());
    if (!selfType)
      return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

    const TypeConverter *typeConverter = this->getTypeConverter();
    auto indicesType =
        dyn_cast<TensorType>(typeConverter->convertType(op.getType(1)));
    if (!indicesType)
      return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

    auto selfElemType = selfType.getElementType();
    auto indicesElemType = indicesType.getElementType();

    // Only statically deducible values are currently supported
    int64_t dim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

    dim = toPositiveDim(dim, selfType.getRank());

    if (!isValidDim(dim, selfType.getRank()))
      return rewriter.notifyMatchFailure(op,
                                         "dim must be less than tensor rank");

    bool keepDim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(op,
                                         "keepdim must be a Scalar constant");

    SmallVector<int64_t> reducedShape, prunedShape;
    for (auto en :
         llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
      if (static_cast<int64_t>(en.index()) == dim) {
        reducedShape.push_back(1);
        continue;
      }
      reducedShape.push_back(en.value());
      prunedShape.push_back(en.value());
    }

    auto dimAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), dim);
    auto prunedShapeValue =
        tosa::getTosaConstShape(rewriter, op->getLoc(), prunedShape);

    Value reduceOp;
    if constexpr (std::is_same<TosaOpT, tosa::ReduceMinOp>() ||
                  std::is_same<TosaOpT, tosa::ReduceMaxOp>()) {
      // Use default NaN Propagation mode "PROPAGATE" for tosa.reduce_min
      // and tosa.reduce_max
      reduceOp = rewriter.create<TosaOpT>(
          op->getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(reducedShape),
                                selfElemType),
          self, dimAttr, /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
    } else {
      reduceOp = rewriter.create<TosaOpT>(
          op->getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(reducedShape),
                                selfElemType),
          self, dimAttr);
    }

    // To handle ReduceMinDim indices, we apply ArgMaxOp on the negate
    // of the input tensor, which will return indices of input's min values
    Value argMaxOp;
    if constexpr (std::is_same<AtenOpT, AtenMinDimOp>()) {
      Value negateOp =
          rewriter.create<tosa::NegateOp>(op->getLoc(), selfType, self);

      // Use default NaN Propagation mode "PROPAGATE" for tosa.argmax
      argMaxOp = rewriter.create<tosa::ArgMaxOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                                indicesElemType),
          negateOp, dimAttr, /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
    } else {
      // Use default NaN Propagation mode "PROPAGATE" for tosa.argmax
      argMaxOp = rewriter.create<tosa::ArgMaxOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                                indicesElemType),
          self, dimAttr, /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
    }

    if (argMaxOp.getType() != indicesType) {
      argMaxOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(), indicesType, argMaxOp,
          tosa::getTosaConstShape(rewriter, op->getLoc(), reducedShape));
    }

    if (!keepDim) {
      reduceOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                                selfElemType),
          reduceOp, prunedShapeValue);
    }

    rewriter.replaceOp(op, {reduceOp, argMaxOp});

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // Only statically deducible values are currently supported
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfType.getRank());

  if (!isValidDim(dim, selfType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim must less than tensor rank");

  int64_t start;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(op, "start must be a Scalar constant");

  if (start < 0) {
    start = toPositiveDim(start, selfType.getShape()[dim]);
    if (!isValidDim(start, selfType.getShape()[dim]))
      return rewriter.notifyMatchFailure(op, "start is not a valid index");
  }
  start = std::min(selfType.getShape()[dim], start);

  int64_t end;
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end))) {
    if (isa<ConstantNoneOp>(op.getEnd().getDefiningOp()))
      end = selfType.getShape()[dim];
    else
      return rewriter.notifyMatchFailure(op, "end must be a Scalar constant");
  }
  // support for end < 0
  end = toPositiveDim(end, selfType.getShape()[dim]);
  // support for end out of upper bound
  end = (end > selfType.getShape()[dim] ? selfType.getShape()[dim] : end);

  // FIXME: add support for start < 0 and end < start
  if (end < start)
    return rewriter.notifyMatchFailure(op,
                                       "Currently unsupported: end < start");

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
    return rewriter.notifyMatchFailure(op, "step must be a Scalar constant");

  if (step != 1)
    return rewriter.notifyMatchFailure(
        op, "step value other than 1 is currently unsupported");

  SmallVector<int64_t> startSlice(selfType.getRank(), 0);
  SmallVector<int64_t> sizeSlice =
      llvm::to_vector(makeShapeTorchCompatible(selfType.getShape()));

  startSlice[dim] = start;
  sizeSlice[dim] = end - start;

  rewriter.replaceOpWithNewOp<tosa::SliceOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), startSlice),
      tosa::getTosaConstShape(rewriter, op->getLoc(), sizeSlice));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenBroadcastToOp>::matchAndRewrite(
    AtenBroadcastToOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();
  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> resultShape;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(resultShape)))
    return rewriter.notifyMatchFailure(op,
                                       "Size must consist of Scalar constants");

  int64_t inputRank = selfType.getRank();
  int64_t outputRank = resultShape.size();
  if (inputRank > outputRank)
    return rewriter.notifyMatchFailure(
        op, "Input tensor rank cannot be greater than output tensor rank");

  // Get the result type
  auto resultType = getTypeConverter()->convertType(op.getType());

  SmallVector<int64_t> inputShape(
      makeShapeTorchCompatible(selfType.getShape()));

  // If input rank is smaller than output rank, we reshape the input tensor to
  // be the same rank as the output tensor by prepending 1s to the input shape
  SmallVector<int64_t> targetInputShape;
  for (int64_t i = 0; i < outputRank - inputRank; i++)
    targetInputShape.push_back(1);
  targetInputShape.append(inputShape);

  // Result dimension -1 means not changing the size of that dimension.
  // Adjust it by assigning its inputShape.
  for (auto shape :
       llvm::enumerate(makeShapeTorchCompatible(targetInputShape))) {
    auto index = shape.index();
    if (resultShape[index] == -1)
      resultShape[index] = shape.value();
  }

  for (int64_t i = 0; i < outputRank; i++) {
    if (targetInputShape[i] != resultShape[i] && targetInputShape[i] != 1)
      return rewriter.notifyMatchFailure(
          op, "Input and result shapes should be equal at each dimension or "
              "input shape should be 1");
  }

  // Check for identity case i.e, for ex: [a, b, c] -> [a, b, c]. If this is
  // true then we can replace the op result with the input operand directly.
  if (llvm::equal(inputShape, resultShape)) {
    // If we reach here, then it means that the broadcasting is not required
    // since the input and result are of same shape.
    op.replaceAllUsesWith(op.getSelf());
    rewriter.eraseOp(op);
  } else {
    // By using reshape and tile ops, support for input rank smaller than result
    // rank is allowed. If the rank is smaller, we reshape the input to be the
    // same rank as the result, then use tile to expand it. The way it was
    // handled before involves adding the input tensor to a const zero tensor of
    // output shape to utilize the innate broadcast feature of the TOSA add op.
    // That poses the danger of sign bit flips for denormalized values.
    // Basically, this approach to broadcast_to legalization allows for more
    // flexibility in rank differences and also offers more safety.
    Value reshapedInput = self;
    if (!llvm::equal(inputShape, targetInputShape))
      reshapedInput = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeTorchCompatible(targetInputShape),
                                selfElemTy),
          self,
          tosa::getTosaConstShape(rewriter, op->getLoc(), targetInputShape));

    SmallVector<int64_t> tileOpShape;
    for (int64_t i = 0; i < outputRank; i++) {
      if (targetInputShape[i] == 1) {
        tileOpShape.push_back(resultShape[i]);
      } else {
        tileOpShape.push_back(1);
      }
    }

    auto tileOpMultiples =
        tosa::getTosaConstShape(rewriter, op->getLoc(), tileOpShape);

    auto result = rewriter.create<tosa::TileOp>(op->getLoc(), resultType,
                                                reshapedInput, tileOpMultiples);

    rewriter.replaceOp(op, {result.getResult()});
  }

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenGatherOp>::matchAndRewrite(
    AtenGatherOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For easy understanding of this algorithm, I will comment the code with an
  // exact example: torch.aten.gather (!torch.vtensor<[1,4,3],f32>,
  // !torch.int-1, !torch.vtensor<[1,4,2],si64>)
  // -> !torch.vtensor<[1,4,2],f32>
  // https://gist.github.com/AmosLewis/2f18434397025211da4491735bcc6db6

  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType input are currently supported");

  auto index = adaptor.getIndex();
  auto indexType = dyn_cast<RankedTensorType>(adaptor.getIndex().getType());
  auto inputShape = inputType.getShape();
  int paramsRank = inputShape.size();

  if (!indexType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType index are currently supported");

  // Check `index` and `input` param should have the same rank
  if (indexType.getRank() != inputType.getRank())
    return rewriter.notifyMatchFailure(
        op, "`index` and `input` param should have the same rank");

  // Dynamic shape check
  if (!inputType.hasStaticShape() || !indexType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "AtenGatherOp: support for dynamic input "
            "shape not implemented");

  // index i64 to i32 for tosa compatitable
  if (indexType.getElementType() != rewriter.getIntegerType(32)) {
    index = tosa::tosaCastTensorToType(
                rewriter, index,
                RankedTensorType::get(indexType.getShape(),
                                      rewriter.getIntegerType(32)))
                .value();
  }

  // Get positive dim
  int64_t dim{0};
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: value `dim` should be a torch constant int");
  dim = toPositiveDim(dim, paramsRank);
  if (!isValidDim(dim, paramsRank))
    return rewriter.notifyMatchFailure(op, "Not dim are invalid");

  // check sparseGrad is bool type
  bool sparseGrad = false;
  if (!matchPattern(op.getSparseGrad(), m_TorchConstantBool(&sparseGrad)))
    return rewriter.notifyMatchFailure(
        op, "only constant boolean `sparse_grad` param supported");
  if (sparseGrad)
    return rewriter.notifyMatchFailure(
        op, "only constant boolean `sparse_grad` == false supported");

  // Get the output type
  auto outType = getTypeConverter()->convertType(op.getType());

  // convert torch style index and dim into tf style indices
  // tensor<[1,4,2],si64> -> tensor<[1,4,2,3],si64>
  auto indicesTf =
      tosa::convertTorchIndexToTfIndices(rewriter, op, input, index, dim);
  if (!indicesTf) {
    return rewriter.notifyMatchFailure(op,
                                       "Convert TorchIndex To TfIndices fail.");
  }

  // do the tf gathernp algorithm with tf style indices as input.
  auto result =
      tosa::convertGatherNdOp(rewriter, op, outType, input, indicesTf.value());

  if (!result) {
    return rewriter.notifyMatchFailure(op, "Convert GatherNdOp fail.");
  }
  rewriter.replaceOp(op, {result.value()});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType inputs are currently supported");

  auto index = adaptor.getIndex();
  auto indexType = dyn_cast<RankedTensorType>(index.getType());
  ArrayRef<int64_t> indexShape = indexType.getShape();

  if (!indexType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType indices are currently supported");

  auto inputShape = inputType.getShape();
  int inputRank = inputType.getRank();

  // indexShape is reference. storing actual data in SmallVector to avoid
  // use-after-free
  SmallVector<int64_t> indexShapeTorchCompatible{};
  if (indexType.getRank() == 0) {
    indexShapeTorchCompatible = makeShapeTorchCompatible({1});
    indexShape = indexShapeTorchCompatible;
    index = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get(indexShape, indexType.getElementType()), index,
        tosa::getTosaConstShape(rewriter, op->getLoc(), indexShape));
  }

  // Dynamic shape check
  if (!inputType.hasStaticShape() || !indexType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "AtenIndexSelectOp: support for dynamic input "
            "shape not implemented");

  // index i64 to i32 for tosa compatible
  if (indexType.getElementType() != rewriter.getIntegerType(32)) {
    index = tosa::tosaCastTensorToType(
                rewriter, index,
                RankedTensorType::get(indexShape, rewriter.getIntegerType(32)))
                .value();
  }

  // Get positive dim
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "Value `dim` should be a torch constant int");
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "Value `dim` is invalid");

  // Get the output type
  auto outType = getTypeConverter()->convertType(op.getType());

  // Reshape and expand the index tensor to have same rank and same dimensions
  // (except for the targeted dim) as the input
  //
  // For example:
  // Input shape = (4, 5, 6)
  // Index vector shape = (2)
  // Targeted dim = 1
  // Reshaped and expanded index vector shape = (4, 2, 6)
  //
  // By reshaping and expanding the index vector, we can supply it into the
  // gather op to mimic the functionality of aten.index_select
  SmallVector<int64_t> indicesInputRankShape;
  for (int64_t i = 0; i < inputRank; i++) {
    if (i == dim) {
      indicesInputRankShape.push_back(indexShape[0]);
    } else {
      indicesInputRankShape.push_back(1);
    }
  }

  auto indicesInputRankType =
      RankedTensorType::get(makeShapeLLVMCompatible(indicesInputRankShape),
                            rewriter.getIntegerType(32));

  auto reshapedIndices = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(), indicesInputRankType, index,
      tosa::getTosaConstShape(rewriter, op->getLoc(), indicesInputRankShape));

  SmallVector<int64_t> tileShape(indicesInputRankShape);
  SmallVector<int64_t> expandedIndicesShape(indicesInputRankShape);
  for (int64_t i = 0; i < inputRank; i++) {
    if (tileShape[i] == 1 && i != dim) {
      tileShape[i] = inputShape[i];
      expandedIndicesShape[i] = inputShape[i];
    } else {
      tileShape[i] = 1;
    }
  }

  auto tileType =
      RankedTensorType::get(makeShapeLLVMCompatible(expandedIndicesShape),
                            rewriter.getIntegerType(32));

  auto tileOpMultiples =
      tosa::getTosaConstShape(rewriter, op->getLoc(), tileShape);

  auto expandedIndices = rewriter.create<tosa::TileOp>(
      op->getLoc(), tileType, reshapedIndices.getResult(), tileOpMultiples);

  // convert torch style index and dim into tf style indices
  // tensor<[1,4,2],si64> -> tensor<[1,4,2,3],si64>
  auto indicesTf = tosa::convertTorchIndexToTfIndices(
      rewriter, op, input, expandedIndices.getResult(), dim);
  if (!indicesTf)
    return rewriter.notifyMatchFailure(
        op, "Convert TorchIndex To TfIndices failed");

  // do the tf gathernd algorithm with tf style indices as input.
  auto result =
      tosa::convertGatherNdOp(rewriter, op, outType, input, indicesTf.value());

  if (!result) {
    return rewriter.notifyMatchFailure(op, "Convert GatherNdOp failed");
  }
  rewriter.replaceOp(op, {result.value()});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIndexPutHackedTwinOp>::matchAndRewrite(
    AtenIndexPutHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto selfType = dyn_cast<TensorType>(input.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  auto fillValues = adaptor.getValues();
  auto valuesType = dyn_cast<TensorType>(fillValues.getType());
  if (!valuesType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  // Deal with torch.prim.ListConstruct of non const value to get the index
  // Index_put-like ops are now decomposed to aten.index_put.hacked_twin with
  // stricter semantics, i.e., no None index in indices argument.
  auto tensorList = op.getIndices();
  SmallVector<Value> tensorsTorchType;
  if (!getListConstructElements(tensorList, tensorsTorchType))
    return op.emitError("Tensor list is not from list construct");
  auto indexTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), tensorsTorchType);

  auto outType = getTypeConverter()->convertType(op.getType());

  bool accumulate{false};
  if (!matchPattern(op.getAccumulate(), m_TorchConstantBool(&accumulate)))
    return rewriter.notifyMatchFailure(
        op, "Accumulate is not a constant bool value");

  // No support for accumulate mode yet
  if (accumulate)
    return rewriter.notifyMatchFailure(
        op, "Accumulate mode is not currently supported");

  SmallVector<Value> indicesTfConcatTensors;
  SmallVector<int64_t> indexesRank;
  SmallVector<SmallVector<int64_t>> indexesShape;

  // concat index tensor into to indices tensor for concat
  for (size_t i = 0; i < indexTensors.size(); i++) {
    auto index = indexTensors[i];

    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    auto indexShape = indexType.getShape();
    indexesShape.push_back(makeShapeTorchCompatible(indexShape));
    indexesRank.push_back(indexType.getRank());

    // index i64 to i32 for tosa compatible
    if (indexType.getElementType() != rewriter.getIntegerType(32))
      index =
          tosa::tosaCastTensorToType(
              rewriter, index,
              RankedTensorType::get(indexShape, rewriter.getIntegerType(32)))
              .value();

    // Expand last dim of index to tf indices [3] -> [3,1]
    // convert [0,0,0]  to [[0],[0],[0]]
    SmallVector<int64_t> indiceShapeOneDim;
    for (auto shape : indexShape)
      indiceShapeOneDim.push_back(shape);
    indiceShapeOneDim.push_back(1);

    auto indicesTfOneDim = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        RankedTensorType::get(indiceShapeOneDim, rewriter.getIntegerType(32)),
        index,
        tosa::getTosaConstShape(rewriter, op->getLoc(), indiceShapeOneDim));

    // create concat tensor for indicesTf
    // ([[0],[0],[0]], [[1],[2],[3]])
    indicesTfConcatTensors.push_back(indicesTfOneDim.getResult());
  }

  // Right now only support multiple indexes with same shape
  // TODO for different shape multiple indexes, add broadcast_to for small
  // shape
  for (auto indexShapeOneDim : indexesShape) {
    if (!llvm::equal(indexesShape[0], indexShapeOneDim)) {
      return rewriter.notifyMatchFailure(
          op, "Only support indices with same shape");
    }
  }

  // concat each indices into indicesTf: shape ([3,1],[3,1]) -> [3,2]
  // ([0,0,0],[1,2,3]) -> [[0,1],[0,2], [0,3]]
  auto indicesShapeConcat = indexesShape[0];
  uint64_t lastDim = indexesRank[0];
  indicesShapeConcat.push_back(indicesTfConcatTensors.size());
  auto indicesTf = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(),
      GetTypeFromTensorShape(indicesShapeConcat, rewriter.getIntegerType(32)),
      indicesTfConcatTensors, lastDim);

  if (!indicesTf)
    return rewriter.notifyMatchFailure(
        op, "Convert PyTorch index to TensorFlow indices failed");

  auto result = tosa::convertScatterNdOp(rewriter, op, outType, input,
                                         indicesTf.getResult(), fillValues);

  if (!result)
    return rewriter.notifyMatchFailure(op, "Convert ScatterNdOp failed");

  rewriter.replaceOp(op, {result.value()});

  return success();
}

std::optional<Value> wrapNegativeIndices(Value index, int maxIndex,
                                         Operation *op,
                                         ConversionPatternRewriter &rewriter) {

  auto zeroValue = tosa::getConstTensor<int32_t>(rewriter, op, 0, {}).value();
  auto maxIndexValue =
      tosa::getConstTensor<int32_t>(rewriter, op, maxIndex, {}).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), index, zeroValue)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), index, maxIndexValue)
          .failed())
    return std::nullopt;

  auto indexType = dyn_cast<RankedTensorType>(index.getType());

  auto wrappedIndicesOp = tosa::CreateOpAndInfer<tosa::AddOp>(
      rewriter, op->getLoc(), indexType, maxIndexValue, index);
  auto boolType = indexType.clone(rewriter.getIntegerType(1));
  auto isNegativeIndices = tosa::CreateOpAndInfer<tosa::GreaterOp>(
      rewriter, op->getLoc(), boolType, zeroValue, index);
  return tosa::CreateOpAndInfer<tosa::SelectOp>(rewriter, op->getLoc(),
                                                indexType, isNegativeIndices,
                                                wrappedIndicesOp, index);
}

template <>
LogicalResult ConvertAtenOp<AtenIndexTensorHackedTwinOp>::matchAndRewrite(
    AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // t        = tf.constant([[1, 2, 3, 4, 5],[6,7,8,9,10],
  //                         [11,12,13,14,15],[16,17,18,19,20]]) # 4*5
  // i        = tf.constant([[1,2,3], [3,2,1]]) # 2*3
  // i_expand = tf.expand_dims(i,axis=2) # 2*3*1
  // IndexTensorOutput = tf.gather_nd(t,tf.i_expand)
  //                   = torch.ops.aten.index(t, (i, )) = t[i] # 2*3*5
  // [[[ 6,  7,  8,  9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
  //  [[16, 17, 18, 19, 20], [11, 12, 13, 14, 15], [ 6,  7,  8,  9, 10]]]
  auto input = adaptor.getSelf();
  auto inputTensorType =
      dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  // Check input is a tensor type.
  if (!inputTensorType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  // Dynamic shape check
  if (!inputTensorType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "AtenIndexTensorHackedTwinOp: support for dynamic input "
            "shape not implemented");

  // Deal with torch.prim.ListConstruct of non const value to get the index
  auto tensorList = op.getIndices();
  SmallVector<Value> tensorsTorchType;
  if (!getListConstructElements(tensorList, tensorsTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");
  auto indexTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), tensorsTorchType);

  auto outType = getTypeConverter()->convertType(op.getType());

  Operation *indicesTf;

  // Support for multiple indexes
  if (indexTensors.size() > 1) {
    // t[i, i]
    // = torch.ops.aten.index(t,(i,i))
    // = tensor([[ t[1,1], t[2,2], t[3,3]],
    //           [ t[3,3], t[2,2],  t[1,1]]])
    // = tensor([[ 7, 13, 19], [19, 13,  7]])
    // = tf.gather_nd(t,tf.ii_expand)
    // ii_expand
    // = tf.concat((i_expand,i_expand), dim=2)
    // = tf.constant([[[1,1],[2,2],[3,3]],
    //                [[3,3],[2,2],[1,1]]]) # 2*3*2
    SmallVector<Value> indicesTfConcatTensors;
    SmallVector<int64_t> indexesRank;
    SmallVector<SmallVector<int64_t>> indexesShape;

    // concat index tensor into to indices tensor for concat
    for (size_t i = 0; i < indexTensors.size(); i++) {
      auto index = indexTensors[i];

      auto indexType = dyn_cast<RankedTensorType>(index.getType());
      auto indexShape = indexType.getShape();
      indexesShape.push_back(makeShapeTorchCompatible(indexShape));
      indexesRank.push_back(indexType.getRank());

      // Make type of index tosa compatible, i64 to i32.
      if (indexType.getElementType() != rewriter.getIntegerType(32)) {
        index =
            tosa::tosaCastTensorToType(
                rewriter, index,
                RankedTensorType::get(indexShape, rewriter.getIntegerType(32)))
                .value();
      }

      index = wrapNegativeIndices(index, inputTensorType.getShape()[i], op,
                                  rewriter)
                  .value();
      // Expand last dim of index to tf indices [2,3] -> [2,3,1]
      SmallVector<int64_t> indiceShapeOneDim;
      for (auto shape : indexShape) {
        indiceShapeOneDim.push_back(shape);
      }
      indiceShapeOneDim.push_back(1);
      auto indicesTfOneDim = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
          rewriter, op->getLoc(),
          RankedTensorType::get(indiceShapeOneDim, rewriter.getIntegerType(32)),
          index,
          tosa::getTosaConstShape(rewriter, op->getLoc(), indiceShapeOneDim));

      // create concat tensor for indicesTf
      indicesTfConcatTensors.push_back(indicesTfOneDim.getResult());
    }

    auto getRankExtendedShape =
        [](SmallVector<int64_t> inputShape,
           SmallVector<int64_t> maxRank1DimShape) -> SmallVector<int64_t> {
      SmallVector<int64_t> rankExtendedShape(maxRank1DimShape);
      auto inputRank = inputShape.size();
      auto maxRank = maxRank1DimShape.size();
      auto startIdx = maxRank - inputRank;
      for (size_t i = startIdx; i < maxRank; i++) {
        rankExtendedShape[i] = inputShape[i - startIdx];
      }
      return rankExtendedShape;
    };

    bool hasDiffShapedIndexes = false;
    for (auto indexShapeOneDim : indexesShape) {
      if (!llvm::equal(indexesShape[0], indexShapeOneDim)) {
        hasDiffShapedIndexes = true;
        break;
      }
    }

    if (hasDiffShapedIndexes) {
      int64_t maxRank = 1;
      for (auto idxRank : indexesRank) {
        if (idxRank > maxRank)
          maxRank = idxRank;
      }
      // Tensor shape of max rank, each dim being 1
      SmallVector<int64_t> maxRank1DimShape;
      for (int i = 0; i < maxRank; i++)
        maxRank1DimShape.push_back(1);
      // Tensor shape of max rank, each dim being the max dim.
      SmallVector<int64_t> maxRankMaxDimShape(maxRank1DimShape);

      auto updateMaxRankMaxDimShape =
          [&](SmallVector<int64_t> broadcastedShape) -> LogicalResult {
        for (size_t i = 0; i < maxRankMaxDimShape.size(); i++) {
          // check for malformed index tensors
          if (broadcastedShape[i] != 1 && maxRankMaxDimShape[i] != 1 &&
              maxRankMaxDimShape[i] != broadcastedShape[i]) {
            return failure();
          }
          if (broadcastedShape[i] > maxRankMaxDimShape[i])
            maxRankMaxDimShape[i] = broadcastedShape[i];
        }
        return success();
      };

      for (size_t i = 0; i < indexesRank.size(); i++) {
        // Reshape all index tensors to same maxRank
        auto idxRank = indexesRank[i];
        auto unreshapedIdxTensor = indicesTfConcatTensors[i];
        SmallVector<int64_t> broadcastedShape =
            getRankExtendedShape(indexesShape[i], maxRank1DimShape);

        if (idxRank < maxRank) {
          auto idxType =
              dyn_cast<RankedTensorType>(indicesTfConcatTensors[i].getType());
          // indicesTfConcatTensors has a trailing [1] dim for the final concat.
          auto broadcastedShapeTf(broadcastedShape);
          broadcastedShapeTf.push_back(1);
          auto reshapeOutputTy = RankedTensorType::get(
              broadcastedShapeTf, idxType.getElementType());
          // Update the tensor array with the max rank-extended form
          indicesTfConcatTensors[i] = rewriter.create<tosa::ReshapeOp>(
              op->getLoc(), reshapeOutputTy, unreshapedIdxTensor,
              tosa::getTosaConstShape(rewriter, op->getLoc(),
                                      broadcastedShapeTf));
        }

        // Construct the max rank broadcasted form of all index tensors with
        // each index tensor.
        if (updateMaxRankMaxDimShape(broadcastedShape).failed()) {
          return rewriter.notifyMatchFailure(
              op, "Malformed index tensors that have mismatched dim shapes");
        }

        // Every index now has the same rank but not yet same shape until
        // tosa.tile below.
        indexesShape[i] = broadcastedShape;
        indexesRank[i] = maxRank;
      }

      auto getTileOpShape = [&](SmallVector<int64_t> indexShape,
                                SmallVector<int64_t> &tileOpShape) -> bool {
        bool needsTiling = false;
        for (size_t i = 0; i < indexShape.size(); i++) {
          if (1 == indexShape[i]) {
            tileOpShape.push_back(maxRankMaxDimShape[i]);
            needsTiling = true;
          } else {
            tileOpShape.push_back(1);
          }
        }
        return needsTiling;
      };

      // Use tosa.tile to broadcast in multiple dims so all index tensors have
      // the same shape. This materializes new tensors.
      for (size_t i = 0; i < indexesRank.size(); i++) {
        SmallVector<int64_t> tileOpShape;
        bool needsTiling = getTileOpShape(indexesShape[i], tileOpShape);

        if (needsTiling) {
          auto idxType =
              dyn_cast<RankedTensorType>(indicesTfConcatTensors[i].getType());

          // indicesTfConcatTensors has a trailing [1] dim for the final concat.
          auto maxRankMaxDimShapeTf(maxRankMaxDimShape);
          maxRankMaxDimShapeTf.push_back(1);

          auto tileOpShapeTf(tileOpShape);
          tileOpShapeTf.push_back(1);

          auto tileOutputTy = RankedTensorType::get(maxRankMaxDimShapeTf,
                                                    idxType.getElementType());
          auto reshapedIdxTensor = indicesTfConcatTensors[i];

          auto tileOpMultiples =
              tosa::getTosaConstShape(rewriter, op->getLoc(), tileOpShapeTf);

          indicesTfConcatTensors[i] = rewriter.create<tosa::TileOp>(
              op->getLoc(), tileOutputTy, reshapedIdxTensor, tileOpMultiples);
        }

        // Every index tensor now has the same rank and shape
        indexesShape[i] = maxRankMaxDimShape;
      }
    }

    // concat each indices into indicesTf: shape [2,3,1],[2,3,1] -> [2,3,2]
    auto indicesShapeConcat = indexesShape[0];
    uint64_t lastDim = indexesRank[0];
    indicesShapeConcat.push_back(indicesTfConcatTensors.size());
    indicesTf = tosa::CreateOpAndInfer<tosa::ConcatOp>(
        rewriter, op->getLoc(),
        GetTypeFromTensorShape(indicesShapeConcat, rewriter.getIntegerType(32)),
        indicesTfConcatTensors, lastDim);

  } else {

    // Single index
    auto index = indexTensors[0];
    auto indexType = dyn_cast<RankedTensorType>(index.getType());
    auto indexShape = indexType.getShape();
    // index i64 to i32 for tosa compatible
    if (indexType.getElementType() != rewriter.getIntegerType(32)) {
      index =
          tosa::tosaCastTensorToType(
              rewriter, index,
              RankedTensorType::get(indexShape, rewriter.getIntegerType(32)))
              .value();
    }

    index =
        wrapNegativeIndices(index, inputTensorType.getShape()[0], op, rewriter)
            .value();

    // Expand last dim of index to tf indices [2,3] -> [2,3,1]
    SmallVector<int64_t> indicesShape;
    for (auto shape : indexShape) {
      indicesShape.push_back(shape);
    }
    indicesShape.push_back(1);
    indicesTf = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        RankedTensorType::get(indicesShape, rewriter.getIntegerType(32)), index,
        tosa::getTosaConstShape(rewriter, op->getLoc(), indicesShape));
  }

  if (!indicesTf) {
    return rewriter.notifyMatchFailure(op,
                                       "Convert TorchIndex To TfIndices fail.");
  }
  // do the tf gathernp algorithm with tf style indices as input.
  auto result = tosa::convertGatherNdOp(rewriter, op, outType, input,
                                        indicesTf->getResult(0));

  if (!result) {
    return rewriter.notifyMatchFailure(
        op, "Convert GatherNdOp fail for index tensor.");
  }
  rewriter.replaceOp(op, {result.value()});

  return success();
}

// Legalization for aten.scatter.src
template <>
LogicalResult ConvertAtenOp<AtenScatterSrcOp>::matchAndRewrite(
    AtenScatterSrcOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType inputs are currently supported");

  auto inputShape = inputType.getShape();
  auto paramsRank = inputType.getRank();

  auto index = adaptor.getIndex();
  auto indexType = dyn_cast<RankedTensorType>(index.getType());
  if (!indexType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType indices are currently supported");

  // Check `index` and `input` param should have the same rank
  if (indexType.getRank() != paramsRank)
    return rewriter.notifyMatchFailure(
        op, "Params index and input should have the same rank");

  auto indexShape = indexType.getShape();

  auto src = adaptor.getSrc();
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  if (!srcType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType sources are currently supported");

  // Check `src` and `input` param should have the same rank
  if (srcType.getRank() != paramsRank)
    return rewriter.notifyMatchFailure(
        op, "Src and input should have the same rank");

  auto srcShape = srcType.getShape();

  // Dynamic shape check
  if (!inputType.hasStaticShape() || !indexType.hasStaticShape() ||
      !srcType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Support for dynamic shape not implemented");

  // index i64 to i32 for tosa compatitable
  if (indexType.getElementType() != rewriter.getIntegerType(32)) {
    index = tosa::tosaCastTensorToType(
                rewriter, index,
                RankedTensorType::get(indexShape, rewriter.getIntegerType(32)))
                .value();
  }

  // Get positive dim
  int64_t dim{0};
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op,
                                       "Dim value should be a constant int");

  dim = toPositiveDim(dim, paramsRank);
  if (!isValidDim(dim, paramsRank))
    return rewriter.notifyMatchFailure(op, "Dim is invalid");

  // It is also required that index.size(d) <= src.size(d) for all dimensions d,
  // and that index.size(d) <= self.size(d) for all dimensions d != dim
  for (int64_t d = 0; d < paramsRank; d++) {
    if (d != dim) {
      if (indexShape[d] > srcShape[d] || indexShape[d] > inputShape[d])
        return rewriter.notifyMatchFailure(
            op, "Index size should be smaller or equal to src or input size "
                "for all dimensions d != dim");
    }
  }

  // Get the output type
  auto outType = getTypeConverter()->convertType(op.getType());

  // convert PyTorch style index and dim into TensorFlows tyle indices
  // tensor<[1,4,2],si64> -> tensor<[1,4,2,3],si64>
  auto indicesTf =
      tosa::convertTorchIndexToTfIndices(rewriter, op, input, index, dim);
  if (!indicesTf)
    return rewriter.notifyMatchFailure(
        op, "Convert PyTorch index and dim to TensorFlow indices failed");

  // Perform the TensorFlow ScatterNd algorithm with TensorFlow style indices as
  // input.
  auto result = tosa::convertScatterNdOp(rewriter, op, outType, input,
                                         indicesTf.value(), src);

  if (!result)
    return rewriter.notifyMatchFailure(op, "Convert ScatterNdOp failed");

  rewriter.replaceOp(op, {result.value()});
  return success();
}

// Legalization for aten.slice_scatter
template <>
LogicalResult ConvertAtenOp<AtenSliceScatterOp>::matchAndRewrite(
    AtenSliceScatterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType inputs are currently supported");

  auto inputShape = inputType.getShape();
  auto paramsRank = inputType.getRank();

  auto src = adaptor.getSrc();
  auto srcType = dyn_cast<RankedTensorType>(src.getType());
  if (!srcType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType sources are currently supported");

  // Check `src` and `input` param should have the same rank
  if (srcType.getRank() != paramsRank)
    return rewriter.notifyMatchFailure(
        op, "Src and input should have the same rank");

  auto srcShape = srcType.getShape();

  // Dynamic shape check
  if (!inputType.hasStaticShape() || !srcType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Support for dynamic shape not implemented");

  // Get positive dim
  int64_t dim{0};
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op,
                                       "Dim value should be a constant int");

  dim = toPositiveDim(dim, paramsRank);
  if (!isValidDim(dim, paramsRank))
    return rewriter.notifyMatchFailure(op, "Dim is invalid");

  // Get start, end, and step params
  // If start and end params are not specified, assign them to 0 and
  // inputShape[dim], respectively.
  int64_t start{0};
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(op,
                                       "Start value should be a constant int");
  if (start < 0)
    start += inputShape[dim];

  int64_t end{inputShape[dim]};
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end)))
    return rewriter.notifyMatchFailure(op,
                                       "End value should be a constant int");
  if (end < 0)
    end += inputShape[dim];

  if (end > inputShape[dim])
    end = inputShape[dim];

  if (start >= end)
    return rewriter.notifyMatchFailure(
        op, "Start value greater than end value not supported");

  int64_t step{1};
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
    return rewriter.notifyMatchFailure(op,
                                       "Step value should be a constant int");

  // Create PyTorch style scatter index based on start, end, and step values
  int64_t outerRepeat{1}, innerRepeat{1};
  for (int64_t i = 0; i < dim; i++)
    outerRepeat *= srcShape[i];

  for (int64_t i = dim + 1; i < paramsRank; i++)
    innerRepeat *= srcShape[i];

  SmallVector<int32_t> indexVec;
  for (int64_t i = 0; i < outerRepeat; i++) {
    for (int32_t indexVal = start; indexVal < end; indexVal += step) {
      for (int64_t j = 0; j < innerRepeat; j++) {
        indexVec.push_back(indexVal);
      }
    }
  }

  Value index =
      tosa::getConstTensor<int32_t>(rewriter, op, indexVec, srcShape).value();

  // Get the output type
  auto outType = getTypeConverter()->convertType(op.getType());

  // convert PyTorch style index and dim into TensorFlows tyle indices
  // tensor<[1,4,2],si64> -> tensor<[1,4,2,3],si64>
  auto indicesTf =
      tosa::convertTorchIndexToTfIndices(rewriter, op, input, index, dim);
  if (!indicesTf)
    return rewriter.notifyMatchFailure(
        op, "Convert PyTorch index and dim to TensorFlow indices failed");

  // Perform the TensorFlow ScatterNd algorithm with TensorFlow style indices as
  // input.
  auto result = tosa::convertScatterNdOp(rewriter, op, outType, input,
                                         indicesTf.value(), src);

  if (!result)
    return rewriter.notifyMatchFailure(op, "Convert ScatterNdOp failed");

  rewriter.replaceOp(op, {result.value()});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenAbsOp>::matchAndRewrite(
    AtenAbsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::AbsOp>(op, outType, adaptor.getSelf());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenWhereSelfOp>::matchAndRewrite(
    AtenWhereSelfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto self = adaptor.getSelf();
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types inputs are currently supported");

  auto cond = adaptor.getCondition();
  auto condType = dyn_cast<TensorType>(cond.getType());
  if (!condType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types conditions are currently supported");

  auto other = adaptor.getOther();
  auto otherType = dyn_cast<TensorType>(other.getType());
  if (!otherType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types inputs are currently supported");

  auto outType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), cond, self).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), cond, other).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, other).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cond, self, other);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIscloseOp>::matchAndRewrite(
    AtenIscloseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // check args
  double rtol, atol;
  bool equalNan;
  if (!matchPattern(op.getRtol(), m_TorchConstantFloat(&rtol)))
    return rewriter.notifyMatchFailure(op, "rtol must be a scalar constant");
  if (!matchPattern(op.getAtol(), m_TorchConstantFloat(&atol)))
    return rewriter.notifyMatchFailure(op, "atol must be a scalar constant");
  if (!matchPattern(op.getEqualNan(), m_TorchConstantBool(&equalNan)))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: equal_nan is expected to be false");

  // check tensor type.
  auto self = adaptor.getSelf();
  auto selfType = dyn_cast<TensorType>(self.getType());
  auto other = adaptor.getOther();
  auto otherType = dyn_cast<TensorType>(other.getType());
  if (!selfType || !otherType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");
  if (!selfType.hasStaticShape() || !otherType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");
  if (!isa<mlir::FloatType>(selfType.getElementType()) ||
      !isa<mlir::FloatType>(otherType.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: only FP element type is supported");
  }
  auto rtolConstOp =
      tosa::getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(rtol));
  auto atolConstOp =
      tosa::getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(atol));

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, other).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, rtolConstOp)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, atolConstOp)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  // Reinitialize selfType and otherType after equalizing ranks
  selfType = dyn_cast<TensorType>(self.getType());
  otherType = dyn_cast<TensorType>(other.getType());

  auto rhsSubOp =
      rewriter.create<tosa::SubOp>(op->getLoc(), selfType, self, other);
  auto rhsAbsOp =
      rewriter.create<tosa::AbsOp>(op->getLoc(), selfType, rhsSubOp);

  auto lhsAbsOp = rewriter.create<tosa::AbsOp>(op->getLoc(), otherType, other);
  auto mulOp = tosa::createMulOpAndCast(rewriter, op, otherType, rtolConstOp,
                                        lhsAbsOp, /*shift=*/0);
  auto addOp =
      rewriter.create<tosa::AddOp>(op->getLoc(), otherType, atolConstOp, mulOp);

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::GreaterEqualOp>(op, outType, addOp,
                                                    rhsAbsOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenClampOp>::matchAndRewrite(
    AtenClampOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "only tensor types input are currently supported");

  auto outType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));
  auto outElemTy = outType.getElementType();

  int64_t minInt, maxInt;
  double minFloat, maxFloat;
  bool isMinNotNone = false;
  bool isMaxNotNone = false;

  auto isMinInt = matchPattern(op.getMin(), m_TorchConstantInt(&minInt));
  auto isMinFloat = matchPattern(op.getMin(), m_TorchConstantFloat(&minFloat));
  if (isMinInt) {
    minFloat = static_cast<float>(minInt);
    isMinNotNone = true;
  } else if (isMinFloat) {
    minInt = static_cast<int64_t>(minFloat);
    isMinNotNone = true;
  } else {
    if (succeeded(checkNotNone(rewriter, op, op.getMin())))
      return rewriter.notifyMatchFailure(op,
                                         "min attr should be a torch constant");
  }

  auto isMaxInt = matchPattern(op.getMax(), m_TorchConstantInt(&maxInt));
  auto isMaxFloat = matchPattern(op.getMax(), m_TorchConstantFloat(&maxFloat));
  if (isMaxInt) {
    maxFloat = static_cast<float>(maxInt);
    isMaxNotNone = true;
  } else if (isMaxFloat) {
    maxInt = static_cast<int64_t>(maxFloat);
    isMaxNotNone = true;
  } else {
    if (succeeded(checkNotNone(rewriter, op, op.getMax())))
      return rewriter.notifyMatchFailure(op,
                                         "max attr should be a torch constant");
  }

  if (!isa<mlir::FloatType>(outElemTy)) {
    IntegerAttr minIntAttr, maxIntAttr;
    if (outElemTy.isInteger(8)) {
      minIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMinNotNone ? minInt : std::numeric_limits<int8_t>::min());
      maxIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMaxNotNone ? maxInt : std::numeric_limits<int8_t>::max());
    } else if (outElemTy.isInteger(16)) {
      minIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMinNotNone ? minInt : std::numeric_limits<int16_t>::min());
      maxIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMaxNotNone ? maxInt : std::numeric_limits<int16_t>::max());
    } else if (outElemTy.isInteger(32)) {
      minIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMinNotNone ? minInt : std::numeric_limits<int32_t>::min());
      maxIntAttr = rewriter.getIntegerAttr(
          outElemTy,
          isMaxNotNone ? maxInt : std::numeric_limits<int32_t>::max());
    } else if (outElemTy.isInteger(64)) {
      minIntAttr = rewriter.getI64IntegerAttr(
          isMinNotNone ? minInt : std::numeric_limits<int64_t>::min());
      maxIntAttr = rewriter.getI64IntegerAttr(
          isMaxNotNone ? maxInt : std::numeric_limits<int64_t>::max());
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported integer type");
    }

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, outType, adaptor.getSelf(), minIntAttr, maxIntAttr,
        /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
  } else {
    FloatAttr minFloatAttr = rewriter.getF32FloatAttr(
        isMinNotNone ? minFloat : std::numeric_limits<float>::lowest());
    FloatAttr maxFloatAttr = rewriter.getF32FloatAttr(
        isMaxNotNone ? maxFloat : std::numeric_limits<float>::max());

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, outType, adaptor.getSelf(), minFloatAttr, maxFloatAttr,
        /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));
  }

  return success();
}

// Legalization for aten.clamp.Tensor
template <>
LogicalResult ConvertAtenOp<AtenClampTensorOp>::matchAndRewrite(
    AtenClampTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // We are not using tosa.clamp to lower aten.clamp.Tensor, as
  // aten.clamp.Tensor's min and max attributes are tensors that can have size
  // greater than 1, which is not compatible with tosa.clamp.
  //
  // Instead, we use the following formula:
  //    yi = min(max(xi, min_valuei), max_valuei)
  auto self = adaptor.getSelf();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");
  auto selfElemTy = selfType.getElementType();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));

  // Get min tensor. If None, there is no lower bound.
  Value min;
  if (succeeded(checkNotNone(rewriter, op, adaptor.getMin()))) {
    min = adaptor.getMin();
  } else {
    min =
        TypeSwitch<Type, Value>(selfElemTy)
            .Case<mlir::FloatType>([&](auto) {
              return tosa::getConstTensor<float>(
                         rewriter, op, std::numeric_limits<float>::lowest(), {},
                         selfElemTy)
                  .value();
            })
            .Case<mlir::IntegerType>([&](auto intType) {
              switch (intType.getWidth()) {
              case 8:
                return tosa::getConstTensor<int8_t>(
                           rewriter, op, std::numeric_limits<int8_t>::min(), {})
                    .value();
              case 32:
                return tosa::getConstTensor<int32_t>(
                           rewriter, op, std::numeric_limits<int32_t>::min(),
                           {})
                    .value();
              case 64:
                return tosa::getConstTensor<int64_t>(
                           rewriter, op, std::numeric_limits<int64_t>::min(),
                           {})
                    .value();
              }
              llvm_unreachable("Invalid integer width");
            });
  }

  // Get max tensor. If None, there is no upper bound.
  Value max;
  if (succeeded(checkNotNone(rewriter, op, adaptor.getMax()))) {
    max = adaptor.getMax();
  } else {
    max =
        TypeSwitch<Type, Value>(selfElemTy)
            .Case<mlir::FloatType>([&](auto) {
              return tosa::getConstTensor<float>(
                         rewriter, op, std::numeric_limits<float>::max(), {},
                         selfElemTy)
                  .value();
            })
            .Case<mlir::IntegerType>([&](auto intType) {
              switch (intType.getWidth()) {
              case 8:
                return tosa::getConstTensor<int8_t>(
                           rewriter, op, std::numeric_limits<int8_t>::max(), {})
                    .value();
              case 32:
                return tosa::getConstTensor<int32_t>(
                           rewriter, op, std::numeric_limits<int32_t>::max(),
                           {})
                    .value();
              case 64:
                return tosa::getConstTensor<int64_t>(
                           rewriter, op, std::numeric_limits<int64_t>::max(),
                           {})
                    .value();
              }
              llvm_unreachable("Invalid integer width");
            });
  }

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, min).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, max).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();
  min = tosa::tosaCastTensorToType(rewriter, min, resultType).value();
  max = tosa::tosaCastTensorToType(rewriter, max, resultType).value();

  // max(xi, min_valuei)
  // Use default NaN Propagation mode "PROPAGATE" for tosa.maximum
  auto minThresholdCheck = rewriter.create<tosa::MaximumOp>(
      op->getLoc(), resultType, self, min,
      /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));

  // yi = min(max(xi, min_valuei), max_valuei)
  // Use default NaN Propagation mode "PROPAGATE" for tosa.minimum
  auto result = rewriter.create<tosa::MinimumOp>(
      op->getLoc(), resultType, minThresholdCheck, max,
      /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"));

  rewriter.replaceOp(op, result);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenArangeStartStepOp>::matchAndRewrite(
    AtenArangeStartStepOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));

  // At this point all tensors should have value semantics, and hence the
  // `layout` check can be ignored.

  // TODO: Add support for pin_memory features.
  // The pin_memory should be either `False` or `none`.
  bool pinMemory;
  if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: pin_memory must be either None or false");
  }

  // Stores a range value (a start, end, or step value) and whether or not it
  // was initiated with a constant integer, an constant float or neither.
  class ConstRangeValue {
  public:
    explicit ConstRangeValue(double v)
        : vDouble(v), fromDouble(true), vInt(static_cast<int64_t>(v)),
          fromInt(false) {}

    explicit ConstRangeValue(int64_t v)
        : vDouble(static_cast<double>(v)), fromDouble(false), vInt(v),
          fromInt(true) {}

    // Constructor for the case where there is no constant value to use.
    ConstRangeValue()
        : vDouble(0), fromDouble(false), vInt(0), fromInt(false) {}

    static ConstRangeValue fromValue(Value v) {
      int64_t intVal{0};
      double floatVal{0.0};
      if (matchPattern(v, m_TorchConstantFloat(&floatVal))) {
        return ConstRangeValue(floatVal);
      } else if (matchPattern(v, m_TorchConstantInt(&intVal))) {
        return ConstRangeValue(intVal);
      }
      return ConstRangeValue();
    }

    bool hasConstInt() const { return fromInt; }
    bool hasConstDouble() const { return fromDouble; }
    bool hasConst() const { return fromInt || fromDouble; }
    double getDouble() const { return vDouble; }
    int64_t getInt() const { return vInt; }

  private:
    double vDouble;
    bool fromDouble;
    int64_t vInt;
    bool fromInt;
  };

  auto start = ConstRangeValue::fromValue(op.getStart());
  if (!start.hasConst()) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: case where `start` is not a constant int or float");
  }

  auto end = ConstRangeValue::fromValue(op.getEnd());
  if (!end.hasConst()) {
    return rewriter.notifyMatchFailure(
        op,
        "unimplemented: case where value `end` is not a constant int or float");
  }

  auto step = ConstRangeValue::fromValue(op.getStep());
  if (!step.hasConst()) {
    return rewriter.notifyMatchFailure(op,
                                       "unimplemented: case where value `step` "
                                       "is not a constant int or float");
  }

  auto getRange = [](auto start, auto end, auto step) {
    // Initialize a small vector of the same type as start:
    using T = decltype(start);
    SmallVector<T> values;

    uint64_t counter{0};
    if (start == end) {
      return values;
    }
    assert(step != T(0));
    values.reserve(
        1 + static_cast<size_t>(std::abs((end - start) / std::abs(step))));
    if (step > 0) {
      while (start + T(counter) * step < end) {
        values.push_back(start + counter * step);
        counter++;
      }
    } else {
      while (start + T(counter) * step > end) {
        values.push_back(start + counter * step);
        counter++;
      }
    }
    return values;
  };

  const auto isIntType =
      dyn_cast_or_null<mlir::IntegerType>(resultType.getElementType());

  const auto isDoubleType =
      dyn_cast_or_null<mlir::FloatType>(resultType.getElementType());

  auto maybeResult = [&]() -> std::optional<Value> {
    // Integer output type, and start / end / range are all integers.
    if (isIntType && start.hasConstInt() && end.hasConstInt() &&
        step.hasConstInt()) {
      auto values = getRange(start.getInt(), end.getInt(), step.getInt());
      return tosa::getConstTensor<int64_t>(rewriter, op, values, values.size());
    }

    // Get a double range.
    auto values =
        getRange(start.getDouble(), end.getDouble(), step.getDouble());
    if (isIntType) {
      SmallVector<int64_t> values_i64;
      values_i64.reserve(values.size());
      for (auto v : values) {
        values_i64.push_back(static_cast<int64_t>(v));
      }
      return tosa::getConstTensor<int64_t>(rewriter, op, values_i64,
                                           values.size());
    }

    if (!isDoubleType) {
      return {};
    }

    SmallVector<float> values_f32;
    values_f32.reserve(values.size());
    for (auto v : values) {
      values_f32.push_back(static_cast<float>(v));
    }
    auto vs = tosa::getConstTensor<float>(rewriter, op, values_f32,
                                          values_f32.size());
    return vs;
  }();

  if (!maybeResult.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "failed to generate constant tensor for arange");
  }
  auto result = maybeResult.value();
  result = tosa::tosaCastTensorToType(rewriter, result, resultType).value();

  rewriter.replaceOp(op, result);

  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));

  // Only supports integer operand type, because for the floating point operand
  // type result tensor has to be of type `f64` which is not supported in the
  // tosa.
  double doubleValue;
  auto isDouble = matchPattern(op.getA(), m_TorchConstantFloat(&doubleValue));
  int64_t intValue;
  auto isInt = matchPattern(op.getA(), m_TorchConstantInt(&intValue));
  if (!isDouble && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  auto outElemTy = resultType.getElementType();
  if (isa<mlir::IntegerType>(outElemTy)) {
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {intValue}));
  } else if (outElemTy.isF64()) {
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {doubleValue}));
  }

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCopyOp>::matchAndRewrite(
    AtenCopyOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  auto srcType = dyn_cast<TensorType>(adaptor.getSrc().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  if (!srcType || !srcType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  auto resultTy =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  // The non_blocking should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking is expected to be false");
  }

  SmallVector<int64_t> selfShape(makeShapeTorchCompatible(selfType.getShape()));
  SmallVector<int64_t> srcShape(makeShapeTorchCompatible(srcType.getShape()));

  if (llvm::equal(selfShape, srcShape) || selfShape.size() == 0) {
    // If we reach here, then it means the given case is handled by implicit
    // broadcasting done by tosa.
    Value result =
        tosa::tosaCastTensorToType(rewriter, adaptor.getSrc(), resultTy)
            .value();
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, "unimplemented: valsem.aten.copy op not supported for this case.");
}

//  Legalizes the torch.aten.to.dtype op
template <>
LogicalResult ConvertAtenOp<AtenToDtypeOp>::matchAndRewrite(
    AtenToDtypeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // The non_blocking arg should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg is expected to be false");
  }

  // The copy arg should be a constant `False`.
  bool copy;
  if (!matchPattern(op.getCopy(), m_TorchConstantBool(&copy))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg must be a constant");
  } else if (copy) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg is expected to be false");
  }

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

  auto resultTy = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));

  Value result =
      tosa::tosaCastTensorToType(rewriter, adaptor.getSelf(), resultTy).value();

  rewriter.replaceOp(op, result);
  return success();
}

template <typename AtenOpT>
class ConvertAtenRemainderFmodOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA Remainder/Fmod");

    auto outType =
        cast<TensorType>(this->getTypeConverter()->convertType(op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value otherTensor;
    if constexpr (std::is_same<AtenOpT, AtenRemainderScalarOp>()) {
      Value other = op.getOther();
      if (failed(torchScalarToTosaTensor(rewriter, op, other, otherTensor,
                                         outElemTy, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA Remainder/Fmod operation");
    } else {
      otherTensor = adaptor.getOther();
      auto otherTy = cast<RankedTensorType>(otherTensor.getType());

      if (!otherTy)
        return rewriter.notifyMatchFailure(
            op, "Only ranked tensor types supported in TOSA Remainder/Fmod");
    }

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, otherTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    constexpr bool isRemainderOp =
        std::is_same<AtenOpT, AtenRemainderScalarOp>() ||
        std::is_same<AtenOpT, AtenRemainderTensorOp>() ||
        std::is_same<AtenOpT, AtenRemainderIntOp>();

    if (selfTy.getElementType() != outElemTy)
      self = tosa::tosaCastTensorToType(rewriter, self, outType).value();

    Value divTensor;
    if (isRemainderOp) {
      // torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
      if (isa<mlir::FloatType>(outElemTy)) {
        auto otherTensorReciprocal = rewriter.create<tosa::ReciprocalOp>(
            op.getLoc(), otherTensor.getType(), otherTensor);
        divTensor = tosa::createMulOpAndCast(
            rewriter, op, outType, self, otherTensorReciprocal, /*shift=*/0);
        divTensor =
            rewriter.create<tosa::FloorOp>(op.getLoc(), outType, divTensor);
      } else {
        divTensor =
            floorIntDiv(rewriter, op, outType, self, otherTensor).value();
      }
    } else {
      // torch.fmod(a, b) == a - a.div(b, rounding_mode="trunc") * b
      if (isa<mlir::FloatType>(outElemTy)) {
        divTensor = truncFloatDiv(rewriter, op, outType, self, otherTensor);
      } else {
        // TOSA IntDiv requires inputs to be i32
        auto i32Type = RankedTensorType::get(outType.getShape(),
                                             rewriter.getIntegerType(32));
        self = tosa::tosaCastTensorToType(rewriter, self, i32Type).value();
        otherTensor =
            tosa::tosaCastTensorToType(rewriter, otherTensor, i32Type).value();

        auto intDivTensor = rewriter.create<tosa::IntDivOp>(
            op->getLoc(), i32Type, self, otherTensor);

        divTensor =
            tosa::tosaCastTensorToType(rewriter, intDivTensor, outType).value();
      }
    }

    auto mulTensor =
        tosa::createMulOpAndCast(rewriter, op, outType, otherTensor, divTensor,
                                 /*shift=*/0);
    rewriter.replaceOpWithNewOp<tosa::SubOp>(op, outType, self, mulTensor);

    return success();
  }
};

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenPoolingBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Different pooling variants need to process inputs differently, e.g.
  // adaptive pooling generates the kernel size rather than receive it. This
  // function also transposes inputs.
  virtual LogicalResult processInputs(AtenOpT op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      Value &input, DenseI64ArrayAttr &kernel,
                                      DenseI64ArrayAttr &stride,
                                      DenseI64ArrayAttr &pad,
                                      Type &outputTy) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented pooling input parsing function");
  }

  static int64_t getOutputDim(PatternRewriter &rewriter, Value &input,
                              Location loc, int64_t inputRank,
                              ArrayRef<int64_t> inputShape, Type inputElemTy,
                              int64_t dimIndex, int64_t kernelDim,
                              int64_t stride, int64_t &padBefore,
                              int64_t &padAfter, int64_t dilation,
                              bool ceilMode = false) {
    int64_t inputDim = inputShape[dimIndex];
    if (inputDim == kUnknownSize) {
      return kUnknownSize;
    } else {
      // TOSA requires dimSize = inputDim + padBefore + padAfter - kernelDim to
      // be fully divisible by stride. We would have to modify the after pad
      // and/ input in order to achieve that.
      // Note: The dimSize calculation below is the same as TOSA's dimSize
      // calculation when dilation = 1, which is the only dilation value that
      // TOSA supports for MaxPool2d (AvgPool2d doesn't have dilation so the
      // value will be defaulted to 1)
      int64_t dimSize =
          inputDim + padBefore + padAfter - dilation * (kernelDim - 1) - 1;
      int64_t remainderDim = dimSize % stride;

      // When PyTorch uses floor mode for output dim calculation, to achieve the
      // TOSA's divisibility requirement, we will remove the unused after pad
      // and slice the unused input rows/columns.
      if (!ceilMode && (remainderDim != 0)) {
        if (remainderDim > padAfter) {
          SmallVector<int64_t> startSlice(inputRank, 0);
          // In cases where we have to do 2 slice operations (one for height and
          // one for width), we need to use the new sliced shape before doing
          // the second slice, not the original inputShape. Therefore, the shape
          // needs to be retrieved again here.
          SmallVector<int64_t> sizeSlice(
              dyn_cast<TensorType>(input.getType()).getShape());
          sizeSlice[dimIndex] = inputDim - (remainderDim - padAfter);
          input = rewriter.create<tosa::SliceOp>(
              loc, RankedTensorType::get(sizeSlice, inputElemTy), input,
              tosa::getTosaConstShape(rewriter, loc, startSlice),
              tosa::getTosaConstShape(rewriter, loc, sizeSlice));
          dimSize = dimSize - padAfter;
          padAfter = 0;
        } else {
          dimSize = dimSize - padAfter;
          padAfter = padAfter - remainderDim;
          dimSize = dimSize + padAfter;
        }
      }

      int64_t outputDim = dimSize / stride + 1;

      // When PyTorch uses ceil mode for output dim calculation, to achieve the
      // TOSA's divisibility requirement, we will remove the unused after pad
      // or add more after pad in case the remainder is more than the after pad
      if (ceilMode && (remainderDim != 0)) {
        if (remainderDim < padAfter) {
          padAfter = padAfter - remainderDim;
        } else {
          padAfter = padAfter + (stride - remainderDim);
        }

        if (outputDim * stride < inputDim + padBefore)
          outputDim++;
      }
      return outputDim;
    }
  }

  // Apply the transposedDims vector on input to generate a transposed form.
  Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposedDims) const {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposedDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
    return rewriter
        .create<tosa::TransposeOp>(
            op->getLoc(), transposedInputType, input,
            rewriter.getDenseI32ArrayAttr(transposedDims))
        .getResult();
  }

  Value transposePoolingInputToHwc(AtenOpT op,
                                   ConversionPatternRewriter &rewriter,
                                   Value input) const {
    auto inputRank = cast<RankedTensorType>(input.getType()).getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? chwToHwc3DTransposeDims
                                          : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(AtenOpT op,
                                    ConversionPatternRewriter &rewriter,
                                    Value input) const {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? hwcToChw3DTransposeDims
                                          : nhwcToNchw4DTransposeDims);
  }

  void
  unsqueezeInputOutputFor2dPool(RankedTensorType inputTy, Value &input,
                                Type &outputTy, Location loc,
                                ConversionPatternRewriter &rewriter) const {
    // 1d pool AtenOps mapped to TosaOp will already have the data in 4D format,
    // here we can have 3D data only if the AtenOp itself is a 2d pool op with
    // data in HWC format.

    // Unsqueeze input tensor in HWC format to NHWC format to be
    // compatible with tosa::AvgPool2dOp, batch is made explicitly 1.
    SmallVector<int64_t> rank4Shape(inputTy.getShape());
    assert(inputTy.getRank() == 3 &&
           "Expected input to be atleast 3 dimensional.");
    rank4Shape.insert(rank4Shape.begin(), 1);
    input = rewriter.create<tosa::ReshapeOp>(
        loc,
        RankedTensorType::get(makeShapeTorchCompatible(rank4Shape),
                              inputTy.getElementType()),
        input, tosa::getTosaConstShape(rewriter, loc, rank4Shape));

    // Unsqueeze output type
    auto outRankedTy = cast<RankedTensorType>(outputTy);
    assert(outRankedTy.getRank() == 3 &&
           "Expected output rank to be same as input.");
    SmallVector<int64_t> rank4ShapeOut(outRankedTy.getShape());
    rank4ShapeOut.insert(rank4ShapeOut.begin(), 1);
    outputTy = outRankedTy.clone(rank4ShapeOut);
  }

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input;
    DenseI64ArrayAttr kernel, stride, pad;
    Type outputTy;

    // Attempts to read input and kernel parameters, or synthesize them in the
    // case of adaptive pooling. Also performs input CHW->HWC transpose.
    if (failed(processInputs(op, adaptor, rewriter, input, kernel, stride, pad,
                             outputTy)))
      return rewriter.notifyMatchFailure(
          op, "Failed to process inputs for pooling");

    // input has already been verified to be RankedTensorType
    auto inputTy = cast<RankedTensorType>(input.getType());
    if (inputTy.getRank() != 4) {
      unsqueezeInputOutputFor2dPool(inputTy, input, outputTy, op->getLoc(),
                                    rewriter);
    }

    Value pooledOutput;
    static_assert(std::is_same<TosaOpT, tosa::MaxPool2dOp>::value ||
                      std::is_same<TosaOpT, tosa::AvgPool2dOp>::value,
                  "Expected either tosa::MaxPool2dOp or tosa::AvgPool2dOp");
    if constexpr (std::is_same<TosaOpT, tosa::MaxPool2dOp>::value) {
      // Use default NaN Propagation mode "PROPAGATE" for tosa.max_pool2d
      pooledOutput = rewriter
                         .create<TosaOpT>(
                             op->getLoc(), outputTy, input, kernel, stride, pad,
                             /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"))
                         .getResult();
    } else if constexpr (std::is_same<TosaOpT, tosa::AvgPool2dOp>::value) {
      TypeAttr accType;
      if (failed(tosa::getAvgPool2dAccType(rewriter, input, accType)))
        return rewriter.notifyMatchFailure(
            op, "Failed to get accumulator type for pooling");
      pooledOutput = rewriter
                         .create<TosaOpT>(op->getLoc(), outputTy, input, kernel,
                                          stride, pad, accType)
                         .getResult();
    }

    auto transposedOutput =
        ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::transposePoolingOutputToChw(
            op, rewriter, pooledOutput);

    Value result = transposedOutput;
    auto resultTy = cast<TensorType>(result.getType());
    auto expectedResultTy = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (resultTy.getRank() != expectedResultTy.getRank()) {
      auto resultShape = expectedResultTy.getShape();
      auto resultElemTy = expectedResultTy.getElementType();

      result = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeTorchCompatible(resultShape),
                                resultElemTy),
          transposedOutput,
          tosa::getTosaConstShape(rewriter, op->getLoc(),
                                  makeShapeTorchCompatible(resultShape)));
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, expectedResultTy, result);

    return success();
  }
};

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenAdaptivePoolingOp
    : public ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT> {
public:
  using ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::ConvertAtenPoolingBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult processInputs(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto inputXchw = adaptor.getSelf();
    auto inputTy = cast<RankedTensorType>(inputXchw.getType());
    if (!inputTy)
      return rewriter.notifyMatchFailure(
          op, "Adaptive avgpool requires ranked tensor input");

    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    // Rank sanity check.
    if (inputRank != 4 && inputRank != 3)
      return rewriter.notifyMatchFailure(
          op, "NCHW->NHWC transpose requires 3D or 4D tensor");

    int64_t inputHDim = inputShape[inputRank - 2];
    int64_t inputWDim = inputShape[inputRank - 1];

    SmallVector<int64_t> outputSize;
    if (!matchPattern(op.getOutputSize(),
                      m_TorchListOfConstantInts(outputSize)))
      return rewriter.notifyMatchFailure(
          op, "Non-const output_size for adaptive pooling unsupported.");

    SmallVector<int64_t> kernelDims;
    int64_t outputHDim, outputWDim;
    if (outputSize.size() == 1) {
      outputHDim = outputWDim = outputSize[0];
    } else {
      if (outputSize.size() != 2)
        return rewriter.notifyMatchFailure(
            op, "Adaptive avgpool output_size not 1 or 2 elements.");

      // Assumes 'None' (e.g. output_size=(None, 5) ) is expressed as <=0.
      outputHDim =
          (outputSize[0] <= 0) ? inputShape[inputRank - 2] : outputSize[0];
      outputWDim =
          (outputSize[1] <= 0) ? inputShape[inputRank - 1] : outputSize[1];
    }

    // In adaptive pooling,
    // stride = inputDim // outputDim
    // kernel = inputDim - (outputDim-1)* stride
    // pad = 0, dilation = 1

    int64_t strideH = inputShape[inputRank - 2] / outputHDim;
    int64_t strideW = inputShape[inputRank - 1] / outputWDim;

    kernelDims.push_back(inputHDim - (outputHDim - 1) * strideH);
    kernelDims.push_back(inputWDim - (outputWDim - 1) * strideW);

    SmallVector<int64_t> outputShape;
    if (inputRank > 3)
      outputShape.push_back(inputShape[0]);
    outputShape.push_back(outputHDim);
    outputShape.push_back(outputWDim);
    outputShape.push_back(inputShape[inputRank - 3]);

    // Transpose to xHWC
    input =
        ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::transposePoolingInputToHwc(
            op, rewriter, inputXchw);
    kernel = rewriter.getDenseI64ArrayAttr(kernelDims);
    stride = rewriter.getDenseI64ArrayAttr({strideH, strideW});
    // Adaptive pooling does unit dilation and zero pad.
    pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
    outputTy = RankedTensorType::get(makeShapeLLVMCompatible(outputShape),
                                     inputElemTy);

    return success();
  }
};

template <typename AtenOpT, typename tosaOp>
static Type getOutputTypeForNonAdaptivePoolingOp(
    PatternRewriter &rewriter, Operation *op, Value &input,
    RankedTensorType inputTy, SmallVectorImpl<int64_t> &kernelSize,
    SmallVectorImpl<int64_t> &strideArray, SmallVectorImpl<int64_t> &padArray,
    SmallVectorImpl<int64_t> &dilationArray, bool ceilMode = false) {
  auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
  auto inputRank = inputTy.getRank();
  auto inputElemTy = inputTy.getElementType();

  // PyTorch uses xCHW, so Height dim index is rank-2 and Width dim index is
  // rank-1
  int64_t outputHDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      rewriter, input, op->getLoc(), inputRank, inputShape, inputElemTy,
      /*dimIndex=*/inputRank - 2, kernelSize[0], strideArray[0], padArray[0],
      padArray[1], dilationArray[0], ceilMode);
  int64_t outputWDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      rewriter, input, op->getLoc(), inputRank, inputShape, inputElemTy,
      /*dimIndex=*/inputRank - 1, kernelSize[1], strideArray[1], padArray[2],
      padArray[3], dilationArray[1], ceilMode);
  SmallVector<int64_t> outputShape;
  if (inputRank > 3)
    outputShape.push_back(inputShape[0]);
  outputShape.push_back(outputHDim);
  outputShape.push_back(outputWDim);
  outputShape.push_back(inputShape[inputRank - 3]);
  return RankedTensorType::get(makeShapeLLVMCompatible(outputShape),
                               inputElemTy);
}

template <typename AtenOpT>
void expandPoolParams(AtenOpT op, SmallVectorImpl<int64_t> &params,
                      int64_t val) {
  // Expand pooling parameter (kernel, stride) to size 2 to be compatible with
  // tosa::MaxPool2dOp or tosa::AvgPool2dOp
  if constexpr (std::is_same<AtenOpT, AtenMaxPool1dOp>() ||
                std::is_same<AtenOpT, AtenAvgPool1dOp>())
    params.push_back(val);

  if constexpr (std::is_same<AtenOpT, AtenMaxPool2dOp>() ||
                std::is_same<AtenOpT, AtenAvgPool2dOp>()) {
    if (params.size() == 1)
      params.push_back(params[0]);
  }
}

// Checks the validity of pooling parameters and stores them in the respective
// vector. Also, gets the output type for the pooling op.
template <typename AtenOpT, typename tosaOp>
static LogicalResult getOutputTypeAndPoolingParameters(
    AtenOpT op, ConversionPatternRewriter &rewriter, Value &inputXchw,
    SmallVectorImpl<int64_t> &dilationArray, Type &outputTy,
    DenseI64ArrayAttr &kernel, DenseI64ArrayAttr &stride,
    DenseI64ArrayAttr &pad) {

  RankedTensorType inputTy = cast<RankedTensorType>(inputXchw.getType());
  if (!inputTy)
    return rewriter.notifyMatchFailure(
        op, "Pooling op requires ranked tensor input");

  auto inputRank = inputTy.getRank();
  // Rank sanity check.
  if (inputTy.getRank() != 4 && inputRank != 3)
    return rewriter.notifyMatchFailure(
        op, "NCHW->NHWC transpose requires 3D or 4D tensor");

  SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts;
  if (!matchPattern(op.getKernelSize(),
                    m_TorchListOfConstantInts(kernelSizeInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const kernel_size for pooling op unsupported");
  expandPoolParams(op, kernelSizeInts, 1);

  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const stride for pooling op unsupported");
  // If `stride` is not specified by the user, it is assigned the value of empty
  // list during import. For such a case, the stride value is the kernel size.
  // See:
  // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
  if (strideInts.empty()) {
    strideInts.assign(kernelSizeInts);
  } else {
    expandPoolParams(op, strideInts, 1);
  }

  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding factor for pooling op unsupported");
  expandPoolParams(op, paddingInts, 0);

  if constexpr (std::is_same<AtenOpT, AtenAvgPool1dOp>() ||
                std::is_same<AtenOpT, AtenAvgPool2dOp>()) {
    // Currently, we can not represent `count_include_pad` with the existing
    // TOSA AvgPool2d specification. Without the below check, we produce silent
    // wrong answer (SWA) when the `count_include_pad` value is `true.`
    //
    // Note: We need to check for `count_include_pad` only when the `padding`
    // value is non-zero.
    bool countIncludePad;
    if ((paddingInts[0] != 0 || paddingInts[1] != 0) &&
        (!matchPattern(op.getCountIncludePad(),
                       m_TorchConstantBool(&countIncludePad)) ||

         countIncludePad)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported `count_include_pad` value, for tosa AvgPool "
              "`count_include_pad` value should be `False`.");
    }
  }

  SmallVector<int64_t, 4> padArr = {paddingInts[0], paddingInts[0],
                                    paddingInts[1], paddingInts[1]};
  kernel = rewriter.getDenseI64ArrayAttr(kernelSizeInts);
  stride = rewriter.getDenseI64ArrayAttr(strideInts);

  bool ceilMode;
  if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(
        op, "only support constant bool ceil_mode for pooling op");

  expandPoolParams(op, dilationArray, 1);
  outputTy = getOutputTypeForNonAdaptivePoolingOp<AtenOpT, tosaOp>(
      rewriter, op, inputXchw, inputTy, kernelSizeInts, strideInts, padArr,
      dilationArray, ceilMode);
  pad = rewriter.getDenseI64ArrayAttr(
      {padArr[0], padArr[1], padArr[2], padArr[3]});
  return success();
}

class ConvertAtenMaxPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenMaxPool2dOp,
                                 tosa::MaxPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenMaxPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto self = adaptor.getSelf();
    SmallVector<int64_t, 2> dilationArray;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const dilation for pooling op unsupported.");
    // TOSA pooling only supports unit dilation.
    if (dilationArray[0] > 1 || dilationArray[1] > 1)
      return rewriter.notifyMatchFailure(
          op, "Cannot process non-unit pooling dilation.");

    if (failed(getOutputTypeAndPoolingParameters<AtenMaxPool2dOp,
                                                 tosa::MaxPool2dOp>(
            op, rewriter, self, dilationArray, outputTy, kernel, stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, self);

    return success();
  }
};

// Legalization for aten.max_pool1d
class ConvertAtenMaxPool1dOp
    : public ConvertAtenPoolingBaseOp<AtenMaxPool1dOp, tosa::MaxPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenMaxPool1dOp,
                                 tosa::MaxPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenMaxPool1dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto self = adaptor.getSelf();

    // Not a RankedTensorType
    auto selfTy = dyn_cast<RankedTensorType>(self.getType());
    if (!selfTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor type inputs are supported");
    auto selfShape = selfTy.getShape();

    // Expected a rank 3 input tensor
    if (selfTy.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "Input tensor for MaxPool1d should have rank 3");

    // Unsqueeze input tensor to rank 4 to be compatible with tosa::MaxPool2dOp
    SmallVector<int64_t> rank4Shape(selfShape);
    rank4Shape.push_back(1);
    auto reshapedSelf =
        rewriter
            .create<tosa::ReshapeOp>(
                op->getLoc(),
                RankedTensorType::get(makeShapeTorchCompatible(rank4Shape),
                                      selfTy.getElementType()),
                self,
                tosa::getTosaConstShape(rewriter, op->getLoc(), rank4Shape))
            .getResult();

    SmallVector<int64_t> dilationArray;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const dilation for pooling op unsupported.");
    // TOSA pooling only supports unit dilation.
    if (dilationArray[0] > 1)
      return rewriter.notifyMatchFailure(
          op, "Cannot process non-unit pooling dilation.");

    // Expand dilation to size 2 to be compatible with tosa::MaxPool2dOp
    dilationArray.push_back(1);

    if (failed(getOutputTypeAndPoolingParameters<AtenMaxPool1dOp,
                                                 tosa::MaxPool2dOp>(
            op, rewriter, reshapedSelf, dilationArray, outputTy, kernel, stride,
            pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenMaxPool1dOp, tosa::MaxPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, reshapedSelf);

    return success();
  }
};

class ConvertAtenAvgPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, tosa::AvgPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenAvgPool2dOp,
                                 tosa::AvgPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenAvgPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto self = adaptor.getSelf();

    // Currently, we can not represent `divisor_override` with the existing TOSA
    // AvgPool2d specification. Without the below check, we produce silent wrong
    // answers (SWA) when the `divisor_override` value is other than `None.`
    if (!isa<Torch::NoneType>(op.getDivisorOverride().getType())) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported `divisor_override` value, for tosa AvgPool2dOp "
              "`divisor_override` value should be `None`.");
    }

    SmallVector<int64_t, 2> dilationArray{1, 1};
    if (failed(getOutputTypeAndPoolingParameters<AtenAvgPool2dOp,
                                                 tosa::AvgPool2dOp>(
            op, rewriter, self, dilationArray, outputTy, kernel, stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, tosa::AvgPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, self);

    return success();
  }
};

// Legalization for aten.avg_pool1d
class ConvertAtenAvgPool1dOp
    : public ConvertAtenPoolingBaseOp<AtenAvgPool1dOp, tosa::AvgPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenAvgPool1dOp,
                                 tosa::AvgPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenAvgPool1dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto self = adaptor.getSelf();

    // Not a RankedTensorType
    auto selfTy = dyn_cast<RankedTensorType>(self.getType());
    if (!selfTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor type inputs are supported");
    auto selfShape = selfTy.getShape();

    // Expected a rank 3 input tensor
    if (selfTy.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "Input tensor for AvgPool1d should have rank 3");

    // Unsqueeze input tensor to rank 4 to be compatible with tosa::AvgPool2dOp
    SmallVector<int64_t> rank4Shape(selfShape);
    rank4Shape.push_back(1);
    auto reshapedSelf =
        rewriter
            .create<tosa::ReshapeOp>(
                op->getLoc(),
                RankedTensorType::get(makeShapeTorchCompatible(rank4Shape),
                                      selfTy.getElementType()),
                self,
                tosa::getTosaConstShape(rewriter, op->getLoc(), rank4Shape))
            .getResult();

    SmallVector<int64_t, 2> dilationArray{1, 1};
    if (failed(getOutputTypeAndPoolingParameters<AtenAvgPool1dOp,
                                                 tosa::AvgPool2dOp>(
            op, rewriter, reshapedSelf, dilationArray, outputTy, kernel, stride,
            pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenAvgPool1dOp, tosa::AvgPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, reshapedSelf);

    return success();
  }
};

// Ref: Error checking based on the Torch to LinAlg lowering
template <typename AtenOpT, int fillVal>
class ConvertAtenConstPatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    // The layout arg should be either `none` or `0` i.e. strided.
    if (!isa<Torch::NoneType>(op.getLayout().getType())) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
    }

    bool pinMemory;
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(shape))) {
      return rewriter.notifyMatchFailure(
          op, "Shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        tosa::getConstTensor<int32_t>(rewriter, op, values, shape).value();

    auto result =
        tosa::tosaCastTensorToType(rewriter, constOp, outType).value();

    rewriter.replaceOp(op, result);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenFillOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value fillValueTargetTensor;
    if constexpr (std::is_same<AtenOpT, AtenFillTensorOp>()) {
      // Reshape value tensor to have same rank and shape as input
      auto inputRank =
          cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();

      auto fillValue = adaptor.getValue();
      auto fillValueType = dyn_cast<TensorType>(fillValue.getType());
      if (!fillValueType)
        return rewriter.notifyMatchFailure(op, "Fill value is not a tensor");
      auto fillValueElemTy = fillValueType.getElementType();

      SmallVector<int64_t> fillValueMatchedInputRankShape(inputRank, 1);

      auto fillValueMatchedInputRankType = RankedTensorType::get(
          makeShapeTorchCompatible(fillValueMatchedInputRankShape),
          fillValueElemTy);

      auto fillValueMatchedInputRankTensor = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(), fillValueMatchedInputRankType, fillValue,
          tosa::getTosaConstShape(rewriter, op->getLoc(),
                                  fillValueMatchedInputRankShape));

      auto tileOpMultiples =
          tosa::getTosaConstShape(rewriter, op->getLoc(), outType.getShape());

      fillValueTargetTensor = rewriter.create<tosa::TileOp>(
          op->getLoc(),
          RankedTensorType::get(makeShapeTorchCompatible(outType.getShape()),
                                fillValueElemTy),
          fillValueMatchedInputRankTensor.getResult(), tileOpMultiples);
    } else {
      if (failed(torchScalarToTosaTensor(
              rewriter, op, op.getValue(), fillValueTargetTensor, outElemTy,
              makeShapeTorchCompatible(outType.getShape()))))
        return rewriter.notifyMatchFailure(
            op, "Fill value must be a scalar constant");
    }

    auto result =
        tosa::tosaCastTensorToType(rewriter, fillValueTargetTensor, outType)
            .value();

    rewriter.replaceOp(op, result);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenMaskedFillOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    // Not a tensor type.
    auto self = adaptor.getSelf();
    auto selfType = dyn_cast<TensorType>(self.getType());
    if (!selfType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "Only tensor types with static shapes input are currently supported");

    auto maskType = dyn_cast<TensorType>(adaptor.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types mask are currently supported");

    Value rhs = adaptor.getValue();
    auto rhsType = dyn_cast<TensorType>(rhs.getType());
    Value rhsAsTensor;
    if (!rhsType) { // scalar
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getValue(),
                                         rhsAsTensor, rhs.getType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    } else { // tensor
      rhsType = dyn_cast<TensorType>(rhs.getType());
    }

    auto rhsTensor = rhsType ? rhs : rhsAsTensor;
    auto rhsTensorType = dyn_cast<TensorType>(rhsTensor.getType());
    if (rhsTensorType.getElementType() != outElemTy)
      rhsTensor =
          tosa::tosaCastTensorToType(rewriter, rhsTensor, outType).value();

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, rhsTensor)
            .failed())
      return rewriter.notifyMatchFailure(
          op, "Failed to equalize ranks among operands and result");

    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, adaptor.getMask(),
                                                rhsTensor, self);
    return success();
  }
};

// Legalizes the torch.clone op.
template <typename AtenOpT>
class ConvertAtenCloneOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t memoryFormat;
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType()) &&
        (!matchPattern(op.getMemoryFormat(),
                       m_TorchConstantInt(&memoryFormat)) ||
         (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::ChannelsLast))) {
      return op.emitError(
          "unimplemented: only contiguous and channels last memory "
          "format is supported");
    }
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    auto result =
        tosa::tosaCastTensorToType(rewriter, adaptor.getSelf(), outType)
            .value();
    rewriter.replaceOp(op, result);

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenConstantPadNdOp>::matchAndRewrite(
    AtenConstantPadNdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  auto selfElemTy = selfTy.getElementType();
  int64_t rank = selfTy.getRank();

  // START the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp) Pattern match against the op's original
  // operands, because otherwise we will get the lowered version of the operands
  // which is harder to pattern match.
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
  for (uint64_t i = 0; i < padRank; ++i) {
    lowPadding[rank - i - 1] = padInts[i * 2];
    highPadding[rank - i - 1] = padInts[i * 2 + 1];
  }
  // END the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp)

  llvm::SmallVector<int64_t> translatePadsList;

  for (unsigned int i = 0; i < rank; i++) {
    translatePadsList.push_back(lowPadding[i]);
    translatePadsList.push_back(highPadding[i]);
  }

  Value padsList1 = tosa::getTosaConstShape(rewriter, loc, translatePadsList);

  Value padValue = adaptor.getValue();
  Operation *padOp = padValue.getDefiningOp();
  padValue = padOp->getOperand(0);

  Value padTensor;
  if (failed(torchScalarToTosaTensor(rewriter, op.getOperation(), padValue,
                                     padTensor, selfElemTy, {})))
    return rewriter.notifyMatchFailure(
        op, "Pad value needs to be a scalar constant for conversion to "
            "TOSA pad operation");

  padTensor = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(), RankedTensorType::get({1}, selfElemTy), padTensor,
      tosa::getTosaConstShape(rewriter, op->getLoc(), {1}));

  rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(
      op, getTypeConverter()->convertType(op.getType()), self, padsList1,
      padTensor);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCatOp>::matchAndRewrite(
    AtenCatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  const TypeConverter *typeConverter = this->getTypeConverter();
  auto outType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  int64_t rank = outType.getRank();
  int64_t dim;

  if (!outType || !outType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, "Only Tensor types with static shapes are currently supported");
  }

  Location loc = op.getLoc();
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op,
                                       "unimplemented: dim is not constant");
  }
  dim = toPositiveDim(dim, rank);
  if (!isValidDim(dim, rank)) {
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");
  }
  auto tensorList = op.getTensors();
  SmallVector<Value> tensorsTorchType;

  if (!getListConstructElements(tensorList, tensorsTorchType)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: the tensor list is not from list construct");
  }
  auto builtinTensors =
      getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

  for (auto &tensor : builtinTensors)
    tensor = tosa::tosaCastTensorToType(rewriter, tensor, outType).value();

  auto result = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, loc, outType, builtinTensors, rewriter.getI32IntegerAttr(dim));
  rewriter.replaceOp(op, result.getResult());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqrtOp>::matchAndRewrite(
    AtenSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Converts AtenSqrtOp into pow(x, 0.5)
  auto self = adaptor.getSelf();
  auto selfTy = dyn_cast<TensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");

  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  auto elementType = resultType.getElementType();

  if (isa<mlir::IntegerType>(selfTy.getElementType()))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  auto oneHalf =
      tosa::getConstTensor<float>(rewriter, op, 0.5, {}, elementType).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, oneHalf).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  rewriter.replaceOpWithNewOp<tosa::PowOp>(op, resultType, self, oneHalf);
  return success();
}

template <>
LogicalResult
ConvertAtenOp<Aten__InterpolateSizeListScaleListOp>::matchAndRewrite(
    Aten__InterpolateSizeListScaleListOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Converts torch.aten.__interpolate.size_list_scale_list to tosa.resize
  auto input = adaptor.getInput();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy)
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");
  auto inputRank = inputTy.getRank();
  if (inputRank != 4)
    return rewriter.notifyMatchFailure(op,
                                       "TOSA resize() takes rank==4 tensors.");

  auto inputShape = inputTy.getShape();
  auto inputElemTy = inputTy.getElementType();
  // TOSA works in NHWC. Perform the necessary transformations.
  SmallVector<int32_t> nchwToNhwcDims({0, 2, 3, 1});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputTy = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
  auto transposedInput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(), getTypeConverter()->convertType(transposedInputTy),
              input, rewriter.getDenseI32ArrayAttr(nchwToNhwcDims))
          .getResult();

  auto inputHeight = transposedInputShape[1];
  auto inputWidth = transposedInputShape[2];

  int outputHeight, outputWidth;
  if (!isa<Torch::NoneType>(op.getScaleFactor().getType())) {
    SmallVector<double, 2> scaleFactor;
    if (!matchPattern(op.getScaleFactor(),
                      m_TorchListOfConstantFloats(scaleFactor)))
      return rewriter.notifyMatchFailure(
          op, "non-const scale_factor parameter unsupported");

    outputHeight = inputHeight * scaleFactor[0];
    outputWidth = inputWidth * scaleFactor[1];

  } else {
    if (!isa<Torch::NoneType>(op.getSize().getType()))
      return rewriter.notifyMatchFailure(
          op, "Scale factor and size are both absent!");

    SmallVector<int64_t, 4> size;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(size)))
      return rewriter.notifyMatchFailure(
          op, "non-const size parameter unsupported");
    outputHeight = size[0];
    outputWidth = size[1];
  }

  std::string pyMode;
  if (!matchPattern(op.getMode(), m_TorchConstantStr(pyMode)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const mode parameter unsupported");

  // All torch modes listed in
  // https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
  if (pyMode != "bilinear" && pyMode != "nearest")
    return rewriter.notifyMatchFailure(
        op, "Only nearest and bilinear interpolation modes supported");

  std::string mode;
  if (pyMode == "bilinear") {
    mode = "BILINEAR";
  } else {
    mode = "NEAREST_NEIGHBOR";
  }

  bool alignCorners;
  if (!matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCorners)))
    return rewriter.notifyMatchFailure(
        op, "non-const align_corners parameter unsupported");

  bool recomputeScaleFactor;
  if (isa<Torch::NoneType>(op.getRecomputeScaleFactor().getType()))
    recomputeScaleFactor = false;
  else if (!matchPattern(op.getRecomputeScaleFactor(),
                         m_TorchConstantBool(&recomputeScaleFactor)))
    return rewriter.notifyMatchFailure(
        op, "non-const recompute_scale_factor parameter unsupported");
  if (recomputeScaleFactor)
    return rewriter.notifyMatchFailure(
        op, "Application of recompute_scale_factor not yet supported");

  bool antialias;
  if (!matchPattern(op.getAntialias(), m_TorchConstantBool(&antialias)))
    return rewriter.notifyMatchFailure(
        op, "non-const antialias parameter unsupported");
  if (antialias)
    return rewriter.notifyMatchFailure(
        op, "Application of antialias not yet supported");

  SmallVector<int64_t> transposedResizedOpShape(
      {inputShape[0], outputHeight, outputWidth, inputShape[1]});
  auto transposedResizedOpTy = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedResizedOpShape), inputElemTy);

  // Formatting snake_case to match TOSA spec names for readability
  int scale_y_n, scale_y_d, offset_y, border_y;
  int scale_x_n, scale_x_d, offset_x, border_x;

  // Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
  // rather than OH / IH. Similarly for width.
  auto normalize = [&](int input, int output, int &n, int &d, int &offset,
                       int &border) {
    // Dimension is length 1, we are just sampling from one value.
    if (input == 1) {
      n = output;
      d = 1;
      offset = 0;
      border = output - 1;
      return;
    }

    // Apply if aligned and capable to be aligned.
    bool apply_aligned = alignCorners && (output > 1);
    n = apply_aligned ? (output - 1) : output;
    d = apply_aligned ? (input - 1) : input;

    // Simplify the scalers, make sure they are even values.
    int gcd = std::gcd(n, d);
    n = 2 * n / gcd;
    d = 2 * d / gcd;

    offset = 0;

    // If nearest neighbours we need to guarantee we round up.
    if (mode == "NEAREST_NEIGHBOR" && alignCorners) {
      offset += n / 2;
    }

    // TBD: impact of antialias parameter here ?

    // We can compute this directly based on previous values.
    border = d * (output - 1) - n * (input - 1) + offset;
  };

  normalize(inputHeight, outputHeight, scale_y_n, scale_y_d, offset_y,
            border_y);
  normalize(inputWidth, outputWidth, scale_x_n, scale_x_d, offset_x, border_x);

  auto scale = tosa::getTosaConstShape(
      rewriter, op->getLoc(), {scale_y_n, scale_y_d, scale_x_n, scale_x_d});
  auto offset =
      tosa::getTosaConstShape(rewriter, op->getLoc(), {offset_y, offset_x});
  auto border =
      tosa::getTosaConstShape(rewriter, op->getLoc(), {border_y, border_x});
  StringAttr modeAttr = rewriter.getStringAttr(mode);

  auto resizeOpResult =
      rewriter
          .create<tosa::ResizeOp>(op->getLoc(), transposedResizedOpTy,
                                  transposedInput, scale, offset, border,
                                  modeAttr)
          .getResult();

  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));

  SmallVector<int32_t> nhwcToNchwDims({0, 3, 1, 2});
  rewriter
      .replaceOpWithNewOp<tosa::TransposeOp>(
          op, getTypeConverter()->convertType(resultType), resizeOpResult,
          rewriter.getDenseI32ArrayAttr(nhwcToNchwDims))
      .getResult();

  return success();
}

// Template to create supporting tril mask tensor for aten.tril
template <typename T>
Value createTrilMask(PatternRewriter &rewriter, Operation *op,
                     ArrayRef<int64_t> shape, int64_t h, int64_t w,
                     int64_t diagonal) {
  SmallVector<T> vec;

  for (int64_t i = 0; i < h; i++) {
    for (int64_t j = 0; j < w; j++) {
      // Positive diagonal value includes as many diagonals above the main
      // diagonal, while negative diagonal value excludes as many diagonals
      // below the main diagonal.
      if (i >= j - diagonal) {
        vec.push_back(static_cast<T>(1));
      } else {
        vec.push_back(static_cast<T>(0));
      }
    }
  }

  return tosa::getConstTensor<T>(rewriter, op, vec, shape).value();
}

// Legalization for aten.tril
template <>
LogicalResult ConvertAtenOp<AtenTrilOp>::matchAndRewrite(
    AtenTrilOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  // Rank below 2 not accepted
  auto selfRank = selfType.getRank();
  if (selfRank <= 1)
    return rewriter.notifyMatchFailure(
        op, "Rank 0 and 1 are not accepted as they cause underflow");

  if (!selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Currently only static shapes are supported");

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "Result type cannot be empty");

  // Get height, width of input tensor, and diagonal arg to create
  // a const mask tensor to multiply with input.
  // This mask tensor has the same height and width of input tensor
  // and consists of 1's for the lower triangle part and 0's for the rest.
  // For example, with h=4, w=6, diagonal=1:
  // tensor([[1, 1, 0, 0, 0, 0],
  //         [1, 1, 1, 0, 0, 0],
  //         [1, 1, 1, 1, 0, 0],
  //         [1, 1, 1, 1, 1, 0]])
  auto selfShape = selfType.getShape();
  int64_t h = selfShape[selfRank - 2];
  int64_t w = selfShape[selfRank - 1];
  int64_t diagonal;

  if (!matchPattern(op.getDiagonal(), m_TorchConstantInt(&diagonal)))
    return rewriter.notifyMatchFailure(op, "Diagonal value is not an integer");

  // Define shape for mask tensor based on rank
  SmallVector<int64_t> maskShape;
  for (auto i = 0; i < selfRank - 2; i++)
    maskShape.push_back(1);
  maskShape.push_back(h);
  maskShape.push_back(w);

  Value trilMask = TypeSwitch<Type, Value>(resultType.getElementType())
                       .Case<mlir::FloatType>([&](auto) {
                         return createTrilMask<float>(rewriter, op, maskShape,
                                                      h, w, diagonal);
                       })
                       .Case<mlir::IntegerType>([&](auto intType) {
                         switch (intType.getWidth()) {
                         case 1:
                           return createTrilMask<bool>(rewriter, op, maskShape,
                                                       h, w, diagonal);
                         case 32:
                           return createTrilMask<int32_t>(
                               rewriter, op, maskShape, h, w, diagonal);
                         case 64:
                           return createTrilMask<int64_t>(
                               rewriter, op, maskShape, h, w, diagonal);
                         }
                         llvm_unreachable("Invalid integer width");
                       });

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, trilMask)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto result =
      tosa::createMulOpAndCast(rewriter, op, resultType, self, trilMask,
                               /*shift=*/0);
  rewriter.replaceOp(op, result.getResult());

  return success();
}

// Legalization for aten.flip
template <>
LogicalResult ConvertAtenOp<AtenFlipOp>::matchAndRewrite(
    AtenFlipOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();

  auto selfTy = dyn_cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are currently supported");

  SmallVector<int64_t> dims;
  if (!matchPattern(adaptor.getDims(), m_TorchListOfConstantInts(dims)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dims are currently supported");

  auto selfRank = selfTy.getRank();

  auto resultTy = getTypeConverter()->convertType(op.getType());
  Value result = self;

  for (auto &dim : dims) {
    dim = toPositiveDim(dim, selfRank);
    if (!isValidDim(dim, selfRank))
      return rewriter.notifyMatchFailure(op, "Not all dims are valid");

    result = rewriter.create<tosa::ReverseOp>(op->getLoc(), resultTy, result,
                                              static_cast<int32_t>(dim));
  }

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for aten.round:
// Rounds elements of input to the nearest integer.
// Implements "round half to even" to break ties when a number is equidistant
// from two integers.
template <>
LogicalResult ConvertAtenOp<AtenRoundOp>::matchAndRewrite(
    AtenRoundOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // To round to the nearest integer, we will consider the fractional part of
  // the input element (= input element - integer part of element). If the
  // fractional part is smaller than 0.5, round the number down. If the
  // fractional part is 0.5, apply "round half to even" rule. If the fractional
  // part is greater than 0.5, round up.
  //
  // if (frac < 0.5 || (frac == 0.5 && floor(input) % 2 == 0)):
  //   res = floor(input)
  // else:
  //   res = ceil(input)

  auto self = adaptor.getSelf();

  auto selfTy = dyn_cast<TensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(op, "Only tensor types supported");

  auto resultTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  auto boolTy =
      RankedTensorType::get(resultTy.getShape(), rewriter.getIntegerType(1));

  auto resultElemTy = resultTy.getElementType();

  auto oneHalf =
      tosa::getConstTensor<float>(rewriter, op, 0.5, {}, resultElemTy).value();

  auto two =
      tosa::getConstTensor<float>(rewriter, op, 2, {}, resultElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, oneHalf)
          .failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, two).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto floorInput =
      rewriter.create<tosa::FloorOp>(op->getLoc(), resultTy, self);

  // input - floor(input)
  auto fractionalPart = rewriter.create<tosa::SubOp>(
      op->getLoc(), resultTy, self, floorInput.getResult());

  auto ceilInput = rewriter.create<tosa::CeilOp>(op->getLoc(), resultTy, self);

  auto floorInputDivByTwo = tosa::createMulOpAndCast(
      rewriter, op, resultTy, floorInput.getResult(), oneHalf, /*shift=*/0);

  auto floorDivResult = rewriter.create<tosa::FloorOp>(
      op->getLoc(), resultTy, floorInputDivByTwo.getResult());

  // (floor(input) // 2) * 2
  auto evenComparison = tosa::createMulOpAndCast(
      rewriter, op, resultTy, floorDivResult.getResult(), two, /*shift=*/0);

  // floor(input) // 2) * 2 == input <=> floor(input) % 2 == 0
  auto floorInputEven = rewriter.create<tosa::EqualOp>(
      op->getLoc(), boolTy, floorInput.getResult(), evenComparison.getResult());

  auto fracEqualOneHalf = rewriter.create<tosa::EqualOp>(
      op->getLoc(), boolTy, fractionalPart.getResult(), oneHalf);

  auto fracLtOneHalf = rewriter.create<tosa::GreaterOp>(
      op->getLoc(), boolTy, oneHalf, fractionalPart.getResult());

  // (frac == 0.5) && (floor(input) % 2 == 0)
  auto fracEqualOneHalfCond = rewriter.create<tosa::LogicalAndOp>(
      op->getLoc(), boolTy, fracEqualOneHalf.getResult(),
      floorInputEven.getResult());

  // (frac < 0.5) || ((frac == 0.5) && (floor(input) % 2 == 0))
  auto floorResultCond = rewriter.create<tosa::LogicalOrOp>(
      op->getLoc(), boolTy, fracLtOneHalf.getResult(),
      fracEqualOneHalfCond.getResult());

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(
      op, resultTy, floorResultCond.getResult(), floorInput.getResult(),
      ceilInput.getResult());

  return success();
}

// Template to create supporting diagonal mask tensor for aten.diagonal
template <typename T>
Value createDiagonalMask(PatternRewriter &rewriter, Operation *op,
                         ArrayRef<int64_t> shape, int64_t h, int64_t w,
                         int64_t offset) {
  SmallVector<T> vec;

  for (int64_t i = 0; i < h; i++) {
    for (int64_t j = 0; j < w; j++) {
      // Positive offset value moves above the main diagonal, while negative
      // diagonal value moves below the main diagonal.
      if (i + offset == j) {
        vec.push_back(static_cast<T>(1));
      } else {
        vec.push_back(static_cast<T>(0));
      }
    }
  }

  return tosa::getConstTensor<T>(rewriter, op, vec, shape).value();
}

// Legalization for aten.diagonal
template <>
LogicalResult ConvertAtenOp<AtenDiagonalOp>::matchAndRewrite(
    AtenDiagonalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  // Rank below 2 not accepted
  auto selfRank = selfType.getRank();
  if (selfRank <= 1)
    return rewriter.notifyMatchFailure(
        op, "Rank 0 and 1 are not accepted as they cause underflow");

  if (!selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Currently only static shapes are supported");

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "Result type cannot be empty");

  auto selfElemTy = selfType.getElementType();
  auto resultElemTy = resultType.getElementType();

  int64_t offset, dim1, dim2;
  if (!matchPattern(op.getOffset(), m_TorchConstantInt(&offset)))
    offset = 0;

  if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1))) {
    dim1 = 0;
  } else {
    dim1 = toPositiveDim(dim1, selfRank);
  }

  if (!matchPattern(op.getDim2(), m_TorchConstantInt(&dim2))) {
    dim2 = 1;
  } else {
    dim2 = toPositiveDim(dim2, selfRank);
  }

  if (dim1 == dim2)
    return rewriter.notifyMatchFailure(op,
                                       "Values dim1 and dim2 cannot be equal");

  auto selfShape = makeShapeTorchCompatible(selfType.getShape());
  int64_t h = selfShape[dim1];
  int64_t w = selfShape[dim2];

  // Overflowing offset not supported
  if ((offset < 0 && std::abs(offset) >= h) || (offset >= 0 && offset >= w))
    return rewriter.notifyMatchFailure(
        op, "Offset greater or equal than shape not supported");

  int64_t targetDim1 = selfRank - 2;
  int64_t targetDim2 = selfRank - 1;

  Value selfTransposed = self;
  SmallVector<int64_t> transposedInputShape = selfShape;
  RankedTensorType transposedInputType = selfType;

  // If (dim1, dim2) != (rank - 2, rank - 1), transpose the input tensor
  // so that dim1 and dim2 become rank - 2 and rank - 1. We do this so that
  // we can consistently create the diagonal mask tensor.
  if (!(dim1 == targetDim1 && dim2 == targetDim2)) {
    SmallVector<int32_t> transposedDims;
    transposedInputShape.clear();

    for (int32_t i = 0; i < selfRank; ++i) {
      if (i == dim1 || i == dim2)
        continue;
      transposedDims.push_back(i);
    }
    transposedDims.push_back(static_cast<int32_t>(dim1));
    transposedDims.push_back(static_cast<int32_t>(dim2));

    for (auto &dim : transposedDims)
      transposedInputShape.push_back(selfShape[dim]);

    transposedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedInputShape), selfElemTy);

    selfTransposed = rewriter.create<tosa::TransposeOp>(
        op->getLoc(), transposedInputType, self,
        rewriter.getDenseI32ArrayAttr(transposedDims));
  }

  // Define shape for mask tensor based on rank
  SmallVector<int64_t> maskShape;
  for (auto i = 0; i < selfRank - 2; i++)
    maskShape.push_back(1);
  maskShape.push_back(h);
  maskShape.push_back(w);

  Value diagonalMask =
      TypeSwitch<Type, Value>(resultElemTy)
          .Case<mlir::FloatType>([&](auto) {
            return createDiagonalMask<float>(rewriter, op, maskShape, h, w,
                                             offset);
          })
          .Case<mlir::IntegerType>([&](auto intType) {
            switch (intType.getWidth()) {
            case 1:
              return createDiagonalMask<bool>(rewriter, op, maskShape, h, w,
                                              offset);
            case 32:
              return createDiagonalMask<int32_t>(rewriter, op, maskShape, h, w,
                                                 offset);
            case 64:
              return createDiagonalMask<int64_t>(rewriter, op, maskShape, h, w,
                                                 offset);
            }
            llvm_unreachable("Invalid integer width");
          });

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, diagonalMask)
          .failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  Value diagonalTensor = tosa::createMulOpAndCast(
      rewriter, op, transposedInputType, selfTransposed, diagonalMask,
      /*shift=*/0);

  auto resultShape = makeShapeTorchCompatible(resultType.getShape());
  auto targetReduceDim = resultShape[resultType.getRank() - 1];

  // If transposedInputShape[targetDim1] (or h) is greater than the innermost
  // dim of the result, we won't get the correct shape when we reduce sum along
  // the innermost dim to get the result. Therefore, we have to slice the
  // transposed tensor so that transposedInputShape[targetDim1] ==
  // targetReduceDim.
  if (h > targetReduceDim) {
    transposedInputShape[targetDim1] = targetReduceDim;
    transposedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedInputShape), selfElemTy);
    SmallVector<int64_t> startSlice(selfRank, 0);
    SmallVector<int64_t> sizeSlice =
        llvm::to_vector(makeShapeTorchCompatible(transposedInputShape));
    if (offset < 0)
      startSlice[targetDim1] = std::abs(offset);
    diagonalTensor = rewriter.create<tosa::SliceOp>(
        op->getLoc(), transposedInputType, diagonalTensor,
        tosa::getTosaConstShape(rewriter, op->getLoc(), startSlice),
        tosa::getTosaConstShape(rewriter, op->getLoc(), sizeSlice));
  }

  // Apply Reduce Sum to get the result
  auto reduceDimType = RankedTensorType::get({1}, rewriter.getI64Type());
  auto reduceDimAttr =
      DenseIntElementsAttr::get(reduceDimType, llvm::ArrayRef({targetDim2}));
  auto result =
      mlir::tosa::convertReduceSumOp(rewriter, op, resultType, diagonalTensor,
                                     reduceDimAttr, /*keep_dims=*/false);

  rewriter.replaceOp(op, result.value());

  return success();
}

// Legalization for aten.diag_embed
template <>
LogicalResult ConvertAtenOp<AtenDiagEmbedOp>::matchAndRewrite(
    AtenDiagEmbedOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // To perform diag_embed, we will apply scatter with a newly created diagonal
  // index tensor over a constant zero tensor.
  // To make it simpler, we will only scatter using the diagonal with respect
  // to the two innermost dimensions, then permute the output tensor to the
  // correct order of dimensions.
  auto self = adaptor.getSelf();

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  auto selfRank = selfType.getRank();
  int64_t outRank = selfRank + 1;

  auto selfShape = makeShapeTorchCompatible(selfType.getShape());
  int64_t diagSize = selfShape[selfRank - 1];

  if (!selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Currently only static shapes are supported");

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "Result type cannot be empty");

  auto selfElemTy = selfType.getElementType();
  auto resultElemTy = resultType.getElementType();

  int64_t offset{0};
  if (!matchPattern(op.getOffset(), m_TorchConstantInt(&offset)))
    return rewriter.notifyMatchFailure(op,
                                       "Offset value should be a constant int");

  // dim1 default is -2
  int64_t dim1{outRank - 2};
  if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
    return rewriter.notifyMatchFailure(op,
                                       "Dim1 value should be a constant int");
  dim1 = toPositiveDim(dim1, outRank);

  // dim2 default is -1
  int64_t dim2{outRank - 1};
  if (!matchPattern(op.getDim2(), m_TorchConstantInt(&dim2)))
    return rewriter.notifyMatchFailure(op,
                                       "Dim2 value should be a constant int");
  dim2 = toPositiveDim(dim2, outRank);

  if (dim1 == dim2)
    return rewriter.notifyMatchFailure(op, "Dim1 and dim2 cannot be equal");

  // If offset is smaller than 0, we will swap dim1 and dim2 and convert offset
  // to a positive value
  if (offset < 0) {
    std::swap(dim1, dim2);
    offset = std::abs(offset);
  }

  // Create the diagonal index tensor
  int64_t repeat = 1;
  for (int64_t i = 0; i < selfRank - 1; i++)
    repeat *= selfShape[i];

  SmallVector<int32_t> indexVec;
  for (int32_t i = 0; i < repeat; i++) {
    for (int32_t j = offset; j < diagSize + offset; j++)
      indexVec.push_back(j);
  }

  SmallVector<int64_t> indexShape = llvm::to_vector(selfShape);
  indexShape.push_back(1);

  auto index = tosa::getConstTensor<int32_t>(rewriter, op,
                                             /*vec=*/indexVec,
                                             /*shape=*/indexShape)
                   .value();

  // Reshape the input tensor to be the same shape as the new index tensor to
  // act as the src for scattering
  auto scatterSrc = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeTorchCompatible(indexShape), selfElemTy),
      self, tosa::getTosaConstShape(rewriter, op->getLoc(), indexShape));

  // Create a const zero tensor to scatter the input onto
  SmallVector<int64_t> zeroShape;
  for (int64_t i = 0; i < selfRank - 1; i++)
    zeroShape.push_back(selfShape[i]);
  zeroShape.push_back(diagSize + offset);
  zeroShape.push_back(diagSize + offset);

  int64_t numElemOfZeroTensor = 1;
  for (int64_t &d : zeroShape)
    numElemOfZeroTensor *= d;

  Value zero =
      TypeSwitch<Type, Value>(selfElemTy)
          .Case<mlir::FloatType>([&](auto) {
            return tosa::getConstTensor<float>(
                       rewriter, op, SmallVector<float>(numElemOfZeroTensor, 0),
                       zeroShape)
                .value();
          })
          .Case<mlir::IntegerType>([&](auto intType) {
            switch (intType.getWidth()) {
            case 1:
              return tosa::getConstTensor<bool>(
                         rewriter, op,
                         SmallVector<bool>(numElemOfZeroTensor, 0), zeroShape)
                  .value();
            case 32:
              return tosa::getConstTensor<int32_t>(
                         rewriter, op,
                         SmallVector<int32_t>(numElemOfZeroTensor, 0),
                         zeroShape)
                  .value();
            case 64:
              return tosa::getConstTensor<int64_t>(
                         rewriter, op,
                         SmallVector<int64_t>(numElemOfZeroTensor, 0),
                         zeroShape)
                  .value();
            }
            llvm_unreachable("Invalid integer width");
          });

  // Convert PyTorch index and dim to TensorFlow-style indices
  auto indicesTf = tosa::convertTorchIndexToTfIndices(rewriter, op, zero, index,
                                                      outRank - 1);
  if (!indicesTf)
    return rewriter.notifyMatchFailure(
        op, "Convert PyTorch index and dim to TensorFlow indices failed");

  // Perform the TensorFlow ScatterNd algorithm with TensorFlow-style indices as
  // input
  auto diagonalTensor = tosa::convertScatterNdOp(
      rewriter, op,
      RankedTensorType::get(makeShapeTorchCompatible(zeroShape), resultElemTy),
      zero, indicesTf.value(), scatterSrc.getResult());
  if (!diagonalTensor)
    return rewriter.notifyMatchFailure(op, "Convert ScatterNdOp failed");

  // Create the final dims order to permute the scattered tensor
  SmallVector<int32_t> permutedDims(outRank, 0);
  int32_t currentDim = 0;
  int32_t i = 0;

  while (i < outRank) {
    if (i == dim1) {
      permutedDims[i] = outRank - 2;
      i++;
      continue;
    }

    if (i == dim2) {
      permutedDims[i] = outRank - 1;
      i++;
      continue;
    }

    permutedDims[i] = currentDim;
    currentDim++;
    i++;
  }

  auto result = rewriter.create<tosa::TransposeOp>(
      op->getLoc(), resultType, diagonalTensor.value(),
      rewriter.getDenseI32ArrayAttr(permutedDims));

  rewriter.replaceOp(op, result.getResult());

  return success();
}

// Legalization for aten.uniform
// Since TOSA hasn't got a built-in random generator yet, we will use
// std::uniform_real_distribution with the std::default_random_engine from C++
// <random> library
template <>
LogicalResult ConvertAtenOp<AtenUniformOp>::matchAndRewrite(
    AtenUniformOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();

  auto generator = adaptor.getGenerator();
  if (!isa<Torch::NoneType>(generator.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "Custom generators are not supported");

  double fromDouble{0.0}, toDouble{1.0};
  auto isFloat =
      matchPattern(op.getFrom(), m_TorchConstantFloat(&fromDouble)) &&
      matchPattern(op.getTo(), m_TorchConstantFloat(&toDouble));

  int64_t fromInt{0}, toInt{1};
  auto isInt = matchPattern(op.getFrom(), m_TorchConstantInt(&fromInt)) &&
               matchPattern(op.getTo(), m_TorchConstantInt(&toInt));

  if (!isFloat && !isInt)
    return rewriter.notifyMatchFailure(
        op, "From and To values are not constant values");

  int64_t numElem = 1;
  for (int64_t i = 0; i < selfType.getRank(); i++)
    numElem *= selfShape[i];

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));

  std::default_random_engine gen;

  auto from = isFloat ? fromDouble : fromInt;
  auto to = isFloat ? toDouble : toInt;

  std::uniform_real_distribution<float> uniformDist(from, to);
  SmallVector<float> uniformVec;

  for (int64_t i = 0; i < numElem; i++)
    uniformVec.push_back(uniformDist(gen));

  auto result = tosa::getConstTensor<float>(rewriter, op, uniformVec, selfShape,
                                            selfType.getElementType())
                    .value();

  result = tosa::tosaCastTensorToType(rewriter, result, resultType).value();

  rewriter.replaceOp(op, {result});

  return success();
}

// Legalization for aten.threshold_backward
// result = self <= threshold ? 0 : grad
template <>
LogicalResult ConvertAtenOp<AtenThresholdBackwardOp>::matchAndRewrite(
    AtenThresholdBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");
  auto selfElemTy = selfType.getElementType();

  auto selfShape = selfType.getShape();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  Value threshold;
  if (failed(torchScalarToTosaTensor(rewriter, op, op.getThreshold(), threshold,
                                     selfElemTy, selfShape)))
    return rewriter.notifyMatchFailure(op,
                                       "Threshold must be a constant scalar");

  auto grad = adaptor.getGradOutput();

  // Not a tensor type
  auto gradType = dyn_cast<TensorType>(grad.getType());
  if (!gradType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  Value zero =
      TypeSwitch<Type, Value>(resultElemTy)
          .Case<mlir::FloatType>([&](auto) {
            return tosa::getConstTensor<float>(rewriter, op, 0, {},
                                               resultElemTy)
                .value();
          })
          .Case<mlir::IntegerType>([&](auto intType) {
            switch (intType.getWidth()) {
            case 1:
              return tosa::getConstTensor<bool>(rewriter, op, 0, {}).value();
            case 8:
              return tosa::getConstTensor<int8_t>(rewriter, op, 0, {}).value();
            case 32:
              return tosa::getConstTensor<int32_t>(rewriter, op, 0, {}).value();
            case 64:
              return tosa::getConstTensor<int64_t>(rewriter, op, 0, {}).value();
            }
            llvm_unreachable("Invalid integer width");
          });

  // Check: input <= threshold
  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      op->getLoc(), RankedTensorType::get(selfShape, rewriter.getI1Type()),
      threshold, self);

  self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();
  grad = tosa::tosaCastTensorToType(rewriter, grad, resultType).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, zero).failed() ||
      mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, grad).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto result = rewriter.create<tosa::SelectOp>(op->getLoc(), resultType,
                                                cond.getResult(), zero, grad);

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.as_strided
template <>
LogicalResult ConvertAtenOp<AtenAsStridedOp>::matchAndRewrite(
    AtenAsStridedOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // To lower aten.as_strided to TOSA, we will first reshape the input tensor to
  // an 1-D tensor, then calculate the indices of result elements based on the
  // output size, stride and storage offset. With the reshaped 1-D tensor and
  // the indices, we can apply Gather to extract the required elements into a
  // new tensor and then reshape it back to the desired output shape.
  auto self = adaptor.getSelf();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");
  auto selfElemTy = selfType.getElementType();
  auto selfShape = selfType.getShape();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  // Get output size
  SmallVector<int64_t> outputSize;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(outputSize)))
    return rewriter.notifyMatchFailure(
        op, "Only a constant list form of output size is supported");

  // Get stride
  SmallVector<int64_t> stride;
  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(
        op, "Only a constant list form of stride is supported");

  // Get storage offset
  int64_t offset;
  if (!matchPattern(op.getStorageOffset(), m_TorchConstantInt(&offset)))
    offset = 0;

  // Reshape input tensor into an 1-D tensor
  int64_t selfNumElems = std::accumulate(selfShape.begin(), selfShape.end(), 1,
                                         std::multiplies<int64_t>());

  auto self1D = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(), RankedTensorType::get({selfNumElems}, selfElemTy), self,
      tosa::getTosaConstShape(rewriter, op->getLoc(), {selfNumElems}));

  // Calculate the target elements indices
  SmallVector<int32_t> targetIndicesVec;
  int64_t outputRank = outputSize.size();
  int64_t outputNumElems = std::accumulate(outputSize.begin(), outputSize.end(),
                                           1, std::multiplies<int64_t>());

  for (int64_t i = 0; i < outputNumElems; i++) {
    // Index formula:
    // index[i] = coord_i_0 * stride[0] + coord_i_1 * stride[1] + ... +
    //              coord_i_n * stride[n]
    int32_t index = offset;
    int64_t coordFinder = i;
    for (int64_t dim = 0; dim < outputRank; dim++) {
      int64_t indexCoord = coordFinder % outputSize[outputRank - dim - 1];
      index += indexCoord * stride[outputRank - dim - 1];
      coordFinder /= outputSize[outputRank - dim - 1];
    }
    targetIndicesVec.push_back(index);
  }

  auto targetIndices =
      tosa::getConstTensor<int32_t>(rewriter, op, targetIndicesVec,
                                    makeShapeTorchCompatible({outputNumElems}))
          .value();

  // Convert PyTorch-style indices and dim into TensorFlow-style indices
  auto targetIndicesTf = tosa::convertTorchIndexToTfIndices(
      rewriter, op, self1D.getResult(), targetIndices, 0);
  if (!targetIndicesTf)
    return rewriter.notifyMatchFailure(op,
                                       "Convert PyTorch-style indices and dim "
                                       "to TensorFlow-style indices failed");

  // Gather the target elements from 1-D input tensor
  // Apply TensorFlow GatherNdOp with TensorFlow-style indices to retrieve the
  // target elements
  auto gatherOp = tosa::convertGatherNdOp(
      rewriter, op,
      RankedTensorType::get(makeShapeTorchCompatible({outputNumElems}),
                            resultElemTy),
      self1D.getResult(), targetIndicesTf.value());

  if (!gatherOp)
    return rewriter.notifyMatchFailure(op, "Convert GatherNdOp failed");

  auto result = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(), resultType, gatherOp.value(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), outputSize));

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for torch.prims.collapse
template <>
LogicalResult ConvertAtenOp<PrimsCollapseOp>::matchAndRewrite(
    PrimsCollapseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getA();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  int64_t start, end;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(
        op, "Only constant int start value is supported");

  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end)))
    return rewriter.notifyMatchFailure(
        op, "Only constant int end value is supported");

  // Identity case
  if (start == end) {
    rewriter.replaceOp(op, self);
    return success();
  }

  // Technically, I should calculate the output shape based on the input shape,
  // start value, and end value. However, that would just give the same result
  // as me taking the result shape straight from resultType and applying
  // tosa::ReshapeOp to the input. Therefore, I'm opting for the latter approach
  // here, which is more simple and quicker.
  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, resultType, self,
      tosa::getTosaConstShape(rewriter, op->getLoc(),
                              makeShapeTorchCompatible(resultShape)));

  return success();
}

Value reflectionPadAlongAxis(Value input, ArrayRef<int64_t> unpaddedShape,
                             int64_t paddingAxisLeft, int64_t paddingAxisRight,
                             int64_t axis, TensorType resultType, Location loc,
                             ConversionPatternRewriter &rewriter) {

  SmallVector<Value> resultTensors;
  auto resultShape = resultType.getShape();

  auto inputType = dyn_cast<TensorType>(input.getType());
  auto inputRank = inputType.getRank();
  auto inputElemTy = inputType.getElementType();

  assert(inputRank == resultType.getRank());
  int64_t axisOffset = inputRank - axis - 1;

  // Use tosa.slice and tosa.reverse to get the reflection pads based on the
  // padding size
  if (paddingAxisLeft > 0) {
    SmallVector<int64_t> leftStartSlice(inputRank, 0);
    SmallVector<int64_t> leftSizeSlice(unpaddedShape.begin(),
                                       unpaddedShape.end() - axisOffset);
    for (int64_t iDim = axisOffset - 1; iDim >= 0; iDim--) {
      leftSizeSlice.push_back(resultShape[inputRank - iDim - 1]);
    }

    leftStartSlice[axis] = 1;
    leftSizeSlice[axis] = paddingAxisLeft;

    SmallVector<int64_t> leftPadShape(unpaddedShape.begin(),
                                      unpaddedShape.end() - (axisOffset + 1));
    leftPadShape.push_back(paddingAxisLeft);
    for (int64_t iDim = axisOffset - 1; iDim >= 0; iDim--) {
      leftPadShape.push_back(resultShape[inputRank - iDim - 1]);
    }

    auto leftPadType = RankedTensorType::get(leftPadShape, inputElemTy);

    auto leftPadSlice = rewriter.create<tosa::SliceOp>(
        loc, leftPadType, input,
        tosa::getTosaConstShape(rewriter, loc, leftStartSlice),
        tosa::getTosaConstShape(rewriter, loc, leftSizeSlice));

    auto leftPad = rewriter.create<tosa::ReverseOp>(
        loc, leftPadType, leftPadSlice.getResult(), static_cast<int32_t>(axis));

    resultTensors.push_back(leftPad.getResult());
  }

  resultTensors.push_back(input);

  if (paddingAxisRight > 0) {
    SmallVector<int64_t> rightStartSlice(inputRank, 0);
    SmallVector<int64_t> rightSizeSlice(unpaddedShape.begin(),
                                        unpaddedShape.end() - axisOffset);
    for (int64_t iDim = axisOffset - 1; iDim >= 0; iDim--) {
      rightSizeSlice.push_back(resultShape[inputRank - iDim - 1]);
    }

    rightStartSlice[axis] = unpaddedShape[axis] - paddingAxisRight - 1;
    rightSizeSlice[axis] = paddingAxisRight;

    SmallVector<int64_t> rightPadShape(unpaddedShape.begin(),
                                       unpaddedShape.end() - (axisOffset + 1));
    rightPadShape.push_back(paddingAxisRight);
    for (int64_t iDim = axisOffset - 1; iDim >= 0; iDim--) {
      rightPadShape.push_back(resultShape[inputRank - iDim - 1]);
    }

    auto rightPadType = RankedTensorType::get(rightPadShape, inputElemTy);

    auto rightPadSlice = rewriter.create<tosa::SliceOp>(
        loc, rightPadType, input,
        tosa::getTosaConstShape(rewriter, loc, rightStartSlice),
        tosa::getTosaConstShape(rewriter, loc, rightSizeSlice));

    auto rightPad = rewriter.create<tosa::ReverseOp>(
        loc, rightPadType, rightPadSlice.getResult(),
        static_cast<int32_t>(axis));

    resultTensors.push_back(rightPad.getResult());
  }

  return tosa::CreateOpAndInfer<tosa::ConcatOp>(rewriter, loc, resultType,
                                                resultTensors, axis);
}

// Legalization for aten.reflection_pad1d
template <>
LogicalResult ConvertAtenOp<AtenReflectionPad1dOp>::matchAndRewrite(
    AtenReflectionPad1dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();
  auto selfRank = selfType.getRank();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));

  SmallVector<int64_t, 2> paddingList;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingList)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding lists are not supported");

  int64_t paddingLeft = paddingList[0];
  int64_t paddingRight = paddingList[1];

  if (paddingLeft >= selfShape[selfRank - 1] ||
      paddingRight >= selfShape[selfRank - 1])
    return rewriter.notifyMatchFailure(
        op, "Padding should be less than input boundary size");

  // Identity case
  if (paddingLeft == 0 && paddingRight == 0) {
    rewriter.replaceOp(op, self);
    return success();
  }

  auto result =
      reflectionPadAlongAxis(self, selfShape, paddingLeft, paddingRight,
                             selfRank - 1, resultType, op->getLoc(), rewriter);

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for aten.reflection_pad2d
template <>
LogicalResult ConvertAtenOp<AtenReflectionPad2dOp>::matchAndRewrite(
    AtenReflectionPad2dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();
  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  SmallVector<int64_t, 4> paddingList;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingList)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding lists are not supported");

  int64_t paddingLeft = paddingList[0];
  int64_t paddingRight = paddingList[1];
  int64_t paddingTop = paddingList[2];
  int64_t paddingBottom = paddingList[3];

  if (paddingLeft >= selfShape[selfRank - 1] ||
      paddingRight >= selfShape[selfRank - 1] ||
      paddingTop >= selfShape[selfRank - 2] ||
      paddingBottom >= selfShape[selfRank - 2])
    return rewriter.notifyMatchFailure(
        op, "Padding must be less than the corresponding input dimension");

  // Identity case
  if (paddingLeft == 0 && paddingRight == 0 && paddingTop == 0 &&
      paddingBottom == 0) {
    rewriter.replaceOp(op, self);
    return success();
  }

  SmallVector<int64_t> selfSidePaddedShape(selfShape.begin(),
                                           selfShape.end() - 1);
  selfSidePaddedShape.push_back(resultShape.back());

  auto selfSidePadded = reflectionPadAlongAxis(
      self, selfShape, paddingLeft, paddingRight, selfRank - 1,
      RankedTensorType::get(selfSidePaddedShape, selfElemTy), op->getLoc(),
      rewriter);

  auto result = reflectionPadAlongAxis(selfSidePadded, selfShape, paddingTop,
                                       paddingBottom, selfRank - 2, resultType,
                                       op->getLoc(), rewriter);

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for aten.reflection_pad3d
template <>
LogicalResult ConvertAtenOp<AtenReflectionPad3dOp>::matchAndRewrite(
    AtenReflectionPad3dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();
  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  SmallVector<int64_t, 6> paddingList;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingList)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding lists are not supported");

  int64_t paddingLeft = paddingList[0];
  int64_t paddingRight = paddingList[1];
  int64_t paddingTop = paddingList[2];
  int64_t paddingBottom = paddingList[3];
  int64_t paddingFront = paddingList[4];
  int64_t paddingBack = paddingList[5];

  if (paddingLeft >= selfShape[selfRank - 1] ||
      paddingRight >= selfShape[selfRank - 1] ||
      paddingTop >= selfShape[selfRank - 2] ||
      paddingBottom >= selfShape[selfRank - 2] ||
      paddingFront >= selfShape[selfRank - 3] ||
      paddingBack >= selfShape[selfRank - 3])
    return rewriter.notifyMatchFailure(
        op, "Padding must be less than the corresponding input dimension");

  // Identity case
  if (paddingLeft == 0 && paddingRight == 0 && paddingTop == 0 &&
      paddingBottom == 0 && paddingFront == 0 && paddingBack == 0) {
    rewriter.replaceOp(op, self);
    return success();
  }

  SmallVector<int64_t> self1dPaddedShape(selfShape.begin(),
                                         selfShape.end() - 1);
  self1dPaddedShape.push_back(resultShape.back());

  auto self1dPadded = reflectionPadAlongAxis(
      self, selfShape, paddingLeft, paddingRight, selfRank - 1,
      RankedTensorType::get(self1dPaddedShape, selfElemTy), op->getLoc(),
      rewriter);

  SmallVector<int64_t> self2dPaddedShape(selfShape.begin(),
                                         selfShape.end() - 2);
  self2dPaddedShape.push_back(resultShape[resultShape.size() - 2]);
  self2dPaddedShape.push_back(resultShape.back());

  auto self2dPadded = reflectionPadAlongAxis(
      self1dPadded, selfShape, paddingTop, paddingBottom, selfRank - 2,
      RankedTensorType::get(self2dPaddedShape, selfElemTy), op->getLoc(),
      rewriter);

  auto result =
      reflectionPadAlongAxis(self2dPadded, selfShape, paddingFront, paddingBack,
                             selfRank - 3, resultType, op->getLoc(), rewriter);

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for aten.replication_pad2d
template <>
LogicalResult ConvertAtenOp<AtenReplicationPad2dOp>::matchAndRewrite(
    AtenReplicationPad2dOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();
  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  SmallVector<int64_t, 4> paddingList;
  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingList)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding lists are not supported");

  int64_t paddingLeft = paddingList[0];
  int64_t paddingRight = paddingList[1];
  int64_t paddingTop = paddingList[2];
  int64_t paddingBottom = paddingList[3];

  // Identity case
  if (paddingLeft == 0 && paddingRight == 0 && paddingTop == 0 &&
      paddingBottom == 0) {
    rewriter.replaceOp(op, self);
    return success();
  }

  // Use tosa.slice to get the reflection pads based on the padding size
  SmallVector<Value> sideTensors;

  if (paddingLeft > 0) {
    SmallVector<int64_t> leftStartSlice(selfRank, 0);
    SmallVector<int64_t> leftSizeSlice(selfShape);

    leftStartSlice[selfRank - 1] = 0;
    leftSizeSlice[selfRank - 1] = 1;

    SmallVector<int64_t> leftPadSliceShape(selfShape.begin(),
                                           selfShape.end() - 1);
    leftPadSliceShape.push_back(1);

    auto leftPadSliceType =
        RankedTensorType::get(leftPadSliceShape, selfElemTy);

    auto leftPadSlice = rewriter.create<tosa::SliceOp>(
        op->getLoc(), leftPadSliceType, self,
        tosa::getTosaConstShape(rewriter, op->getLoc(), leftStartSlice),
        tosa::getTosaConstShape(rewriter, op->getLoc(), leftSizeSlice));

    for (int64_t i = 0; i < paddingLeft; i++)
      sideTensors.push_back(leftPadSlice.getResult());
  }

  sideTensors.push_back(self);

  if (paddingRight > 0) {
    SmallVector<int64_t> rightStartSlice(selfRank, 0);
    SmallVector<int64_t> rightSizeSlice(selfShape);

    rightStartSlice[selfRank - 1] = selfShape[selfRank - 1] - 1;
    rightSizeSlice[selfRank - 1] = 1;

    SmallVector<int64_t> rightPadSliceShape(selfShape.begin(),
                                            selfShape.end() - 1);
    rightPadSliceShape.push_back(1);

    auto rightPadSliceType =
        RankedTensorType::get(rightPadSliceShape, selfElemTy);

    auto rightPadSlice = rewriter.create<tosa::SliceOp>(
        op->getLoc(), rightPadSliceType, self,
        tosa::getTosaConstShape(rewriter, op->getLoc(), rightStartSlice),
        tosa::getTosaConstShape(rewriter, op->getLoc(), rightSizeSlice));

    for (int64_t i = 0; i < paddingRight; i++)
      sideTensors.push_back(rightPadSlice.getResult());
  }

  SmallVector<int64_t> selfSidePaddedShape(selfShape.begin(),
                                           selfShape.end() - 1);
  selfSidePaddedShape.push_back(resultShape.back());

  auto selfSidePadded = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(selfSidePaddedShape, selfElemTy), sideTensors,
      selfRank - 1);

  SmallVector<Value> resultTensors;

  if (paddingTop > 0) {
    SmallVector<int64_t> topStartSlice(selfRank, 0);
    SmallVector<int64_t> topSizeSlice(selfShape.begin(), selfShape.end() - 1);
    topSizeSlice.push_back(resultShape.back());

    topStartSlice[selfRank - 2] = 0;
    topSizeSlice[selfRank - 2] = 1;

    SmallVector<int64_t> topPadSliceShape(selfShape.begin(),
                                          selfShape.end() - 2);
    topPadSliceShape.push_back(1);
    topPadSliceShape.push_back(resultShape.back());

    auto topPadSliceType = RankedTensorType::get(topPadSliceShape, selfElemTy);

    auto topPadSlice = rewriter.create<tosa::SliceOp>(
        op->getLoc(), topPadSliceType, selfSidePadded,
        tosa::getTosaConstShape(rewriter, op->getLoc(), topStartSlice),
        tosa::getTosaConstShape(rewriter, op->getLoc(), topSizeSlice));

    for (int64_t i = 0; i < paddingTop; i++)
      resultTensors.push_back(topPadSlice.getResult());
  }

  resultTensors.push_back(selfSidePadded.getResult());

  if (paddingBottom > 0) {
    SmallVector<int64_t> bottomStartSlice(selfRank, 0);
    SmallVector<int64_t> bottomSizeSlice(selfShape.begin(),
                                         selfShape.end() - 1);
    bottomSizeSlice.push_back(resultShape.back());

    bottomStartSlice[selfRank - 2] = selfShape[selfRank - 2] - 1;
    bottomSizeSlice[selfRank - 2] = 1;

    SmallVector<int64_t> bottomPadSliceShape(selfShape.begin(),
                                             selfShape.end() - 2);
    bottomPadSliceShape.push_back(1);
    bottomPadSliceShape.push_back(resultShape.back());

    auto bottomPadSliceType =
        RankedTensorType::get(bottomPadSliceShape, selfElemTy);

    auto bottomPadSlice = rewriter.create<tosa::SliceOp>(
        op->getLoc(), bottomPadSliceType, selfSidePadded,
        tosa::getTosaConstShape(rewriter, op->getLoc(), bottomStartSlice),
        tosa::getTosaConstShape(rewriter, op->getLoc(), bottomSizeSlice));

    for (int64_t i = 0; i < paddingBottom; i++)
      resultTensors.push_back(bottomPadSlice.getResult());
  }

  auto result = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, op->getLoc(), resultType, resultTensors, selfRank - 2);

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for torch.prims.split_dim
template <>
LogicalResult ConvertAtenOp<PrimsSplitDimOp>::matchAndRewrite(
    PrimsSplitDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getA();

  // Not a tensor type
  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  int64_t dim, outerLength;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "Only constant int dim value is supported");

  auto selfRank = selfType.getRank();
  dim = toPositiveDim(dim, selfRank);
  if (!isValidDim(dim, selfRank))
    return rewriter.notifyMatchFailure(op, "Dim is invalid");

  if (!matchPattern(op.getOuterLength(), m_TorchConstantInt(&outerLength)))
    return rewriter.notifyMatchFailure(
        op, "Only constant int outer length value is supported");

  // Technically, I should calculate the output shape based on the dim and
  // outer length values. However, that would just give the same result as me
  // taking the result shape straight from resultType and applying
  // tosa::ReshapeOp to the input. Therefore, I'm opting for the latter
  // approach here, which is more simple and quicker.
  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, resultType, self,
      tosa::getTosaConstShape(rewriter, op->getLoc(),
                              makeShapeTorchCompatible(resultShape)));

  return success();
}

// Legalization for aten.outer
template <>
LogicalResult ConvertAtenOp<AtenOuterOp>::matchAndRewrite(
    AtenOuterOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  if (selfType.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "Only rank 1 vectors are supported");

  auto vec2 = adaptor.getVec2();

  auto vec2Type = dyn_cast<TensorType>(vec2.getType());
  if (!vec2Type)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  if (vec2Type.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "Only rank 1 vectors are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultShape = resultType.getShape();

  self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();
  vec2 = tosa::tosaCastTensorToType(rewriter, vec2, resultType).value();

  SmallVector<int64_t, 2> resultShapeIndex1Replaced({resultShape[0], 1});
  SmallVector<int64_t, 2> resultShapeIndex0Replaced({1, resultShape[1]});

  // Reshape and tile self to shape {selfShape[0], resultShape[1]}
  auto selfReshaped = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(resultShapeIndex1Replaced,
                            resultType.getElementType()),
      self,
      tosa::getTosaConstShape(rewriter, op->getLoc(),
                              resultShapeIndex1Replaced));

  auto selfTileOpMultiples = tosa::getTosaConstShape(rewriter, op->getLoc(),
                                                     resultShapeIndex0Replaced);

  auto selfTiled = rewriter.create<tosa::TileOp>(
      op->getLoc(), resultType, selfReshaped.getResult(), selfTileOpMultiples);

  // Reshape and tile vec2 to shape {resultShape[0], vec2Shape[0]}
  auto vec2Reshaped = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(resultShapeIndex0Replaced,
                            resultType.getElementType()),
      vec2,
      tosa::getTosaConstShape(rewriter, op->getLoc(),
                              resultShapeIndex0Replaced));

  auto vec2TileOpMultiples = tosa::getTosaConstShape(rewriter, op->getLoc(),
                                                     resultShapeIndex1Replaced);

  auto vec2Tiled = rewriter.create<tosa::TileOp>(
      op->getLoc(), resultType, vec2Reshaped.getResult(), vec2TileOpMultiples);

  auto result =
      tosa::createMulOpAndCast(rewriter, op, resultType, selfTiled.getResult(),
                               vec2Tiled.getResult(), /*shift=*/0);

  rewriter.replaceOp(op, result);
  return success();
}

// Legalization for aten.upsample_nearest2d
template <typename AtenOpT>
class ConvertUpsampleNearest2dForward : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // aten.upsample_nearest2d lowering process:
    // 1. Reshape input: (N, C, H, W) -> (N, C, H x W)
    // 2. Calculate PyTorch-styled gather op indices based on the following
    // formula (based on Torch to Linalg UpsampleNearest2d lowering formula):
    //    for i in range(N x C):
    //      for heightIndex in range(scaledHeight):
    //        for widthIndex in range(scaledWidth):
    //          indices.append(int(heightIndex // scalesH * selfWidth +
    //                         widthIndex // scalesW))
    // 3. Convert PyTorch-styled indices to TensorFlow-styled indices
    // 4. Apply TensorFlow-styled ConverGatherOpNd to retrieve the output
    // 5. Reshape output to desired output shape
    Value self;
    if constexpr (std::is_same<AtenOpT, AtenUpsampleNearest2dOp>()) {
      self = adaptor.getSelf();
    } else if constexpr (std::is_same<AtenOpT, AtenUpsampleNearest2dVecOp>()) {
      self = adaptor.getInput();
    } else {
      return rewriter.notifyMatchFailure(
          op, "Expected either AtenUpsampleNearest2dOp or "
              "AtenUpsampleNearest2dVecOp");
    }

    auto selfType = dyn_cast<TensorType>(self.getType());
    if (!selfType)
      return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

    auto selfShape = selfType.getShape();
    auto selfRank = selfType.getRank();
    auto selfElemTy = selfType.getElementType();

    auto selfHeight = selfShape[selfRank - 2];
    auto selfWidth = selfShape[selfRank - 1];

    auto resultType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    auto resultShape = resultType.getShape();
    auto resultElemTy = resultType.getElementType();

    // Get op's parameters
    SmallVector<int64_t> outputSize;
    SmallVector<double> scaleFactors;
    double scalesH;
    double scalesW;
    int64_t outputHeight;
    int64_t outputWidth;
    if constexpr (std::is_same<AtenOpT, AtenUpsampleNearest2dOp>()) {
      if (!matchPattern(op.getOutputSize(),
                        m_TorchListOfConstantInts(outputSize)))
        return rewriter.notifyMatchFailure(
            op, "Non-constant output size not supported");

      outputHeight = outputSize[0];
      outputWidth = outputSize[1];

      if (isa<Torch::NoneType>(op.getScalesH().getType())) {
        scalesH =
            static_cast<double>(outputHeight) / static_cast<double>(selfHeight);
      } else {
        if (!matchPattern(op.getScalesH(), m_TorchConstantFloat(&scalesH)))
          return rewriter.notifyMatchFailure(
              op, "Non-constant height scales not supported");

        scalesH = std::ceil(scalesH);
      }

      if (isa<Torch::NoneType>(op.getScalesW().getType())) {
        scalesW =
            static_cast<double>(outputWidth) / static_cast<double>(selfWidth);
      } else {
        if (!matchPattern(op.getScalesW(), m_TorchConstantFloat(&scalesW)))
          return rewriter.notifyMatchFailure(
              op, "Non-constant width scales not supported");

        scalesW = std::ceil(scalesW);
      }
    } else if constexpr (std::is_same<AtenOpT, AtenUpsampleNearest2dVecOp>()) {
      auto isOutputSizeNone =
          isa<Torch::NoneType>(op.getOutputSize().getType());
      auto isScaleFactorsNone =
          isa<Torch::NoneType>(op.getScaleFactors().getType());

      if ((isOutputSizeNone && isScaleFactorsNone) ||
          (!isOutputSizeNone && !isScaleFactorsNone))
        return rewriter.notifyMatchFailure(
            op, "Must specify exactly one of output size and scale factors");

      if (!isOutputSizeNone) {
        if (!matchPattern(op.getOutputSize(),
                          m_TorchListOfConstantInts(outputSize)))
          return rewriter.notifyMatchFailure(
              op, "Non-constant output size not supported");

        outputHeight = outputSize[0];
        outputWidth = outputSize[1];

        // Output size values being provided implies that scale values are not
        // provided
        scalesH =
            static_cast<double>(outputHeight) / static_cast<double>(selfHeight);
        scalesW =
            static_cast<double>(outputWidth) / static_cast<double>(selfWidth);
      } else {
        if (!matchPattern(op.getScaleFactors(),
                          m_TorchListOfConstantFloats(scaleFactors)))
          return rewriter.notifyMatchFailure(
              op, "Non-constant output size not supported");

        scalesH = std::ceil(scaleFactors[0]);
        scalesW = std::ceil(scaleFactors[1]);

        // Scale values being provided implies that output size values are not
        // provided
        outputHeight = static_cast<int64_t>(scalesH * selfHeight);
        outputWidth = static_cast<int64_t>(scalesW * selfWidth);
      }
    }

    // Reshape input
    SmallVector<int64_t> reshapedSelfShape(selfShape.begin(),
                                           selfShape.end() - 2);
    reshapedSelfShape.push_back(selfHeight * selfWidth);

    auto reshapedSelf = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), RankedTensorType::get(reshapedSelfShape, selfElemTy),
        self,
        tosa::getTosaConstShape(rewriter, op->getLoc(), reshapedSelfShape));

    // Calculate PyTorch-styled gather indices
    SmallVector<int32_t> targetIndicesVec;
    int64_t indexRepeat = std::accumulate(
        selfShape.begin(), selfShape.end() - 2, 1, std::multiplies<int64_t>());
    for (int64_t i = 0; i < indexRepeat; i++) {
      for (int64_t heightIndex = 0; heightIndex < outputHeight; heightIndex++) {
        for (int64_t widthIndex = 0; widthIndex < outputWidth; widthIndex++) {
          targetIndicesVec.push_back(static_cast<int32_t>(
              std::floor(heightIndex / scalesH) * selfWidth +
              std::floor(widthIndex / scalesW)));
        }
      }
    }

    SmallVector<int64_t> targetIndicesShape(selfShape.begin(),
                                            selfShape.end() - 2);
    targetIndicesShape.push_back(outputHeight * outputWidth);
    auto targetIndicesTorch =
        tosa::getConstTensor<int32_t>(rewriter, op, targetIndicesVec,
                                      targetIndicesShape)
            .value();

    // Convert PyTorch-styled indices to TensorFlow-styled indices
    auto targetIndicesTF = tosa::convertTorchIndexToTfIndices(
        rewriter, op, reshapedSelf.getResult(), targetIndicesTorch,
        selfRank - 2);
    if (!targetIndicesTF)
      return rewriter.notifyMatchFailure(
          op, "Convert PyTorch-styled indices and dim "
              "to TensorFlow-styled indices failed");
    // Apply TensorFlow GatherNdOp with TensorFlow-style indices to retrieve
    // target elements
    auto gatherOp = tosa::convertGatherNdOp(
        rewriter, op, RankedTensorType::get(targetIndicesShape, resultElemTy),
        reshapedSelf.getResult(), targetIndicesTF.value());
    if (!gatherOp)
      return rewriter.notifyMatchFailure(op, "Convert GatherNdOp failed");

    auto result = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), resultType, gatherOp.value(),
        tosa::getTosaConstShape(rewriter, op->getLoc(), resultShape));

    rewriter.replaceOp(op, {result.getResult()});

    return success();
  }
};

// Legalization for aten.logit
template <>
LogicalResult ConvertAtenOp<AtenLogitOp>::matchAndRewrite(
    AtenLogitOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Logit formula:
  // result = log(zi / (1 - zi))
  // Where: if eps is not None:
  //          zi = input clampled to [eps, 1 - eps]
  //        else:
  //          zi = input
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  if (!isa<mlir::FloatType>(resultElemTy))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype result types are supported");

  // If input is not a float type then cast it to result element type
  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  bool isEpsNone = isa<Torch::NoneType>(op.getEps().getType());

  double eps;
  if (!isEpsNone && !matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
    return rewriter.notifyMatchFailure(op,
                                       "Non-const eps value is not supported");

  auto zi = self;

  // Clamp input to [eps, 1 - eps] when eps is not None
  // Use default NaN Propagation mode "PROPAGATE" for tosa.clamp
  if (!isEpsNone) {
    zi = rewriter
             .create<tosa::ClampOp>(
                 op->getLoc(), resultType, self,
                 rewriter.getF32FloatAttr(static_cast<float>(eps)),
                 rewriter.getF32FloatAttr(static_cast<float>(1 - eps)),
                 /*nan_mode=*/rewriter.getStringAttr("PROPAGATE"))
             .getResult();
  }

  auto one =
      tosa::getConstTensor<float>(rewriter, op, 1.0f, {}, resultElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, one).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto oneMinusZi =
      rewriter.create<tosa::SubOp>(op->getLoc(), resultType, one, zi);

  auto oneMinusZiReciprocal = rewriter.create<tosa::ReciprocalOp>(
      op->getLoc(), resultType, oneMinusZi.getResult());

  auto mulOp = tosa::createMulOpAndCast(rewriter, op, resultType, zi,
                                        oneMinusZiReciprocal.getResult(),
                                        /*shift=*/0);

  auto result =
      rewriter.create<tosa::LogOp>(op->getLoc(), resultType, mulOp.getResult());

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.log1p
template <>
LogicalResult ConvertAtenOp<AtenLog1pOp>::matchAndRewrite(
    AtenLog1pOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // log1p formula:
  // yi = log(xi + 1)
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  if (!isa<mlir::FloatType>(resultElemTy))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype result types are supported");

  // If input is not a float type then cast it to result element type
  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  auto one =
      tosa::getConstTensor<float>(rewriter, op, 1.0f, {}, resultElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, one).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto addOp =
      rewriter.create<tosa::AddOp>(op->getLoc(), resultType, self, one);

  auto result =
      rewriter.create<tosa::LogOp>(op->getLoc(), resultType, addOp.getResult());

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.log10
template <>
LogicalResult ConvertAtenOp<AtenLog10Op>::matchAndRewrite(
    AtenLog10Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // log10 formula (using log base changing formula since TOSA doesn't have a
  // builtin log10 op):
  // yi = log(xi) / log(10)
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  if (!isa<mlir::FloatType>(resultElemTy))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype result types are supported");

  // If input is not a float type then cast it to result element type
  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  auto ten = tosa::getConstTensor<float>(rewriter, op, 10.0f, {}, resultElemTy)
                 .value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, ten).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto logOfSelf = rewriter.create<tosa::LogOp>(op->getLoc(), resultType, self);

  auto constTenType = RankedTensorType::get(
      dyn_cast<TensorType>(ten.getType()).getShape(), resultElemTy);

  auto logOfTen = rewriter.create<tosa::LogOp>(op->getLoc(), constTenType, ten);

  auto reciprocalOp = rewriter.create<tosa::ReciprocalOp>(
      op->getLoc(), constTenType, logOfTen.getResult());

  auto result = tosa::createMulOpAndCast(
      rewriter, op, resultType, logOfSelf.getResult(), reciprocalOp.getResult(),
      /*shift=*/0);

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.expm1
template <>
LogicalResult ConvertAtenOp<AtenExpm1Op>::matchAndRewrite(
    AtenExpm1Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // expm1 formula:
  // yi = exp(x) - 1
  // Note: This lowering might not provide as great precision as aten.expm1
  // since TOSA doesn't have a built-in expm1 op.
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  if (!isa<mlir::FloatType>(resultElemTy))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype result types are supported");

  // If input is not a float type then cast it to result element type
  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  auto one =
      tosa::getConstTensor<float>(rewriter, op, 1.0f, {}, resultElemTy).value();

  if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), self, one).failed())
    return rewriter.notifyMatchFailure(
        op, "Failed to equalize ranks among operands and result");

  auto expOp = rewriter.create<tosa::ExpOp>(op->getLoc(), resultType, self);

  auto result = rewriter.create<tosa::SubOp>(op->getLoc(), resultType,
                                             expOp.getResult(), one);

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.tan
template <>
LogicalResult ConvertAtenOp<AtenTanOp>::matchAndRewrite(
    AtenTanOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // tan = sin / cos
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));

  if (!isa<mlir::FloatType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype result types are supported");

  // Non floating point inputs are not supported in TOSA so we cast the input
  // to result type
  if (!isa<mlir::FloatType>(selfType.getElementType()))
    self = tosa::tosaCastTensorToType(rewriter, self, resultType).value();

  auto sinOp = rewriter.create<tosa::SinOp>(op->getLoc(), resultType, self);

  auto cosOp = rewriter.create<tosa::CosOp>(op->getLoc(), resultType, self);

  auto reciprocalOp =
      rewriter.create<tosa::ReciprocalOp>(op->getLoc(), resultType, cosOp);

  auto result = tosa::createMulOpAndCast(
      rewriter, op, resultType, sinOp.getResult(), reciprocalOp.getResult(),
      /*shift=*/0);

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

// Legalization for aten.unfold
template <>
LogicalResult ConvertAtenOp<AtenUnfoldOp>::matchAndRewrite(
    AtenUnfoldOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Approach: Use GatherOp to retrieve target elements from target dim and then
  // reshape the output into slices according to the output shape
  //
  // Lowering steps:
  // 1. Create PyTorch-style indices tensor corresponding to target elements and
  // reshape them to (d_0, d_1, ..., nWindows * size, ..., d_(rank - 1))
  // with d_x being the dimension size of the input at dim x.
  // The indices vector will be calculated using the following formula:
  //    for i in range(d_0 * d_1 * ... * d_(target_dim - 1)):
  //      for window in range(nWindows):
  //        for elementIndex in range(size):
  //          for j in range(d_(target_dim + 1) * ... * d_(rank-1)):
  //            indices_vec.push_back(elementIndex + window * step)
  // 2. Convert PyTorch-style indices and target dim to TensorFlow-style indices
  // 3. Apply TensorFlow GatherNdOp with TensorFlow-style indices to retrieve
  // target elements
  // 4. Reshape result from above to correct output shape
  auto self = adaptor.getSelf();

  auto selfType = dyn_cast<TensorType>(self.getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfShape = selfType.getShape();
  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();

  auto resultType =
      dyn_cast<TensorType>(typeConverter->convertType(op.getType()));
  auto resultElemTy = resultType.getElementType();

  int64_t dim;
  if (!matchPattern(op.getDimension(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op,
                                       "Only constant int dims are supported");

  int64_t size;
  if (!matchPattern(op.getSize(), m_TorchConstantInt(&size)))
    return rewriter.notifyMatchFailure(op,
                                       "Only constant int sizes are supported");

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
    return rewriter.notifyMatchFailure(op,
                                       "Only constant int steps are supported");

  if (step <= 0)
    return rewriter.notifyMatchFailure(op, "Step value must be greater than 0");

  // Handle rank zero
  if (selfRank == 0) {
    if (dim != 0)
      return rewriter.notifyMatchFailure(
          op, "Unsupported dim value for rank zero input");

    if (size != 1)
      return rewriter.notifyMatchFailure(
          op, "Unsupported size value for rank zero input");

    auto result = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), RankedTensorType::get({1}, selfElemTy), self,
        tosa::getTosaConstShape(rewriter, op->getLoc(), {1}));

    rewriter.replaceOp(op, {result.getResult()});
    return success();
  }

  dim = toPositiveDim(dim, selfRank);
  if (!isValidDim(dim, selfRank))
    return rewriter.notifyMatchFailure(op, "Dim value is invalid");

  // Size of dimension 'dim' in the returned tensor (or number of windows within
  // the dimension that got sliced)
  int64_t nWindows = (selfShape[dim] - size) / step + 1;

  // Find number of times that each base index value gets repeated for target
  // dim based on dim values before and after target dim i.e. preDimAccumulate =
  // d_0 * d_1 * ... * d_(target_dim - 1)
  //      postDimAccumulate = d_(target_dim + 1) * ... * d_(rank - 1)
  int64_t preDimAccumulate =
      std::accumulate(selfShape.begin(), selfShape.begin() + dim, 1,
                      std::multiplies<int64_t>());
  int64_t postDimAccumulate =
      std::accumulate(selfShape.begin() + dim + 1, selfShape.end(), 1,
                      std::multiplies<int64_t>());

  // Calculate PyTorch-style gather indices vector
  // Example: shape = (2, 4, 3), dim = 1, size = 3, step = 1
  //          -> preDimAccumulate = 2, postDimAccummulate = 3, nWindows = 2
  // pyTorchIndicesBaseVec = [0, 0, 0, 1, 1, 1, 2, 2, 2,
  //                          1, 1, 1, 2, 2, 2, 3, 3, 3]
  // pyTorchIndicesVec = [0, 0, 0, 1, 1, 1, 2, 2, 2,
  //                      1, 1, 1, 2, 2, 2, 3, 3, 3,
  //                      0, 0, 0, 1, 1, 1, 2, 2, 2,
  //                      1, 1, 1, 2, 2, 2, 3, 3, 3]
  SmallVector<int32_t> pyTorchIndicesBaseVec;
  SmallVector<int32_t> pyTorchIndicesVec;

  for (int64_t window = 0; window < nWindows; window++) {
    for (int64_t elementIndex = 0; elementIndex < size; elementIndex++) {
      int32_t baseIndex = static_cast<int32_t>(elementIndex + window * step);
      for (int64_t i = 0; i < postDimAccumulate; i++)
        pyTorchIndicesBaseVec.push_back(baseIndex);
    }
  }

  for (int64_t i = 0; i < preDimAccumulate; i++)
    pyTorchIndicesVec.insert(pyTorchIndicesVec.end(),
                             pyTorchIndicesBaseVec.begin(),
                             pyTorchIndicesBaseVec.end());

  // Create the PyTorch-style indices tensor
  // Continuing with the previous example:
  // pyTorchIndicesShape = (2, nWindows * size, 3) = (2, 6, 3)
  // pyTorchIndices = tensor([[[0, 0, 0],
  //                           [1, 1, 1],
  //                           [2, 2, 2],
  //                           [1, 1, 1],
  //                           [2, 2, 2],
  //                           [3, 3, 3]],
  //                          [[0, 0, 0],
  //                           [1, 1, 1],
  //                           [2, 2, 2],
  //                           [1, 1, 1],
  //                           [2, 2, 2],
  //                           [3, 3, 3]]])
  SmallVector<int64_t> pyTorchIndicesShape(selfShape);
  pyTorchIndicesShape[dim] = nWindows * size;
  auto pyTorchIndices =
      tosa::getConstTensor<int32_t>(rewriter, op, pyTorchIndicesVec,
                                    pyTorchIndicesShape)
          .value();

  // Convert PyTorch-style indices to TensorFlow-style indices
  auto tfIndices = tosa::convertTorchIndexToTfIndices(rewriter, op, self,
                                                      pyTorchIndices, dim);
  if (!tfIndices)
    return rewriter.notifyMatchFailure(op,
                                       "Convert PyTorch-style indices and dim "
                                       "to TensorFlow-style indices failed");

  // Apply TensorFlow GatherNdOp with TensorFlow-style indices to retrieve
  // target elements
  auto gatherNdOp = tosa::convertGatherNdOp(
      rewriter, op, RankedTensorType::get(pyTorchIndicesShape, resultElemTy),
      self, tfIndices.value());
  if (!gatherNdOp)
    return rewriter.notifyMatchFailure(op, "Convert GatherNdOp failed");

  // Reshape to an intermediary shape where the gathered elements in dimension
  // 'dim' are split back into 2 dimensions of sizes 'nWindows' and 'size'
  SmallVector<int64_t> intermediaryShape;
  for (int64_t currentDim = 0; currentDim < selfRank; currentDim++) {
    if (currentDim == dim) {
      intermediaryShape.push_back(nWindows);
      intermediaryShape.push_back(size);
    } else {
      intermediaryShape.push_back(pyTorchIndicesShape[currentDim]);
    }
  }

  auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(), RankedTensorType::get(intermediaryShape, resultElemTy),
      gatherNdOp.value(),
      tosa::getTosaConstShape(rewriter, op->getLoc(), intermediaryShape));

  // Permute dims to the correct result order
  SmallVector<int32_t> permutedDims;
  for (int64_t currentDim = 0; currentDim < selfRank + 1; currentDim++) {
    if (currentDim != dim + 1)
      permutedDims.push_back(static_cast<int32_t>(currentDim));
  }
  permutedDims.push_back(static_cast<int32_t>(dim + 1));

  auto result = rewriter.create<tosa::TransposeOp>(
      op->getLoc(), resultType, reshapeOp.getResult(),
      rewriter.getDenseI32ArrayAttr(permutedDims));

  rewriter.replaceOp(op, {result.getResult()});

  return success();
}

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosa Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosa : public ConvertTorchToTosaBase<ConvertTorchToTosa> {
public:
  ConvertTorchToTosa() = default;
  ConvertTorchToTosa(bool requireFullTosaConversion) {
    this->requireFullTosaConversion = requireFullTosaConversion;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    if (this->requireFullTosaConversion) {
      target.addIllegalDialect<Torch::TorchDialect>();
    } else {
      target.addLegalDialect<Torch::TorchDialect>();
    }

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    populateTorchToTosaConversionLegalOps(target);

    RewritePatternSet patterns(context);

    auto illegalOps = populateTorchToTosaConversionPatternsAndIllegalOps(
        typeConverter, patterns);

    for (auto op : illegalOps) {
      target.addIllegalOp(OperationName(op, context));
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void torch::populateTorchToTosaConversionLegalOps(ConversionTarget &target) {
  // The following ops are never the primary reason why lowering fails.
  // The backend contract only allows functions to return tensors thus there
  // is always another op using them.
  // When we have a chain of torch.constant.int followed by a unsupported
  // torch op, we want the pass to mention the unsupported torch op
  // in the error message.
  target.addLegalOp<ConstantNoneOp>();
  target.addLegalOp<ConstantBoolOp>();
  target.addLegalOp<ConstantIntOp>();
  target.addLegalOp<ConstantFloatOp>();
  target.addLegalOp<ConstantStrOp>();
  target.addLegalOp<ConstantDeviceOp>();
  target.addLegalOp<PrimListConstructOp>();
  target.addLegalOp<PrimTupleConstructOp>();
}

std::set<StringRef> torch::populateTorchToTosaConversionPatternsAndIllegalOps(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {

  MLIRContext *context = patterns.getContext();
  std::set<StringRef> illegalOps;

#define INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenOp, TosaOp)                     \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenUnaryPromoteToFPOp<AtenOp, TosaOp>>(typeConverter,   \
                                                              context);
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenLogOp, tosa::LogOp)
  INSERT_UNARY_PROMOTE_TO_FP_PATTERN(AtenExpOp, tosa::ExpOp)
#undef INSERT_UNARY_PROMOTE_TO_FP_PATTERN

#define INSERT_UNARY_PATTERN(AtenOp, TosaOp)                                   \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenUnaryOp<AtenOp, TosaOp>>(typeConverter, context);
  INSERT_UNARY_PATTERN(AtenNegOp, tosa::NegateOp)
  INSERT_UNARY_PATTERN(AtenFloorOp, tosa::FloorOp)
  INSERT_UNARY_PATTERN(AtenRsqrtOp, tosa::RsqrtOp)
  INSERT_UNARY_PATTERN(AtenBitwiseNotOp, tosa::BitwiseNotOp)
  INSERT_UNARY_PATTERN(AtenCeilOp, tosa::CeilOp)
  INSERT_UNARY_PATTERN(AtenReciprocalOp, tosa::ReciprocalOp)
  INSERT_UNARY_PATTERN(AtenCosOp, tosa::CosOp)
  INSERT_UNARY_PATTERN(AtenSinOp, tosa::SinOp)
  INSERT_UNARY_PATTERN(AtenLogicalNotOp, tosa::LogicalNotOp)
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_PATTERN(AtenOp, TosaOp)                                  \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenBinaryOp<AtenOp, TosaOp>>(typeConverter, context);
  INSERT_BINARY_PATTERN(AtenMaximumOp, tosa::MaximumOp)
  INSERT_BINARY_PATTERN(AtenMinimumOp, tosa::MinimumOp)
  INSERT_BINARY_PATTERN(AtenLogicalOrOp, tosa::LogicalOrOp)
  INSERT_BINARY_PATTERN(AtenLogicalXorOp, tosa::LogicalXorOp)
  INSERT_BINARY_PATTERN(AtenLogicalAndOp, tosa::LogicalAndOp)
  INSERT_BINARY_PATTERN(AtenBitwiseLeftShiftTensorOp, tosa::LogicalLeftShiftOp)
  INSERT_BINARY_PATTERN(AtenBitwiseRightShiftTensorOp,
                        tosa::ArithmeticRightShiftOp)
#undef INSERT_BINARY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, TosaOp)                           \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, TosaOp>>(typeConverter, context);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, tosa::AddOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, tosa::AddOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, tosa::SubOp)
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, tosa::SubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp, TosaOp)                          \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenCompareOp<AtenOp, TosaOp>>(typeConverter, context);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp, tosa::GreaterOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenGeScalarOp, tosa::GreaterEqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenGeTensorOp, tosa::GreaterEqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp, tosa::GreaterOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp, tosa::GreaterOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp, tosa::GreaterOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLeTensorOp, tosa::GreaterEqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenLeScalarOp, tosa::GreaterEqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp, tosa::EqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp, tosa::EqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp, tosa::EqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp, tosa::EqualOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseAndTensorOp, tosa::BitwiseAndOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseAndScalarOp, tosa::BitwiseAndOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseOrTensorOp, tosa::BitwiseOrOp)
  INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseXorTensorOp, tosa::BitwiseXorOp)
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_MUL_PATTERN(AtenOp)                                      \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMulOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_MUL_PATTERN(AtenMulTensorOp);
  INSERT_BINARY_MUL_PATTERN(AtenMulScalarOp);
#undef INSERT_BINARY_MUL_PATTERN

#define INSERT_BINARY_DIV_PATTERN(AtenOp)                                      \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenDivOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp);
  INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp);
  INSERT_BINARY_DIV_PATTERN(AtenDivTensorModeOp);
  INSERT_BINARY_DIV_PATTERN(AtenDivScalarModeOp);
#undef INSERT_BINARY_DIV_PATTERN

#define INSERT_REMAINDER_FMOD_OP_PATTERN(AtenOp)                               \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenRemainderFmodOp<AtenOp>>(typeConverter, context);
  INSERT_REMAINDER_FMOD_OP_PATTERN(AtenRemainderScalarOp);
  INSERT_REMAINDER_FMOD_OP_PATTERN(AtenRemainderTensorOp);
  INSERT_REMAINDER_FMOD_OP_PATTERN(AtenFmodScalarOp);
  INSERT_REMAINDER_FMOD_OP_PATTERN(AtenFmodTensorOp);
#undef INSERT_REMAINDER_FMOD_OP_PATTERN

#define INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)              \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMultipleDimsReductionOp<AtenOp, ConversionFunc>>(    \
      typeConverter, context);
  INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenMeanDimOp,
                                    mlir::tosa::convertReduceMeanOp)
  INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenSumDimIntListOp,
                                    mlir::tosa::convertReduceSumOp)
  INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenLinalgVectorNormOp,
                                    mlir::tosa::convertLinalgVectorNormOp)
#undef INSERT_NDIMS_REDUCTION_OP_PATTERN

#define INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)             \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenOneDimReductionOp<AtenOp, ConversionFunc>>(          \
      typeConverter, context);
  INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenAnyDimOp,
                                     mlir::tosa::convertReduceAnyOp)
  INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenAllDimOp,
                                     mlir::tosa::convertReduceAllOp)
  INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenProdDimIntOp,
                                     mlir::tosa::convertReduceProdOp)
#undef INSERT_ONEDIM_REDUCTION_OP_PATTERN

#define INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)            \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenAllDimsReductionOp<AtenOp, ConversionFunc>>(         \
      typeConverter, context);
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAllOp, mlir::tosa::convertReduceAllOp)
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAnyOp, mlir::tosa::convertReduceAnyOp)
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenSumOp, mlir::tosa::convertReduceSumOp)
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenMaxOp, mlir::tosa::convertReduceMaxOp)
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenMinOp, mlir::tosa::convertReduceMinOp)
  INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenProdOp,
                                      mlir::tosa::convertReduceProdOp)
#undef INSERT_ALLDIMS_REDUCTION_OP_PATTERN

#define INSERT_INDICES_REDUCTION_OP_PATTERN(AtenOp, TosaOp)                    \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMinMaxDimOp<AtenOp, TosaOp>>(typeConverter, context);
  INSERT_INDICES_REDUCTION_OP_PATTERN(AtenMaxDimOp, tosa::ReduceMaxOp);
  INSERT_INDICES_REDUCTION_OP_PATTERN(AtenMinDimOp, tosa::ReduceMinOp);
#undef INSERT_INDICES_REDUCTION_OP_PATTERN

#define INSERT_SQUEEZE_OP_PATTERN(AtenOp, TemplateForm)                        \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<TemplateForm<AtenOp>>(typeConverter, context);
  INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeOp, ConvertAtenSqueezeAllDimsOp)
  INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeDimOp, ConvertAtenSqueezeOneDimOp)
#undef INSERT_SQUEEZE_OP_PATTERN

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp)                                   \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
  INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATENOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp)                                       \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
  INSERT_MM_ATENOP_PATTERN(AtenMmOp);
  INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATENOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp)                                   \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
  INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATENOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp)                                   \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
  INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATEMOP_PATTERN

#define INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenOp, TosaOpT)                \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenAdaptivePoolingOp<AtenOp, TosaOpT>>(typeConverter,   \
                                                              context);
  INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenAdaptiveAvgPool2dOp,
                                         tosa::AvgPool2dOp);
#undef INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN

  illegalOps.insert(AtenMaxPool2dOp::getOperationName());
  patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);

  illegalOps.insert(AtenMaxPool1dOp::getOperationName());
  patterns.add<ConvertAtenMaxPool1dOp>(typeConverter, context);

  illegalOps.insert(AtenAvgPool2dOp::getOperationName());
  patterns.add<ConvertAtenAvgPool2dOp>(typeConverter, context);

  illegalOps.insert(AtenAvgPool1dOp::getOperationName());
  patterns.add<ConvertAtenAvgPool1dOp>(typeConverter, context);

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
  INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
  INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
  INSERT_CONSTANT_FILL_PATTERN(AtenEmptyMemoryFormatOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_FILL_PATTERN(AtenOp)                                            \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenFillOp<AtenOp>>(typeConverter, context);
  INSERT_FILL_PATTERN(AtenFill_ScalarOp);
  INSERT_FILL_PATTERN(AtenFillScalarOp);
  INSERT_FILL_PATTERN(AtenFillTensorOp);
#undef INSERT_FILL_PATTERN

#define INSERT_MASKED_FILL_PATTERN(AtenOp)                                     \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenMaskedFillOp<AtenOp>>(typeConverter, context);
  INSERT_MASKED_FILL_PATTERN(AtenMaskedFillScalarOp);
  INSERT_MASKED_FILL_PATTERN(AtenMaskedFillTensorOp);
#undef INSERT_MASKED_FILL_PATTERN

#define INSERT_POW_OP_PATTERN(AtenOp)                                          \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenPowOp<AtenOp>>(typeConverter, context);
  INSERT_POW_OP_PATTERN(AtenPowTensorScalarOp);
  INSERT_POW_OP_PATTERN(AtenPowTensorTensorOp);
  INSERT_POW_OP_PATTERN(AtenPowScalarOp);
#undef INSERT_POW_OP_PATTERN

#define INSERT_UPSAMPLE_NEAREST_2D_FORWARD_OP_PATTERN(AtenOp)                  \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertUpsampleNearest2dForward<AtenOp>>(typeConverter, context);
  INSERT_UPSAMPLE_NEAREST_2D_FORWARD_OP_PATTERN(AtenUpsampleNearest2dOp);
  INSERT_UPSAMPLE_NEAREST_2D_FORWARD_OP_PATTERN(AtenUpsampleNearest2dVecOp);
#undef INSERT_UPSAMPLE_NEAREST_2D_FORWARD_OP_PATTERN

#define INSERT_ACTIVATION_FUNCTION_OP_PATTERN(AtenOp, TosaOp)                  \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenActivationFunctionOp<AtenOp, TosaOp>>(typeConverter, \
                                                                context);
  INSERT_ACTIVATION_FUNCTION_OP_PATTERN(AtenTanhOp, tosa::TanhOp);
  INSERT_ACTIVATION_FUNCTION_OP_PATTERN(AtenSigmoidOp, tosa::SigmoidOp);
  INSERT_ACTIVATION_FUNCTION_OP_PATTERN(AtenErfOp, tosa::ErfOp);
#undef INSERT_ACTIVATION_FUNCTION_OP_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
  INSERT_ATENOP_PATTERN(AtenHardtanhBackwardOp);
  INSERT_ATENOP_PATTERN(AtenReluOp);
  INSERT_ATENOP_PATTERN(AtenLeakyReluOp);
  INSERT_ATENOP_PATTERN(AtenArgmaxOp);
  INSERT_ATENOP_PATTERN(AtenRsubScalarOp);
  INSERT_ATENOP_PATTERN(AtenConvolutionOp);
  INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATENOP_PATTERN(AtenReshapeOp);
  INSERT_ATENOP_PATTERN(AtenBatchNormOp);
  INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
  INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
  INSERT_ATENOP_PATTERN(AtenUnflattenIntOp);
  INSERT_ATENOP_PATTERN(AtenPermuteOp);
  INSERT_ATENOP_PATTERN(AtenLog2Op);
  INSERT_ATENOP_PATTERN(AtenThresholdOp);
  INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
  INSERT_ATENOP_PATTERN(AtenContiguousOp);
  INSERT_ATENOP_PATTERN(AtenDropoutOp);
  INSERT_ATENOP_PATTERN(AtenViewOp);
  INSERT_ATENOP_PATTERN(AtenGeluOp);
  INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
  INSERT_ATENOP_PATTERN(AtenEmbeddingOp);
  INSERT_ATENOP_PATTERN(AtenTransposeIntOp);
  INSERT_ATENOP_PATTERN(AtenSliceTensorOp);
  INSERT_ATENOP_PATTERN(AtenBroadcastToOp);
  INSERT_ATENOP_PATTERN(AtenGatherOp);
  INSERT_ATENOP_PATTERN(AtenIndexPutHackedTwinOp);
  INSERT_ATENOP_PATTERN(AtenIndexTensorHackedTwinOp);
  INSERT_ATENOP_PATTERN(AtenAbsOp);
  INSERT_ATENOP_PATTERN(AtenWhereSelfOp);
  INSERT_ATENOP_PATTERN(AtenClampOp);
  INSERT_ATENOP_PATTERN(AtenArangeStartStepOp);
  INSERT_ATENOP_PATTERN(PrimNumToTensorScalarOp);
  INSERT_ATENOP_PATTERN(AtenCopyOp);
  INSERT_ATENOP_PATTERN(AtenToDtypeOp);
  INSERT_ATENOP_PATTERN(AtenConstantPadNdOp);
  INSERT_ATENOP_PATTERN(AtenCatOp);
  INSERT_ATENOP_PATTERN(AtenSqrtOp);
  INSERT_ATENOP_PATTERN(AtenIscloseOp);
  INSERT_ATENOP_PATTERN(Aten__InterpolateSizeListScaleListOp);
  INSERT_ATENOP_PATTERN(AtenTrilOp);
  INSERT_ATENOP_PATTERN(AtenDiagonalOp);
  INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
  INSERT_ATENOP_PATTERN(AtenFlipOp);
  INSERT_ATENOP_PATTERN(AtenRoundOp);
  INSERT_ATENOP_PATTERN(AtenScatterSrcOp);
  INSERT_ATENOP_PATTERN(AtenSliceScatterOp);
  INSERT_ATENOP_PATTERN(AtenDiagEmbedOp);
  INSERT_ATENOP_PATTERN(AtenUniformOp);
  INSERT_ATENOP_PATTERN(AtenThresholdBackwardOp);
  INSERT_ATENOP_PATTERN(AtenAsStridedOp);
  INSERT_ATENOP_PATTERN(AtenClampTensorOp);
  INSERT_ATENOP_PATTERN(PrimsCollapseOp);
  INSERT_ATENOP_PATTERN(AtenReflectionPad1dOp);
  INSERT_ATENOP_PATTERN(AtenReflectionPad2dOp);
  INSERT_ATENOP_PATTERN(AtenReflectionPad3dOp);
  INSERT_ATENOP_PATTERN(AtenReplicationPad2dOp);
  INSERT_ATENOP_PATTERN(PrimsSplitDimOp);
  INSERT_ATENOP_PATTERN(AtenOuterOp);
  INSERT_ATENOP_PATTERN(AtenLogitOp);
  INSERT_ATENOP_PATTERN(AtenLog1pOp);
  INSERT_ATENOP_PATTERN(AtenLog10Op);
  INSERT_ATENOP_PATTERN(AtenExpm1Op);
  INSERT_ATENOP_PATTERN(AtenTanOp);
  INSERT_ATENOP_PATTERN(AtenUnfoldOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_CLONE_ATENOP_PATTERN(AtenOp)                                    \
  illegalOps.insert(AtenOp::getOperationName());                               \
  patterns.add<ConvertAtenCloneOp<AtenOp>>(typeConverter, context);
  INSERT_CLONE_ATENOP_PATTERN(AtenCloneOp);
#undef INSERT_CLONE_ATENOP_PATTERN

  return illegalOps;
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaPass() {
  return std::make_unique<ConvertTorchToTosa>(true);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaPass(bool requireFullTosaConversion) {
  return std::make_unique<ConvertTorchToTosa>(requireFullTosaConversion);
}
