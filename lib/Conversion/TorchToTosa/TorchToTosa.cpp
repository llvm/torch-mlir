//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

namespace {

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only Tensor types supported in TOSA");

    if (selfTy.getElementType().isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<TosaOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          self);
      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
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
    rewriter.replaceOpWithNewOp<TosaOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self());
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
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().cast<TensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("Add: input datatypes mismatched");

    rewriter.replaceOpWithNewOp<TosaOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        lhs, rhs);
    return success();
  }
};

template <typename T>
static bool isInValidRange(bool isFloat, const double &doubleValue, bool isInt,
                           const int64_t &intValue) {
  if (isFloat) {
    // Do a round-trip check here instead of numeric limits due to
    // compiler warnings around double <-> int conversion.
    return (doubleValue == static_cast<double>(static_cast<T>(doubleValue)));
  } else {
    assert(isInt);
    return (intValue >= std::numeric_limits<T>::min()) &&
           (intValue <= std::numeric_limits<T>::max());
  }
  return true;
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
    return op->emitError("Unable to extract the scalar constant");

  if (dtype.isa<mlir::FloatType>()) {
    tosaTensor = tosa::getConstTensor<float>(
                     rewriter, op, (isFloat ? doubleValue : intValue), dshape)
                     .getValue();
  } else if (auto intType = dtype.dyn_cast<mlir::IntegerType>()) {
    auto w = intType.getWidth();
    if (w != 32 && w != 64)
      return op->emitError("Unsupported integer type") << intType;

    if (w == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError("Supplied value of scalar constant exceeds limits "
                             "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      tosaTensor =
          tosa::getConstTensor<int32_t>(rewriter, op, {d}, dshape).getValue();
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError("Supplied value of scalar constant exceeds limits "
                             "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      tosaTensor =
          tosa::getConstTensor<int64_t>(rewriter, op, {d}, dshape).getValue();
    }
  } else
    return op->emitError("Usupported element type");

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
    return op->emitError("Currently only scalar constants are supported for "
                         "alpha in TOSA operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return op->emitError("Unsupported integer value for alpha");

  alphaTensor =
      mlir::tosa::getTosaConstTensorSingleF32(rewriter, op, alphaValue);

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
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in TOSA");

    if (auto lhsElemTy = lhsType.getElementType().dyn_cast<IntegerType>()) {
      if (lhsElemTy.getWidth() > 32)
        return op.emitError(
            "Integers with widths greater than 32 are not supported");
    }

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsType) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.other(), rhsAsTensor,
                                         outElemTy, {})))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in TOSA operation");
    }
    auto rhsTensor = rhsType ? rhs : rhsAsTensor;

    // Handle alpha.
    Value alphaTensor;
    if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(), op.alpha(),
                                      alphaTensor, outElemTy,
                                      /*checkForUnity=*/false))) {
      return op.emitError("Currently only scalar constants are supported for "
                          "alpha in conversion to TOSA operation");
    }

    auto multTensor = rewriter.create<tosa::MulOp>(
        op.getLoc(), rhsType ? rhsType : RankedTensorType::get({}, outElemTy),
        rhsTensor, alphaTensor, /*shift=*/0);

    if (outElemTy.isa<mlir::FloatType>()) {
      if (lhsType.getElementType() != outElemTy)
        lhs = rewriter.create<tosa::CastOp>(op.getLoc(), outType, lhs);

      rewriter.replaceOpWithNewOp<TosaOpT>(op, outType, lhs, multTensor);

      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
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
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    // For bitwise operators, only integer datatype legalization is supported
    if (lhsElemTy.isa<mlir::FloatType>() &&
        std::is_same<AtenOpT, AtenBitwiseAndTensorOp>()) {
      return op.emitError("For bitwise operators, only integer datatype "
                          "legalization is supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.other(), rhsAsTensor,
                                         lhsElemTy, {})))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in TOSA operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    // There is no Lesser operator in TOSA.
    auto swapLhsRhs = (std::is_same<AtenOpT, AtenLtTensorOp>() ||
                       std::is_same<AtenOpT, AtenLtScalarOp>());

    auto resultOp = rewriter.create<TosaOpT>(
        op.getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        (swapLhsRhs ? rhsTensor : lhs), (swapLhsRhs ? lhs : rhsTensor));

    // There is no NE operator in TOSA.
    if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
        std::is_same<AtenOpT, AtenNeScalarOp>())
      rewriter.replaceOpWithNewOp<tosa::LogicalNotOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          resultOp.getResult());
    else
      rewriter.replaceOp(op, resultOp.getResult());

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
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in TOSA");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    Value rhsTensor;
    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      Value rhsAsTensor;
      Value rhs = adaptor.other();
      auto rhsType = rhs.getType().dyn_cast<TensorType>();
      if (!rhsType) {
        if (failed(torchScalarToTosaTensor(rewriter, op, op.other(),
                                           rhsAsTensor, outElemTy, {})))
          return op.emitError(
              "Currently only scalar constants are supported for "
              "conversion in TOSA operation");
      }
      rhsTensor = rhsType ? rhs : rhsAsTensor;
    }

    if (outElemTy.isa<mlir::FloatType>() ||
        outElemTy.isa<mlir::IntegerType>()) {
      if (lhsType.getElementType() != outElemTy)
        lhs = rewriter.create<tosa::CastOp>(op.getLoc(), outType, lhs);

      rewriter.replaceOpWithNewOp<tosa::MulOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs, rhsTensor,
          /*shift=*/0);
      return success();
    } else {
      // Quantized multiplication may need to rescale inputs.
      return op.emitError("Only floating-point or integer datatype "
                          "legalization currently supported");
    }
  }
};

template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.other(), rhsAsTensor,
                                         lhsElemTy, {})))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in TOSA operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;

    if (lhsElemTy.isa<mlir::FloatType>()) {
      auto rcpOp = rewriter.create<tosa::ReciprocalOp>(
          op->getLoc(), rhsTy ? rhsTy : RankedTensorType::get({}, lhsElemTy),
          rhsTensor);
      rewriter.replaceOpWithNewOp<tosa::MulOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs, rcpOp.getResult(), /*shift=*/0);
    } else {
      rewriter.replaceOpWithNewOp<tosa::DivOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs, rhsTensor);
    }
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

template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<tosa::TanhOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  } else {
    // Sigmoid legalization in TOSA for quantized element-type uses
    // specialized tosa.table construct.
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

template <>
LogicalResult ConvertAtenOp<AtenSigmoidOp>::matchAndRewrite(
    AtenSigmoidOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<tosa::SigmoidOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  } else {
    // Sigmoid legalization in TOSA for quantized element-type uses
    // specialized tosa.table construct.
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();

  // Maps to tosa.clamp which has both int and fp limits.
  int64_t clampMin = 0;
  Value clampIn = self;
  if (selfTy) {
    // Rescale the clampIn for quantized types. TBD
    if (!selfTy.getElementType().isa<mlir::FloatType>()) {
      return op.emitError(
          "Only floating-point datatype legalization currently supported");
    }
    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, getTypeConverter()->convertType(op.getType()), clampIn,
        rewriter.getI64IntegerAttr(clampMin),
        rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
        rewriter.getF32FloatAttr(0.0f),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
    return success();
  } else {
    return op.emitError("Only Tensor types supported in TOSA");
  }
}

using ReductionConvFunc = llvm::Optional<Value> (*)(PatternRewriter &,
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
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only Tensor types supported in TOSA");

    auto outputTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<RankedTensorType>();
    if (!outputTy)
      return op.emitError(
          "Only ranked tensor type outputs permitted for reduce_mean");

    ElementsAttr reduceDimsAttr;
    bool keepDims;

    if (failed(readReduceDimsAndKeepDims(op, adaptor, rewriter, reduceDimsAttr,
                                         keepDims)))
      return failure();

    llvm::Optional<Value> result =
        ConversionFuncT(rewriter, op, outputTy, self, reduceDimsAttr, keepDims);

    if (!result)
      return failure();

    // TBD - support dtype casting.

    rewriter.replaceOp(op, {result.getValue()});

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
    SmallVector<int64_t, 4> reduceDims;
    if (!matchPattern(op.dim(), m_TorchConstantIntList(reduceDims)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t N = reduceDims.size();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(reduceDimsType,
                                               llvm::makeArrayRef(reduceDims));

    keepDims = false;
    if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDims)))
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
    if (!matchPattern(op.dim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    auto reduceDimsType = RankedTensorType::get({1}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(reduceDimsType,
                                               llvm::makeArrayRef({reduceDim}));

    keepDims = false;
    if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDims)))
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
    auto self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();

    // Select all dims to reduce
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < selfTy.getRank(); i++)
      reduceDims.push_back(i);
    int64_t N = selfTy.getRank();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(reduceDimsType,
                                               llvm::makeArrayRef(reduceDims));
    keepDims = false;

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenArgmaxOp>::matchAndRewrite(
    AtenArgmaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();

  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in TOSA argmax");

  int64_t reduceDim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&reduceDim))) {
    // NoneType indicates reduce on all dims
    reduceDim = -1;
  }

  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim)))
    return rewriter.notifyMatchFailure(
        op, "non-const keepdim parameter unsupported");

  auto resultTy = getTypeConverter()
                      ->convertType(op.getResult().getType())
                      .cast<RankedTensorType>();
  auto outputETy = resultTy.getElementType();

  // Create a single instance of tosa.argmax.
  // Multiple dims require chained construct.
  auto buildArgmax = [&](int64_t reduceDim, Value input) -> Value {
    auto inputTy = input.getType().cast<RankedTensorType>();
    auto inputShape = inputTy.getShape();
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
        ArrayRef<int64_t>(outputShapeArr), rewriter.getI32Type());
    auto reduceDimAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), reduceDim);
    return rewriter
        .create<tosa::ArgMaxOp>(op->getLoc(),
                                getTypeConverter()->convertType(outputReduceTy),
                                input, reduceDimAttr)
        .getResult();
  };

  // Convert the final index to i64 for backend finalization, However, i64
  // is not a defined type for tosa.cast, so using arith.extsi instead.
  auto castToInt64 = [&](Value result) -> LogicalResult {
    auto resTy = result.getType().cast<ShapedType>();
    if (!resTy)
      return op.emitError("Argmax: Result is not a shaped type");

    auto resShape = resTy.getShape();
    auto outTy =
        RankedTensorType::get(resShape, outputETy); // rewriter.getI64Type());

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
    Value self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();

    if (!selfTy)
      return op.emitError("Only ranked tensor types supported in TOSA argmax");

    SmallVector<int64_t> newOutputShape;
    if (failed(generateSqueezedShape(op, selfTy, rewriter, newOutputShape)))
      return op.emitError("Squeeze could not compute new shape");

    auto resultTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getResult().getType())
                        .template cast<RankedTensorType>();
    auto resultElemTy = resultTy.getElementType();

    auto newOutputTy = RankedTensorType::get(newOutputShape, resultElemTy);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            newOutputTy),
        self, rewriter.getI64ArrayAttr(newOutputShape));
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
    if (!matchPattern(op.dim(), m_TorchConstantInt(&squeezeDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");

    // Handle negative dim
    if (squeezeDim < 0)
      squeezeDim = squeezeDim + selfTy.getRank();

    auto selfShape = selfTy.getShape();

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
    auto selfShape = selfTy.getShape();

    // Dims that may dynamically resolve to 1 are not reduced here. Only
    // compile-time resolvable dims are handled here.
    for (auto &dim : selfShape) {
      if (dim != 1)
        squeezedShape.push_back(dim);
    }
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenPowTensorScalarOp>::matchAndRewrite(
    AtenPowTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();

  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in TOSA Pow");

  if (!selfTy.getElementType().isa<mlir::FloatType>())
    return op.emitError("Only floating-point datatype legalization supported");

  Value expTensor;
  Value expScalar = op.exponent();
  if (failed(torchScalarToTosaTensor(rewriter, op, expScalar, expTensor,
                                     selfTy.getElementType(), {})))
    return op.emitError("Currently only scalar constants are supported for "
                        "conversion in TOSA Pow operation");

  rewriter.replaceOpWithNewOp<tosa::PowOp>(
      op, getTypeConverter()->convertType(op.getType()), self, expTensor);

  return success();
}

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

    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = lhsTy.getShape();
    auto rhsShape = rhsTy.getShape();

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("Matmul: input datatypes mismatched");

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
      auto tensorTy = tensor.getType().cast<TensorType>();
      auto tensorShape = tensorTy.getShape();
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
    auto lhsBroadcastedTy =
        RankedTensorType::get(lhsBroadcastedShape, lhsElemTy);
    auto rhsBroadcastedShape = getRankBroadcastedShape(rhs, true);
    auto rhsBroadcastedTy =
        RankedTensorType::get(rhsBroadcastedShape, rhsElemTy);

    auto rankBroadcastedLhs =
        lhsRank == maxInputRank
            ? lhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      lhsBroadcastedTy),
                  lhs, rewriter.getI64ArrayAttr(lhsBroadcastedShape));

    auto rankBroadcastedRhs =
        rhsRank == maxInputRank
            ? rhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      rhsBroadcastedTy),
                  rhs, rewriter.getI64ArrayAttr(rhsBroadcastedShape));

    // TOSA matmul is performed on two 3D inputs and generates a 3D output.
    // Lower ranked tensors are dim-1 reshaped up to 3D
    auto reshapeUpTo3DTensor = [&](Value tensor) -> Value {
      auto tensorTy = tensor.getType().cast<TensorType>();
      auto rank = tensorTy.getRank();

      assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
      if (rank == 3)
        return tensor;

      auto shape = tensorTy.getShape();
      SmallVector<int64_t> newShape({1, 1, 1});

      if (rank == 2) { // batchsize = 1
        newShape[1] = shape[0];
        newShape[2] = shape[1];
      } else { // rank 1
        newShape[2] = shape[0];
      }
      auto newType = RankedTensorType::get(newShape, tensorTy.getElementType());

      return rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newType),
          tensor, rewriter.getI64ArrayAttr(newShape));
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

    // Transpose needs to done if transposeDims are not non-monotonically
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

    SmallVector<TensorShape_t> commonElems, lhsSqueezedElems, rhsSqueezedElems;

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
          commonElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }

      // TODO: Handle the case when there are dynamic batch dimensions.
      if (hasDynamicDims)
        commonValue = ShapedType::kDynamicSize;

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

      // Step: Create the tosa.transpose array. If this array has a
      // non-monotonic series of dims, perform transpose.
      // First the common_elems
      for (uint32_t i = 0; i < commonElems.size(); i++) {
        transposedLhsShape.push_back(commonElems[i].shape);
        transposedLhsDims.push_back(commonElems[i].dim);
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
        auto transposedLhsType =
            RankedTensorType::get(transposedLhsShape, rhsElemTy);

        llvm::Optional<Value> transposedLhsDimsConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedLhsDims,
                /*shape=*/{static_cast<int32_t>(transposedLhsDims.size())});

        lhsReshapeInput =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedLhsType),
                    rankBroadcastedLhs, transposedLhsDimsConst.getValue())
                .getResult();
      }

      // LHS = {common, lhs_squeezed, matmul_dim}
      SmallVector<int64_t> newLhsShape(
          {1, 1, lhsBroadcastedShape[maxInputRank - 1]});
      newLhsShape[0] = commonValue;
      newLhsShape[1] =
          hasDynamicDims ? ShapedType::kDynamicSize : lhsSqueezedValue;

      auto newLhsType = RankedTensorType::get(newLhsShape, lhsElemTy);

      matmulLhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newLhsType),
          lhsReshapeInput, rewriter.getI64ArrayAttr(newLhsShape));

      SmallVector<int64_t> transposedRhsShape;
      SmallVector<int32_t> transposedRhsDims;

      // Step: Create the RHS transpose sequence
      // RHS = {common, matmul_dim, rhs_squeezed}
      // first the common_dims
      for (uint32_t i = 0; i < commonElems.size(); i++) {
        transposedRhsShape.push_back(commonElems[i].shape);
        transposedRhsDims.push_back(commonElems[i].dim);
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

      auto transposedRhsType =
          RankedTensorType::get(transposedRhsShape, rhsElemTy);

      if (hasDynamicDims)
        rhsSqueezedValue = ShapedType::kDynamicSize;

      SmallVector<int64_t> newRhsShape({commonValue,
                                        rhsBroadcastedShape[maxInputRank - 2],
                                        rhsSqueezedValue});
      auto newRhsType = RankedTensorType::get(newRhsShape, rhsElemTy);

      bool rhsNeedsTranspose = isTransposeRequired(transposedRhsDims);

      auto transposedRhsValue = rankBroadcastedRhs;

      if (rhsNeedsTranspose) {
        llvm::Optional<Value> transposedRhsDimsConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedRhsDims,
                /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

        transposedRhsValue =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedRhsType),
                    rankBroadcastedRhs, transposedRhsDimsConst.getValue())
                .getResult();
      }

      // reshape
      matmulRhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newRhsType),
          transposedRhsValue, rewriter.getI64ArrayAttr(newRhsShape));
    }

    auto matmulLhsShape =
        matmulLhs.getType().template cast<RankedTensorType>().getShape();
    auto matmulRhsShape =
        matmulRhs.getType().template cast<RankedTensorType>().getShape();

    // The reshape/transpose should ensure the tosa.matmul always has same
    // batch size for either matrix. If if shapes are dynamic, they'll be
    // appropriately handled.
    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "tosa.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});
    Type outputElemTy;
    if (lhsElemTy.isa<mlir::FloatType>()) {
      outputElemTy = lhsElemTy;
    } else { // qint8 emits i32 matmul output
      outputElemTy = rewriter.getIntegerType(32);
    }

    auto mmOutputTy = RankedTensorType::get(matmulOutputShape, outputElemTy);
    auto mmOpResult =
        rewriter
            .create<tosa::MatMulOp>(
                op->getLoc(),
                OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                    mmOutputTy),
                matmulLhs, matmulRhs)
            .getResult();

    // Perform the reshape to output shape. This is always required unless both
    // inputs are rank=3, in which case the tosa.matmul output itself is
    // correctly shaped.
    bool performOpReshape =
        !(lhsRank == 3 && rhsRank == 3 && lhsShape[0] == rhsShape[0]);

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
        for (uint32_t i = 0; i < commonElems.size(); i++) {
          reshapedOpShape.push_back(commonElems[i].shape);
          transposedOpDims.push_back(commonElems[i].dim);
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

        // Final transposed output shape construction
        for (uint32_t i = 0; i < maxInputRank - 2; i++) {
          if (lhsBroadcastedTy.isDynamicDim(i)) {
            transposedOpShapes.push_back(ShapedType::kDynamicSize);
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
      auto reshapedOpType =
          RankedTensorType::get(reshapedOpShape, outputElemTy);
      auto reshapedOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              reshapedOpType),
          mmOpResult, rewriter.getI64ArrayAttr(reshapedOpShape));

      if (opNeedsTranspose) {

        llvm::Optional<Value> transposedOpShapeConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedOpDims,
                /*shape=*/{static_cast<int32_t>(transposedOpDims.size())});

        auto transposedOpType =
            RankedTensorType::get(transposedOpShape, outputElemTy);
        output =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedOpType),
                    reshapedOp.getResult(), transposedOpShapeConst.getValue())
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
      return op.emitError("Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return op.emitError("Failed to perform matmul operation");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>(),
        output);

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
    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.other();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in TOSA matmul");

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

    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.mat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in TOSA matmul");

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

    lhs = adaptor.input();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.weight();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    // Protection against crash due to unguarded code in TOSA->LinAlg.
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return op.emitError("aten.Linear needs statically shaped input");

    return success();
  }
  // Override the default rewriter to perform RHS transpose and bias addition as
  // well.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return op.emitError("Failed to read matmul inputs");

    // The aten.Linear op has a bias tensor that is added to the matmul output.
    auto bias = adaptor.bias();
    auto biasTy = bias.getType();

    // TOSA does not mandate that elementwise op tensors need to be ranked.
    if (!biasTy.template isa<Torch::NoneType>() &&
        !biasTy.template isa<TensorType>())
      return op.emitError("Only tensor types supported in GEMM to "
                          "TOSA for bias tensor");

    // RHS must have its last two dims transposed prior to matrix
    // multiplication.
    auto rhsTy = rhs.getType().cast<RankedTensorType>();
    auto rhsRank = rhsTy.getRank();
    auto rhsShape = rhsTy.getShape();
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

    llvm::Optional<Value> transposedRhsShapeConst =
        tosa::getConstTensor<int32_t>(
            rewriter, op,
            /*vec=*/transposedRhsDims,
            /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

    auto transposedRhsType =
        RankedTensorType::get(transposedRhsShape, rhsElemTy);
    rhs = rewriter.create<tosa::TransposeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            transposedRhsType),
        rhs, transposedRhsShapeConst.getValue());

    Value matmulOutput;
    if (failed(
            this->performMatmul(op, adaptor, rewriter, lhs, rhs, matmulOutput)))
      return op.emitError("Failed to perform matmul operation");

    Value matmulPlusBias = matmulOutput;
    if (!biasTy.template isa<Torch::NoneType>()) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias =
          rewriter
              .create<tosa::AddOp>(op->getLoc(), matmulOutput.getType(),
                                   matmulOutput, bias)
              .getResult();
    }

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>(),
        matmulPlusBias);

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenRsubScalarOp>::matchAndRewrite(
    AtenRsubScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.self();
  auto otherScalar = op.other();
  auto alphaScalar = op.alpha();

  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in TOSA Rsub");

  if (!selfTy.getElementType().isa<mlir::FloatType>())
    return op.emitError("Only floating-point datatype legalization supported");

  Value otherTensor, alphaTensor;

  if (failed(torchScalarToTosaTensor(rewriter, op, otherScalar, otherTensor,
                                     selfTy.getElementType(), {})))
    return op.emitError("Currently only scalar constants are supported for "
                        "conversion in TOSA Rsub operation");

  if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(), alphaScalar,
                                    alphaTensor, selfTy.getElementType(),
                                    /*checkForUnity=*/true)))
    return failure();

  auto multTensor = rewriter.create<tosa::MulOp>(
      op->getLoc(), getTypeConverter()->convertType(op.getType()), self,
      alphaTensor, /*shift=*/0);

  rewriter.replaceOpWithNewOp<tosa::SubOp>(
      op, getTypeConverter()->convertType(op.getType()), otherTensor,
      multTensor);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto input = adaptor.input();
  auto weight = adaptor.weight();

  auto inputTy = input.getType().template cast<RankedTensorType>();
  auto weightTy = weight.getType().template cast<RankedTensorType>();
  auto outputTy = getTypeConverter()
                      ->convertType(op.getType())
                      .template cast<RankedTensorType>();

  if (!inputTy || !weightTy || !outputTy)
    return op.emitError(
        "Input, weight and output to Convolution must be ranked tensors");

  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto inputShape = inputTy.getShape();
  auto weightShape = weightTy.getShape();

  if (inputTy.getRank() != 4)
    return op.emitError("Unimplemented: only 2D convolutions supported");

  if (!weightTy.hasStaticShape())
    return op.emitError("Unimplemented: TOSA only supports static weight");

  // Bias is optional. TOSA mandates a zero tensor here, so construct one if
  // required.
  auto bias = adaptor.bias();
  if (adaptor.bias().getType().template isa<Torch::NoneType>()) {
    // TBD: This is only valid for quantized 8-bit. For 16-bit, the bias (and
    // accumulator) are 48-bit and not 32-bit, and requires the use of APInt to
    // define a 48-bit int.
    if (inputElemTy.isa<quant::QuantizedType>()) {
      SmallVector<int32_t> zeroVec(weightShape[0], 0);
      bias = tosa::getConstTensor<int32_t>(
                 rewriter, op, zeroVec, {static_cast<int32_t>(weightShape[0])})
                 .getValue();
    } else {
      SmallVector<float> zeroVec(weightShape[0], 0);
      bias = tosa::getConstTensor<float>(rewriter, op, zeroVec,
                                         {static_cast<int32_t>(weightShape[0])})
                 .getValue();
    }
  } else {
    if (!bias.getType().cast<RankedTensorType>())
      return op.emitError("Bias provided but not a ranked tensor");
  }
  auto biasElemTy = inputElemTy.template isa<mlir::FloatType>()
                        ? inputElemTy
                        : rewriter.getI32Type();

  SmallVector<int64_t, 2> stride;
  if (!matchPattern(adaptor.stride(), m_TorchConstantIntList(stride)))
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");

  SmallVector<int64_t, 2> padding_2d;
  if (!matchPattern(adaptor.padding(), m_TorchConstantIntList(padding_2d)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");
  // TOSA uses 4D padding {t, b, l, r} while Torch defines 2D padding {t, l}.
  // The Torch OFM computation uses 2*pad in each spatial direction, implying
  // the same t=b and l=r values for TOSA.
  SmallVector<int64_t> padding(
      {padding_2d[0], padding_2d[0], padding_2d[1], padding_2d[1]});

  SmallVector<int64_t, 2> dilation;
  if (!matchPattern(adaptor.dilation(), m_TorchConstantIntList(dilation)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");

  // TOSA works in NHWC and takes OHWI weights. Perform the necessary transpose.
  llvm::Optional<Value> nchwToNhwcTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 2, 3, 1},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputType =
      RankedTensorType::get(transposedInputShape, inputElemTy);
  auto transposedInput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedInputType), input,
              nchwToNhwcTransposeConst.getValue())
          .getResult();

  SmallVector<int64_t> transposedWeightShape(
      {weightShape[0], weightShape[2], weightShape[3], weightShape[1]});
  auto transposedWeightType =
      RankedTensorType::get(transposedWeightShape, weightElemTy);
  auto transposedWeight =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedWeightType), weight,
              nchwToNhwcTransposeConst.getValue())
          .getResult();

  int64_t outputHDim, outputWDim;
  if (inputTy.hasStaticShape()) {
    outputHDim = (transposedInputShape[1] + padding[0] + padding[1] -
                  dilation[0] * (transposedWeightShape[1] - 1) - 1) /
                     stride[0] +
                 1;
    outputWDim = (transposedInputShape[2] + padding[2] + padding[3] -
                  dilation[1] * (transposedWeightShape[2] - 1) - 1) /
                     stride[1] +
                 1;
  } else {
    outputHDim = ShapedType::kDynamicSize;
    outputWDim = ShapedType::kDynamicSize;
  }

  // Output shape is NHWC, to be transposed back to NCHW. Output elemTy for
  // quantized input is i32, which gets rescaled down to quantized output range.
  SmallVector<int64_t> outputShape = {transposedInputShape[0], outputHDim,
                                      outputWDim, transposedWeightShape[0]};
  auto convOpTy = RankedTensorType::get(outputShape, biasElemTy);

  Value convOpResult =
      rewriter
          .create<tosa::Conv2DOp>(op->getLoc(),
                                  getTypeConverter()->convertType(convOpTy),
                                  transposedInput, transposedWeight, bias,
                                  rewriter.getI64ArrayAttr(padding),
                                  rewriter.getI64ArrayAttr(stride),
                                  rewriter.getI64ArrayAttr(dilation))
          .getResult();

  llvm::Optional<Value> nhwcToNchwTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 3, 1, 2},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedOutputShape(
      {outputShape[0], outputShape[3], outputShape[1], outputShape[2]});
  auto transposedOutputType =
      RankedTensorType::get(transposedOutputShape, biasElemTy);
  auto transposedOutput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedOutputType),
              convOpResult, nhwcToNchwTransposeConst.getValue())
          .getResult();

  Value rescaledResult = transposedOutput;
  if (inputElemTy.template isa<quant::QuantizedType>()) {
    rescaledResult = tosa::buildRescaleOpConvOutput(
        rewriter, op, transposedOutput, inputTy, weightTy, outputTy);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), rescaledResult);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenReshapeOp>::matchAndRewrite(
    AtenReshapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.self();

  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in TOSA Reshape");

  // Check that at most one dimension is -1
  SmallVector<int64_t> newShape;
  if (!matchPattern(op.shape(), m_TorchConstantIntList(newShape)))
    return op.emitError("Only constant shape supported in TOSA Reshape");

  int auto_sz = 0;
  for (auto s : newShape)
    auto_sz += (s == -1 ? 1 : 0);
  if (auto_sz > 1)
    op.emitError("At most one dimension may be specified as -1 to "
                 "automatically calculate its size");

  auto newType = RankedTensorType::get(newShape, selfTy.getElementType());

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), self,
      rewriter.getI64ArrayAttr(newShape));

  return success();
}

Value computeBatchNorm(Operation *op, ConversionPatternRewriter &rewriter,
                       Type outType, Value input, Value variance, Value eps,
                       Value mean, Value weight, Value bias) {
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

  auto op1SubInputMean =
      rewriter.create<tosa::SubOp>(op->getLoc(), outType, input, mean);

  auto op2AddVarEpsilon = rewriter.create<tosa::AddOp>(
      op->getLoc(), variance.getType(), variance, eps);

  auto op3RsqrtOp2 = rewriter.create<tosa::RsqrtOp>(
      op->getLoc(), variance.getType(), op2AddVarEpsilon.getResult());

  auto op4MulOp1Op3 = rewriter.create<tosa::MulOp>(op->getLoc(), outType,
                                                   op1SubInputMean.getResult(),
                                                   op3RsqrtOp2.getResult(), 0);

  auto op5MulOp4Scale = rewriter.create<tosa::MulOp>(
      op->getLoc(), outType, op4MulOp1Op3.getResult(), weight, 0);

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
  if (!adaptor.input().getType().dyn_cast<RankedTensorType>())
    return op.emitError("Only ranked tensor types are supported");

  auto outType = getTypeConverter()->convertType(op.getType());

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle training and momentum.
  if (op.momentum().getType().isa<Torch::NoneType>())
    op.emitError("Unsupported None for momentum");

  auto meanType = adaptor.running_mean().getType().dyn_cast<TensorType>();
  auto varianceType = adaptor.running_var().getType().dyn_cast<TensorType>();
  if (!varianceType || !meanType)
    return op.emitError("Only ranked tensor types are supported");

  // Normalization ops perform elementwise ops of a single mean/stdev value
  // against the feature map and because input is NCHW, the rank-1 value must be
  // reshaped so it sits on the same dim as 'C'.
  auto reshapeToNormInputDim = [&](Operation *op,
                                   ConversionPatternRewriter &rewriter,
                                   TypeConverter *converter, Type outType,
                                   const Value toBcast, Value &result) {
    RankedTensorType toBcastType =
        toBcast.getType().dyn_cast<RankedTensorType>();
    if (toBcastType.getRank() > 1)
      op->emitError("Rank cannot be more than 1");

    RankedTensorType outTensorType = outType.cast<RankedTensorType>();
    SmallVector<int64_t> newShape = {toBcastType.getShape()[0]};
    for (auto i = 2; i < outTensorType.getRank(); ++i)
      newShape.push_back(1);
    auto newType =
        RankedTensorType::get(newShape, outTensorType.getElementType());

    result = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), newType, toBcast, rewriter.getI64ArrayAttr(newShape));

    return success();
  };

  Value meanVal, varianceVal, weightVal, biasVal;
  assert(meanType.getNumElements() != 0 && varianceType.getNumElements() != 0);
  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.running_mean(), meanVal)))
    op.emitError("Failed to reshape running mean");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.running_var(), varianceVal)))
    op.emitError("Failed to reshape running variance");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.weight(), weightVal)))
    op.emitError("Failed to reshape weight");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType, adaptor.bias(),
                                   biasVal)))
    op.emitError("Failed to reshape bias");

  double eps;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps)))
    return op.emitError("eps must be a scalar constant");

  auto epsilonConst =
      mlir::tosa::getTosaConstTensorSingleF32(rewriter, op, eps);

  auto batchNorm =
      computeBatchNorm(op, rewriter, outType, adaptor.input(), varianceVal,
                       epsilonConst, meanVal, weightVal, biasVal);

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
  if (!adaptor.input().getType().dyn_cast<RankedTensorType>())
    return op.emitError("Only ranked tensor types are supported");

  auto inputType = adaptor.input().getType().cast<RankedTensorType>();
  if (inputType.getRank() > 4)
    return op.emitError("Only up to 4D tensors are supported");

  auto outType = getTypeConverter()->convertType(op.getType(0));

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle the None cases for the optional parameters.
  if (adaptor.weight().getType().isa<Torch::NoneType>())
    return op.emitError("Unsupported None for weight");
  if (adaptor.bias().getType().isa<Torch::NoneType>())
    return op.emitError("Unsupported None for bias");

  auto weightType = adaptor.weight().getType().cast<RankedTensorType>();
  auto biasType = adaptor.bias().getType().cast<RankedTensorType>();
  int64_t inputRank = inputType.getRank();
  Type elemTy = inputType.getElementType();

  // Check if all the arguments meet the requirements.
  SmallVector<int64_t> normalizedShapeSizesInt;
  if (!matchPattern(op.normalized_shape(),
                    m_TorchConstantIntList(normalizedShapeSizesInt))) {
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
    if (inputType.getShape()[index + meanAndVarShapeRank] != value ||
        weightType.getShape()[index] != value ||
        biasType.getShape()[index] != value)
      return op.emitError("mismatching contracting dimension");
  }

  // Helper for computing mean and variance.
  auto computeSumAndReshape = [&](Value toReduce, RankedTensorType toReduceType,
                                  Type outType, SmallVector<int64_t> outShape) {
    Value sumDiv = toReduce;
    SmallVector<int64_t> toReduceShape(toReduceType.getShape().begin(),
                                       toReduceType.getShape().end());
    while (static_cast<int64_t>(toReduceShape.size()) != meanAndVarShapeRank) {
      toReduceShape.back() = 1;
      sumDiv = rewriter.create<tosa::ReduceSumOp>(
          op.getLoc(),
          RankedTensorType::get(toReduceShape, inputType.getElementType()),
          sumDiv, rewriter.getI64IntegerAttr(toReduceShape.size() - 1));
      toReduceShape.pop_back();
    }

    return rewriter.create<tosa::ReshapeOp>(op.getLoc(), outType, sumDiv,
                                            rewriter.getI64ArrayAttr(outShape));
  };

  // TOSA has integer Div so, compute reciprocal of element count to be used in
  // mul.
  int64_t elemCnt = 1;
  for (auto i : normalizedShapeSizesInt)
    elemCnt *= i;

  auto elemCntConst =
      tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                  {static_cast<float>(elemCnt)}, {1})
          .getValue();
  Value elemCntRcp = rewriter.create<tosa::ReciprocalOp>(
      op.getLoc(), elemCntConst.getType(), elemCntConst);

  // Broadcast type and shape for various intermediate values.
  SmallVector<int64_t> bcastOutShape;
  for (auto en : llvm::enumerate(inputType.getShape())) {
    bcastOutShape.push_back(
        static_cast<int64_t>(en.index()) >= meanAndVarShapeRank ? 1
                                                                : en.value());
  }
  auto bcastOutType = RankedTensorType::get(bcastOutShape, elemTy);

  // Compute mean.
  Value sum = computeSumAndReshape(adaptor.input(), inputType, bcastOutType,
                                   bcastOutShape);
  Value meanVal = rewriter.create<tosa::MulOp>(op.getLoc(), bcastOutType, sum,
                                               elemCntRcp, /*shift=*/0);

  // Compute variance.
  Value squareSumSub = rewriter.create<tosa::SubOp>(op.getLoc(), inputType,
                                                    adaptor.input(), meanVal);
  Value squareSum = rewriter.create<tosa::MulOp>(op.getLoc(), inputType,
                                                 squareSumSub, squareSumSub, 0);

  Value squareSumReduced =
      computeSumAndReshape(squareSum, inputType, bcastOutType, bcastOutShape);
  Value varianceVal = rewriter.create<tosa::MulOp>(
      op.getLoc(), bcastOutType, squareSumReduced, elemCntRcp, /*shift=*/0);

  // Reshape weight and bias.
  SmallVector<int64_t> weightAndBiasBcastShape;
  for (auto en : llvm::enumerate(inputType.getShape())) {
    weightAndBiasBcastShape.push_back(
        static_cast<int64_t>(en.index()) < meanAndVarShapeRank ? 1
                                                               : en.value());
  }
  auto weightAndMeanBcastType =
      RankedTensorType::get(weightAndBiasBcastShape, elemTy);

  Value weightVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, adaptor.weight(),
      rewriter.getI64ArrayAttr(weightAndBiasBcastShape));

  Value biasVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, adaptor.bias(),
      rewriter.getI64ArrayAttr(weightAndBiasBcastShape));

  double eps;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps)))
    return op.emitError("eps must be a scalar constant");
  auto epsilonConst =
      mlir::tosa::getTosaConstTensorSingleF32(rewriter, op, eps);

  // Compute layer norm.
  auto layerNorm =
      computeBatchNorm(op, rewriter, outType, adaptor.input(), varianceVal,
                       epsilonConst, meanVal, weightVal, biasVal);

  rewriter.replaceOp(op, {layerNorm, meanVal, varianceVal});

  return success();
}

// Torch constants are converted to tosa.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto outputTy = getTypeConverter()
                      ->convertType(op.getType())
                      .template cast<RankedTensorType>();
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputTy, adaptor.value());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenFlattenUsingIntsOp>::matchAndRewrite(
    AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType || !selfType.hasStaticShape())
    return op.emitError(
        "Only ranked tensor types with static shapes are currently supported");

  int64_t selfRank = selfType.getRank();

  int64_t start_dim, end_dim;

  if (!matchPattern(op.start_dim(), m_TorchConstantInt(&start_dim)))
    return op.emitError("start_dim must be a Scalar constant");
  start_dim = toPositiveDim(start_dim, selfRank);

  if (!matchPattern(op.end_dim(), m_TorchConstantInt(&end_dim)))
    return op.emitError("end_dim must be a Scalar constant");
  end_dim = toPositiveDim(end_dim, selfRank);

  if (selfRank > 0 && !isValidDim(start_dim, selfRank))
    return op.emitError("start_dim is statically invalid");
  if (selfRank > 0 && !isValidDim(end_dim, selfRank))
    return op.emitError("end_dim is statically invalid");
  if (end_dim < start_dim)
    return op.emitError("end_dim must be larger than start_dim");

  SmallVector<int64_t> newShape;
  for (auto s : llvm::enumerate(selfType.getShape())) {
    int64_t idx = s.index();
    if (idx < start_dim || idx > end_dim) {
      newShape.push_back(s.value());
    } else {
      if (idx == start_dim)
        newShape.push_back(s.value());
      else
        newShape.back() *= s.value();
    }
  }

  // Handle the Scalar case
  if (newShape.size() == 0)
    newShape.push_back(1);

  auto newType = RankedTensorType::get(newShape, selfType.getElementType());
  auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), newType, adaptor.self(), rewriter.getI64ArrayAttr(newShape));

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), reshapeOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPermuteOp>::matchAndRewrite(
    AtenPermuteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType)
    return op.emitError(
        "Only ranked tensor types with static shapes are currently supported");

  SmallVector<int64_t> dimListInt;
  if (!matchPattern(adaptor.dims(), m_TorchConstantIntList(dimListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dimensions are currently supported");

  int64_t selfRank = selfType.getRank();
  for (auto &d : dimListInt) {
    d = toPositiveDim(d, selfRank);
    if (!isValidDim(d, selfRank))
      return op.emitError("Not all dims are valid");
  }

  auto transposeDimsConst = mlir::tosa::getConstTensor<int64_t>(
      rewriter, op.getOperation(), dimListInt, {selfRank});

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self(),
      transposeDimsConst.getValue());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTransposeIntOp>::matchAndRewrite(
    AtenTransposeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType)
    return op.emitError(
        "Only ranked tensor types with static shapes are currently supported");

  int64_t dim0, dim1;
  if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0)))
    return op->emitError("dim must be a Scalar constant");
  if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1)))
    return op->emitError("dim must be a Scalar constant");

  int64_t selfRank = selfType.getRank();
  dim0 = toPositiveDim(dim0, selfRank);
  if (!isValidDim(dim0, selfRank))
    return op.emitError("Not all dims are valid");
  dim1 = toPositiveDim(dim1, selfRank);
  if (!isValidDim(dim1, selfRank))
    return op.emitError("Not all dims are valid");

  SmallVector<int64_t> dimListInt;
  for ( int64_t i = 0; i < selfRank; ++i ) {
    dimListInt.push_back( i );
  }
  //swap dimensions to be transposed  
  std::swap( dimListInt[ dim0 ], dimListInt[ dim1 ] );

  auto transposeDimsConst = mlir::tosa::getConstTensor<int64_t>(
      rewriter, op.getOperation(), dimListInt, { selfRank });

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self(),
      transposeDimsConst.getValue());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLog2Op>::matchAndRewrite(
    AtenLog2Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  // Constant value of ln2.
  SmallVector<int64_t> ln2Shape(selfType.getRank(), 1);
  auto ln2Op =
      tosa::getConstTensor<float>(rewriter, op, {0.69314718056}, ln2Shape)
          .getValue();
  auto rcpOp =
      rewriter.create<tosa::ReciprocalOp>(op.getLoc(), ln2Op.getType(), ln2Op);

  auto outType = getTypeConverter()->convertType(op.getType());
  auto logOp =
      rewriter.create<tosa::LogOp>(op.getLoc(), outType, adaptor.self());
  rewriter.replaceOpWithNewOp<tosa::MulOp>(op, outType, logOp, rcpOp,
                                           /*shift=*/0);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenThresholdOp>::matchAndRewrite(
    AtenThresholdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat())
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");

  // Integer types with width > 32 are not supported
  auto selfIntType = selfElemTy.dyn_cast<IntegerType>();
  if (selfIntType && selfIntType.getWidth() > 32) {
    return op.emitError(
        "Integer types with width greater than 32 are not supported");
  }

  SmallVector<int64_t> constTypeShape(selfType.getRank(), 1);
  Value threshold, value;
  if (failed(torchScalarToTosaTensor(rewriter, op, op.threshold(), threshold,
                                     selfElemTy, constTypeShape)))
    return op.emitError("Only scalar constant is supported for threshold");

  if (failed(torchScalarToTosaTensor(rewriter, op, op.value(), value,
                                     selfElemTy, constTypeShape)))
    return op.emitError("Only scalar constant is supported for value");

  // Threshold only clamps the upper values. tosa::ClampOp has the same
  // value for both threshold and clamped value so cannot be used.
  auto outType = getTypeConverter()->convertType(op.getType());

  auto cmpOp = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      adaptor.self(), threshold);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cmpOp,
                                              adaptor.self(), value);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType) {
    return op.emitError("Only tensor types are currently supported");
  }

  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return op->emitError("dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfRank);
  if (!isValidDim(dim, selfRank))
    return op.emitError("dim is statically invalid");

  SmallVector<int64_t> outShape;
  for (auto en : llvm::enumerate(selfType.getShape())) {
    if (static_cast<int64_t>(en.index()) == dim)
      outShape.push_back(1);

    outShape.push_back(en.value());
  }

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self(),
      rewriter.getI64ArrayAttr(outShape));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenContiguousOp>::matchAndRewrite(
    AtenContiguousOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  // FIXME: memory_format is not handled.

  rewriter.replaceOp(op, adaptor.self());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenDropoutOp>::matchAndRewrite(
    AtenDropoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.input().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  // FIXME: train and p are not handled.

  bool train;
  if (!matchPattern(op.train(), m_TorchConstantBool(&train)))
    op.emitError("train must be a Scalar constant");

  if (train)
    op.emitError("train must be false");

  rewriter.replaceOpWithNewOp<tosa::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.input());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewOp>::matchAndRewrite(
    AtenViewOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> outShape;
  if (!matchPattern(op.size(), m_TorchConstantIntList(outShape)))
    return op.emitError("size must consist of Scalar constants");

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self(),
      rewriter.getI64ArrayAttr(outShape));

  return success();
}

static Value approximateErfOp(ConversionPatternRewriter &rewriter,
                              Operation *op, Value x) {
  // Using:
  // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations with
  // maximum error as 5 x 10^-4 where a1 = 0.278393, a2 = 0.230389, a3 =
  // 0.000972, a4 = 0.078108.
  //
  // Erf = 1 - 1 / (1 + a1X + a2X + a3X + a4X)^4

  auto outType = x.getType().cast<TensorType>();
  auto loc = op->getLoc();
  auto absX = rewriter.create<tosa::AbsOp>(loc, outType, x);
  auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}).getValue();
  auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}).getValue();

  auto a1 = tosa::getConstTensor<float>(rewriter, op, 0.278393, {}).getValue();
  auto a1X = rewriter.create<tosa::MulOp>(loc, outType, a1, absX, /*shift=*/0);
  auto sum = rewriter.create<tosa::AddOp>(loc, outType, a1X, one);

  auto a2 = tosa::getConstTensor<float>(rewriter, op, 0.230389, {}).getValue();
  auto x2 = rewriter.create<tosa::MulOp>(loc, outType, absX, absX, /*shift=*/0);
  auto a2X = rewriter.create<tosa::MulOp>(loc, outType, a2, x2, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a2X);

  auto a3 = tosa::getConstTensor<float>(rewriter, op, 0.000972, {}).getValue();
  auto x3 = rewriter.create<tosa::MulOp>(loc, outType, x2, absX, /*shift=*/0);
  auto a3X = rewriter.create<tosa::MulOp>(loc, outType, a3, x3, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a3X);

  auto a4 = tosa::getConstTensor<float>(rewriter, op, 0.078108, {}).getValue();
  auto x4 = rewriter.create<tosa::MulOp>(loc, outType, x3, absX, /*shift=*/0);
  auto a4X = rewriter.create<tosa::MulOp>(loc, outType, a4, x4, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a4X);

  auto rcprl = rewriter.create<tosa::ReciprocalOp>(loc, outType, sum);
  auto rcprl2 =
      rewriter.create<tosa::MulOp>(loc, outType, rcprl, rcprl, /*shift=*/0);
  auto rcprl4 =
      rewriter.create<tosa::MulOp>(loc, outType, rcprl2, rcprl2, /*shift=*/0);
  auto erf = rewriter.create<tosa::SubOp>(loc, outType, one, rcprl4);

  // Deal with negative x.
  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      loc,
      RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1)), x,
      zero);
  auto negateErf = rewriter.create<tosa::NegateOp>(loc, outType, erf);

  return rewriter.create<tosa::SelectOp>(loc, outType, cond, erf, negateErf);
}

static Value buildUnitNormalCdf(ConversionPatternRewriter &rewriter,
                                Operation *op, Value x) {
  auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}).getValue();
  auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}).getValue();
  auto loc = op->getLoc();

  // buildNormalCdf, mean = zero, sigma = one
  auto outType = x.getType();
  auto mean = zero;
  Value xMinusMean = rewriter.create<tosa::SubOp>(loc, outType, x, mean);
  // rsqrt of 2
  Value rsqrt2 =
      tosa::getConstTensor<float>(rewriter, op, 0.70710678, {}).getValue();
  Value erfArg = rewriter.create<tosa::MulOp>(loc, outType, xMinusMean, rsqrt2,
                                              /*shift=*/0);
  Value erf = approximateErfOp(rewriter, op, erfArg);
  Value erfPlus1 = rewriter.create<tosa::AddOp>(loc, outType, one, erf);
  Value oneHalf = tosa::getConstTensor<float>(rewriter, op, 0.5, {}).getValue();
  Value normalCdf = rewriter.create<tosa::MulOp>(loc, outType, oneHalf,
                                                 erfPlus1, /*shift=*/0);
  return normalCdf;
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isa<mlir::FloatType>()) {
    return op.emitError("Only floating-point datatype legalization supported");
  }

  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.approximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return op.emitError("Unsupported value of approximate");
  }

  Value cdf = buildUnitNormalCdf(rewriter, op, adaptor.self());
  rewriter.replaceOpWithNewOp<tosa::MulOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self(), cdf,
      /*shift=*/0);

  return success();
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluBackwardOp>::matchAndRewrite(
    AtenGeluBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isa<mlir::FloatType>()) {
    return op.emitError("Only floating-point datatype legalization supported");
  }

  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.approximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return op.emitError("Unsupported value of approximate");
  }

  auto loc = op->getLoc();

  const double cstAlpha0 = 1.12837916709551257390;
  const double cstAlpha1 = 0.70710678118654752440;
  const double oneHalf = 0.5;
  const double kAlpha = cstAlpha0 * cstAlpha1;

  Value kAlphaHalf =
      tosa::getConstTensor<float>(rewriter, op, kAlpha * oneHalf, {})
          .getValue();
  Value negOneHalf =
      tosa::getConstTensor<float>(rewriter, op, -0.5, {}).getValue();
  Value inputSquared = rewriter.create<tosa::MulOp>(
      loc, selfType, adaptor.self(), adaptor.self(), /*shift=*/0);
  Value negHalfInputSquared = rewriter.create<tosa::MulOp>(
      loc, selfType, inputSquared, negOneHalf, /*shift=*/0);
  Value dinput =
      rewriter.create<tosa::ExpOp>(loc, selfType, negHalfInputSquared);
  Value cdf = buildUnitNormalCdf(rewriter, op, adaptor.self());
  Value dinputInput = rewriter.create<tosa::MulOp>(loc, selfType, dinput,
                                                   adaptor.self(), /*shift=*/0);
  Value dinputInputAlpha = rewriter.create<tosa::MulOp>(
      loc, selfType, dinputInput, kAlphaHalf, /*shift=*/0);
  Value cdfExt =
      rewriter.create<tosa::AddOp>(loc, selfType, dinputInputAlpha, cdf);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.grad_output(),
      cdfExt,
      /*shift=*/0);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenEmbeddingOp>::matchAndRewrite(
    AtenEmbeddingOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value weight = adaptor.weight();
  Value indices = adaptor.indices();
  RankedTensorType outType =
      typeConverter->convertType(op.getType()).cast<RankedTensorType>();

  auto indicesType = indices.getType().dyn_cast<RankedTensorType>();
  if (!indicesType || !indicesType.getElementType().isa<IntegerType>())
    return op.emitError("Indices must be of integer tensor type");

  if (indicesType.getRank() != 2)
    return op.emitError("indices must be of rank 2");

  auto weightType = weight.getType().cast<RankedTensorType>();
  if (weightType.getRank() != 2)
    return op.emitError("weight must be of rank 2");

  // FIXME: padding_idx, scale_grad_by_freq and sparse are not handled yet.
  int64_t paddingIdx;
  if (!matchPattern(op.padding_idx(), m_TorchConstantInt(&paddingIdx)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant int padding_idx for embedding op");

  bool scaleGradByFreq;
  if (!matchPattern(op.scale_grad_by_freq(),
                    m_TorchConstantBool(&scaleGradByFreq)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant bool scale_grad_by_freq for embedding op");
  if (scaleGradByFreq)
    return rewriter.notifyMatchFailure(
        op,
        "only supports scale_grad_by_freq equals to False for embedding op");

  bool isSparse;
  if (!matchPattern(op.sparse(), m_TorchConstantBool(&isSparse)))
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
  auto indicesShape = indicesType.getShape();
  auto weightShape = weightType.getShape();

  SmallVector<int64_t> newWeightShape = {1};
  for (auto s : weightShape)
    newWeightShape.push_back(s);

  auto reshapedWeight = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(newWeightShape, weightType.getElementType()),
      weight, rewriter.getI64ArrayAttr(newWeightShape));

  int64_t numIndices = 1;
  if (indicesType.hasStaticShape()) {
    for (auto s : indicesShape)
      numIndices *= s;
  } else {
    numIndices = ShapedType::kDynamicSize;
  }

  SmallVector<int64_t> newIndicesShape = {1, numIndices};
  auto reshapedIndices = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(newIndicesShape, indicesType.getElementType()),
      indices, rewriter.getI64ArrayAttr(newIndicesShape));

  auto castIndices = rewriter.create<tosa::CastOp>(
      op->getLoc(),
      RankedTensorType::get(newIndicesShape, rewriter.getIntegerType(32)),
      reshapedIndices);

  SmallVector<int64_t> intermediateOutShape = {1, numIndices, weightShape[1]};
  auto gatherOp = rewriter.create<tosa::GatherOp>(
      op->getLoc(),
      RankedTensorType::get(intermediateOutShape, weightType.getElementType()),
      reshapedWeight, castIndices);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, outType, gatherOp, rewriter.getI64ArrayAttr(outType.getShape()));

  return success();
}

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
                                      Value &input, ArrayAttr &kernel,
                                      ArrayAttr &stride, ArrayAttr &pad,
                                      Type &outputTy) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented pooling input parsing function");
  }

  static int64_t getOutputDim(int64_t inputDim, int64_t kernelDim,
                              int64_t stride, int64_t padBefore,
                              int64_t padAfter, int64_t dilation) {
    if (inputDim == ShapedType::kDynamicSize) {
      return ShapedType::kDynamicSize;
    } else {
      return (
          (inputDim + padBefore + padAfter - dilation * (kernelDim - 1) - 1) /
              stride +
          1);
    }
  }

  // Apply the transposeDims vector on input to generate a transposed form.
  Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposeDims) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();

    llvm::Optional<Value> transposeDimsConst = tosa::getConstTensor<int32_t>(
        rewriter, op,
        /*vec=*/transposeDims,
        /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType =
        RankedTensorType::get(transposedInputShape, inputElemTy);
    return rewriter
        .create<tosa::TransposeOp>(op->getLoc(), transposedInputType, input,
                                   transposeDimsConst.getValue())
        .getResult();
  }

  Value transposePoolingInputToHwc(AtenOpT op,
                                   ConversionPatternRewriter &rewriter,
                                   Value input) const {
    auto inputRank =
        input.getType().template cast<RankedTensorType>().getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? chwToHwc3DTransposeDims
                                          : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(AtenOpT op,
                                    ConversionPatternRewriter &rewriter,
                                    Value input) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? hwcToChw3DTransposeDims
                                          : nhwcToNchw4DTransposeDims);
  }

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input;
    ArrayAttr kernel, stride, pad;
    Type outputTy;

    // Attempts to read input and kernel parameters, or synthesize them in the
    // case of adaptive pooling. Also performs input CHW->HWC transpose.
    if (failed(processInputs(op, adaptor, rewriter, input, kernel, stride, pad,
                             outputTy)))
      return op.emitError("Failed to process inputs for pooling");

    auto pooledOutput =
        rewriter
            .create<TosaOpT>(op->getLoc(), outputTy, input, kernel, stride, pad)
            .getResult();

    auto transposedOutput =
        ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::transposePoolingOutputToChw(
            op, rewriter, pooledOutput);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        transposedOutput);

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
                              ArrayAttr &kernel, ArrayAttr &stride,
                              ArrayAttr &pad, Type &outputTy) const override {
    auto inputXchw = adaptor.self();
    auto inputTy = inputXchw.getType().template cast<RankedTensorType>();
    if (!inputTy)
      return op.emitError("Adaptive avgpool requires ranked tensor input");

    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    // Rank sanity check.
    if (inputTy.getRank() != 4 && inputRank != 3)
      return op.emitError("NCHW->NHWC transpose requires 3D or 4D tensor");

    int64_t inputHDim = inputShape[inputRank - 2];
    int64_t inputWDim = inputShape[inputRank - 1];

    SmallVector<int64_t> outputSize;
    if (!matchPattern(op.output_size(), m_TorchConstantIntList(outputSize)))
      return rewriter.notifyMatchFailure(
          op, "Non-const output_size for adaptive pooling unsupported.");

    SmallVector<int64_t> kernelDims;
    int64_t outputHDim, outputWDim;
    if (outputSize.size() == 1) {
      outputHDim = outputWDim = outputSize[0];
    } else {
      if (outputSize.size() != 2)
        return op.emitError(
            "Adaptive avgpool output_size not 1 or 2 elements.");

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
    kernel = rewriter.getI64ArrayAttr(kernelDims);
    stride = rewriter.getI64ArrayAttr({strideH, strideW});
    // Adaptive pooling does unit dilation and zero pad.
    pad = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    outputTy = RankedTensorType::get(outputShape, inputElemTy);

    return success();
  }
};

template <typename AtenOpT, typename tosaOp>
static Type getOutputTypeForNonAdaptivePoolingOp(
    RankedTensorType inputTy, SmallVectorImpl<int64_t> &kernelSize,
    SmallVectorImpl<int64_t> &strideArray, SmallVectorImpl<int64_t> &padArray,
    SmallVectorImpl<int64_t> &dilationArray) {
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  auto inputElemTy = inputTy.getElementType();

  int64_t outputHDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 2], kernelSize[0], strideArray[0], padArray[0],
      padArray[0], dilationArray[0]);
  int64_t outputWDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 1], kernelSize[1], strideArray[1], padArray[1],
      padArray[1], dilationArray[1]);
  SmallVector<int64_t> outputShape;
  if (inputRank > 3)
    outputShape.push_back(inputShape[0]);
  outputShape.push_back(outputHDim);
  outputShape.push_back(outputWDim);
  outputShape.push_back(inputShape[inputRank - 3]);
  return RankedTensorType::get(outputShape, inputElemTy);
}

// Checks the validity of pooling parameters and stores them in the respective
// vector. Also, gets the output type for the pooling op.
template <typename AtenOpT, typename tosaOp>
static LogicalResult getOutputTypeAndPoolingParameters(
    AtenOpT op, ConversionPatternRewriter &rewriter, Value inputXchw,
    SmallVectorImpl<int64_t> &dilationArray, Type &outputTy, ArrayAttr &kernel,
    ArrayAttr &stride, ArrayAttr &pad) {

  RankedTensorType inputTy = inputXchw.getType().cast<RankedTensorType>();
  if (!inputTy)
    return op.emitError("Pooling op requires ranked tensor input");

  auto inputRank = inputTy.getRank();
  // Rank sanity check.
  if (inputTy.getRank() != 4 && inputRank != 3)
    return op.emitError("NCHW->NHWC transpose requires 3D or 4D tensor");

  SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts;
  if (!matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSizeInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const kernel_size for pooling op unsupported");
  if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const stride for pooling op unsupported");
  if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding factor for pooling op unsupported");

  kernel = rewriter.getI64ArrayAttr(kernelSizeInts);
  stride = rewriter.getI64ArrayAttr(strideInts);
  pad = rewriter.getI64ArrayAttr(
      {paddingInts[0], paddingInts[0], paddingInts[1], paddingInts[1]});

  // FIXME: add ceil_mode support.
  bool ceilMode;
  if (!matchPattern(op.ceil_mode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(
        op, "only support constant bool ceil_mode for pooling op");
  if (ceilMode)
    return rewriter.notifyMatchFailure(
        op, "only support ceil_mode equals to False for pooling op");

  outputTy = getOutputTypeForNonAdaptivePoolingOp<AtenOpT, tosaOp>(
      inputTy, kernelSizeInts, strideInts, paddingInts, dilationArray);

  return success();
}

class ConvertAtenMaxPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenMaxPool2dOp,
                                 tosa::MaxPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenMaxPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              ArrayAttr &kernel, ArrayAttr &stride,
                              ArrayAttr &pad, Type &outputTy) const override {
    SmallVector<int64_t, 2> dilationArray;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const dilation for pooling op unsupported.");
    // TOSA pooling only supports unit dilation.
    if (dilationArray[0] > 1 || dilationArray[1] > 1)
      return op.emitError("Cannot process non-unit pooling dilation.");

    if (failed(getOutputTypeAndPoolingParameters<AtenMaxPool2dOp,
                                                 tosa::MaxPool2dOp>(
            op, rewriter, adaptor.self(), dilationArray, outputTy, kernel,
            stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, adaptor.self());

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
                              ArrayAttr &kernel, ArrayAttr &stride,
                              ArrayAttr &pad, Type &outputTy) const override {
    SmallVector<int64_t, 2> dilationArray{1, 1};
    if (failed(getOutputTypeAndPoolingParameters<AtenAvgPool2dOp,
                                                 tosa::AvgPool2dOp>(
            op, rewriter, adaptor.self(), dilationArray, outputTy, kernel,
            stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, tosa::AvgPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, adaptor.self());

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

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType)
      return op.emitError("Only Tensor types supported in TOSA");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    if (!op.layout().getType().template isa<Torch::NoneType>())
      return op.emitError("Only default layout is supported");

    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return op.emitError(
          "Unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.size(), m_TorchConstantIntList(shape))) {
      return op.emitError("Shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        tosa::getConstTensor<int32_t>(rewriter, op, values, shape).getValue();

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, outType, constOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenFillScalarOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType || !outType.hasStaticShape())
      return op.emitError(
          "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }
    Value constOp;
    if (failed(torchScalarToTosaTensor(rewriter, op, op.value(), constOp,
                                       outElemTy, outType.getShape())))
      return op.emitError("Supplied value must be a Scalar constant");

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, outType, constOp);

    return success();
  }
};

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosa Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosa : public ConvertTorchToTosaBase<ConvertTorchToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, TosaOp)                            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, TosaOp>>(typeConverter,        \
                                                         context);
    INSERT_UNARY_FPONLY_PATTERN(AtenLogOp, tosa::LogOp)
    INSERT_UNARY_FPONLY_PATTERN(AtenExpOp, tosa::ExpOp)
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_UNARY_PATTERN(AtenOp, TosaOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_UNARY_PATTERN(AtenNegOp, tosa::NegateOp)
    INSERT_UNARY_PATTERN(AtenFloorOp, tosa::FloorOp)
    INSERT_UNARY_PATTERN(AtenRsqrtOp, tosa::RsqrtOp)
    INSERT_UNARY_PATTERN(AtenBitwiseNotOp, tosa::BitwiseNotOp)
    INSERT_UNARY_PATTERN(AtenCeilOp, tosa::CeilOp)
    INSERT_UNARY_PATTERN(AtenReciprocalOp, tosa::ReciprocalOp)
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_PATTERN(AtenOp, TosaOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenBinaryOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_PATTERN(AtenMaximumOp, tosa::MaximumOp)
    INSERT_BINARY_PATTERN(AtenMinimumOp, tosa::MinimumOp)
#undef INSERT_BINARY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, TosaOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, tosa::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, tosa::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, tosa::SubOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, tosa::SubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp, TosaOp)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseAndTensorOp, tosa::BitwiseAndOp)
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_MUL_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMulOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_MUL_PATTERN(AtenMulTensorOp);
    INSERT_BINARY_MUL_PATTERN(AtenMulScalarOp);
#undef INSERT_BINARY_MUL_PATTERN

#define INSERT_BINARY_DIV_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenDivOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp);
    INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp);
#undef INSERT_BINARY_DIV_PATTERN

#define INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)              \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMultipleDimsReductionOp<AtenOp, ConversionFunc>>(    \
      typeConverter, context);
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenMeanDimOp,
                                      mlir::tosa::convertReduceMeanOp)
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenSumDimIntListOp,
                                      mlir::tosa::convertReduceSumOp)
#undef INSERT_NDIMS_REDUCTION_OP_PATTERN

#define INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)             \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOneDimReductionOp<AtenOp, ConversionFunc>>(          \
      typeConverter, context);
    INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenAnyDimOp,
                                       mlir::tosa::convertReduceAnyOp)
#undef INSERT_ONEDIM_REDUCTION_OP_PATTERN

#define INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAllDimsReductionOp<AtenOp, ConversionFunc>>(         \
      typeConverter, context);
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAllOp,
                                        mlir::tosa::convertReduceAllOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAnyOp,
                                        mlir::tosa::convertReduceAnyOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenSumOp,
                                        mlir::tosa::convertReduceSumOp)
#undef INSERT_ALLDIMS_REDUCTION_OP_PATTERN

#define INSERT_SQUEEZE_OP_PATTERN(AtenOp, TemplateForm)                        \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<TemplateForm<AtenOp>>(typeConverter, context);
    INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeOp, ConvertAtenSqueezeAllDimsOp)
    INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeDimOp, ConvertAtenSqueezeOneDimOp)
#undef INSERT_SQUEEZE_OP_PATTERN

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
    INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATEMOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp)                                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
    INSERT_MM_ATENOP_PATTERN(AtenMmOp);
    INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATEMOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
    INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATEMOP_PATTERN

#define INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenOp, TosaOpT)                \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAdaptivePoolingOp<AtenOp, TosaOpT>>(typeConverter,   \
                                                              context);
    INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenAdaptiveAvgPool2dOp,
                                           tosa::AvgPool2dOp);
#undef INSERT_ADAPTIVE_POOLING_ATEMOP_PATTERN

    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);

    target.addIllegalOp<AtenAvgPool2dOp>();
    patterns.add<ConvertAtenAvgPool2dOp>(typeConverter, context);

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
    INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
    INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_FILL_SCALAR_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenFillScalarOp<AtenOp>>(typeConverter, context);
    INSERT_FILL_SCALAR_PATTERN(AtenFill_ScalarOp);
#undef INSERT_FILL_SCALAR_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenTanhOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenReluOp);
    INSERT_ATENOP_PATTERN(AtenArgmaxOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenRsubScalarOp);
    INSERT_ATENOP_PATTERN(AtenConvolutionOp);
    INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
    INSERT_ATENOP_PATTERN(AtenReshapeOp);
    INSERT_ATENOP_PATTERN(AtenBatchNormOp);
    INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
    INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
    INSERT_ATENOP_PATTERN(AtenPermuteOp);
    INSERT_ATENOP_PATTERN(AtenTransposeIntOp);
    INSERT_ATENOP_PATTERN(AtenLog2Op);
    INSERT_ATENOP_PATTERN(AtenThresholdOp);
    INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
    INSERT_ATENOP_PATTERN(AtenContiguousOp);
    INSERT_ATENOP_PATTERN(AtenDropoutOp);
    INSERT_ATENOP_PATTERN(AtenViewOp);
    INSERT_ATENOP_PATTERN(AtenGeluOp);
    INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
    INSERT_ATENOP_PATTERN(AtenEmbeddingOp);
#undef INSERT_ATENOP_PATTERN

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaPass() {
  return std::make_unique<ConvertTorchToTosa>();
}
