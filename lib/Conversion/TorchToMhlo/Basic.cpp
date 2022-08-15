//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "../PassDetail.h"
#include "./MhloLegalizeUtils.h"
#include "./PopulatePatterns.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <iostream>
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

bool skipMultiplyAlpha(Value alphaValue) {
  double doubleValue;
  auto isFloat = matchPattern(alphaValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(alphaValue, m_TorchConstantInt(&intValue));

  return ((isFloat && doubleValue == 1.0) || (isInt && intValue == 1.0));
}

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
namespace {
template <typename AtenOpT, typename MhloOpT>
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
      return op.emitError("only Tensor types supported in MHLO");

    if (selfTy.getElementType().isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<MhloOpT>(
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

// ConvertAtenUnaryConvertOp legalize genearl unary ops into Mhlo ConverOp
namespace {
template <typename AtenOpT>
class ConvertAtenUnaryConvertOp: public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self());
    return success();
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
      return op.emitError("only Tensor types supported in MHLO");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    if (!op.layout().getType().template isa<Torch::NoneType>())
      return op.emitError("only default layout is supported");

    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return op.emitError(
          "unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.size(), m_TorchConstantIntList(shape))) {
      return op.emitError("shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        mhlo::getConstTensor<int32_t>(rewriter, op, values, shape).value();

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, outType, constOp);
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
    Value lhs = adaptor.self();
    RankedTensorType lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    Value rhs = adaptor.other();
    RankedTensorType rhsType = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsType)
      return op.emitError("only Tensor types supported in MHLO");

    TensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(op.getType())
                             .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (!rhsType) {
      rhs = mhlo::scalarToMhloTensor(rewriter, op, adaptor.other(), outElemTy);
    }

    lhs = mhlo::promoteType(rewriter, lhs, outType);
    rhs = mhlo::promoteType(rewriter, rhs, outType);

    if (!skipMultiplyAlpha(op.alpha())) {
      Value alpha =
          mhlo::scalarToMhloTensor(rewriter, op, adaptor.alpha(), outElemTy);
      DenseIntElementsAttr bcastDimensions;
      rhs = rewriter.create<chlo::BroadcastMulOp>(op->getLoc(), rhs, alpha,
                                                  bcastDimensions);
    }

    DenseIntElementsAttr bcastDimensions;
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
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    TensorType rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("only Tensor types supported in MHLO");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhs = lhs;
    } else if (!rhsType) {
      rhs = mhlo::scalarToMhloTensor(rewriter, op, adaptor.other(), outElemTy);
    }
    DenseIntElementsAttr bcastDimensions;
    lhs = mhlo::promoteType(rewriter, lhs, outType);
    rhs = mhlo::promoteType(rewriter, rhs, outType);
    auto loc = op.getLoc();
    Value result =
        rewriter.create<ChloOpT>(loc, outType, lhs, rhs, bcastDimensions);

    if (!isa<AtenDivTensorModeOp>(op)) {
      rewriter.replaceOp(op, result);
      return success();
    }

    AtenDivTensorModeOp divTensorModeOp =
        llvm::dyn_cast<AtenDivTensorModeOp>(op.getOperation());
    std::string roundingMode;
    if (!matchPattern(divTensorModeOp.rounding_mode(),
                      m_TorchConstantStr(roundingMode)))
      return rewriter.notifyMatchFailure(
          op, "only support constant str rounding mode");

    if (roundingMode == "trunc") {
      // "trunc" - rounds the results of the division towards zero. Equivalent
      // to C-style integer division.
      auto sign = rewriter.create<mhlo::SignOp>(loc, result);
      auto abs = rewriter.create<mhlo::AbsOp>(loc, result);
      auto floor = rewriter.create<mhlo::FloorOp>(loc, abs);
      result = rewriter.create<mhlo::MulOp>(loc, sign, floor).getResult();
    }
    if (roundingMode == "floor") {
      // "floor" - rounds the results of the division down. Equivalent to
      // floor division in Python (the // operator)
      result = rewriter.create<mhlo::FloorOp>(loc, result).getResult();
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
    Value lhs = adaptor.self();
    Value rhs = adaptor.other();
    RankedTensorType lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhsTy = rhs.getType().dyn_cast<RankedTensorType>();

    if (!lhsTy)
      return op.emitError("only Tensor types supported in MHLO");

    RankedTensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                                   ->convertType(op.getType())
                                   .template cast<RankedTensorType>();

    Type lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (!rhsTy) {
      rhs = mhlo::scalarToMhloTensor(rewriter, op, adaptor.other(), lhsElemTy);
    }

    // TODO: what is the PyTorch default type promotion?
    rhs = mhlo::promoteType(rewriter, rhs, lhsTy);

    mhlo::ComparisonTypeAttr compareTypeAttr;
    mhlo::ComparisonDirectionAttr compareDirectionAttr;

    if (lhsElemTy.isa<mlir::FloatType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          op->getContext(), mhlo::ComparisonType::FLOAT);
    } else if (lhsElemTy.isa<mlir::IntegerType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          op->getContext(), mhlo::ComparisonType::SIGNED);
    }

    if (std::is_same<AtenOpT, AtenLtTensorOp>() ||
        std::is_same<AtenOpT, AtenLtScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::LT);
    } else if (std::is_same<AtenOpT, AtenGtTensorOp>() ||
               std::is_same<AtenOpT, AtenGtScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::GT);
    } else if (std::is_same<AtenOpT, AtenGeScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::GE);
    } else if (std::is_same<AtenOpT, AtenEqTensorOp>() ||
               std::is_same<AtenOpT, AtenEqScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::EQ);
    } else if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
               std::is_same<AtenOpT, AtenNeScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::NE);
    }
    DenseIntElementsAttr bcastDimensions;
    rewriter.replaceOpWithNewOp<chlo::BroadcastCompareOp>(
        op, outType, lhs, rhs, bcastDimensions, compareDirectionAttr,
        compareTypeAttr);
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
    Value self = adaptor.self();
    int64_t dim0;
    if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0))) {
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    }
    int64_t dim1;
    if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1))) {
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");
    }

    auto inType = self.getType().cast<RankedTensorType>();
    auto inputRank = inType.getRank();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

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
    DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(permValues.size())},
                              rewriter.getI64Type()),
        permValues);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(op, outType, self,
                                                   permutation);
    return success();
  }
};
} // namespace

// AtenBroadcastToOp
namespace {
class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<RankedTensorType>();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

#ifdef TORCH_MLIR_ENABLE_MHLO_STATIC_SHAPE
    if (selfTy.hasStaticShape()) {
      Value bcastOp = mhlo::promoteAndBroadcast(rewriter, self, outType);
      rewriter.replaceOp(op, bcastOp);
      return success();
    }
#endif

    SmallVector<Value> shape;
    if (!(getListConstructElements(adaptor.size(), shape))) {
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
      if (!(matchPattern(dValue, m_TorchConstantInt(&dInt)))) {
        return op->emitError("element of desired shape must be a scalar");
      }
      if (i >= leadingRank && dInt == -1) {
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

#ifdef TORCH_MLIR_ENABLE_MHLO_TRUNC_DIMSIZE_TO_I32
    for (auto &dsize : bcastShapeVec) {
      auto dsizeI64 = rewriter.create<mlir::arith::IndexCastOp>(
          op->getLoc(), rewriter.getI64Type(), dsize);
      dsize = rewriter.create<arith::TruncIOp>(op->getLoc(),
                                               rewriter.getI32Type(), dsizeI64);
    }
#endif

    Value bcastShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), ValueRange{bcastShapeVec});
    auto dimensionNumbers =
        llvm::to_vector<4>(llvm::seq<int64_t>(leadingRank, totalRank));
    rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op, outType, self, bcastShapeTensor,
        rewriter.getI64TensorAttr(dimensionNumbers));
    return success();
  }
};
} // namespace

// AtenPermuteOp
namespace {
class ConvertAtenPermuteOp : public OpConversionPattern<AtenPermuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenPermuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.self();
    // Not a ranked tensor type
    auto inType = self.getType().dyn_cast<RankedTensorType>();
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();
    if (!inType)
      return op.emitError("only ranked tensor types with static shapes are "
                          "currently supported");

    SmallVector<int64_t> permValues;
    if (!matchPattern(adaptor.dims(), m_TorchConstantIntList(permValues)))
      return rewriter.notifyMatchFailure(
          op, "only constant dimensions are currently supported");

    int64_t inRank = inType.getRank();
    for (auto &d : permValues) {
      d = toPositiveDim(d, inRank);
      if (!isValidDim(d, inRank))
        return op.emitError("not all dims are valid");
    }

    DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(permValues.size())},
                              rewriter.getI64Type()),
        permValues);
    rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(op, outType, self,
                                                   permutation);
    return success();
  }
};

} // namespace

namespace {
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

// AtenTanhOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<mhlo::TanhOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  } else {
    return op.emitError(
        "only floating-point datatype legalization currently supported");
  }
}
} // namespace

// ValueTensorLiteralOp
namespace {
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType resultType = getTypeConverter()
                                    ->convertType(op->getResult(0).getType())
                                    .cast<RankedTensorType>();

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  // TODO: what about unsigned integer?
  if (auto elements = op.valueAttr().dyn_cast<DenseIntElementsAttr>()) {
    Type builtinTensorElemTy = resultType.getElementType();
    unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();

    DenseElementsAttr valueAttr =
        elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
          return APInt(bitWidth, v.getSExtValue());
        });
    rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, resultType, valueAttr);
    return success();
  }

  rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, resultType,
                                                adaptor.value());
  return success();
}

} // namespace

// AtenReciprocalOp
// Reciprocal(x) = Div(1, x)
namespace {
template <>
LogicalResult ConvertAtenOp<AtenReciprocalOp>::matchAndRewrite(
    AtenReciprocalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto outTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  if (!inputTy.getElementType().isa<mlir::FloatType>()) {
    return op.emitError("only floating-point datatype legalization supported "
                        "for AtenReciprocalOp");
  }

  Value oneTensor = chlo::getConstantLike(rewriter, op->getLoc(), 1, input);
  rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, outTy, oneTensor, input);
  return success();
}
} // namespace

// PrimNumToTensorScalarOp
namespace {
template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  RankedTensorType outputType = getTypeConverter()
                                    ->convertType(op->getResult(0).getType())
                                    .cast<RankedTensorType>();
  auto outputElemType = outputType.getElementType();
  Value mhloTensor =
      mhlo::scalarToMhloTensor(rewriter, op, adaptor.a(), outputElemType);
  rewriter.replaceOp(op, mhloTensor);
  return success();
}
} // namespace

// AtenContiguousOp
// Ref: TosaToTosa.cpp for implementation details
namespace {
template <>
LogicalResult ConvertAtenOp<AtenContiguousOp>::matchAndRewrite(
    AtenContiguousOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("only tensor types are currently supported");

  // FIXME: memory_format is not handled.

  rewriter.replaceOp(op, adaptor.self());

  return success();
}

} // namespace

// AtenReluOp
// Relu(x) = Max(0, x)
namespace {
template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.self();
  auto lhsTy = lhs.getType().cast<RankedTensorType>();
  auto lhsElemTy = lhsTy.getElementType();

  if (!lhsElemTy.isa<mlir::FloatType>()) {
    return op->emitError("only float tensor in relu op is supported");
  }

  Value zeroTensor;
  zeroTensor = chlo::getConstantLike(
      rewriter, op->getLoc(),
      APFloat::getZero(lhsElemTy.cast<mlir::FloatType>().getFloatSemantics(),
                       false),
      lhs);
  rewriter.replaceOpWithNewOp<mhlo::MaxOp>(op, lhs, zeroTensor);
  return success();
}

} // namespace

// Convert a Aten::GELU to HLO
// Gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
namespace {
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("only ranked tensor type is supported.");
  }

  Value one = chlo::getConstantLike(rewriter, loc, 1.0, input);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, input);
  Value half = chlo::getConstantLike(rewriter, loc, 0.5, input);
  auto rsqrtTwo = rewriter.create<mlir::mhlo::RsqrtOp>(loc, two);
  auto erfElement = rewriter.create<mhlo::MulOp>(loc, input, rsqrtTwo);
  auto erf = rewriter.create<mlir::chlo::ErfOp>(loc, erfElement);
  auto erfAdd = rewriter.create<mhlo::AddOp>(loc, erf, one);
  auto halfMul = rewriter.create<mhlo::MulOp>(loc, erfAdd, half);
  rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, input, halfMul);
  return success();
}
} // namespace

// AtenErfOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenErfOp>::matchAndRewrite(
    AtenErfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.self();
  auto inputType = input.getType().cast<TensorType>();
  if (!inputType.getElementType().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(op, "only float tensor is supported");
  }
  rewriter.replaceOpWithNewOp<chlo::ErfOp>(
      op, getTypeConverter()->convertType(op.getType()), input);
  return success();
}

} // namespace

// AtenBatchNormOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.input();
  // shape = [N, C, H, W]
  auto inputTy = input.getType().cast<RankedTensorType>();
  Value weight = adaptor.weight();
  Value bias = adaptor.bias();
  Value runningMean = adaptor.running_mean();
  Value runningVar = adaptor.running_var();
  // momentum is ignored
  Value momentum = adaptor.momentum();
  (void)momentum;

  if (inputTy.getRank() <= 2) {
    return rewriter.notifyMatchFailure(op,
                                       "input should have rank larger than 2");
  }
  if (!inputTy.getElementType().template isa<mlir::FloatType>()) {
    return op.emitError("only input tensor of float type is supported");
  }
  auto inputElemTy = inputTy.getElementType().cast<mlir::FloatType>();

  Value channelDim = rewriter.create<tensor::DimOp>(op->getLoc(), input, 1);

#ifdef TORCH_MLIR_ENABLE_MHLO_TRUNC_DIMSIZE_TO_I32
  auto channelDimI64 = rewriter.create<mlir::arith::IndexCastOp>(
      op->getLoc(), rewriter.getI64Type(), channelDim);
  channelDim = rewriter.create<arith::TruncIOp>(
      op->getLoc(), rewriter.getI32Type(), channelDimI64);
#endif

  Value channelShape = rewriter.create<tensor::FromElementsOp>(
      op->getLoc(), ValueRange{channelDim});
  if (failed(checkNotNone(rewriter, op, weight))) {
    weight = mhlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 1),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, bias))) {
    bias = mhlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 0),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, runningVar))) {
    runningVar = mhlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 1),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }
  if (failed(checkNotNone(rewriter, op, runningMean))) {
    runningMean = mhlo::getConstantOfShape(
        rewriter, op->getLoc(), APFloat(inputElemTy.getFloatSemantics(), 0),
        channelShape,
        RankedTensorType::get({inputTy.getShape()[1]},
                              inputTy.getElementType()));
  }

  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto biasTy = bias.getType().cast<RankedTensorType>();
  auto runningMeanTy = runningMean.getType().cast<RankedTensorType>();
  auto runningVarTy = runningVar.getType().cast<RankedTensorType>();

  if (weightTy.getRank() != 1 || biasTy.getRank() != 1 ||
      runningMeanTy.getRank() != 1 || runningVarTy.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "expect weight, bias, running_mean and running_var to be rank 1");
  }
  if (!weightTy.getElementType().template isa<mlir::FloatType>() ||
      !biasTy.getElementType().template isa<mlir::FloatType>() ||
      !runningMeanTy.getElementType().template isa<mlir::FloatType>() ||
      !runningVarTy.getElementType().template isa<mlir::FloatType>()) {
    return op.emitError("only float weight/bias/runningMean/runningVar tensor "
                        "of float type is supported");
  }

  double eps = 0.0;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op, "non-float(double) eps unsupported");
  }
  bool training = false;
  if (!matchPattern(op.training(), m_TorchConstantBool(&training))) {
    return rewriter.notifyMatchFailure(op, "non-bool training unsupported");
  }
  // TODO: handle cudnnEnabled parameter. Here, we just ignore it!
  bool cudnnEnabled = false;
  if (!matchPattern(op.cudnn_enabled(), m_TorchConstantBool(&cudnnEnabled))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-bool cudnn_enabled unsupported");
  }
  if (training) {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    Type batchMeanOrVarTy =
        RankedTensorType::get(weightTy.getShape(), inputTy.getElementType());
    auto batchNormTrainingResult = rewriter.create<mhlo::BatchNormTrainingOp>(
        op.getLoc(), outputTy, batchMeanOrVarTy, batchMeanOrVarTy, input,
        weight, bias, rewriter.getF32FloatAttr(eps),
        rewriter.getI64IntegerAttr(1));
    rewriter.replaceOp(op, batchNormTrainingResult.getResult(0));
    return success();
  } else {
    Type outputTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mhlo::BatchNormInferenceOp>(
        op, outputTy, input, weight, bias, runningMean, runningVar,
        rewriter.getFloatAttr(inputTy.getElementType(), eps),
        rewriter.getI64IntegerAttr(1));
    return success();
  }
}

} // namespace

// AtenNativeLayerNormOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenNativeLayerNormOp>::matchAndRewrite(
    AtenNativeLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.input();
  auto inputTy = input.getType().cast<RankedTensorType>();
  auto inputShape = inputTy.getShape();
  auto inputRank = inputTy.getRank();
  Value weight = adaptor.weight();
  Value bias = adaptor.bias();

  if (!inputTy.hasStaticShape()) {
    return op->emitError("dynamic shaped input is not supported");
  }

  SmallVector<int64_t> normalizedShape;
  if (!matchPattern(op.normalized_shape(),
                    m_TorchConstantIntList(normalizedShape))) {
    return rewriter.notifyMatchFailure(
        op, "normalized_shape must be a list of const int");
  }
  double eps = 0;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op,
                                       "non const float eps is unsupported");
  }
  if (failed(checkNotNone(rewriter, op, weight)) ||
      failed(checkNotNone(rewriter, op, bias))) {
    return op->emitError("none weight or bias is unsupported");
  }
  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto biasTy = bias.getType().cast<RankedTensorType>();

  if (!inputTy.getElementType().isa<mlir::FloatType>() ||
      !biasTy.getElementType().isa<mlir::FloatType>() ||
      !weightTy.getElementType().isa<mlir::FloatType>()) {
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
  SmallVector<int64_t> meanOrVarMhloOutShape{numFeatureDimSize};

  auto mhloBatchNormOutTy =
      RankedTensorType::get(inputFlattenShape, inputTy.getElementType());
  auto mhloBathNormOutMeanOrVarTy =
      RankedTensorType::get(meanOrVarMhloOutShape, inputTy.getElementType());

  // Reshape input
  auto mhloInput = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), mhloBatchNormOutTy, input,
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(inputFlattenShape),
                           {static_cast<int64_t>(inputFlattenShape.size())})
          .value());

  // Generate "scale" and "offset" Value for mhlo.BatchNormTrainingOp.
  SmallVector<APFloat> zeroConstVec(
      numFeatureDimSize, APFloat::getZero(inputTy.getElementType()
                                              .cast<mlir::FloatType>()
                                              .getFloatSemantics()));
  SmallVector<APFloat> oneConstVec(
      numFeatureDimSize,
      APFloat(
          inputTy.getElementType().cast<mlir::FloatType>().getFloatSemantics(),
          1));
  auto oneOrZeroConstType =
      RankedTensorType::get({numFeatureDimSize}, inputTy.getElementType());

  Value scale = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), oneOrZeroConstType,
      DenseElementsAttr::get(oneOrZeroConstType, oneConstVec));
  Value offset = rewriter.create<mhlo::ConstantOp>(
      op->getLoc(), oneOrZeroConstType,
      DenseElementsAttr::get(oneOrZeroConstType, zeroConstVec));
  auto batchNormTrainingResult = rewriter.create<mhlo::BatchNormTrainingOp>(
      op->getLoc(), mhloBatchNormOutTy, mhloBathNormOutMeanOrVarTy,
      mhloBathNormOutMeanOrVarTy, mhloInput, scale, offset,
      rewriter.getF32FloatAttr(eps), rewriter.getI64IntegerAttr(1));

  // Reshape back
  auto outputTy =
      getTypeConverter()->convertType(op.getType(0)).cast<RankedTensorType>();
  auto outputMeanOrVarTy =
      getTypeConverter()->convertType(op.getType(1)).cast<RankedTensorType>();

  auto output = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputTy, batchNormTrainingResult.getResult(0),
      mhlo::getConstTensor(rewriter, op, outputTy.getShape(),
                           {static_cast<int64_t>(outputTy.getShape().size())})
          .value());
  auto mean = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(1),
      mhlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .value());
  auto var = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(2),
      mhlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .value());

  // Apply affine transform: output x weight + bias [element-wise]
  auto bcastedWeight = mhlo::promoteAndBroadcast(rewriter, weight, outputTy);
  auto bcastedBias = mhlo::promoteAndBroadcast(rewriter, bias, outputTy);
  auto outputMulWeight =
      rewriter.create<mhlo::MulOp>(op->getLoc(), output, bcastedWeight);
  auto finalOuput =
      rewriter.create<mhlo::AddOp>(op->getLoc(), outputMulWeight, bcastedBias);
  rewriter.replaceOp(op, {finalOuput, mean, var});
  return success();
}

} // namespace

// AtenCatOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenCatOp>::matchAndRewrite(
    AtenCatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto outType =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op,
                                       "only constant dim param is supported");
  }

  SmallVector<Value> torchTensors;
  if (!getListConstructElements(op.tensors(), torchTensors)) {
    return rewriter.notifyMatchFailure(
        op, "input should comes from a PrimListConstructOp");
  }
  SmallVector<Value> builtinTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), torchTensors);

  // Promote type
  for (auto &v : builtinTensors) {
    v = mhlo::promoteType(rewriter, v, outType);
  }

  rewriter.replaceOpWithNewOp<mhlo::ConcatenateOp>(
      op, ValueRange(builtinTensors), static_cast<uint64_t>(dim));
  return success();
}
} // namespace

// AtenSizeIntOp
namespace {
template <>
LogicalResult ConvertAtenOp<AtenSizeIntOp>::matchAndRewrite(
    AtenSizeIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");
  auto dim = rewriter.create<arith::IndexCastOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.dim());
  auto dimSize = rewriter.create<tensor::DimOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.self(), dim);

  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      op, getTypeConverter()->convertType(op.getType()), dimSize);

  return success();
}
} // namespace

// ValsemVariantAtenUniformOp
namespace {
template <>
LogicalResult ConvertAtenOp<ValsemVariantAtenUniformOp>::matchAndRewrite(
    ValsemVariantAtenUniformOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto inputTy = adaptor.self().getType().template cast<RankedTensorType>();
  auto loc = op.getLoc();
  if (!inputTy) {
    op.emitError("input should be ranked tensor type.");
  }
  auto definingOp = op.self().getDefiningOp();
  auto shape = definingOp->getOperand(0);
  SmallVector<Value, 4> dimSizes;
  getListConstructElements(shape, dimSizes);
  std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
    dSize = rewriter.create<torch::TorchConversion::ToI64Op>(loc, dSize).getResult();
    return dSize;
  });

  auto mhloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), dimSizes);

  double fromDoubleValue, toDoubleValue;
  if (!matchPattern(op.from(), m_TorchConstantFloat(&fromDoubleValue))) {
    op.emitError("operand #1 should be scalar");
  }
  if (!matchPattern(op.to(), m_TorchConstantFloat(&toDoubleValue))) {
    op.emitError("operand #2 should be scalar");
  }
  Value fromTensor = rewriter.create<mhlo::ConstantOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), fromDoubleValue));
  Value toTensor = rewriter.create<mhlo::ConstantOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), toDoubleValue));

  auto outType = getTypeConverter()
                     ->convertType(op.getType())
                     .template dyn_cast<TensorType>();
  rewriter.replaceOpWithNewOp<mhlo::RngOp>(
      op, inputTy, fromTensor, toTensor, mhloShape, mhlo::RngDistribution::UNIFORM);
  return success();
}
}
void mlir::torch::torch_to_mhlo::populateBasicOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenTransposeIntOp>();
  patterns.add<ConvertAtenTransposeIntOp>(typeConverter, context);

  target.addIllegalOp<AtenBroadcastToOp>();
  patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);

  target.addIllegalOp<AtenPermuteOp>();
  patterns.add<ConvertAtenPermuteOp>(typeConverter, context);

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, MhloOp)                            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, MhloOp>>(typeConverter,        \
                                                         context);
  INSERT_UNARY_FPONLY_PATTERN(AtenLogOp, mhlo::LogOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenExpOp, mhlo::ExpOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenCloneOp, mhlo::CopyOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenSqrtOp, mhlo::SqrtOp);
  INSERT_UNARY_FPONLY_PATTERN(AtenNegOp, mhlo::NegOp);
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_UNARY_CONVERT_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                              \
  patterns.add<ConvertAtenUnaryConvertOp<AtenOp>>(typeConverter,              \
                                                  context);
  INSERT_UNARY_CONVERT_PATTERN(AtenContiguousOp);
  INSERT_UNARY_CONVERT_PATTERN(AtenToDtypeOp);
  INSERT_UNARY_CONVERT_PATTERN(AtenTypeAsOp);
#undef INSERT_UNARY_CONVERT_PATTERN

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
  INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
  INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, ChloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, ChloOp>>(typeConverter, context);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, chlo::BroadcastAddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, chlo::BroadcastAddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, chlo::BroadcastSubOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, chlo::BroadcastSubOp);
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_MULDIV_PATTERN(AtenOp, ChloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMulDivOp<AtenOp, ChloOp>>(typeConverter, context);
  INSERT_BINARY_MULDIV_PATTERN(AtenMulTensorOp, chlo::BroadcastMulOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenMulScalarOp, chlo::BroadcastMulOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivTensorOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivTensorModeOp, chlo::BroadcastDivOp);
  INSERT_BINARY_MULDIV_PATTERN(AtenDivScalarOp, chlo::BroadcastDivOp);
#undef INSERT_BINARY_MULDIV_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp>>(typeConverter, context);

  INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGeScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp);
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
  INSERT_ATENOP_PATTERN(AtenTanhOp);
  INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATENOP_PATTERN(AtenReciprocalOp);
  INSERT_ATENOP_PATTERN(PrimNumToTensorScalarOp);
  INSERT_ATENOP_PATTERN(AtenContiguousOp);

  INSERT_ATENOP_PATTERN(AtenReluOp);
  INSERT_ATENOP_PATTERN(AtenGeluOp);
  INSERT_ATENOP_PATTERN(AtenErfOp);

  INSERT_ATENOP_PATTERN(AtenCatOp);

  INSERT_ATENOP_PATTERN(AtenBatchNormOp);
  INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
  INSERT_ATENOP_PATTERN(AtenSizeIntOp);
  INSERT_ATENOP_PATTERN(ValsemVariantAtenUniformOp);
#undef INSERT_ATENOP_PATTERN
}
