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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
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
      return op.emitError("Only Tensor types supported in MHLO");

    if (selfTy.getElementType().isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<MhloOpT>(
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
      return op.emitError("Only Tensor types supported in MHLO");

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
        mhlo::getConstTensor<int32_t>(rewriter, op, values, shape).getValue();

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, outType, constOp);
    return success();
  }
};

} // namespace

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
namespace {
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    TensorType lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    TensorType rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    TensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(op.getType())
                             .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsType) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsAsTensor, outElemTy,
                                               outType.getShape())))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
    }
    Value lhsTensor = lhs;
    Value rhsTensor = rhsType ? rhs : rhsAsTensor;

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    // Handle alpha.
    Value multTensor;
    if (skipMultiplyAlpha(op.alpha())) {
      multTensor = rhsTensor;
    } else {
      Value alphaTensor;
      if (failed(mhlo::torchAlphaToMhloTensor(rewriter, op.getOperation(),
                                              op.alpha(), alphaTensor,
                                              outElemTy, outType.getShape(),
                                              /*checkForUnity=*/false))) {
        return op.emitError("Currently only scalar constants are supported for "
                            "alpha in conversion to MHLO operation");
      }

      multTensor = rewriter.create<mhlo::MulOp>(op.getLoc(), outType, rhsTensor,
                                                alphaTensor);
    }

    rewriter.replaceOpWithNewOp<MhloOpT>(op, outType, lhsTensor, multTensor);
    return success();
  }
};
} // namespace

// Binary op legalizations for Mul variants.
namespace {
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
    Value rhs = adaptor.other();
    TensorType rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value lhsTensor = lhs;
    Value rhsTensor;
    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      if (!rhsType) {
        if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                                 rhsTensor, outElemTy,
                                                 outType.getShape())))
          return op.emitError(
              "Currently only scalar constants are supported for "
              "conversion in MHLO operation");
      } else {
        rhsTensor = rhs;
      }
    }

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, outType, lhsTensor, rhsTensor);
    return success();
  }
};
} // namespace

// Binary op legalizations for Div variants.
namespace {
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
      return op.emitError("Only Tensor types supported.");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();
    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value lhsTensor = lhs;
    Value rhsTensor;
    if (!rhsTy) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsTensor, outElemTy,
                                               outType.getShape())))
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
    } else {
      rhsTensor = rhs;
    }

    // Handle broadcasting. Since we have the output type already, here we
    // just broodcast operands' shape to output shape.
    lhsTensor = mhlo::promoteAndBroadcast(rewriter, lhsTensor, outType);
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, outType);

    rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, outType, lhsTensor, rhsTensor);
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
      return op.emitError("Only Tensor types supported in MHLO");

    RankedTensorType outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                                   ->convertType(op.getType())
                                   .template cast<RankedTensorType>();

    Type lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.other(),
                                               rhsAsTensor, lhsElemTy, {}))) {
        return op.emitError("Currently only scalar constants are supported for "
                            "conversion in MHLO operation");
      }
    }

    Value lhsTensor = lhs;
    Value rhsTensor = rhsTy ? rhs : rhsAsTensor;
    rhsTensor = mhlo::promoteAndBroadcast(rewriter, rhsTensor, lhsTy);

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
    } else if (std::is_same<AtenOpT, AtenEqTensorOp>() ||
               std::is_same<AtenOpT, AtenEqScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::EQ);
    } else if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
               std::is_same<AtenOpT, AtenNeScalarOp>()) {
      compareDirectionAttr = mhlo::ComparisonDirectionAttr::get(
          op->getContext(), mhlo::ComparisonDirection::NE);
    }

    rewriter.replaceOpWithNewOp<mhlo::CompareOp>(
        op, outType, lhsTensor, rhsTensor, compareDirectionAttr,
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
    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

    Value bcastOp = mhlo::promoteAndBroadcast(rewriter, self, outType);
    rewriter.replaceOp(op, bcastOp);
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
      return op.emitError("Only ranked tensor types with static shapes are "
                          "currently supported");

    SmallVector<int64_t> permValues;
    if (!matchPattern(adaptor.dims(), m_TorchConstantIntList(permValues)))
      return rewriter.notifyMatchFailure(
          op, "Only constant dimensions are currently supported");

    int64_t inRank = inType.getRank();
    for (auto &d : permValues) {
      d = toPositiveDim(d, inRank);
      if (!isValidDim(d, inRank))
        return op.emitError("Not all dims are valid");
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
        "Only floating-point datatype legalization currently supported");
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
    return op.emitError("Only floating-point datatype legalization supported "
                        "for AtenReciprocalOp");
  }
  Value oneTensor =
      mhlo::getConstTensor<float>(rewriter, op, {static_cast<float>(1.0)}, {})
          .getValue();
  oneTensor = mhlo::promoteAndBroadcast(rewriter, oneTensor, inputTy);
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
  auto outputShape = outputType.getShape();
  auto outputElemType = outputType.getElementType();
  Value mhloTensor;
  if (failed(mhlo::torchScalarToMhloTensor(rewriter, op, op.a(), mhloTensor,
                                           outputElemType, outputShape,
                                           false))) {
    return op->emitError("Failed lowering PrimNumToTensorScalarOp to MHLO");
  }
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
    return op.emitError("Only tensor types are currently supported");

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

  int64_t lhsSize = 1;
  for (auto &en : llvm::enumerate(lhsTy.getShape())) {
    lhsSize *= en.value();
  }
  auto constTy = RankedTensorType::get(lhsTy.getShape(), lhsElemTy);
  DenseElementsAttr constAttr;
  if (lhsElemTy.isa<mlir::FloatType>()) {
    std::vector<APFloat> constVec(
        lhsSize,
        APFloat::getZero(lhsElemTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false));
    constAttr = DenseElementsAttr::get(constTy, constVec);
  } else if (lhsElemTy.isa<mlir::IntegerType>()) {
    std::vector<APInt> constVec(
        lhsSize, APInt::getZero(lhsElemTy.getIntOrFloatBitWidth()));
    constAttr = DenseElementsAttr::get(constTy, constVec);
  }
  Value rhs =
      rewriter.create<mhlo::ConstantOp>(op.getLoc(), constTy, constAttr);

  rewriter.replaceOpWithNewOp<mhlo::MaxOp>(op, lhs, rhs);
  return success();
}

} // namespace

// Convert a Aten::GELU to HLO
// Gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
namespace {
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
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
  auto inputType = input.getType().cast<RankedTensorType>();
  if (!inputType.getElementType().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Only support float data type");
  }
  auto outType =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

  // Using:
  // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations with
  // maximum error as 5 x 10^-4 where a1 = 0.278393, a2 = 0.230389, a3 =
  // 0.000972, a4 = 0.078108.
  // Erf = 1 - 1 / (1 + a1X + a2X^2 + a3X^3 + a4X^4)^4

  auto loc = op->getLoc();
  auto zeroConst =
      mhlo::getConstTensor<float>(rewriter, op, {0.0}, {}).getValue();
  auto zero = mhlo::promoteAndBroadcast(rewriter, zeroConst, outType);
  auto oneConst =
      mhlo::getConstTensor<float>(rewriter, op, {1.0}, {}).getValue();
  auto one = mhlo::promoteAndBroadcast(rewriter, oneConst, outType);
  auto a1Const =
      mhlo::getConstTensor<float>(rewriter, op, {0.278393}, {}).getValue();
  auto a1 = mhlo::promoteAndBroadcast(rewriter, a1Const, outType);
  auto a2Const =
      mhlo::getConstTensor<float>(rewriter, op, {0.230389}, {}).getValue();
  auto a2 = mhlo::promoteAndBroadcast(rewriter, a2Const, outType);
  auto a3Const =
      mhlo::getConstTensor<float>(rewriter, op, {0.000972}, {}).getValue();
  auto a3 = mhlo::promoteAndBroadcast(rewriter, a3Const, outType);
  auto a4Const =
      mhlo::getConstTensor<float>(rewriter, op, {0.078108}, {}).getValue();
  auto a4 = mhlo::promoteAndBroadcast(rewriter, a4Const, outType);

  auto absX = rewriter.create<mhlo::AbsOp>(loc, outType, input);
  auto a1X = rewriter.create<mhlo::MulOp>(loc, outType, a1, absX);
  auto sum = rewriter.create<mhlo::AddOp>(loc, outType, a1X, one);

  auto x2 = rewriter.create<mhlo::MulOp>(loc, outType, absX, absX);
  auto a2X = rewriter.create<mhlo::MulOp>(loc, outType, a2, x2);
  sum = rewriter.create<mhlo::AddOp>(loc, outType, sum, a2X);

  auto x3 = rewriter.create<mhlo::MulOp>(loc, outType, x2, absX);
  auto a3X = rewriter.create<mhlo::MulOp>(loc, outType, a3, x3);
  sum = rewriter.create<mhlo::AddOp>(loc, outType, sum, a3X);

  auto x4 = rewriter.create<mhlo::MulOp>(loc, outType, x3, absX);
  auto a4X = rewriter.create<mhlo::MulOp>(loc, outType, a4, x4);
  sum = rewriter.create<mhlo::AddOp>(loc, outType, sum, a4X);

  auto rcprl = rewriter.create<mhlo::DivOp>(loc, outType, one, sum);
  auto rcprl2 = rewriter.create<mhlo::MulOp>(loc, outType, rcprl, rcprl);
  auto rcprl4 = rewriter.create<mhlo::MulOp>(loc, outType, rcprl2, rcprl2);
  auto erf = rewriter.create<mhlo::SubtractOp>(loc, outType, one, rcprl4);

  // Deal with negative x.
  mhlo::ComparisonDirectionAttr compareDirectionAttr =
      mhlo::ComparisonDirectionAttr::get(op->getContext(),
                                         mhlo::ComparisonDirection::GE);
  mhlo::ComparisonTypeAttr compareTypeAttr = mhlo::ComparisonTypeAttr::get(
      op->getContext(), mhlo::ComparisonType::FLOAT);
  auto geZero = rewriter.create<mhlo::CompareOp>(
      loc, RankedTensorType::get(outType.getShape(), rewriter.getI1Type()),
      input, zero, compareDirectionAttr, compareTypeAttr);
  auto negaErf = rewriter.create<mhlo::NegOp>(loc, erf);
  rewriter.replaceOpWithNewOp<mhlo::SelectOp>(op, outType, geZero, erf,
                                              negaErf);
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

  // init weight, bias, runningVar, runningMean if they are none
  auto initNoneValue = [&](Value &input, bool zero) {
    SmallVector<APFloat> constVec(inputTy.getShape()[1],
                                  APFloat::getZero(inputTy.getElementType()
                                                       .cast<mlir::FloatType>()
                                                       .getFloatSemantics()));
    if (!zero) {
      for (auto &item : constVec) {
        item = APFloat(inputTy.getElementType()
                           .cast<mlir::FloatType>()
                           .getFloatSemantics(),
                       1);
      }
    }
    auto constType = RankedTensorType::get({inputTy.getShape()[1]},
                                           inputTy.getElementType());
    auto constAttr = DenseElementsAttr::get(constType, constVec);
    input =
        rewriter.create<mhlo::ConstantOp>(op.getLoc(), constType, constAttr);
  };
  if (failed(checkNotNone(rewriter, op, weight))) {
    initNoneValue(weight, false);
  }
  if (failed(checkNotNone(rewriter, op, bias))) {
    initNoneValue(bias, true);
  }
  if (failed(checkNotNone(rewriter, op, runningVar))) {
    initNoneValue(runningVar, false);
  }
  if (failed(checkNotNone(rewriter, op, runningMean))) {
    initNoneValue(runningMean, true);
  }

  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto biasTy = bias.getType().cast<RankedTensorType>();
  auto runningMeanTy = runningMean.getType().cast<RankedTensorType>();
  auto runningVarTy = runningVar.getType().cast<RankedTensorType>();
  if (inputTy.getRank() <= 2) {
    return rewriter.notifyMatchFailure(op,
                                       "input should have rank larger than 2");
  }
  if (weightTy.getRank() != 1 || biasTy.getRank() != 1 ||
      runningMeanTy.getRank() != 1 || runningVarTy.getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "expect weight, bias, running_mean and running_var to be rank 1");
  }
  if (!inputTy.getElementType().template isa<mlir::FloatType>() ||
      !weightTy.getElementType().template isa<mlir::FloatType>() ||
      !biasTy.getElementType().template isa<mlir::FloatType>() ||
      !runningMeanTy.getElementType().template isa<mlir::FloatType>() ||
      !runningVarTy.getElementType().template isa<mlir::FloatType>()) {
    return op.emitError(
        "Only float element type is supported in MHLO BatchNormOp");
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

  SmallVector<int64_t> normalizedShape;
  if (!matchPattern(op.normalized_shape(),
                    m_TorchConstantIntList(normalizedShape))) {
    return rewriter.notifyMatchFailure(
        op, "normalized_shape must be a list of const int");
  }
  double eps = 0;
  if (!matchPattern(op.eps(), m_TorchConstantFloat(&eps))) {
    return rewriter.notifyMatchFailure(op, "non const float eps unsupported");
  }
  if (failed(checkNotNone(rewriter, op, weight)) ||
      failed(checkNotNone(rewriter, op, bias))) {
    return op->emitError("Unsupported None for weight or bias");
  }
  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto biasTy = bias.getType().cast<RankedTensorType>();

  if (!inputTy.getElementType().isa<mlir::FloatType>() ||
      !biasTy.getElementType().isa<mlir::FloatType>() ||
      !weightTy.getElementType().isa<mlir::FloatType>()) {
    return op->emitError("For now, only float data type are supported");
  }
  int64_t normalizedShapeRank = normalizedShape.size();
  if (weightTy.getRank() != normalizedShapeRank ||
      biasTy.getRank() != normalizedShapeRank ||
      inputRank < normalizedShapeRank || normalizedShapeRank < 1) {
    return rewriter.notifyMatchFailure(op, "Input or weight or bias shape or"
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

  // flatten dims to fit batch_norm operation.
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

  // reshape input
  auto mhloInput = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), mhloBatchNormOutTy, input,
      mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(inputFlattenShape),
                           {static_cast<int64_t>(inputFlattenShape.size())})
          .getValue());

  // generate "scale" and "offset" Value for mhlo.BatchNormTrainingOp.
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

  // reshape back
  auto outputTy =
      getTypeConverter()->convertType(op.getType(0)).cast<RankedTensorType>();
  auto outputMeanOrVarTy =
      getTypeConverter()->convertType(op.getType(1)).cast<RankedTensorType>();

  auto output = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputTy, batchNormTrainingResult.getResult(0),
      mhlo::getConstTensor(rewriter, op, outputTy.getShape(),
                           {static_cast<int64_t>(outputTy.getShape().size())})
          .getValue());
  auto mean = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(1),
      mhlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .getValue());
  auto var = rewriter.create<mhlo::DynamicReshapeOp>(
      op->getLoc(), outputMeanOrVarTy, batchNormTrainingResult.getResult(2),
      mhlo::getConstTensor(
          rewriter, op, outputMeanOrVarTy.getShape(),
          {static_cast<int64_t>(outputMeanOrVarTy.getShape().size())})
          .getValue());

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

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
  INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
  INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, MhloOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, MhloOp>>(typeConverter, context);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, mhlo::AddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, mhlo::AddOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, mhlo::SubtractOp);
  INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, mhlo::SubtractOp);
#undef INSERT_BINARY_ADDSUB_PATTERN

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

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp>>(typeConverter, context);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp);
  INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp);
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

  INSERT_ATENOP_PATTERN(AtenBatchNormOp);
  INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
#undef INSERT_ATENOP_PATTERN
}
