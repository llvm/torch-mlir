//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
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
  LogicalResult
  matchAndRewrite(AtenOpT op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename AtenOpT::Adaptor adaptor(operands);
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
  LogicalResult
  matchAndRewrite(AtenOpT op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename AtenOpT::Adaptor adaptor(operands);
    rewriter.replaceOpWithNewOp<TosaOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self());
    return success();
  }
};

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  LogicalResult matchAndRewrite(AtenOpT op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    typename AtenOpT::Adaptor adaptor(operands);

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

    // FIXME: Handle alpha.
    // Needs extraction of floating point constant.

    if (lhsElemTy.isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<TosaOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs, rhs);
      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
  }
};

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOpT op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  AtenTanhOp::Adaptor adaptor(operands);
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
    AtenSigmoidOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  AtenSigmoidOp::Adaptor adaptor(operands);
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
    AtenReluOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  AtenReluOp::Adaptor adaptor(operands);
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

template <>
LogicalResult ConvertAtenOp<AtenMulTensorOp>::matchAndRewrite(
    AtenMulTensorOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  AtenMulTensorOp::Adaptor adaptor(operands);

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

  if (lhsElemTy.isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op, getTypeConverter()->convertType(op.getType()), lhs, rhs,
        /*shift=*/0);
    return success();
  } else {
    // Quantized multiplication may need to rescale inputs.
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

template <>
LogicalResult ConvertAtenOp<AtenDivTensorOp>::matchAndRewrite(
    AtenDivTensorOp op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  AtenDivTensorOp::Adaptor adaptor(operands);

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

  if (lhsElemTy.isa<mlir::FloatType>()) {
    auto rcpOp = rewriter.create<tosa::ReciprocalOp>(
        op->getLoc(), getTypeConverter()->convertType(op.getType()), rhs);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op, getTypeConverter()->convertType(op.getType()), lhs,
        rcpOp.getResult(), /*shift=*/0);
  } else {
    rewriter.replaceOpWithNewOp<tosa::DivOp>(
        op, getTypeConverter()->convertType(op.getType()), lhs, rhs);
  }
  return success();
}

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosa Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosa : public ConvertTorchToTosaBase<ConvertTorchToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect>();

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
    INSERT_UNARY_PATTERN(AtenBitwiseNotOp, tosa::BitwiseNotOp)
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, TosaOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, tosa::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, tosa::SubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenTanhOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenReluOp);
    INSERT_ATENOP_PATTERN(AtenMulTensorOp);
    INSERT_ATENOP_PATTERN(AtenDivTensorOp);
#undef INSERT_ATENOP_PATTERN

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToTosaPass() {
  return std::make_unique<ConvertTorchToTosa>();
}
