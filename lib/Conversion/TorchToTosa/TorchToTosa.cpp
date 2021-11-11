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
// These legalizations are for unary ops with only for FP datatypes.
// There is no supported quantized integer mode for these.
#define DEF_FULLCONV_FPONLY_UNARY_ATENOP(aten_op, tosa_op)                     \
  class ConvertAten##aten_op##Op                                               \
      : public OpConversionPattern<Aten##aten_op##Op> {                        \
  public:                                                                      \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult                                                              \
    matchAndRewrite(Aten##aten_op##Op op, ArrayRef<Value> operands,            \
                    ConversionPatternRewriter &rewriter) const override {      \
      Aten##aten_op##Op::Adaptor adaptor(operands);                            \
      Value self = adaptor.self();                                             \
      auto selfTy = self.getType().cast<TensorType>();                         \
      if (selfTy) {                                                            \
        if (selfTy.getElementType().isa<mlir::FloatType>()) {                  \
          rewriter.replaceOpWithNewOp<tosa::tosa_op##Op>(                      \
              op, getTypeConverter()->convertType(op.getType()), self);        \
          return success();                                                    \
        } else {                                                               \
          return op.emitError("Only FP type legalization supported");          \
        }                                                                      \
      } else {                                                                 \
        return op.emitError("Only Tensor types supported in TOSA");            \
      }                                                                        \
    }                                                                          \
  };
DEF_FULLCONV_FPONLY_UNARY_ATENOP(Log, Log)
DEF_FULLCONV_FPONLY_UNARY_ATENOP(Exp, Exp)
#undef DEF_FULLCONV_FPONLY_UNARY_ATENOP

// These unary op legalizations are identical for FP or quantized types
#define DEF_FULLCONV_UNARY_ATENOP(aten_op, tosa_op)                            \
  class ConvertAten##aten_op##Op                                               \
      : public OpConversionPattern<Aten##aten_op##Op> {                        \
  public:                                                                      \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult                                                              \
    matchAndRewrite(Aten##aten_op##Op op, ArrayRef<Value> operands,            \
                    ConversionPatternRewriter &rewriter) const override {      \
      Aten##aten_op##Op::Adaptor adaptor(operands);                            \
      rewriter.replaceOpWithNewOp<tosa::tosa_op##Op>(                          \
          op, getTypeConverter()->convertType(op.getType()), adaptor.self());  \
      return success();                                                        \
    }                                                                          \
  };
DEF_FULLCONV_UNARY_ATENOP(Neg, Negate)
DEF_FULLCONV_UNARY_ATENOP(Floor, Floor)
DEF_FULLCONV_UNARY_ATENOP(BitwiseNot, BitwiseNot)
#undef DEF_FULLCONV_UNARY_ATENOP

// These binary op legalizations are identical for FP or quantized types
#define DEF_FULLCONV_ADDSUB_ATENOP(aten_op, tosa_op)                           \
  class ConvertAten##aten_op##Op                                               \
      : public OpConversionPattern<Aten##aten_op##Op> {                        \
  public:                                                                      \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult matchAndRewrite(Aten##aten_op##Op op,                        \
                                  ArrayRef<Value> operands,                    \
                                  ConversionPatternRewriter &rewriter) const { \
      Aten##aten_op##Op::Adaptor adaptor(operands);                            \
                                                                               \
      Value lhs = adaptor.self();                                              \
      auto lhsTy = lhs.getType().cast<TensorType>();                           \
      Value rhs = adaptor.other();                                             \
      auto rhsTy = rhs.getType().cast<TensorType>();                           \
                                                                               \
      if (!lhsTy || !rhsTy)                                                    \
        return op.emitError("Only Tensor types supported in TOSA");            \
                                                                               \
      auto lhsElemTy = lhsTy.getElementType();                                 \
      auto rhsElemTy = rhsTy.getElementType();                                 \
                                                                               \
      if (lhsElemTy != rhsElemTy)                                              \
        return op.emitError("Add: input datatypes mismatched");                \
                                                                               \
      /* FIXME: Handle alpha.                                                  \
         Needs extraction of floating point constant. */                       \
                                                                               \
      if (lhsElemTy.isa<mlir::FloatType>()) {                                  \
        rewriter.replaceOpWithNewOp<tosa::tosa_op##Op>(                        \
            op, getTypeConverter()->convertType(op.getType()), lhs, rhs);      \
        return success();                                                      \
      } else {                                                                 \
        return op.emitError("Only FP type legalization supported");            \
      }                                                                        \
    }                                                                          \
  };
DEF_FULLCONV_ADDSUB_ATENOP(AddTensor, Add)
DEF_FULLCONV_ADDSUB_ATENOP(SubTensor, Sub)
#undef DEF_FULLCONV_ADDSUB_ATENOP

// These legalizations have both FP and quantized type supported modes.
// Their rewriters are expressed below
#define DECL_CONVERT_ATENOP(aten_op)                                           \
  class ConvertAten##aten_op##Op                                               \
      : public OpConversionPattern<Aten##aten_op##Op> {                        \
  public:                                                                      \
    using OpConversionPattern::OpConversionPattern;                            \
    LogicalResult                                                              \
    matchAndRewrite(Aten##aten_op##Op op, ArrayRef<Value> operands,            \
                    ConversionPatternRewriter &rewriter) const override;       \
  };
DECL_CONVERT_ATENOP(Tanh)
DECL_CONVERT_ATENOP(Sigmoid)
DECL_CONVERT_ATENOP(Relu)
#undef DECL_CONVERT_ATENOP

LogicalResult
ConvertAtenTanhOp::matchAndRewrite(AtenTanhOp op, ArrayRef<Value> operands,
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
    return op.emitError("Only FP type legalization currently supported");
  }
}

LogicalResult ConvertAtenSigmoidOp::matchAndRewrite(
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
    return op.emitError("Only FP type legalization currently supported");
  }
} // namespace

LogicalResult
ConvertAtenReluOp::matchAndRewrite(AtenReluOp op, ArrayRef<Value> operands,
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
      return op.emitError("Only FP type legalization currently supported");
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

#define INSERT_NEW_PATTERN(aten_op)                                            \
  target.addIllegalOp<Aten##aten_op##Op>();                                    \
  patterns.add<ConvertAten##aten_op##Op>(typeConverter, context);
    INSERT_NEW_PATTERN(Log);
    INSERT_NEW_PATTERN(Exp);
    INSERT_NEW_PATTERN(Neg);
    INSERT_NEW_PATTERN(Floor);
    INSERT_NEW_PATTERN(BitwiseNot);
    INSERT_NEW_PATTERN(AddTensor);
    INSERT_NEW_PATTERN(SubTensor);
    INSERT_NEW_PATTERN(Tanh);
    INSERT_NEW_PATTERN(Sigmoid);
    INSERT_NEW_PATTERN(Relu);
#undef INSERT_NEW_PATTERN

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
