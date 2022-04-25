//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStd/TorchToStd.h"

#include "../PassDetail.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// Patterns (as this grows, it should be organized into multiple files)
// -----------------------------------------------------------------------------
// This is going to eventually be O(#torch operators), which is in the 100s.

namespace {
// Note: Confusingly, ATen's "dim" means "number of dimensions" which is what
// MLIR calls "rank".
class ConvertAtenDimOp : public OpConversionPattern<AtenDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = rewriter.create<tensor::RankOp>(op->getLoc(), adaptor.self());
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
        op, getTypeConverter()->convertType(op.getType()), rank);
    return success();
  }
};
} // namespace

namespace {
class ConvertRuntimeAssertOp : public OpConversionPattern<RuntimeAssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(RuntimeAssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::AssertOp>(op, adaptor.condition(),
                                              adaptor.message());
    return success();
  }
};
} // namespace

namespace {
template <typename AtenOp, typename BinOp>
class ConvertAtenBinaryOp : public OpConversionPattern<AtenOp> {
public:
  using OpConversionPattern<AtenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOp op,
                  typename OpConversionPattern<AtenOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<BinOp>(op, adaptor.a(), adaptor.b());
    return success();
  }
};
} // namespace

namespace {
// Lowers aten integer comparison ops.
template <typename AtenOp, arith::CmpIPredicate Pred>
class ConvertAtenIntComparisonOp : public OpConversionPattern<AtenOp> {
public:
  using OpConversionPattern<AtenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOp op,
                  typename OpConversionPattern<AtenOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, Pred, adaptor.a(),
                                               adaptor.b());
    return success();
  }
};
} // namespace

namespace {
// Lowers aten float and float_int comparison ops.
template <typename AtenOp, arith::CmpFPredicate Pred>
class ConvertAtenFloatComparisonOp : public OpConversionPattern<AtenOp> {
public:
  using OpConversionPattern<AtenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOp op,
                  typename OpConversionPattern<AtenOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.a(), rhs = adaptor.b();
    rhs = convertScalarToDtype(rewriter, op.getLoc(), rhs, lhs.getType());
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, Pred, lhs, rhs);
    return success();
  }
};
} // namespace

// Tensors with integer types need to be converted to signless integer
// element type. All tensors with element types other than integer can reuse
// existing elements attribute.
namespace {
class ConvertTorchTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  using OpConversionPattern<ValueTensorLiteralOp>::OpConversionPattern;
  using OpAdaptor = ValueTensorLiteralOp::Adaptor;
  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    if (auto elements = op.valueAttr().dyn_cast<DenseIntElementsAttr>()) {
      Type elemTy = op.valueAttr().getElementType();
      unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
      Type builtinTensorElemTy = IntegerType::get(context, bitWidth);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
            return APInt(bitWidth, v.getSExtValue());
          }));
      return success();
    }
    if (auto elements = op.valueAttr().dyn_cast<OpaqueElementsAttr>()) {
      if (auto type = elements.getType().dyn_cast<RankedTensorType>()) {
        if (auto intType = type.getElementType().dyn_cast<IntegerType>()) {
          Type builtinTensorElemTy =
              IntegerType::get(context, intType.getIntOrFloatBitWidth());
          auto shapedType =
              RankedTensorType::get(type.getShape(), builtinTensorElemTy);
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(
              op, OpaqueElementsAttr::get(elements.getDialect(), shapedType,
                                          elements.getValue()));
          return success();
        }
      }
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.valueAttr());
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy>
class ConvertTorchConstantOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.valueAttr());
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToStd : public ConvertTorchToStdBase<ConvertTorchToStd> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<cf::ControlFlowDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, func::FuncDialect,
                           arith::ArithmeticDialect, tensor::TensorDialect,
                           cf::ControlFlowDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenDimOp>();
    patterns.add<ConvertAtenDimOp>(typeConverter, context);
    target.addIllegalOp<RuntimeAssertOp>();
    patterns.add<ConvertRuntimeAssertOp>(typeConverter, context);
    target.addIllegalOp<AtenNeIntOp, AtenEqIntOp, AtenGtIntOp>();
    patterns
        .add<ConvertAtenIntComparisonOp<AtenNeIntOp, arith::CmpIPredicate::ne>>(
            typeConverter, context);
    patterns
        .add<ConvertAtenIntComparisonOp<AtenEqIntOp, arith::CmpIPredicate::eq>>(
            typeConverter, context);
    patterns.add<
        ConvertAtenIntComparisonOp<AtenGtIntOp, arith::CmpIPredicate::sgt>>(
        typeConverter, context);
    target.addIllegalOp<AtenGeFloatOp, AtenGeFloatIntOp, AtenNeFloatIntOp>();
    patterns.add<
        ConvertAtenFloatComparisonOp<AtenGeFloatOp, arith::CmpFPredicate::UGE>>(
        typeConverter, context);
    patterns.add<ConvertAtenFloatComparisonOp<AtenGeFloatIntOp,
                                              arith::CmpFPredicate::UGE>>(
        typeConverter, context);
    patterns.add<ConvertAtenFloatComparisonOp<AtenNeFloatIntOp,
                                              arith::CmpFPredicate::UNE>>(
        typeConverter, context);
    target.addIllegalOp<ValueTensorLiteralOp>();
    patterns.add<ConvertTorchTensorLiteralOp>(typeConverter, context);

    target.addIllegalOp<ConstantBoolOp>();
    patterns.add<ConvertTorchConstantOp<ConstantBoolOp>>(typeConverter,
                                                         context);
    target.addIllegalOp<Torch::ConstantFloatOp>();
    patterns.add<ConvertTorchConstantOp<Torch::ConstantFloatOp>>(typeConverter,
                                                                 context);
    target.addIllegalOp<Torch::ConstantIntOp>();
    patterns.add<ConvertTorchConstantOp<Torch::ConstantIntOp>>(typeConverter,
                                                               context);
    target.addIllegalOp<AtenAddIntOp, AtenSubIntOp, AtenMulIntOp>();
    patterns.add<ConvertAtenBinaryOp<AtenAddIntOp, arith::AddIOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenBinaryOp<AtenSubIntOp, arith::SubIOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenBinaryOp<AtenMulIntOp, arith::MulIOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenSubFloatOp>();
    patterns.add<ConvertAtenBinaryOp<AtenSubFloatOp, arith::SubFOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenDivFloatOp>();
    patterns.add<ConvertAtenBinaryOp<AtenDivFloatOp, arith::DivFOp>>(
        typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToStdPass() {
  return std::make_unique<ConvertTorchToStd>();
}
