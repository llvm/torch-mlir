//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
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
    auto rank =
        rewriter.create<tensor::RankOp>(op->getLoc(), adaptor.getSelf());
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
    rewriter.replaceOpWithNewOp<cf::AssertOp>(op, adaptor.getCondition(),
                                              adaptor.getMessage());
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
    rewriter.template replaceOpWithNewOp<BinOp>(op, adaptor.getA(),
                                                adaptor.getB());
    return success();
  }
};
} // namespace

namespace {
template <typename AtenOp, typename UnaryOp>
class ConvertAtenUnaryOpToFloatMathOp : public OpConversionPattern<AtenOp> {
public:
  using OpConversionPattern<AtenOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOp op,
                  typename OpConversionPattern<AtenOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getA();
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!input.getType().isa<mlir::FloatType>())
      input = convertScalarToDtype(rewriter, loc, input, rewriter.getF64Type());
    Value result = rewriter.create<UnaryOp>(loc, input);
    rewriter.replaceOp(op,
                       convertScalarToDtype(rewriter, loc, result, resultType));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenDivIntOp : public OpConversionPattern<AtenDivIntOp> {
public:
  using OpConversionPattern<AtenDivIntOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenDivIntOp op,
                  typename OpConversionPattern<AtenDivIntOp>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = convertScalarToDtype(rewriter, loc, adaptor.getA(),
                                   rewriter.getF64Type());
    Value b = convertScalarToDtype(rewriter, loc, adaptor.getB(),
                                   rewriter.getF64Type());
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, a, b);
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
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, Pred, adaptor.getA(),
                                               adaptor.getB());
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
    Value lhs = adaptor.getA(), rhs = adaptor.getB();
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
    if (auto elements = op.getValueAttr().dyn_cast<DenseIntElementsAttr>()) {
      if (auto type = elements.getType().dyn_cast<RankedTensorType>()) {
        Type elemTy = op.getValueAttr().getElementType();
        unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
        Type builtinTensorElemTy = IntegerType::get(context, bitWidth);
        auto shapedType =
            RankedTensorType::get(type.getShape(), builtinTensorElemTy);
        auto rawData = elements.getRawData();
        DenseElementsAttr newAttr =
            DenseElementsAttr::getFromRawBuffer(shapedType, rawData);
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newAttr);
        return success();
      }
    }
    if (auto elements =
            op.getValueAttr().dyn_cast<DenseResourceElementsAttr>()) {
      if (auto type = elements.getType().dyn_cast<RankedTensorType>()) {
        if (auto intType = type.getElementType().dyn_cast<IntegerType>()) {
          Type builtinTensorElemTy =
              IntegerType::get(context, intType.getIntOrFloatBitWidth());
          auto shapedType =
              RankedTensorType::get(type.getShape(), builtinTensorElemTy);
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(
              op, DenseResourceElementsAttr::get(shapedType,
                                                 elements.getRawHandle()));
          return success();
        }
      }
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValueAttr());
    return success();
  }
};

class ConvertTorchConstantIntOp
    : public OpConversionPattern<Torch::ConstantIntOp> {
public:
  using OpConversionPattern<Torch::ConstantIntOp>::OpConversionPattern;
  using OpAdaptor = Torch::ConstantIntOp::Adaptor;
  LogicalResult
  matchAndRewrite(Torch::ConstantIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // note: arith.constant only accept signless integer, so convert signed to
    // signless
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getI64Type(),
                                    op.getValueAttr().getValue()));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenFloatScalarOp : public OpConversionPattern<AtenFloatScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenFloatScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    Value result =
        convertScalarToDtype(rewriter, op.getLoc(), adaptor.getA(), resultType);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAddOp : public OpConversionPattern<AtenAddOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    Value operandA =
        convertScalarToDtype(rewriter, loc, adaptor.getA(), resultType);
    Value operandB =
        convertScalarToDtype(rewriter, loc, adaptor.getB(), resultType);
    if (resultType.isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, operandA, operandB);
    } else if (resultType.isa<mlir::IntegerType>()) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, operandA, operandB);
    } else {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only support integer or float result type");
    }
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy, typename BinOp>
class ConvertAtenAnyOrAllBoolOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  virtual bool reductionFunction(ArrayRef<bool> inputArray) const = 0;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value result;
    SmallVector<Value> inputListTorchBool;
    if (!getListConstructElements(op.getSelf(), inputListTorchBool)) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: input list not constructed from ListConstruct");
    }
    SmallVector<Value> inputList = getTypeConvertedValues(
        rewriter, loc, this->getTypeConverter(), inputListTorchBool);
    result = inputList[0];
    for (unsigned i = 1; i < inputList.size(); i++)
      result = rewriter.create<BinOp>(loc, result, inputList[i]);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertAtenAnyOp
    : public ConvertAtenAnyOrAllBoolOp<AtenAnyBoolOp, arith::OrIOp> {
  using ConvertAtenAnyOrAllBoolOp<AtenAnyBoolOp,
                                  arith::OrIOp>::ConvertAtenAnyOrAllBoolOp;
  bool reductionFunction(ArrayRef<bool> inputArray) const override {
    return llvm::any_of(inputArray,
                        [](bool inputListElem) { return inputListElem; });
  }
};

class ConvertAtenAllOp
    : public ConvertAtenAnyOrAllBoolOp<AtenAllBoolOp, arith::AndIOp> {
  using ConvertAtenAnyOrAllBoolOp<AtenAllBoolOp,
                                  arith::AndIOp>::ConvertAtenAnyOrAllBoolOp;
  bool reductionFunction(ArrayRef<bool> inputArray) const override {
    return llvm::all_of(inputArray,
                        [](bool inputListElem) { return inputListElem; });
  }
};
} // namespace

namespace {
template <typename OpTy, typename CmpOpTy, typename CmpOpPred, CmpOpPred Pred>
class ConvertAtenBoolLikeOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type inputType = adaptor.getA().getType();
    Value cstZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(inputType));
    Value cstTrue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    Value cstFalse =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));

    Value cmpPred;
    cmpPred = rewriter.create<CmpOpTy>(loc, Pred, adaptor.getA(), cstZero);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, cmpPred, cstTrue,
                                                 cstFalse);
    return success();
  }
};
} // namespace

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToArith
    : public ConvertTorchToArithBase<ConvertTorchToArith> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<math::MathDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, func::FuncDialect,
                           arith::ArithDialect, tensor::TensorDialect,
                           cf::ControlFlowDialect, math::MathDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenDimOp>();
    patterns.add<ConvertAtenDimOp>(typeConverter, context);
    target.addIllegalOp<RuntimeAssertOp>();
    patterns.add<ConvertRuntimeAssertOp>(typeConverter, context);
    target.addIllegalOp<AtenNeIntOp, AtenEqIntOp, AtenGtIntOp, AtenGeIntOp>();
    patterns
        .add<ConvertAtenIntComparisonOp<AtenNeIntOp, arith::CmpIPredicate::ne>>(
            typeConverter, context);
    patterns
        .add<ConvertAtenIntComparisonOp<AtenEqIntOp, arith::CmpIPredicate::eq>>(
            typeConverter, context);
    patterns.add<
        ConvertAtenIntComparisonOp<AtenGtIntOp, arith::CmpIPredicate::sgt>>(
        typeConverter, context);
    patterns.add<
        ConvertAtenIntComparisonOp<AtenGeIntOp, arith::CmpIPredicate::sge>>(
        typeConverter, context);
    target.addIllegalOp<AtenGeFloatOp, AtenGeFloatIntOp, AtenNeFloatIntOp,
                        AtenGtFloatIntOp>();
    patterns.add<
        ConvertAtenFloatComparisonOp<AtenGeFloatOp, arith::CmpFPredicate::UGE>>(
        typeConverter, context);
    patterns.add<ConvertAtenFloatComparisonOp<AtenGeFloatIntOp,
                                              arith::CmpFPredicate::UGE>>(
        typeConverter, context);
    patterns.add<ConvertAtenFloatComparisonOp<AtenNeFloatIntOp,
                                              arith::CmpFPredicate::UNE>>(
        typeConverter, context);
    patterns.add<ConvertAtenFloatComparisonOp<AtenGtFloatIntOp,
                                              arith::CmpFPredicate::UGT>>(
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
    patterns.add<ConvertTorchConstantIntOp>(typeConverter, context);

    target.addIllegalOp<AtenFloatScalarOp>();
    patterns.add<ConvertAtenFloatScalarOp>(typeConverter, context);

    target.addIllegalOp<AtenAddOp>();
    patterns.add<ConvertAtenAddOp>(typeConverter, context);

    target.addIllegalOp<AtenAddIntOp, AtenSubIntOp, AtenMulIntOp>();
    patterns.add<ConvertAtenBinaryOp<AtenAddIntOp, arith::AddIOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenBinaryOp<AtenSubIntOp, arith::SubIOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenBinaryOp<AtenMulIntOp, arith::MulIOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenSubFloatOp, AtenMulFloatOp>();
    patterns.add<ConvertAtenBinaryOp<AtenSubFloatOp, arith::SubFOp>>(
        typeConverter, context);
    patterns.add<ConvertAtenBinaryOp<AtenMulFloatOp, arith::MulFOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenDivIntOp>();
    patterns.add<ConvertAtenDivIntOp>(typeConverter, context);
    target.addIllegalOp<AtenDivFloatOp>();
    patterns.add<ConvertAtenBinaryOp<AtenDivFloatOp, arith::DivFOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenFloordivIntOp>();
    patterns.add<ConvertAtenBinaryOp<AtenFloordivIntOp, arith::FloorDivSIOp>>(
        typeConverter, context);
    target.addIllegalOp<PrimMaxIntOp>();
    patterns.add<ConvertAtenBinaryOp<PrimMaxIntOp, arith::MaxSIOp>>(
        typeConverter, context);
    target.addIllegalOp<PrimMinIntOp>();
    patterns.add<ConvertAtenBinaryOp<PrimMinIntOp, arith::MinSIOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenCeilFloatOp>();
    patterns
        .add<ConvertAtenUnaryOpToFloatMathOp<AtenCeilFloatOp, math::CeilOp>>(
            typeConverter, context);
    target.addIllegalOp<AtenSqrtIntOp>();
    patterns.add<ConvertAtenUnaryOpToFloatMathOp<AtenSqrtIntOp, math::SqrtOp>>(
        typeConverter, context);
    target.addIllegalOp<AtenAnyBoolOp, AtenAllBoolOp>();
    patterns.add<ConvertAtenAnyOp>(typeConverter, context);
    patterns.add<ConvertAtenAllOp>(typeConverter, context);
    target.addIllegalOp<AtenBoolFloatOp, AtenBoolIntOp>();
    patterns.add<
        ConvertAtenBoolLikeOp<AtenBoolFloatOp, arith::CmpFOp,
                              arith::CmpFPredicate, arith::CmpFPredicate::UNE>>(
        typeConverter, context);
    patterns.add<
        ConvertAtenBoolLikeOp<AtenBoolIntOp, arith::CmpIOp,
                              arith::CmpIPredicate, arith::CmpIPredicate::ne>>(
        typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToArithPass() {
  return std::make_unique<ConvertTorchToArith>();
}
