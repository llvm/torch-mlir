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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertBackendLegalAtenOpToCustomOp : public ConversionPattern {
public:
  SetVector<StringRef> customOps;

  ConvertBackendLegalAtenOpToCustomOp(TypeConverter &typeConverter,
                                      MLIRContext *context,
                                      ArrayRef<std::string> customOps)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, 0, context) {
    this->customOps = SetVector<StringRef>(customOps.begin(), customOps.end());
  }

  Value convertOperandToTensor(Value inputOperand, PatternRewriter &rewriter,
                               Operation *backendLegalOp) const {
    // Get the Torch Op to find the constant attributes attached to the
    // backendLegalOp
    Value torchInputOperand = inputOperand;
    if (auto unrealizedCastOp = dyn_cast_or_null<UnrealizedConversionCastOp>(
            inputOperand.getDefiningOp())) {
      torchInputOperand = unrealizedCastOp.getInputs()[0];
    }
    // Handle the special case where input operand is an argument to the module
    // function
    if (!torchInputOperand.getDefiningOp())
      return inputOperand;

    return TypeSwitch<Operation *, Value>(torchInputOperand.getDefiningOp())
        .Case<Torch::ConstantBoolOp>([&](Operation *boolOperand) -> Value {
          bool boolConstAttr;
          if (matchPattern(boolOperand, m_TorchConstantBool(&boolConstAttr))) {
            return tosa::getConstTensor<int64_t>(rewriter, backendLegalOp,
                                                 boolConstAttr, {})
                .value();
          }
          return nullptr;
        })
        // TODO Add support for converting "torch.constant.device"
        .Case<Torch::ConstantDeviceOp>(
            [&](Operation *strOperand) -> Value { return nullptr; })
        .Case<Torch::ConstantIntOp>([&](Operation *intOperand) -> Value {
          int64_t intConstAttr;
          if (matchPattern(intOperand, m_TorchConstantInt(&intConstAttr))) {
            return tosa::getConstTensor<int64_t>(rewriter, backendLegalOp,
                                                 intConstAttr, {})
                .value();
          }
          return nullptr;
        })
        .Case<Torch::ConstantFloatOp>([&](Operation *floatOperand) -> Value {
          double floatConstAttr;
          if (matchPattern(floatOperand,
                           m_TorchConstantFloat(&floatConstAttr))) {
            return tosa::getConstTensor<float>(rewriter, backendLegalOp,
                                               floatConstAttr, {})
                .value();
          }
          return nullptr;
        })
        .Case<Torch::ConstantNoneOp>([&](Operation *noneOperand) -> Value {
          auto noneCustomOp = rewriter.create<tosa::CustomOp>(
              backendLegalOp->getLoc(),
              RankedTensorType::get({}, rewriter.getIntegerType(1)),
              rewriter.getStringAttr("constant.none"),
              rewriter.getStringAttr("torch_mlir"), rewriter.getStringAttr(""),
              ValueRange{});
          return noneCustomOp.getResult(0);
        })
        // TODO Add support for converting "torch.constant.number"
        .Case<Torch::ConstantNumberOp>(
            [&](Operation *strOperand) -> Value { return nullptr; })
        .Case<Torch::ConstantStrOp>([&](Operation *strOperand) -> Value {
          std::string strConstAttr;
          if (matchPattern(strOperand, m_TorchConstantStr(strConstAttr))) {
            auto strCustomOp = rewriter.create<tosa::CustomOp>(
                backendLegalOp->getLoc(),
                RankedTensorType::get({}, rewriter.getIntegerType(8)),
                rewriter.getStringAttr("constant.str"),
                rewriter.getStringAttr("torch_mlir"),
                rewriter.getStringAttr(""), ValueRange{});
            return strCustomOp.getResult(0);
          }
          return nullptr;
        })
        .Case<Torch::PrimListConstructOp>(
            [&](Operation *intListConstructOperand) -> Value {
              SmallVector<int64_t> intConstListAttr;
              if (matchPattern(intListConstructOperand,
                               m_TorchListOfConstantInts(intConstListAttr))) {
                return tosa::getConstTensor<int64_t>(
                           rewriter, backendLegalOp, intConstListAttr,
                           {static_cast<int64_t>(intConstListAttr.size())})
                    .value();
              }
              return nullptr;
            })
        .Default([&](Operation *defaultOperand) { return inputOperand; });
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    if (customOps.contains(op->getName().getStringRef())) {
      SmallVector<Value> customOpInputOperands;

      for (auto operand : operands) {
        Value customOpInputOperand =
            convertOperandToTensor(operand, rewriter, op);
        if (!customOpInputOperand) {
          return rewriter.notifyMatchFailure(
              op,
              "failed to match the constant operand of the backend-legal Op");
        }
        customOpInputOperands.push_back(customOpInputOperand);
      }
      SmallVector<Type> customOpResultTypes;
      auto convertTypesResult = getTypeConverter()->convertTypes(
          op->getResultTypes(), customOpResultTypes);
      if (convertTypesResult.failed())
        return rewriter.notifyMatchFailure(
            op, "failed to convert TOSA CustomOp result types; Only tensor "
                "types are supported for the resutls.");
      rewriter.replaceOpWithNewOp<tosa::CustomOp>(
          op, TypeRange{customOpResultTypes},
          llvm::StringRef(op->getName().stripDialect()), // identifier
          llvm::StringRef("torch_mlir"),                 // config
          llvm::StringRef(""),                           // implementation_attrs
          ValueRange{customOpInputOperands});
      return success();
    }
    return failure();
  }
};

} // namespace

// -----------------------------------------------------------------------------
// TorchBackendLegalToTosaCustom Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchBackendLegalToTosaCustom
    : public ConvertTorchBackendLegalToTosaCustomBase<
          ConvertTorchBackendLegalToTosaCustom> {
public:
  ConvertTorchBackendLegalToTosaCustom() = default;
  ConvertTorchBackendLegalToTosaCustom(ArrayRef<std::string> customOps) {
    this->customOps = customOps;
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

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    patterns.add<ConvertBackendLegalAtenOpToCustomOp>(typeConverter, context,
                                                      customOps);

    for (std::string opName : customOps) {
      target.addIllegalOp(OperationName(opName, context));
    }

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchBackendLegalToTosaCustomPass(
    ArrayRef<std::string> customOps) {
  return std::make_unique<ConvertTorchBackendLegalToTosaCustom>(customOps);
}
