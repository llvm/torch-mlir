//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch;
using namespace mlir::torch::TorchConversion;

void mlir::torch::TorchConversion::getBackendTypeConversionDependentDialects(
    DialectRegistry &registry) {
  registry.insert<TorchConversionDialect>();
}

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

static void
setupValueTensorToBuiltinTensorConversion(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToBuiltinTensorOp,
                    TorchConversion::FromBuiltinTensorOp>();
  typeConverter.addConversion(
      [](Torch::ValueTensorType type) -> Optional<Type> {
        return type.toBuiltinTensor();
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<ToBuiltinTensorOp>(loc, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder,
                                  Torch::ValueTensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<FromBuiltinTensorOp>(loc, type, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

static void setupTorchBoolToI1Conversion(ConversionTarget &target,
                                         TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToI1Op, TorchConversion::FromI1Op>();
  typeConverter.addConversion([](Torch::BoolType type) -> Optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    // Other builtin integer types could be handled by other materializers.
    if (!(type.getWidth() == 1 && type.isSignless()))
      return None;
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BoolType>());
    return builder.create<ToI1Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::BoolType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<FromI1Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

static void setupTorchIntToI64Conversion(ConversionTarget &target,
                                         TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToI64Op, TorchConversion::FromI64Op>();
  typeConverter.addConversion([](Torch::IntType type) -> Optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            IntegerType type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    // Other builtin integer types could be handled by other materializers.
    if (!(type.getWidth() == 64 && type.isSignless()))
      return None;
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::IntType>());
    return builder.create<ToI64Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::IntType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<FromI64Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

static void setupTorchFloatToF64Conversion(ConversionTarget &target,
                                           TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::ToF64Op, TorchConversion::FromF64Op>();
  typeConverter.addConversion([](Torch::FloatType type) -> Optional<Type> {
    return Float64Type::get(type.getContext());
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            Float64Type type, ValueRange inputs,
                                            Location loc) -> Optional<Value> {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::FloatType>());
    return builder.create<ToF64Op>(loc, inputs[0]).getResult();
  });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::FloatType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Float64Type>());
    return builder.create<FromF64Op>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

void mlir::torch::TorchConversion::setupBackendTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupValueTensorToBuiltinTensorConversion(target, typeConverter);
  setupTorchBoolToI1Conversion(target, typeConverter);
  setupTorchIntToI64Conversion(target, typeConverter);
  setupTorchFloatToF64Conversion(target, typeConverter);
}

//===----------------------------------------------------------------------===//
// FuncBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
struct FuncBackendTypeConversionPass
    : public FuncBackendTypeConversionBase<FuncBackendTypeConversionPass> {
  using FuncBackendTypeConversionBase<
      FuncBackendTypeConversionPass>::FuncBackendTypeConversionBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    populateFuncOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<CallOp>(
        [&](CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::TorchConversion::createFuncBackendTypeConversionPass() {
  return std::make_unique<FuncBackendTypeConversionPass>();
}

//===----------------------------------------------------------------------===//
// FinalizingBackendTypeConversionPass
//===----------------------------------------------------------------------===//

namespace {
// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};
} // namespace

template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(typeConverter,
                                              patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

namespace {
struct FinalizingBackendTypeConversionPass
    : public FinalizingBackendTypeConversionBase<
          FinalizingBackendTypeConversionPass> {
  using FinalizingBackendTypeConversionBase<
      FinalizingBackendTypeConversionPass>::FinalizingBackendTypeConversionBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    setupFinalization<ToBuiltinTensorOp, FromBuiltinTensorOp, FromI1Op, ToI1Op,
                      FromI64Op, ToI64Op, FromF64Op, ToF64Op>(target, patterns,
                                                              typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::TorchConversion::createFinalizingBackendTypeConversionPass() {
  return std::make_unique<FinalizingBackendTypeConversionPass>();
}
