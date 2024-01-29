//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
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
      [](Torch::ValueTensorType type) -> std::optional<Type> {
        return type.toBuiltinTensor();
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    if (!inputs[0].getType().isa<Torch::BaseTensorType>())
      return {};
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
  typeConverter.addConversion([](Torch::BoolType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 1);
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, IntegerType type, ValueRange inputs,
         Location loc) -> std::optional<Value> {
        // Other builtin integer types could be handled by other materializers.
        if (!(type.getWidth() == 1 && type.isSignless()))
          return std::nullopt;
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
  typeConverter.addConversion([](Torch::IntType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, IntegerType type, ValueRange inputs,
         Location loc) -> std::optional<Value> {
        // Other builtin integer types could be handled by other materializers.
        if (!(type.getWidth() == 64 && type.isSignless()))
          return std::nullopt;
        // Other input type to be converted to i64 are handled by other
        // materializers.
        if (!inputs[0].getType().isa<Torch::IntType>())
          return std::nullopt;
        assert(inputs.size() == 1);
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
  typeConverter.addConversion([](Torch::FloatType type) -> std::optional<Type> {
    return Float64Type::get(type.getContext());
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, Float64Type type, ValueRange inputs,
         Location loc) -> std::optional<Value> {
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

static void setupTorchGeneratorToI64Conversion(ConversionTarget &target,
                                               TypeConverter &typeConverter) {
  target.addLegalOp<TorchConversion::GeneratorToI64Op,
                    TorchConversion::I64ToGeneratorOp>();
  typeConverter.addConversion(
      [](Torch::GeneratorType type) -> std::optional<Type> {
        return IntegerType::get(type.getContext(), 64);
      });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &builder, IntegerType type, ValueRange inputs,
         Location loc) -> std::optional<Value> {
        // Other builtin integer types could be handled by other materializers.
        if (!(type.getWidth() == 64 && type.isSignless()))
          return std::nullopt;
        // Other input type to be converted to i64 are handled by other
        // materializers.
        if (!inputs[0].getType().isa<Torch::GeneratorType>())
          return std::nullopt;
        assert(inputs.size() == 1);
        return builder.create<GeneratorToI64Op>(loc, inputs[0]).getResult();
      });
  auto sourceMaterialization = [](OpBuilder &builder, Torch::GeneratorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IntegerType>());
    return builder.create<I64ToGeneratorOp>(loc, inputs[0]);
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
  setupTorchGeneratorToI64Conversion(target, typeConverter);
}
