//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchUtils.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Torch;

void mlir::NPCOMP::Torch::setupValueTensorToBuiltinTensorConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  target.addLegalOp<Torch::ToBuiltinTensorOp, Torch::FromBuiltinTensorOp>();
  typeConverter.addConversion(
      [](Torch::ValueTensorType type) -> Optional<Type> {
        return type.toBuiltinTensor();
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, TensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<BaseTensorType>());
    return builder.create<ToBuiltinTensorOp>(loc, inputs[0]);
  });
  auto sourceMaterialization = [](OpBuilder &builder, ValueTensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<FromBuiltinTensorOp>(loc, inputs[0]);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}
