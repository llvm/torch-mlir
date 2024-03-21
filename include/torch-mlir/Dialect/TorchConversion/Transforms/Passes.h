//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
#define TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace torch {
namespace TorchConversion {

/// Creates a pipeline that lowers from the torch backend contract to the
/// linalg-on-tensors backend contract.
void createTorchBackendToLinalgOnTensorsBackendPipeline(OpPassManager &pm);

/// Creates a pipeline that lowers from the torch backend contract to the
/// TOSA backend contract.
void createTorchBackendToTosaBackendPipeline(OpPassManager &pm);

// Do not register the stablehlo options if the stablehlo target is disabled
#ifdef TORCH_MLIR_ENABLE_STABLEHLO
struct StablehloBackendPipelineOptions
    : public PassPipelineOptions<StablehloBackendPipelineOptions> {
  Option<bool> enableStaticShape{
      *this, "enable-static-shape",
      llvm::cl::desc("Enable static shape conversion."), llvm::cl::init(false)};
  // The i64 calculation is much slower than i32 on some devices, such as
  // Nvidia GPU. One can truncate from i64 to i32 since dimension sizes
  // are unlikely to exceed the range of i32(4GiB)
  Option<bool> enableI32Index{
      *this, "enable-i32-index",
      llvm::cl::desc("Enable truncate index from i64 to i32(unsafely)"),
      llvm::cl::init(false)};
};

void createTorchBackendToStablehloBackendPipeline(
    OpPassManager &pm, const StablehloBackendPipelineOptions &options);
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyStablehloBackendContractPass();
#endif

std::unique_ptr<OperationPass<ModuleOp>> createFuncBackendTypeConversionPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFinalizingBackendTypeConversionPass();

// These passes do a one-off conversion of a specific kind of quantized group
// matmul as a prototype. Generalized quantized operation handling will likely
// obviate them but that are being carried for now in order to unblock progress
// on full integrations. See https://github.com/llvm/torch-mlir/issues/2417 for
// the plan to support a more generalized lowering for these graphs.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createUnpackQuantTensorPass();
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertCustomQuantOpPass();

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgOnTensorsBackendContractPass();

std::unique_ptr<OperationPass<ModuleOp>> createVerifyTosaBackendContractPass();

} // namespace TorchConversion

/// Registers all Torch transformation passes.
void registerTorchConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
