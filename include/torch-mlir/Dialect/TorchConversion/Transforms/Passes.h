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

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

// Do not register the torch-to-mhlo pipeline if mhlo target is disabled
#ifdef TORCH_MLIR_ENABLE_MHLO
struct MhloBackendPipelineOptions
    : public PassPipelineOptions<MhloBackendPipelineOptions> {
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

void createTorchBackendToMhloBackendPipeline(
    OpPassManager &pm, const MhloBackendPipelineOptions &options);
std::unique_ptr<OperationPass<ModuleOp>> createVerifyMhloBackendContractPass();
#endif

#ifdef TORCH_MLIR_ENABLE_TCP
// Creates a pipeline that lowers from the torch backend contract to the TCP
// backend contract.
void createTorchBackendToTcpBackendPipeline(OpPassManager &pm);
std::unique_ptr<OperationPass<ModuleOp>> createVerifyTcpBackendContractPass();
#endif // TORCH_MLIR_ENABLE_TCP

std::unique_ptr<OperationPass<ModuleOp>> createFuncBackendTypeConversionPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createFinalizingBackendTypeConversionPass();

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgOnTensorsBackendContractPass();

std::unique_ptr<OperationPass<ModuleOp>> createVerifyTosaBackendContractPass();

} // namespace TorchConversion

/// Registers all Torch transformation passes.
void registerTorchConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
