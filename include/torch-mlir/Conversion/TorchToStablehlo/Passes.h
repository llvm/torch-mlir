//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_STABLEHLO_PASSES_H
#define TORCHMLIR_CONVERSION_STABLEHLO_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {

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

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToStablehloPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToStablehloPass(bool enableStaticShape, bool enableI32Index);

/// Registers all torch-mlir conversion passes.
void registerStablehloConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_STABLEHLO_PASSES_H
