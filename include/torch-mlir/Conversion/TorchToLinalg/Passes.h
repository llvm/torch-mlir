//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_LINALG_PASSES_H
#define TORCHMLIR_CONVERSION_LINALG_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {

/// Creates a pipeline that lowers from the torch backend contract to the
/// linalg-on-tensors backend contract.
void createTorchBackendToLinalgOnTensorsBackendPipeline(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgOnTensorsBackendContractPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToLinalgPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTorchConversionToMLProgramPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToTMTensorPass();

/// Registers all torch-mlir conversion passes.
void registerLinalgConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_PASSES_H
