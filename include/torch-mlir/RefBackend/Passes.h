//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_REFBACKEND_PASSES_H
#define TORCHMLIR_REFBACKEND_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
class ModuleOp;

namespace torch {
namespace RefBackend {

/// Registers all RefBackend passes.
void registerRefBackendPasses();

std::unique_ptr<OperationPass<ModuleOp>> createMungeCallingConventionsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createExpandOpsForLLVMPass();

std::unique_ptr<OperationPass<ModuleOp>> createMLProgramBufferizePass();

std::unique_ptr<OperationPass<func::FuncOp>> createMungeMemrefCopyPass();

std::unique_ptr<OperationPass<func::FuncOp>> createGeneralizeTensorPadPass();
} // namespace RefBackend
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_REFBACKEND_PASSES_H
