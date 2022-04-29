//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
#define TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace torch {
namespace Torch {

std::unique_ptr<OperationPass<ModuleOp>> createGlobalizeObjectGraphPass();

std::unique_ptr<OperationPass<ModuleOp>>
createPrepareForGlobalizeObjectGraphPass();

struct TorchLoweringPipelineOptions
    : public PassPipelineOptions<TorchLoweringPipelineOptions> {
  // If this option is true, then perform optimizations.
  // If this option is false, only do the bare minimum for correctness.
  Option<bool> optimize{*this, "optimize", llvm::cl::desc("Do optimizations."),
                        llvm::cl::init(true)};

  Option<bool> decomposeEarly{*this, "decompose-complex-ops-early",
                              llvm::cl::desc("Decompose complex operations."),
                              llvm::cl::init(true)};
  // If this option is false, decompose complex operations.
  // If this option is true, skip decomposition of complex operations.
  Option<bool> decompose{*this, "decompose-complex-ops",
                         llvm::cl::desc("Decompose complex operations."),
                         llvm::cl::init(true)};
};

/// Creates a pipeline that lowers the object graph IR that is produced by
/// TorchScript import into the form expected by torch-verify-backend-contract.
void createTorchScriptModuleToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

/// Creates a pipeline that lowers a flat list of funcs and global slots
/// with the torch and aten dialects and mutable arrays and converts it to
/// the form required by torch-verify-backend-contract.
void createTorchFunctionToTorchBackendPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

/// Creates a pipeline that refines shapes of tensor operations in the program.
void createTorchShapeRefinementPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

std::unique_ptr<OperationPass<ModuleOp>> createAdjustCallingConventionsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createRefineTypesPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineGlobalSlotsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createReduceOpVariantsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMaximizeValueSemanticsPass();

std::unique_ptr<OperationPass<ModuleOp>> createRefinePublicReturnPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComplexOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposeComplexOpsEarlyPass();

std::unique_ptr<OperationPass<ModuleOp>> createPreprocessShapeLibraryPass();

std::unique_ptr<OperationPass<ModuleOp>> createReifyShapeCalculationsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyShapeCalculationsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDropShapeCalculationsPass();

StringRef getShapeLibrary();

} // namespace Torch

/// Registers all Torch transformation passes.
void registerTorchPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
