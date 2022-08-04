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
  // The maximum number of invocations of the simplification pipeline in
  // LowerToBackendContract.
  Option<int> maxIterations{
      *this, "max-iterations",
      llvm::cl::desc(
          "Maximum number of invocations of the simplification pipeline."),
      llvm::cl::init(10)};
  // If this option is false, decompose complex operations.
  // If this option is true, skip decomposition of complex operations.
  // TODO: This should be replaced with a list of operations to decompose.
  // (or some other way to specify the set of allowed ops in the backend
  // contract)
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

/// Creates a pipeline that simplifies the computations in the program.
/// This pass does not do any global program restructuring -- it works entirely
/// within a single semantic model of a `builtin.module` with
/// `torch.global_slot` ops and `func.func` ops.
void createTorchSimplificationPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

/// Creates a pipeline that refines shapes of tensor operations in the program.
void createTorchShapeRefinementPipeline(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>> createAdjustCallingConventionsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createRefineTypesPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineGlobalSlotsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createReduceOpVariantsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createMaximizeValueSemanticsPass();

std::unique_ptr<OperationPass<ModuleOp>> createRefinePublicReturnPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeComplexOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createPreprocessShapeLibraryPass();

std::unique_ptr<OperationPass<ModuleOp>> createReifyShapeCalculationsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyShapeCalculationsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDropShapeCalculationsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyConversionToValueSemanticsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createEraseModuleInitializerPass();

std::unique_ptr<OperationPass<ModuleOp>>
createLowerToBackendContractPass(int maxIterations, bool decompose);

StringRef getShapeLibrary();

} // namespace Torch

/// Registers all Torch transformation passes.
void registerTorchPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
