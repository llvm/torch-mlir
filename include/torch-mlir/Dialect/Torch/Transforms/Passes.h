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

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

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
  // If this option is true, decompose complex operations.
  // If this option is false, skip decomposition of complex operations.
  Option<bool> decompose{*this, "decompose-complex-ops",
                         llvm::cl::desc("Decompose complex operations."),
                         llvm::cl::init(true)};
  // A list of ops that should be considered legal for the backend.
  // TODO: The meaning of this list should be formalized.
  // A sketch of the semantics would be:
  // - In torch_ods_gen.py, we mark each op as "legal in backend contract",
  // "illegal in backend contract", or "conditionally legal in backend
  // contract".
  // This option would be a list of ops from the "conditionally legal" set
  // which should be considered legal for a particular invocation of the
  // lowering pipeline.
  // TODO: The "decompose" flag should be expanded into this formulation
  // of legality for the backend. Ultimately we will want LowerToBackendContract
  // to check for a specific set of legal ops to stop its iteration.
  ListOption<std::string> backendLegalOps{
      *this, "backend-legal-ops",
      llvm::cl::desc("List of ops to be considered legal for the backend, such "
                     "as 'aten.foo'.")};

  Option<std::string> extraLibrary{
      *this, "extra-library",
      llvm::cl::desc("Filename of MLIR module for splicing into the abstract "
                     "interpretation library.")};
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
void createTorchShapeRefinementPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

/// Creates a pipeline that refines dtype of tensor operations in the program.
void createTorchDtypeRefinementPipeline(
    OpPassManager &pm, const TorchLoweringPipelineOptions &options);

std::unique_ptr<OperationPass<ModuleOp>> createAdjustCallingConventionsPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineGlobalSlotsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createReduceOpVariantsPass(StringRef extraLibrary);

std::unique_ptr<OperationPass<func::FuncOp>> createMaximizeValueSemanticsPass();

std::unique_ptr<OperationPass<ModuleOp>> createRefinePublicReturnPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposeComplexOpsPass(ArrayRef<std::string> legalOps);

std::unique_ptr<OperationPass<func::FuncOp>> createRecomposeComplexOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createFuseQuantizedOpsPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createMatchQuantizedCustomOpsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createReifyShapeCalculationsPass(StringRef extraLibrary);

std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyShapeCalculationsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createReifyDtypeCalculationsPass(StringRef extraLibrary);

std::unique_ptr<OperationPass<func::FuncOp>>
createSimplifyDtypeCalculationsPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createDropAbstractInterpCalculationsPass();

std::unique_ptr<OperationPass<ModuleOp>> createEraseModuleInitializerPass();

std::unique_ptr<OperationPass<ModuleOp>>
createLowerToBackendContractPass(int maxIterations, bool decompose,
                                 ArrayRef<std::string> backendLegalOps,
                                 StringRef extraLibrary);

std::unique_ptr<OperationPass<ModuleOp>>
createVerifyBackendContractNoDecompositionsPass();

StringRef getAbstractInterpLibrary();

static const char kTorchOpPrefix[] = R"(torch.)";

} // namespace Torch

/// Registers all Torch transformation passes.
void registerTorchPasses();

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
