//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_TRANSFORMS_PASSES_H
#define NPCOMP_DIALECT_TORCH_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {
namespace Torch {

std::unique_ptr<OperationPass<ModuleOp>> createGlobalizeObjectGraphPass();

std::unique_ptr<OperationPass<ModuleOp>>
createPrepareForGlobalizeObjectGraphPass();

/// Creates a pipeline that "globalizes" the given program.
/// See the documentation on torch-globalize-object-graph for more details.
void createGlobalizePipeline(OpPassManager &pm);

/// Creates a pipeline that lowers the object graph IR that is produced by
/// TorchScript import into the form expected by npcomp-verify-backend-contract.
void createLowerObjectGraphPipeline(OpPassManager &pm);

/// Creates a pipeline that lowers a flat list of funcs and global slots
/// with the torch and aten dialects and mutable arrays and converts it to
/// the form required by npcomp-verify-backend-contract, in particular
/// lowering most arrays to ranked tensors of known dtype, lowering aten ops to
/// linalg, converting torch.prim.* ops to elementary math operations.
void createLowerToNpcompBackendPipeline(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>> createAdjustCallingConventionsPass();

std::unique_ptr<OperationPass<FuncOp>> createRefineTypesPass();

std::unique_ptr<OperationPass<ModuleOp>> createInlineGlobalSlotsPass();

} // namespace Torch

/// Registers all Torch transformation passes.
void registerTorchPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_TRANSFORMS_PASSES_H
