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

std::unique_ptr<OperationPass<ModuleOp>> createAdjustCallingConventionsPass();

std::unique_ptr<OperationPass<FuncOp>> createRefineTypesPass();

} // namespace Torch

/// Registers all Torch transformation passes.
void registerTorchPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TORCH_TRANSFORMS_PASSES_H
