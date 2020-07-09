//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_E2E_E2E_H
#define NPCOMP_E2E_E2E_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace NPCOMP {

// Look in createE2ELoweringPipeline for more information about how these
// passes fit together.
//
// Pass summaries are in Passes.td.

std::unique_ptr<OperationPass<FuncOp>> createLowerBroadcastToToLoopsPass();

std::unique_ptr<OperationPass<FuncOp>>
createLowerLinalgOnTensorToLinalgOnMemrefPass();

std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOfOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createResolveTensorLoadStoreOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerLinalgLoopDimOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerRankedShapesPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToNpcomprtABIPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerAllocMemRefOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();

void createLowerToHybridTensorMemRefPipeline(OpPassManager &pm);

struct E2ELoweringPipelineOptions
    : public PassPipelineOptions<E2ELoweringPipelineOptions> {
  // If this option is true, then perform optimizations.
  // If this option is false, only do the bare minimum for correctness.
  Option<bool> optimize{*this, "optimize", llvm::cl::desc("Do optimizations."),
                        llvm::cl::init(false)};
};

// The main pipeline that encapsulates the full E2E lowering.
void createE2ELoweringPipeline(OpPassManager &pm,
                               const E2ELoweringPipelineOptions &options);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_E2E_E2E_H
