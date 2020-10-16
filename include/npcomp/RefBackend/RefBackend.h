//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_REFBACKEND_REFBACKEND_H
#define NPCOMP_REFBACKEND_REFBACKEND_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace NPCOMP {

/// Registers all RefBackend passes.
void registerRefBackendPasses();

// Look in createRefBackendLoweringPipeline for more information about how these
// passes fit together.
//
// Pass summaries are in Passes.td.

std::unique_ptr<OperationPass<ModuleOp>>
createLowerConstantTensorsToMemrefPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerStructuralToMemrefPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToRefbackrtABIPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerAllocMemRefOpsPass();

std::unique_ptr<OperationPass<ModuleOp>> createLowerToLLVMPass();

std::unique_ptr<Pass> createRestrictedCanonicalizerPass();

struct RefBackendLoweringPipelineOptions
    : public PassPipelineOptions<RefBackendLoweringPipelineOptions> {
  // If this option is true, then perform optimizations.
  // If this option is false, only do the bare minimum for correctness.
  Option<bool> optimize{*this, "optimize", llvm::cl::desc("Do optimizations."),
                        llvm::cl::init(false)};
};

// The main pipeline that encapsulates the full RefBackend lowering.
void createRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options);

// Helper pipeline that runs TCF->TCP lowering before invoking
// RefBackendLoweringPipeline.
// For now, just piggy-back on the same set of options since this is such a
// thin wrapper.
// Longer-term, the reference backend should fit into some sort of
// "target interface" and this helper won't be needed.
void createTCFRefBackendLoweringPipeline(
    OpPassManager &pm, const RefBackendLoweringPipelineOptions &options);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_REFBACKEND_REFBACKEND_H
