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

std::unique_ptr<OperationPass<FuncOp>>
createResolveShapeOfOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createResolveTensorLoadStoreOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerLinalgLoopDimOpsPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerRankedShapesPass();

std::unique_ptr<OperationPass<FuncOp>> createLowerToMemRefABIPass();

void createLowerToHybridTensorMemRefPipeline(OpPassManager &pm);

// The main pipeline that encapsulates the full E2E lowering.
void createE2ELoweringPipeline(OpPassManager &pm);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_E2E_E2E_H
