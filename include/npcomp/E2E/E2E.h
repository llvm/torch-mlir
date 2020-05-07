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

std::unique_ptr<OperationPass<FuncOp>> createLowerBroadcastToToLoopsPass();

std::unique_ptr<OperationPass<FuncOp>>
createLowerLinalgOnTensorToLinalgOnMemrefPass();

std::unique_ptr<OperationPass<FuncOp>>
createResolveShapeOfOpsPass();

void createLowerToHybridTensorMemRefPipeline(OpPassManager &pm);

// The main pipeline that encapsulates the full E2E lowering.
void createE2ELoweringPipeline(OpPassManager &pm);

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_E2E_E2E_H
