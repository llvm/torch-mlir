//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BACKEND_IREE_PASSES_H
#define NPCOMP_BACKEND_IREE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace NPCOMP {
namespace IREEBackend {
/// Registers all IREEBackend passes.
void registerIREEBackendPasses();

/// Create a pipeline that runs all passes needed to lower the npcomp backend
/// contract to IREE's frontend contract.
void createNpcompBackendToIreeFrontendPipeline(OpPassManager &pm);

std::unique_ptr<OperationPass<ModuleOp>> createLowerLinkagePass();

} // namespace IREEBackend
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_BACKEND_IREE_PASSES_H
