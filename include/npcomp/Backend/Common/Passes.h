//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_BACKEND_COMMON_PASSES_H
#define NPCOMP_BACKEND_COMMON_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace NPCOMP {
namespace CommonBackend {
/// Registers all CommonBackend passes.
void registerCommonBackendPasses();

std::unique_ptr<OperationPass<ModuleOp>> createVerifyBackendContractPass();

} // namespace CommonBackend
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_BACKEND_COMMON_PASSES_H
