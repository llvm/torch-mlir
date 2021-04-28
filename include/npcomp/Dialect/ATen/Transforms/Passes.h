//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_TRANSFORMS_PASSES_H
#define NPCOMP_DIALECT_ATEN_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {
namespace aten {

std::unique_ptr<OperationPass<FuncOp>> createRecognizeKernelsPass();

std::unique_ptr<OperationPass<ModuleOp>> createATenOpReportPass();
// Return the report in the given output string.
std::unique_ptr<OperationPass<ModuleOp>>
createATenOpReportPass(std::string &output);

std::unique_ptr<OperationPass<ModuleOp>> createATenLayerNamePass();
std::unique_ptr<OperationPass<ModuleOp>> createReturnEliminationPass();
} // namespace aten

/// Registers all ATen transformation passes.
void registerATenPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_ATEN_TRANSFORMS_PASSES_H
