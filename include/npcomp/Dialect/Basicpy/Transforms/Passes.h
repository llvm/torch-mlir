//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSES_H
#define NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

std::unique_ptr<OperationPass<FuncOp>> createFunctionTypeInferencePass();

} // namespace Basicpy

/// Registers all Basicpy transformation passes.
void registerBasicpyPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSES_H
