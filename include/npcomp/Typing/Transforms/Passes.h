//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_TRANSFORMS_PASSES_H
#define NPCOMP_TYPING_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace NPCOMP {
namespace Typing {

std::unique_ptr<OperationPass<FuncOp>> createCPAFunctionTypeInferencePass();

} // namespace Typing

/// Registers all typing passes.
void registerTypingPasses();

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_TYPING_TRANSFORMS_PASSES_H
