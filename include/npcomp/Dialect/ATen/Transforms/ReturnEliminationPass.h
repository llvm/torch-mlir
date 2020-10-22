//===- ReturnEliminationPass.h ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_RETURNELIMINATIONPASS_H
#define NPCOMP_DIALECT_ATEN_RETURNELIMINATIONPASS_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace NPCOMP {
namespace aten {

std::unique_ptr<mlir::Pass> createReturnEliminationPass();
void registerReturnEliminationPass();
} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif
