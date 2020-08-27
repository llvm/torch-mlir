//===- ATenPasses.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_PASSES_H
#define NPCOMP_DIALECT_ATEN_PASSES_H

#include "npcomp/Dialect/ATen/ATenLayerNamePass.h"
#include "npcomp/Dialect/ATen/ATenLoweringPass.h"
#include "npcomp/Dialect/ATen/ATenOpReport.h"
#include "npcomp/Dialect/ATen/ReturnEliminationPass.h"

namespace mlir {
namespace NPCOMP {
namespace aten {
// #define GEN_PASS_CLASSES
// #include "npcomp/Dialect/ATen/ATenPasses.h.inc"

void registerATenPasses();
} // namespace aten
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_ATEN_PASSES_H
