//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H
#define NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

#define GEN_PASS_CLASSES
#include "npcomp/Dialect/Basicpy/Transforms/Passes.h.inc"

} // namespace Basicpy
} // namespace NPCOMP
} // end namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_TRANSFORMS_PASSDETAIL_H
