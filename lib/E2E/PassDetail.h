//===- PassDetail.h - E2E Pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef E2E_PASSDETAIL_H
#define E2E_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace NPCOMP {

#define GEN_PASS_CLASSES
#include "npcomp/E2E/Passes.h.inc"

} // namespace NPCOMP
} // end namespace mlir

#endif // E2E_PASSDETAIL_H
