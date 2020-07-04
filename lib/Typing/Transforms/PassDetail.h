//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_TYPING_TRANSFORMS_PASSDETAIL_H
#define NPCOMP_TYPING_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace NPCOMP {
namespace Typing {

#define GEN_PASS_CLASSES
#include "npcomp/Typing/Transforms/Passes.h.inc"

} // namespace Typing
} // namespace NPCOMP
} // end namespace mlir

#endif // NPCOMP_TYPING_TRANSFORMS_PASSDETAIL_H
