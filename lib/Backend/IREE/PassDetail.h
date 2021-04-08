//===- PassDetail.h - Pass class details ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BACKEND_IREE_PASSDETAIL_H
#define BACKEND_IREE_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace NPCOMP {
namespace IREEBackend {

#define GEN_PASS_CLASSES
#include "npcomp/Backend/IREE/Passes.h.inc"

} // namespace IREEBackend
} // namespace NPCOMP
} // end namespace mlir

#endif // BACKEND_IREE_PASSDETAIL_H
