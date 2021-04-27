//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "npcomp/Backend/Common/Passes.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::CommonBackend;

namespace {
#define GEN_PASS_REGISTRATION
#include "npcomp/Backend/Common/Passes.h.inc"
} // end namespace

void mlir::NPCOMP::CommonBackend::registerCommonBackendPasses() {
  ::registerPasses();
}
