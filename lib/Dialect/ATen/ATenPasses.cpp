//===- ATenPasses.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/ATenPasses.h"

using namespace mlir::NPCOMP::aten;

void mlir::NPCOMP::aten::registerATenPasses() {
  // TODO: Use the automatically generated pass registration.
  // #define GEN_PASS_REGISTRATION
  //   #include "npcomp/Dialect/ATen/ATenPasses.h.inc"
  registerATenLayerNamePass();
  registerATenOpReportPass();
  registerATenLoweringPass();
  registerReturnEliminationPass();
}
