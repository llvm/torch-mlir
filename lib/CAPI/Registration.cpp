//===- Registration.cpp - C Interface for MLIR Registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "npcomp/InitAll.h"

void npcompRegisterAllDialects(MlirContext context) {
  mlir::NPCOMP::registerAllDialects(unwrap(context)->getDialectRegistry());
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->getDialectRegistry().loadAll(unwrap(context));
}
