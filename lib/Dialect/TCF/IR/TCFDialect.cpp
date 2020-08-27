//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::tcf;

void TCFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/TCF/IR/TCFOps.cpp.inc"
      >();
}
