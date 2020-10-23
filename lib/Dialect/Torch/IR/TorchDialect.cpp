//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Torch/IR/TorchDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::NPCOMP::Torch;

void TorchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Torch/IR/TorchOps.cpp.inc"
      >();
}

#include "npcomp/Dialect/Torch/IR/OpInterfaces.h"
