//===- Traits.cpp ------------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Interfaces/Traits.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::NPCOMP;
bool mlir::NPCOMP::allowsTypeRefinement(Operation *op) {
  if (op->hasTrait<NPCOMP::OpTrait::AllowsTypeRefinement>())
    return true;
  return isa<RankOp>(op);
}
