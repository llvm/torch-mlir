//===- TorchOps.cpp - C Interface for torch ops ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-c/TorchOps.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;
using namespace mlir::torch;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

MlirValue torchMlirAdjustStaticInformation(MlirBlock block_,
                                           MlirOperation insertBefore_,
                                           MlirValue value_,
                                           MlirType desiredType_,
                                           bool userAllowsRefinement) {
  Block *block = unwrap(block_);
  Operation *insertBefore = unwrap(insertBefore_);
  OpBuilder builder(unwrap(mlirTypeGetContext(desiredType_)));
  builder.setInsertionPoint(block, insertBefore ? insertBefore->getIterator()
                                                : block->end());
  Value value = unwrap(value_);
  Type desiredType = unwrap(desiredType_);
  return wrap(Torch::adjustStaticInformation(
      builder, value.getLoc(), value, desiredType, userAllowsRefinement));
}
