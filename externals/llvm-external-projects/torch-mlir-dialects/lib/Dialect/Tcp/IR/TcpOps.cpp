//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

#define GET_OP_CLASSES
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.cpp.inc"

namespace mlir::tcp {
LogicalResult BroadcastOp::verify() {
  auto compareIntAttr = [](Attribute v1, Attribute v2) {
    return v1.cast<IntegerAttr>().getInt() < v2.cast<IntegerAttr>().getInt();
  };

  if (!llvm::is_sorted(getAxes(), compareIntAttr))
    return emitOpError(
        "failed to verify that attribute `axes` must be in increasing order");

  if (std::adjacent_find(std::begin(getAxes()), std::end(getAxes())) !=
      std::end(getAxes()))
    return emitOpError(
        "failed to verify that attribute `axes` must not have any duplicates");

  if (getNewDimSizes().size() != getAxes().size())
    return emitOpError("failed to verify that argument `new_dim_sizes` has the "
                       "same size as the attribute `axes`");

  return success();
}
} // namespace mlir::tcp
