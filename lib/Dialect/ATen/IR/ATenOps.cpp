//===- ATenDialect.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::aten;

//===----------------------------------------------------------------------===//
// IsOp
//===----------------------------------------------------------------------===//

OpFoldResult IsOp::fold(ArrayRef<Attribute> operands) {
  auto lhsType = self().getType();
  auto rhsType = obj().getType();
  // If either type is a NoneType, make it be the lhsType.
  if (rhsType.isa<Basicpy::NoneType>())
    std::swap(lhsType, rhsType);
  // TODO: Implement and use subtype infra for this.
  // If neither type is a subtype of the other, then the result is false.
  if (lhsType.isa<Basicpy::NoneType>() && !rhsType.isa<Torch::OptionalType>())
    return IntegerAttr::get(IntegerType::get(getContext(), 1), 0);
  return nullptr;
}

#define GET_OP_CLASSES
#include "npcomp/Dialect/ATen/IR/ATenOps.cpp.inc"

#include "npcomp/Dialect/ATen/IR/GeneratedATenOps.cpp.inc"
