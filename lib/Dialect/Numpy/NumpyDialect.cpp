//===- NumpyDialect.cpp - Core numpy dialect --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/NumpyOps.h"

using namespace mlir;
using namespace mlir::npcomp::NUMPY;

NumpyDialect::NumpyDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "npcomp/Dialect/Numpy/NumpyOps.cpp.inc"
      >();
}
