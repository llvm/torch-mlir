//===- NumpyOps.cpp - Core numpy dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"

namespace mlir {
namespace NPCOMP {
namespace Numpy {
#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/IR/NumpyOps.cpp.inc"
} // namespace Numpy
} // namespace NPCOMP
} // namespace mlir
