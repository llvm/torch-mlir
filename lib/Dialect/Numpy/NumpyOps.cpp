//===- NumpyOps.cpp - Core numpy dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Numpy/NumpyOps.h"
#include "mlir/IR/OpImplementation.h"
#include "npcomp/Dialect/Numpy/NumpyDialect.h"

namespace mlir {
namespace npcomp {
namespace NUMPY {
#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/NumpyOps.cpp.inc"
} // namespace NUMPY
} // namespace npcomp
} // namespace mlir
