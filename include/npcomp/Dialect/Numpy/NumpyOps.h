//===- NumpyOps.h - Core numpy dialect ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_OPS_H
#define NPCOMP_DIALECT_NUMPY_NUMPY_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffects.h"

namespace mlir {
namespace npcomp {
namespace NUMPY {

#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/NumpyOps.h.inc"

} // namespace NUMPY
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_OPS_H
