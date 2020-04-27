//===- NumpyDialect.h - Core numpy dialect ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H
#define NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace npcomp {
namespace NUMPY {

#include "npcomp/Dialect/Numpy/NumpyOpsDialect.h.inc"

} // namespace NUMPY
} // namespace npcomp
} // namespace mlir

#endif // NPCOMP_DIALECT_NUMPY_NUMPY_DIALECT_H
