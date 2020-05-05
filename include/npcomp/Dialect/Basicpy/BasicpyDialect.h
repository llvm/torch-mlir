//===- BasicPyDialect.h - Basic Python --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H
#define NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "npcomp/Dialect/Common.h"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

namespace BasicpyTypes {
enum Kind {
  PlaceholderType = TypeRanges::Basicpy,
  LAST_BASICPY_TYPE = PlaceholderType
};
} // namespace BasicpyTypes

#include "npcomp/Dialect/Basicpy/BasicpyOpsDialect.h.inc"

} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_BASICPY_DIALECT_H
