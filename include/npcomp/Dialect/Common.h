//===- Common.h - Common definitions for all dialects -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_COMMON_H
#define NPCOMP_DIALECT_COMMON_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace NPCOMP {

namespace TypeRanges {
enum {
  Basicpy = Type::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
  Numpy = Basicpy + 50,
};
}

} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_COMMON_H
