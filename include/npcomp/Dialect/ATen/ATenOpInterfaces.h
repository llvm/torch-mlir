//===- ATenOpInterfaces.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_ATEN_OPINTERFACES_H
#define NPCOMP_DIALECT_ATEN_OPINTERFACES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace NPCOMP {
#include "npcomp/Dialect/ATen/ATenOpInterfaces.h.inc"
} // namespace NPCOMP
} // namespace mlir

#endif
