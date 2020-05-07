//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TCP_IR_TCPDIALECT_H
#define NPCOMP_DIALECT_TCP_IR_TCPDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace NPCOMP {
namespace tcp {

#include "npcomp/Dialect/TCP/IR/TCPOpsDialect.h.inc"

} // namespace tcp
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TCP_IR_TCPDIALECT_H
