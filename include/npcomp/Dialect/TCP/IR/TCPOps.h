//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TCP_IR_TCPOPS_H
#define NPCOMP_DIALECT_TCP_IR_TCPOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace NPCOMP {
namespace tcp {

#define GET_OP_CLASSES
#include "npcomp/Dialect/TCP/IR/TCPOps.h.inc"

} // namespace tcp
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_TCP_IR_TCPOPS_H
