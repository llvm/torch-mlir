//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_RD_IR_RDOPS_H
#define NPCOMP_DIALECT_RD_IR_RDOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "npcomp/Dialect/RD/IR/RDDatasetInterface.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/RD/IR/RDOps.h.inc"

#endif // NPCOMP_DIALECT_RD_IR_RDOPS_H
