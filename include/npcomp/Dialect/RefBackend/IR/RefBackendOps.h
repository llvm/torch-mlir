//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_REFBACKEND_IR_REFBACKENDOPS_H
#define NPCOMP_DIALECT_REFBACKEND_IR_REFBACKENDOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/RefBackend/IR/RefBackendOps.h.inc"

#endif // NPCOMP_DIALECT_REFBACKEND_IR_REFBACKENDOPS_H
