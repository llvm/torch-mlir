//===- NumpyOps.h - Core numpy dialect ops ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS_H
#define NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "npcomp/Interfaces/Traits.h"
#include "npcomp/Typing/Analysis/CPA/Interfaces.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h.inc"

#endif // NPCOMP_DIALECT_NUMPY_IR_NUMPY_OPS_H
