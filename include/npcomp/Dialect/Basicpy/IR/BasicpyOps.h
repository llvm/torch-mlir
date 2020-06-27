//===- BasicPyOps.h - Basic python ops --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_BASICPY_IR_BASICPY_OPS_H
#define NPCOMP_DIALECT_BASICPY_IR_BASICPY_OPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "npcomp/Dialect/Basicpy/IR/BasicpyOpsEnums.h.inc"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {

#define GET_OP_CLASSES
#include "npcomp/Dialect/Basicpy/IR/BasicpyOps.h.inc"

} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir

#endif // NPCOMP_DIALECT_BASICPY_IR_BASICPY_OPS_H
