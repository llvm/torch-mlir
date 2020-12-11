//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TCF_IR_TCFOPS_H
#define NPCOMP_DIALECT_TCF_IR_TCFOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/TCF/IR/TCFOps.h.inc"

#endif // NPCOMP_DIALECT_TCF_IR_TCFOPS_H
