//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H
#define NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "npcomp/Dialect/Torch/IR/OpInterfaces.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/Torch/IR/TorchOps.h.inc"

#endif // NPCOMP_DIALECT_TORCH_IR_TORCHOPS_H
