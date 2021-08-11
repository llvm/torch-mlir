//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_TORCHCONVERSION_IR_TORCHOPS_H
#define NPCOMP_DIALECT_TORCHCONVERSION_IR_TORCHOPS_H

#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "npcomp/Dialect/Torch/IR/TorchTypes.h"
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#define GET_OP_CLASSES
#include "npcomp/Dialect/TorchConversion/IR/TorchConversionOps.h.inc"

#endif // NPCOMP_DIALECT_TORCHCONVERSION_IR_TORCHOPS_H
