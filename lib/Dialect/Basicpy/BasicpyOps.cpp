//===- BasicpyOps.cpp - Core numpy dialect ops --------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Dialect/Basicpy/BasicpyOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "npcomp/Dialect/Basicpy/BasicpyDialect.h"

namespace mlir {
namespace NPCOMP {
namespace Basicpy {
#define GET_OP_CLASSES
#include "npcomp/Dialect/Basicpy/BasicpyOps.cpp.inc"
} // namespace Basicpy
} // namespace NPCOMP
} // namespace mlir
