//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NPCOMP_DIALECT_RD_IR_RDDATASETINTERFACE_H
#define NPCOMP_DIALECT_RD_IR_RDDATASETINTERFACE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace NPCOMP {
namespace rd {
/// A mapping from arguments in the @definition function to function arguments
/// in the init function.
using InitArgMap = llvm::DenseMap<Value, Value>;
}  // namespace rd
}  // namespace NPCOMP
}  // namepsace mlir

// Include the generated interface declaration.
#include "npcomp/Dialect/RD/IR/RDDatasetInterface.h.inc"

#endif // NPCOMP_DIALECT_RD_IR_RDDATASETINTERFACE_H
