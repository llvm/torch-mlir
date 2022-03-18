//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities referenced by GeneratedTorchOps.cpp.
//
// The utilities defined here are only meant for use by GeneratedTorchOps.cpp.
// If something is of wider use, then it should be moved elsewhere.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace torch {
namespace Torch {

// Parse a generated Torch op in the default format.
ParseResult parseDefaultTorchOp(OpAsmParser &parser, OperationState &result,
                                int numOperands, int numResults);

// Print a generated Torch op in the default format.
void printDefaultTorchOp(OpAsmPrinter &p, Operation *op, int numOperands,
                         int numResults);

} // namespace Torch
} // namespace torch
} // namespace mlir
