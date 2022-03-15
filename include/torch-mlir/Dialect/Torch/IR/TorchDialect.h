//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H
#define TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H

#include "mlir/IR/Dialect.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h.inc"

namespace mlir {
namespace torch {
namespace Torch {

/// Parse a type registered to this dialect.
Type parseTorchDialectType(AsmParser &parser);

/// Print a type registered to this dialect.
void printTorchDialectType(Type type, AsmPrinter &printer);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_IR_TORCHDIALECT_H
