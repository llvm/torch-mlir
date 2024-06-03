//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_INITALL_H
#define TORCH_MLIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace torch {

// Registers all dialects that this project produces and any dependencies.
void registerAllDialects(mlir::DialectRegistry &registry);

// Registers all necessary dialect extensions for this project
void registerAllExtensions(mlir::DialectRegistry &registry);

// Registers dialects that may be needed to parse torch-mlir inputs and
// test cases.
void registerOptionalInputDialects(mlir::DialectRegistry &registry);

void registerAllPasses();

} // namespace torch
} // namespace mlir

#endif // TORCH_MLIR_INITALL_H
