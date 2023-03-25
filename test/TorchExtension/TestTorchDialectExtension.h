//===- TestTorchDialectExtension.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Torch dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTORCHDIALECTEXTENSION_H
#define MLIR_TESTTORCHDIALECTEXTENSION_H

#include "mlir/IR/OpImplementation.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTraits.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

#define GET_OP_CLASSES
#include "TestTorchDialectExtension.h.inc"

namespace test {
/// Registers the test extension to the Torch dialect.
void registerTestTorchDialectExtension(::mlir::DialectRegistry &registry);
} // namespace test

#endif // MLIR_TESTTORCHDIALECTEXTENSION_H
