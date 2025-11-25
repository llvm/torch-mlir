//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_REFBACKEND_PASSES_H
#define TORCHMLIR_REFBACKEND_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace torch {
namespace RefBackend {

/// Registers all RefBackend passes.
void registerRefBackendPasses();

} // namespace RefBackend
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_REFBACKEND_PASSES_H
