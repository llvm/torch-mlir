//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToArithPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToSCFPass();

// Note that this only registers common conversion passes. Backend
// specific passes with their own Passes.h in a subdirectory must be
// included/registered explicitly as they are all optional.
void registerConversionPasses();

} // namespace torch
} // namespace mlir
