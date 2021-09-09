//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/InitAll.h"

#include "mlir/IR/Dialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

void mlir::torch::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::torch::Torch::TorchDialect>();
}

void mlir::torch::registerAllPasses() { mlir::torch::registerTorchPasses(); }
