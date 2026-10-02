//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_H
#define TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::torch::onnx_c {

/// Target-dependent choices that some conversions consult. The defaults keep
/// the target-independent lowering.
struct OnnxConversionOptions {
  /// See the gru-split-gates-min-elements option of
  /// convert-torch-onnx-to-torch.
  int64_t gruSplitGatesMinElements = 0;
};

std::unique_ptr<OperationPass<func::FuncOp>> createTorchOnnxToTorchPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createTorchOnnxToTorchPass(const OnnxConversionOptions &options);

/// Registers all torch-mlir conversion passes.
void registerTorchOnnxToTorchPasses();

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNX_TO_TORCH_H
