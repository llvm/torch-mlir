//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
#define TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace torch {
namespace TorchConversion {

std::unique_ptr<OperationPass<ModuleOp>> createFuncBackendTypeConversionPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createFinalizingBackendTypeConversionPass();

// These passes do a one-off conversion of a specific kind of quantized group
// matmul as a prototype. Generalized quantized operation handling will likely
// obviate them but that are being carried for now in order to unblock progress
// on full integrations. See https://github.com/llvm/torch-mlir/issues/2417 for
// the plan to support a more generalized lowering for these graphs.
std::unique_ptr<OperationPass<func::FuncOp>> createUnpackQuantTensorPass();
std::unique_ptr<OperationPass<func::FuncOp>> createConvertCustomQuantOpPass();

} // namespace TorchConversion

/// Registers all Torch transformation passes.
void registerTorchConversionPasses();

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCHCONVERSION_TRANSFORMS_PASSES_H
