//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_TORCHTOSTABLEHLO_H
#define TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_TORCHTOSTABLEHLO_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {

#define GEN_PASS_DECL_CONVERTTORCHTOSTABLEHLO
#include "torch-mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToStablehloPass();

// Convenience wrapper for users who want to pass options as individual
// parameters
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToStablehloPass(bool enableStaticShape, bool enableI32Index,
                                  bool supportsNonFinites);

} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOSTABLEHLO_TORCHTOSTABLEHLO_H
