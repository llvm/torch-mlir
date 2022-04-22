//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_TRANSFORMS_PASSES_H_
#define TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {
namespace TMTensor {

std::unique_ptr<OperationPass<func::FuncOp>> createTMTensorToLoopsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createTMTensorBufferizePass();

void registerPasses();

} // namespace TMTensor
} // namespace torch
} // namespace mlir

#endif // TORCH_MLIR_DIALECTS_DIALECT_TMTENSOR_TRANSFORMS_PASSES_H_
