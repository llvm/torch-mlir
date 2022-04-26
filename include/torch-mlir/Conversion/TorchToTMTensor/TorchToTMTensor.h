//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOTMTENSOR_TORCHTOTMTENSOR_H
#define TORCHMLIR_CONVERSION_TORCHTOTMTENSOR_TORCHTOTMTENSOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToTMTensorPass();
}
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTMTENSOR_TORCHTOTMTENSOR_H
