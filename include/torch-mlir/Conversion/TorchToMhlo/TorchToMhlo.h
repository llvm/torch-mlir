//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H
#define TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace torch {
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToMhloPass();
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOMHLO_TORCHTOMHLO_H
