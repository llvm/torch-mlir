//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H
#define TORCHMLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
namespace torch {
void populateTorchToLinalgOnTensorsPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns);
void populateTorchToLinalgOnTensorsOpsLegality(ConversionTarget &target);
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToLinalgPass();
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_ATENTOLINALG_ATENTOLINALG_H
