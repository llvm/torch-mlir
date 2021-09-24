//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOSCF_TORCHTOSCF_H
#define TORCHMLIR_CONVERSION_TORCHTOSCF_TORCHTOSCF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {
std::unique_ptr<OperationPass<FuncOp>> createConvertTorchToSCFPass();
}
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOSCF_TORCHTOSCF_H
