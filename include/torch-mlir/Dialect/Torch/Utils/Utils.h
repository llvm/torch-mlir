//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_UTILS_H

#include "mlir/Support/LLVM.h"

namespace mlir {
namespace torch {
namespace Torch {

int64_t toPositiveDim(int64_t dim, int64_t inputRank);
bool isValidDim(int64_t dim, int64_t inputRank);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_UTILS_H
