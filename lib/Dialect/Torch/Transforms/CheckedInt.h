//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_TRANSFORMS_CHECKEDINT_H
#define TORCHMLIR_DIALECT_TORCH_TRANSFORMS_CHECKEDINT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

namespace mlir::torch::Torch {

/// Return failure instead of wrapping on signed int64 overflow.
inline FailureOr<int64_t> checkedAdd(int64_t lhs, int64_t rhs) {
  int64_t result;
  if (llvm::AddOverflow(lhs, rhs, result))
    return failure();
  return result;
}

/// Return failure instead of wrapping on signed int64 underflow/overflow.
inline FailureOr<int64_t> checkedSub(int64_t lhs, int64_t rhs) {
  int64_t result;
  if (llvm::SubOverflow(lhs, rhs, result))
    return failure();
  return result;
}

/// Return failure instead of wrapping on signed int64 multiplication overflow.
inline FailureOr<int64_t> checkedMul(int64_t lhs, int64_t rhs) {
  int64_t result;
  if (llvm::MulOverflow(lhs, rhs, result))
    return failure();
  return result;
}

/// Return failure unless `lhs * rhs + acc` fits in signed int64.
inline FailureOr<int64_t> checkedMulAdd(int64_t lhs, int64_t rhs, int64_t acc) {
  FailureOr<int64_t> product = checkedMul(lhs, rhs);
  if (failed(product))
    return failure();
  return checkedAdd(*product, acc);
}

/// Return failure unless the product of all values fits in signed int64.
inline FailureOr<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    FailureOr<int64_t> product = checkedMul(result, value);
    if (failed(product))
      return failure();
    result = *product;
  }
  return result;
}

} // namespace mlir::torch::Torch

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_CHECKEDINT_H
