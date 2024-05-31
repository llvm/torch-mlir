//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_SPARSITY_UTILS_H
#define TORCHMLIR_DIALECT_TORCH_SPARSITY_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace torch {
namespace Torch {

// Create a new SparseTensorEncodingAttr based on the provided `attr`, but with
// a new dense level inserted at `dim`.
FailureOr<Attribute> getSparsityWithDenseLTAtDim(Attribute attr, Value dim);

} // namespace Torch
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_SPARSITY_UTILS_H
