//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#ifndef TORCHMLIR_DIALECT_TORCH_TRANSFORMS_STORAGEVIEWUTILS_H
#define TORCHMLIR_DIALECT_TORCH_TRANSFORMS_STORAGEVIEWUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mlir::torch::Torch {

/// Storage base for a traced tensor view chain.
///
/// `base` is the value before the suffix of supported view-like ops.
/// `storageOffset` is a PyTorch element offset relative to `base` storage.
struct StorageViewBase {
  Value base;
  int64_t storageOffset = 0;
};

/// Walk from `input` to the tensor whose storage `input` reads.
///
/// The walk crosses view ops only when their PyTorch storage offset can be
/// computed statically. A tensor with no modeled view producer returns
/// `{input, 0}` without requiring a static tensor shape.
///
/// Returns failure when offset/bounds computation needs dynamic metadata, when
/// int64 arithmetic overflows, or when the walk reaches an unmodeled view op.
FailureOr<StorageViewBase> traceViewLikeStorageBase(Value input);

/// Return success if PyTorch dense-view stride computation for `sizes` fits.
LogicalResult checkDenseViewShape(llvm::ArrayRef<int64_t> sizes);

} // namespace mlir::torch::Torch

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_STORAGEVIEWUTILS_H
