//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace torch {
namespace Torch {

inline bool isTransformerEncoderOperatorName(llvm::StringRef name) {
  if (!name.consume_front(kTorchOpPrefix))
    return false;
  if (!name.consume_front("aten._transformer_encoder_layer_fwd"))
    return false;
  return name.empty() || name == ".default";
}

inline bool isTransformerEncoderOperator(Torch::OperatorOp op) {
  auto nameAttr = op.getNameAttr();
  if (!nameAttr)
    return false;
  return isTransformerEncoderOperatorName(nameAttr.getValue());
}

} // namespace Torch
} // namespace torch
} // namespace mlir
