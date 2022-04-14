//===- mlir_node.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/torch/csrc/lazy/ts_backend/ts_node.cpp
//===----------------------------------------------------------------------===//

#include <torch/csrc/lazy/core/cache.h>

#include "../utils/exception.h"
#include "mlir_node.h"

namespace torch {
namespace lazy {

TorchMlirOpVector TorchMlirNode::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  return {};
}

} // namespace lazy
} // namespace torch
