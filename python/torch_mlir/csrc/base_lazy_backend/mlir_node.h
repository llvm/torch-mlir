//===- mlir_node.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_node.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>

#include "../utils/debug.h"
#include "../utils/exception.h"
#include "mlir_lowering_context.h"

namespace torch {
namespace lazy {

class TORCH_API TorchMlirNode : public torch::lazy::Node {
public:
  TorchMlirNode(OpKind op, OpList operands, std::vector<Shape>&& shapes,
                size_t num_outputs, hash_t hash_seed = kHashSeed);

  TorchMlirNode(OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
                size_t num_outputs, hash_t hash_seed = kHashSeed);

  TorchMlirNode(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed = kHashSeed);

  TorchMlirNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed = kHashSeed);

  hash_t hash() const override;

  hash_t shapeHash() const override;

  virtual TorchMlirOpVector
  Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const;

private:
  // The hash of the dag WITH size info. Used for shape caching
  hash_t shape_hash_;
  // The hash of the dag used to look up the compiled graph by a hash
  // in this case, we will use the dag hash WITHOUT size info if dynamic shape is enabled
  // and use the dag hash WITH size info otherwise.
  hash_t dag_hash_;
};

} // namespace lazy
} // namespace torch
