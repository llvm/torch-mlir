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

namespace {

hash_t OperandHashes(const OpList& operands, const hash_t& seed, bool bakeInSizes) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash = operand.hash();
    hash = HashCombine(hash, operand_hash);
  }
  return hash;
}

hash_t GetOpHash(OpKind op, const Shape& shape, hash_t hash_seed, bool bakeInSizes) {
  hash_t h = HashCombine(op.hash(), shape.hash(bakeInSizes));
  return HashCombine(h, hash_seed);
}

} // namespace

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands, std::vector<Shape>&& shapes, size_t num_outputs, hash_t hash_seed)
    : Node(op, operands, std::move(shapes), num_outputs) {
  hash_seed = HashCombine(op.hash(), hash_seed);
  shape_hash_ = OperandHashes(operands, hash_seed, true);
  dag_hash_ = (enableDynamicShape() ? OperandHashes(operands, hash_seed, false) : shape_hash_);
}

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
               size_t num_outputs, hash_t hash_seed)
    : TorchMlirNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  addComputedShape(shape_fn);
}

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
    : TorchMlirNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

TorchMlirNode::TorchMlirNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs),
      shape_hash_(GetOpHash(op, shape, hash_seed, true)),
      dag_hash_(enableDynamicShape() ? GetOpHash(op, shape, hash_seed, false) : shape_hash_) {
  shapes_.push_back(std::move(shape));
}

hash_t TorchMlirNode::hash() const { return dag_hash_; }

hash_t TorchMlirNode::shapeHash() const { return shape_hash_; }

TorchMlirOpVector TorchMlirNode::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  return {};
}

} // namespace lazy
} // namespace torch
