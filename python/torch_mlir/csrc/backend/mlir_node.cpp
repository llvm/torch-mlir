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

hash_t OperandHashes(
    const OpList& operands, const hash_t& seed, const bool bakeInSizes) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash =
        bakeInSizes ? operand.hash_with_sizes() : operand.hash_without_sizes();
    hash = HashCombine(hash, operand_hash);
  }
  return hash;
}

hash_t GetOpHash(
    OpKind op, const Shape& shape, hash_t hash_seed, const bool bakeInSizes) {
  hash_t h = HashCombine(op.hash(), shape.hash(bakeInSizes));
  return HashCombine(h, hash_seed);
}

} // namespace

MlirNode::MlirNode(
    OpKind op, OpList operands, std::vector<Shape>&& shapes, size_t num_outputs,
    hash_t hash_seed)
    : Node(
          op, num_outputs,
          /* node_hash */ HashCombine(op.hash(), hash_seed),
          /* dag_hash */
          [&](bool bakeInSizes) -> hash_t {
            return OperandHashes(
                operands, HashCombine(op.hash(), hash_seed), bakeInSizes);
          }),
      shapes_(std::move(shapes)) {
  for (auto& operand : operands) {
    // Ideally, optional operands should be filtered by the leaf node classes,
    // but it's just much easier to do it here.
    if (!operand) {
      continue;
    }

    AddOperand(operand.node, operand.index);
  }
}

MlirNode::MlirNode(
    OpKind op, OpList operands, const std::function<Shape()>& shape_fn,
    size_t num_outputs, hash_t hash_seed)
    : MlirNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  shapes_.push_back(GetOpShape(shape_fn));
}

MlirNode::MlirNode(
    OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
    : MlirNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

void MlirNode::SetShapeDeferred(const std::function<Shape()>& shape_fn) {
  shapes_.push_back(GetOpShape(shape_fn));
}

MlirNode::MlirNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : Node(op, num_outputs, [&](bool bakeInSizes) -> hash_t {
        return GetOpHash(op, shape, hash_seed, bakeInSizes);
      }) {
  shapes_.push_back(std::move(shape));
}

using ShapeCache = Cache<hash_t, Shape, HashReducer>;

constexpr const int torch_lazy_shape_cache_size = 4096;

ShapeCache* GetShapeCache() {
  static ShapeCache* cache = new ShapeCache(torch_lazy_shape_cache_size);
  return cache;
}

Shape MlirNode::GetOpShape(const std::function<Shape()>& shape_fn) const {
  ShapeCache* shape_cache = GetShapeCache();
  auto shape = shape_cache->Get(hash());
  if (shape == nullptr) {
    shape = shape_cache->Add(hash(), std::make_shared<Shape>(shape_fn()));
  }
  return *shape;
}

c10::ArrayRef<Shape> MlirNode::shapes() const { return shapes_; }

const Shape& MlirNode::shape(size_t output_index) const {
  return shapes_.at(output_index);
}

const std::vector<Output>& MlirNode::operands() const {
  return operands_as_outputs_;
}

const Output& MlirNode::operand(size_t i) const {
  return operands_as_outputs_.at(i);
}

void MlirNode::AddOperand(NodePtr node, size_t index) {
  CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.emplace_back(operands_.back().get(), index);
}

} // namespace lazy
} // namespace torch
