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

#include "mlir_node.h"
#include "utils/exception.h"

namespace torch {
namespace lazy {

namespace {

hash_t OperandHashes(const OpList &operands, const c10::ArrayRef<Shape> &shapes,
                     const hash_t &seed, bool bakeInSizes) {
  hash_t hash = seed;
  for (auto &operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash = bakeInSizes ? operand.shapeHash() : operand.hash();
    hash = HashCombine(hash, operand_hash);
  }
  for (auto &shape : shapes) {
    hash = HashCombine(hash, shape.hash(bakeInSizes));
  }
  return hash;
}

} // namespace

// Adds a static hook that is run after every single TorchMlirNode is
// initialized
static std::vector<std::function<void(TorchMlirNode *)>> constructor_hooks;
void TorchMlirNode::addConstructorHook(std::function<void(TorchMlirNode *)> f) {
  constructor_hooks.emplace_back(f);
}

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands,
                             std::vector<Shape> &&shapes, size_t num_outputs,
                             hash_t hash_seed)
    : Node(op, operands, std::move(shapes), num_outputs) {
  hash_seed = HashCombine(op.hash(), hash_seed);
  shape_hash_ = OperandHashes(operands, this->shapes(), hash_seed, true);
  dag_hash_ = (enableDynamicShape()
                   ? OperandHashes(operands, this->shapes(), hash_seed, false)
                   : shape_hash_);

  for (std::function<void(TorchMlirNode *)> &f : constructor_hooks) {
    f(this);
  }
}

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands,
                             const std::function<Shape()> &shape_fn,
                             size_t num_outputs, hash_t hash_seed)
    : TorchMlirNode(op, operands, std::vector<Shape>{}, num_outputs,
                    hash_seed) {
  addComputedShape(shape_fn);
}

TorchMlirNode::TorchMlirNode(OpKind op, OpList operands, size_t num_outputs,
                             hash_t hash_seed)
    : TorchMlirNode(op, operands, std::vector<Shape>{}, num_outputs,
                    hash_seed) {}

TorchMlirNode::TorchMlirNode(OpKind op, Shape shape, size_t num_outputs,
                             hash_t hash_seed)
    : TorchMlirNode(op, {}, {std::move(shape)}, num_outputs, hash_seed) {}

hash_t TorchMlirNode::hash() const { return dag_hash_; }

hash_t TorchMlirNode::shapeHash() const { return shape_hash_; }

TorchMlirNode *TorchMlirNode::mlir_node(int index) const {
  return dynamic_cast<TorchMlirNode *>(operands_.at(index).get());
}

///////////////////////////////////////////////////////////////////////////////
// TorchMlirTensorList
///////////////////////////////////////////////////////////////////////////////

OpKind TorchMlirTensorList::ClassOpKind() {
  // Note: this OpKind is separate from ltc_ops.h since it would be a circular
  // import otherwise
  static const OpKind tensor_list_opkind =
      OpKind::Get("lazy_tensors::tensor_list");
  return tensor_list_opkind;
}

TorchMlirTensorList::TorchMlirTensorList(OpList values)
    : TorchMlirNode(
          /*op=*/TorchMlirTensorList::ClassOpKind(),
          /*operands=*/values,
          /*shapes=*/std::vector<Shape>(),
          /*num_outputs=*/1,
          /*hash_seed=*/kHashSeed) {}

torch::lazy::TorchMlirOpVector
TorchMlirTensorList::Lower(TorchMlirFunction function,
                           TorchMlirLoweringContext *loctx) const {
  std::vector<torch::jit::Value *> tensor_list;
  CHECK(!operands().empty());
  for (const torch::lazy::Output &operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  auto graph = function->graph();
  auto listnode =
      graph->insertNode(graph->createList(c10::TensorType::get(), tensor_list));
  return {listnode->output()};
}

///////////////////////////////////////////////////////////////////////////////
// TorchMlirOptionalTensorList
///////////////////////////////////////////////////////////////////////////////

OpKind TorchMlirOptionalTensorList::ClassOpKind() {
  // Note: this OpKind is separate from ltc_ops.h since it would be a circular
  // import otherwise
  static const OpKind tensor_list_opkind =
      OpKind::Get("lazy_tensors::optional_tensor_list");
  return tensor_list_opkind;
}

TorchMlirOptionalTensorList::TorchMlirOptionalTensorList(OpList values)
    : TorchMlirNode(
          /*op=*/TorchMlirOptionalTensorList::ClassOpKind(),
          /*operands=*/values,
          /*shapes=*/std::vector<Shape>(),
          /*num_outputs=*/1,
          /*hash_seed=*/kHashSeed) {}

torch::lazy::TorchMlirOpVector
TorchMlirOptionalTensorList::Lower(TorchMlirFunction function,
                                   TorchMlirLoweringContext *loctx) const {
  std::vector<torch::jit::Value *> tensor_list;
  CHECK(!operands().empty());
  for (const torch::lazy::Output &operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  auto graph = function->graph();
  auto listnode = graph->insertNode(graph->createList(
      c10::OptionalType::create(c10::TensorType::get()), tensor_list));
  return {listnode->output()};
}

} // namespace lazy
} // namespace torch
