//===- dynamic_ir.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/dynamic_ir.cpp
//===----------------------------------------------------------------------===//

#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>

namespace torch {
namespace lazy {

DimensionNode::DimensionNode(OpKind op, OpList operands, hash_t hash_seed)
    : TorchMlirNode(
          op, operands, /*num_outputs=*/1,
          /* hash_seed */ HashCombine(op.hash(), hash_seed)) {}

std::string DimensionNode::ToString() const { return "DimensionNode"; }

SizeNode::SizeNode(Value input, size_t dim)
    : DimensionNode(
          OpKind{c10::Symbol::fromQualString("aten::size")}, {input},
          MHash(dim)),
      dim_(dim){};

int64_t SizeNode::getStaticValue() const {
  return dynamic_cast<const TorchMlirNode*>(operand(0).node)
      ->shape(0)
      .size(dim_);
}

std::string SizeNode::ToString() const { return "SizeNode"; }

SizeAdd::SizeAdd(Value a, Value b)
    : DimensionNode(OpKind{c10::Symbol::fromQualString("aten::add")}, {a, b}){};

int64_t SizeAdd::getStaticValue() const {
  return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() +
         dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeAdd::ToString() const { return "SizeAdd"; }

SizeMul::SizeMul(Value a, Value b)
    : DimensionNode(OpKind{c10::Symbol::fromQualString("aten::mul")}, {a, b}){};

int64_t SizeMul::getStaticValue() const {
  return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() *
         dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeMul::ToString() const { return "SizeMul"; }

SizeDiv::SizeDiv(Value a, Value b)
    : DimensionNode(OpKind{c10::Symbol::fromQualString("aten::div")}, {a, b}){};

int64_t SizeDiv::getStaticValue() const {
  TORCH_CHECK(
      dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue() !=
          0,
      "Can't divide a dimension by zero");
  return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() /
         dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeDiv::ToString() const { return "SizeDiv"; }

} // namespace lazy
} // namespace torch
