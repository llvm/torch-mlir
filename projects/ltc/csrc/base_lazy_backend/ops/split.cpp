//===- split.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "split.h"

namespace torch {
namespace lazy {

SplitWithSizesCopy::SplitWithSizesCopy(
    const torch::lazy::Value &self, const ::std::vector<int64_t> &split_sizes,
    const int64_t &dim, std::vector<torch::lazy::Shape> &&shapes)
    : torch::lazy::TorchMlirNode(SplitWithSizesCopy::ClassOpKind(),
                                 OpList{self}, std::move(shapes),
                                 split_sizes.size() /* num_outputs */,
                                 torch::lazy::MHash(split_sizes, dim)),
      split_sizes(split_sizes), dim(dim) {}

std::string SplitWithSizesCopy::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  ss << ", split_sizes=" << split_sizes;
  ss << ", dim=" << dim;
  return ss.str();
}

bool SplitWithSizesCopy::CanBeReused(const torch::lazy::Value &self,
                                     const ::std::vector<int64_t> &split_sizes,
                                     const int64_t &dim) const {
  return false;
}

TorchMlirOpVector
SplitWithSizesCopy::Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const {
  PRINT_FUNCTION();
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(3);
  kwarguments.reserve(0);
  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back("split_sizes", split_sizes);
  arguments.emplace_back("dim", dim);

  torch::lazy::TorchMlirOpVector split_with_sizes_copy_out =
      torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(), arguments,
                                         kwarguments);

  return split_with_sizes_copy_out;
}

SplitCopyTensor::SplitCopyTensor(const torch::lazy::Value &self,
                                 const torch::lazy::Value &split_size,
                                 const int64_t &dim,
                                 std::vector<torch::lazy::Shape> &&shapes,
                                 const size_t num_outputs)
    : torch::lazy::TorchMlirNode(SplitCopyTensor::ClassOpKind(),
                                 OpList{self, split_size}, std::move(shapes),
                                 num_outputs, torch::lazy::MHash(dim)),
      dim(dim) {}

std::string SplitCopyTensor::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  ss << ", dim=" << dim;
  return ss.str();
}

bool SplitCopyTensor::CanBeReused(const torch::lazy::Value &self,
                                  const torch::lazy::Value &split_size,
                                  const int64_t &dim) const {
  return false;
}

TorchMlirOpVector
SplitCopyTensor::Lower(TorchMlirFunction function,
                       TorchMlirLoweringContext *loctx) const {
  PRINT_FUNCTION();
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(3);
  kwarguments.reserve(0);
  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back("dim", dim);

  torch::lazy::TorchMlirOpVector split_copy_out =
      torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(), arguments,
                                         kwarguments);
  return split_copy_out;
}

} // namespace lazy
} // namespace torch
