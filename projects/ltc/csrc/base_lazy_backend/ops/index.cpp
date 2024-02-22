//===- index.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "index.h"

namespace torch {
namespace lazy {

IndexTensor::IndexTensor(const torch::lazy::Value &self,
                         const torch::lazy::Value &indices,
                         std::vector<torch::lazy::Shape> &&shapes)
    : torch::lazy::TorchMlirNode(IndexTensor::ClassOpKind(),
                                 OpList{self, indices}, std::move(shapes),
                                 /* num_outputs */ 1, torch::lazy::MHash()) {}

std::string IndexTensor::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  return ss.str();
}

bool IndexTensor::CanBeReused(const torch::lazy::Value &self,
                              const torch::lazy::Value &indices) const {
  return false;
}

TorchMlirOpVector IndexTensor::Lower(TorchMlirFunction function,
                                     TorchMlirLoweringContext *loctx) const {
  PRINT_FUNCTION();
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(2);
  kwarguments.reserve(0);

  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));

  torch::lazy::TorchMlirOpVector index_out = torch::lazy::LowerTorchMlirBuiltin(
      function, op().op, shapes(), arguments, kwarguments);
  TORCH_CHECK_EQ(index_out.size(), 1);

  return index_out;
}

IndexPut::IndexPut(const torch::lazy::Value &self,
                   const torch::lazy::Value &indices,
                   const torch::lazy::Value &values, bool accumulate,
                   std::vector<torch::lazy::Shape> &&shapes)
    : torch::lazy::TorchMlirNode(
          IndexPut::ClassOpKind(), OpList{self, indices, values},
          std::move(shapes),
          /* num_outputs */ 1, torch::lazy::MHash(accumulate)),
      accumulate(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  ss << ", accumulate=" << accumulate;
  return ss.str();
}

bool IndexPut::CanBeReused(const torch::lazy::Value &self,
                           const torch::lazy::Value &indices,
                           const torch::lazy::Value &values,
                           bool accumulate) const {
  return false;
}

TorchMlirOpVector IndexPut::Lower(TorchMlirFunction function,
                                  TorchMlirLoweringContext *loctx) const {
  PRINT_FUNCTION();
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(4);
  kwarguments.reserve(0);

  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back("accumulate", accumulate);

  torch::lazy::TorchMlirOpVector index_out = torch::lazy::LowerTorchMlirBuiltin(
      function, op().op, shapes(), arguments, kwarguments);

  TORCH_CHECK_EQ(index_out.size(), 1);

  return index_out;
}

} // namespace lazy
} // namespace torch
