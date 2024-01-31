//===- unbind_int.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "unbind_int.h"

namespace torch {
namespace lazy {

UnbindCopyInt::UnbindCopyInt(const torch::lazy::Value &self, const int64_t &dim,
                             std::vector<torch::lazy::Shape> &&shapes)
    : torch::lazy::TorchMlirNode(UnbindCopyInt::ClassOpKind(), OpList{self},
                                 std::move(shapes),
                                 self.shape().size(dim), /* num_outputs */
                                 torch::lazy::MHash(dim)),
      dim(dim) {}

std::string UnbindCopyInt::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  ss << ", dim=" << dim;
  return ss.str();
}

bool UnbindCopyInt::CanBeReused(const torch::lazy::Value &self,
                                const int64_t &dim) const {
  return false;
}

TorchMlirOpVector UnbindCopyInt::Lower(TorchMlirFunction function,
                                       TorchMlirLoweringContext *loctx) const {
  PRINT_FUNCTION();
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(2);
  kwarguments.reserve(0);
  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back("dim", dim);

  torch::lazy::TorchMlirOpVector unbind_copy_out =
      torch::lazy::LowerTorchMlirBuiltin(function, op().op, shapes(), arguments,
                                         kwarguments);

  return unbind_copy_out;
}

} // namespace lazy
} // namespace torch
