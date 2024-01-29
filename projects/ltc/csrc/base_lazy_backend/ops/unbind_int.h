//===- unbind_int.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../mlir_node.h"

namespace torch {
namespace lazy {

class UnbindCopyInt : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::aten::unbind_copy);
  }

  UnbindCopyInt(const torch::lazy::Value &self, const int64_t &dim,
                std::vector<torch::lazy::Shape> &&shapes);

  std::string ToString() const override;

  bool CanBeReused(const torch::lazy::Value &self, const int64_t &dim) const;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;

  int64_t dim;
};

} // namespace lazy
} // namespace torch