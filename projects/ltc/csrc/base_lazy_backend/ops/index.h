//===- index.h ------------------------------------------------------------===//
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

class IndexTensor : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::aten::index);
  }

  IndexTensor(const torch::lazy::Value &self, const torch::lazy::Value &indices,
              std::vector<torch::lazy::Shape> &&shapes);

  std::string ToString() const override;

  bool CanBeReused(const torch::lazy::Value &self,
                   const torch::lazy::Value &indices) const;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;
};

class IndexPut : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::aten::index_put);
  }

  IndexPut(const torch::lazy::Value &self, const torch::lazy::Value &indices,
           const torch::lazy::Value &values, bool accumulate,
           std::vector<torch::lazy::Shape> &&shapes);

  std::string ToString() const override;

  bool CanBeReused(const torch::lazy::Value &self,
                   const torch::lazy::Value &indices,
                   const torch::lazy::Value &values, bool accumulate) const;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;

  bool accumulate;
};

} // namespace lazy
} // namespace torch
