//===- split.h ------------------------------------------------------------===//
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

class SplitWithSizesCopy : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::aten::split_with_sizes_copy);
  }

  SplitWithSizesCopy(const torch::lazy::Value &self,
                     const ::std::vector<int64_t> &split_sizes,
                     const int64_t &dim,
                     std::vector<torch::lazy::Shape> &&shapes);

  std::string ToString() const override;

  bool CanBeReused(const torch::lazy::Value &self,
                   const ::std::vector<int64_t> &split_sizes,
                   const int64_t &dim) const;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;

  std::vector<int64_t> split_sizes;
  int64_t dim;
};

class SplitCopyTensor : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::aten::split_copy);
  }

  SplitCopyTensor(const torch::lazy::Value &self,
                  const torch::lazy::Value &split_size, const int64_t &dim,
                  std::vector<torch::lazy::Shape> &&shapes,
                  const size_t num_outputs = 1);

  std::string ToString() const override;

  bool CanBeReused(const torch::lazy::Value &self,
                   const torch::lazy::Value &split_size,
                   const int64_t &dim) const;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;

  int64_t dim;
};

} // namespace lazy
} // namespace torch
