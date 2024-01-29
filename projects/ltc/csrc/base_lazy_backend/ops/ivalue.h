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

// IValueConstant IR Node represents a `prim::Constant` constructed with IValue
// parameter which is helpful in different usecases when we need custom
// native ops lowering to torch-mlir IR nodes.
class IValueConstant : public torch::lazy::TorchMlirNode {
public:
  static torch::lazy::OpKind ClassOpKind() {
    return torch::lazy::OpKind(at::prim::Constant);
  }

  IValueConstant(const c10::IValue &value);

  std::string ToString() const override;

  TorchMlirOpVector Lower(TorchMlirFunction function,
                          TorchMlirLoweringContext *loctx) const override;

  c10::IValue value;
};

} // namespace lazy
} // namespace torch
