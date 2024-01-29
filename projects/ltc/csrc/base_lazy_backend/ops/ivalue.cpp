//===- ivalue.cpp
//----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "ivalue.h"

#include <ATen/core/ivalue.h>

namespace torch {
namespace lazy {

IValueConstant::IValueConstant(const c10::IValue &value)
    : torch::lazy::TorchMlirNode(IValueConstant::ClassOpKind(), OpList{},
                                 std::vector<Shape>{},
                                 /* num_outputs */ 1, torch::lazy::MHash()),
      value(value) {}

std::string IValueConstant::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TorchMlirNode::ToString();
  return ss.str();
}

TorchMlirOpVector IValueConstant::Lower(TorchMlirFunction function,
                                        TorchMlirLoweringContext *loctx) const {
  return {loctx->graph()->insertConstant(value)};
}

} // namespace lazy
} // namespace torch
