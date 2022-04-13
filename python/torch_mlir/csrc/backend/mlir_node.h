//===- mlir_node.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_node.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>

#include "../utils/debug.h"
#include "../utils/exception.h"
#include "mlir_lowering_context.h"

namespace torch {
namespace lazy {

typedef std::vector<torch::jit::Value*> TorchMlirOpVector;
typedef std::shared_ptr<torch::jit::GraphFunction> TorchMlirFunction;

class TORCH_API TorchMlirNode : public torch::lazy::Node {
public:
  using torch::lazy::Node::Node;

  virtual TorchMlirOpVector
  Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const;
};

} // namespace lazy
} // namespace torch
