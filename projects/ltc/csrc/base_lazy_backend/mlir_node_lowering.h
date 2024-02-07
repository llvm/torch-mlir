//===- mlir_node_lowering.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_node_lowering.h
//===----------------------------------------------------------------------===//

#pragma once

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/lazy/backend/lowering_context.h>

namespace torch {
namespace lazy {

typedef std::vector<torch::jit::Value *> TorchMlirOpVector;
typedef std::shared_ptr<torch::jit::GraphFunction> TorchMlirFunction;

TORCH_API TorchMlirOpVector LowerTorchMlirBuiltin(
    TorchMlirFunction function, c10::Symbol sym,
    const c10::ArrayRef<Shape> result_shapes,
    const std::vector<torch::jit::NamedValue> &arguments,
    const std::vector<torch::jit::NamedValue> &kwarguments = {});

} // namespace lazy
} // namespace torch
