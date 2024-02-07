//===- generic.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/generic.cpp
//===----------------------------------------------------------------------===//

#include "generic.h"

namespace torch {
namespace lazy {

Generic::Generic(OpKind op, OpList operands, Shape shape, size_t num_outputs,
                 hash_t hash_seed)
    : TorchMlirNode(op, operands, {std::move(shape)}, num_outputs, hash_seed),
      hash_seed_(hash_seed) {}

} // namespace lazy
} // namespace torch
