//===- aten_eager_fallback.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// Facilitates eager fallback behaviour
//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/csrc/ts_backend/ts_eager_fallback.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/native/CPUFallback.h>

namespace torch {
namespace lazy {

bool force_eager_fallback(c10::Symbol op);
void ltc_eager_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack);

// The MLIR LTC backend does not register itself with pytorch dispatcher
// until it is explicitly initialized.  This function should only be called
// by the main MLIR LTC backend init function.
TORCH_API void register_mlir_ltc_eager_fallback();

} // namespace lazy
} // namespace torch
