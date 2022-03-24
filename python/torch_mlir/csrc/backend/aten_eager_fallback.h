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
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.h
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/native/CPUFallback.h>

namespace torch_lazy_tensors {

bool force_eager_fallback(c10::Symbol op);
void ltc_eager_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack);

extern TORCH_API std::function<void(void)> register_mlir_ltc_eager_fallback;

} // namespace torch_lazy_tensors
