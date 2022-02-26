//===- aten_eager_fallback.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/aten_eager_fallback.cpp
//===----------------------------------------------------------------------===//

#include <unordered_map>

#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/metrics.h>

#include "aten_eager_fallback.h"
#include "../utils/exception.h"

namespace torch_lazy_tensors {

static std::unordered_map<std::string, ::torch::lazy::Counter*> _eager_fallback_counters;

bool force_eager_fallback(c10::Symbol op) {
    return false;  // Never force eager fallback
}

void ltc_eager_fallback(const c10::OperatorHandle& op,
                        torch::jit::Stack* stack) {
    UNSUPPORTED_ERROR("ltc_eager_fallback is not supported");
}

TORCH_LIBRARY_IMPL(_, Lazy, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&ltc_eager_fallback>());
}

}  // namespace torch_lazy_tensors
