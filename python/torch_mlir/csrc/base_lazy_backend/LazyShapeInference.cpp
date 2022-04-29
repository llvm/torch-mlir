//===- LazyShapeInference.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "LazyShapeInference.h"
#include "../utils/exception.h"

namespace torch {
namespace lazy {

std::vector<Shape> compute_shape_native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  std::vector<Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());
  if (running_mean.has_value()) {
    shapes.emplace_back(
        running_mean.value().scalar_type(), running_mean.value().sizes().vec());
    if (running_var.has_value()) {
      shapes.emplace_back(
          running_var.value().scalar_type(), running_var.value().sizes().vec());
    }
  }
  return shapes;
}

} // namespace lazy
} // namespace torch
