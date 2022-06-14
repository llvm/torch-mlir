//===- LazyShapeInference.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cmath>

#include "../utils/exception.h"
#include "generated/shape_inference.h"

namespace torch {
namespace lazy {

// TODO(henrytu): Upstream these shape inference functions to PyTorch in the future.

std::vector<torch::lazy::Shape>
compute_shape_div(const at::Tensor& self, const at::Scalar & other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_mul(const at::Tensor& self, const at::Scalar& other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_new_empty(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (dtype.has_value()) {
    return {Shape(*dtype, size)};
  }
  return {Shape(self.scalar_type(), size)};
}

} // namespace lazy
} // namespace torch
