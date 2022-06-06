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
#include "LazyShapeInference.h"

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

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each channel.
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  int64_t num_features = input.sizes().vec()[1];

  if (running_mean.has_value()) {
    shapes.emplace_back(
        running_mean.value().scalar_type(), running_mean.value().sizes().vec());
  } else {
    shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{num_features});
  }

  if (running_var.has_value()) {
    shapes.emplace_back(
        running_var.value().scalar_type(), running_var.value().sizes().vec());
  } else {
    shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{num_features});
  }
  return shapes;
}

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    ::std::array<bool, 3> output_mask) {
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each channel.
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  int64_t num_features = input.sizes().vec()[1];

  // `weight` and `bias` are vectors of length C (number of channels)`
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});

  return shapes;
}

std::vector<torch::lazy::Shape> compute_shape_new_empty(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  if (dtype.has_value()) {
    return {Shape(*dtype, size)};
  }
  return {Shape(self.scalar_type(), size)};
}

} // namespace lazy
} // namespace torch
