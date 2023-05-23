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

#include "generated/shape_inference.h"
#include "utils/exception.h"

namespace torch {
namespace lazy {

// TODO(henrytu): Upstream these shape inference functions to PyTorch in the future.

std::vector<torch::lazy::Shape>
compute_shape_div(const at::Tensor& self, const at::Scalar& other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_mse_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_mul(const at::Tensor& self, const at::Scalar& other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_var(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction, bool keepdim) {
  // Result of variance is scalar tensor.
  return {Shape(self.scalar_type(), {})};
}

std::vector<torch::lazy::Shape> compute_shape_hardtanh(
    const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val
) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_where(
  const at::Tensor & condition,
  const at::Tensor & self,
  const at::Tensor & other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_bucketize(
    const at::Tensor& self, const at::Tensor& boundaries, bool out_int32,
    bool right) {
  auto dtype = out_int32 ? at::kInt : at::kLong;
  return {Shape(dtype, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_copy(
  const at::Tensor& self,
  const at::Tensor& src,
  bool non_blocking) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_native_group_norm(
  const at::Tensor& input,
  const c10::optional<at::Tensor>& weight,
  const c10::optional<at::Tensor>& bias,
  int64_t N, int64_t C, int64_t HxW,
  int64_t group, double eps) {

  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each group per N.
  shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{N, group});

  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{N, group});
  return shapes;
}

std::vector<torch::lazy::Shape> compute_shape_native_group_norm_backward(
  const at::Tensor& grad_out,
  const at::Tensor& input,
  const at::Tensor& mean,
  const at::Tensor& rstd,
  const c10::optional<at::Tensor>& weight,
  int64_t N, int64_t C, int64_t HxW,
  int64_t group, ::std::array<bool, 3> output_mask) {

  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  int64_t num_features = input.size(1);

  // `weight` and `bias` are vectors of length C (number of channels)`
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});

  return shapes;
}

} // namespace lazy
} // namespace torch
