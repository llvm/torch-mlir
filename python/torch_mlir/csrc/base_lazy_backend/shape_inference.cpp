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

} // namespace lazy
} // namespace torch
