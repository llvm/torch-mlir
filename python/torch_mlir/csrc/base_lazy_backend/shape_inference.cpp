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

} // namespace lazy
} // namespace torch
