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
#include <cmath>

namespace torch {
namespace lazy {

// TODO(henrytu): Upstream these shape inference functions to PyTorch in the future.

// Turns any negative index positive (assuming it's valid)
int64_t normalize_index(int64_t index, unsigned dims) {
  return index < 0 ? (int64_t)dims + index : index;
}

std::vector<Shape>
compute_shape_dropout(const at::Tensor& input, double p, bool train) {
  return {Shape(input.scalar_type(), input.sizes().vec())};
}

std::vector<Shape> compute_shape_layer_norm(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias, double eps, bool cudnn_enable) {
  return {Shape(input.scalar_type(), input.sizes().vec())};
}

std::vector<Shape>
compute_shape_matmul(const at::Tensor& self, const at::Tensor& other) {
  std::vector<int64_t> sizes;

  auto self_sizes = self.sizes().vec();
  auto other_sizes = other.sizes().vec();

  // For tensors with dimensions >2, the leading dimensions are for batch info.
  // The last 2 (or 1 in the case of a single dim tensor) dimensions are the
  // matrix dimensions themselves, which is checked to ensure the matmul op
  // is legal.
  //
  // Example:
  // [1, 2, 3, 4] -> [1, 2] batch dims and [3, 4] matrix
  //    [1, 4, 5] ->    [1] batch dims and [4, 5] matrix
  //       [4, 5] ->     [] batch dims and [4, 5] matrix
  //          [5] ->     [] batch dims and    [5] matrix
  //
  // We'll start by splitting the shapes as described above.
  auto partition_shape = [](at::ArrayRef<int64_t> sizes) {
    if (sizes.size() <= 2) {
      return std::make_pair(
          std::vector<int64_t>(),
          std::vector<int64_t>(sizes.begin(), sizes.end()));
    } else {
      std::size_t partition_idx = sizes.size() - 2;
      return std::make_pair(
          std::vector<int64_t>(sizes.begin(), sizes.begin() + partition_idx),
          std::vector<int64_t>(sizes.begin() + partition_idx, sizes.end()));
    }
  };
  auto [self_batch_sizes, self_matrix_sizes] = partition_shape(self_sizes);
  auto [other_batch_sizes, other_matrix_sizes] = partition_shape(other_sizes);

  // Insert batch dimensions.
  // The final list of sizes will be based on the tensor w/ more dims.
  // Individual dimension sizes are "right justified" as we iterate thru
  // to pick the larger dimension between them.
  // 0 1 1 3 4
  //     5 1 2
  // ---------
  // 0 1 5 3 4 <- Result
  int64_t self_size, other_size;
  std::size_t num_batch_dim =
      std::max(self_batch_sizes.size(), other_batch_sizes.size());
  auto get_batch_dim = [&](std::vector<int64_t> batch_sizes, std::size_t dim) {
    long idx = dim - num_batch_dim + batch_sizes.size();
    // Negative index means out of bounds, which defaults to a dim size of 1.
    return idx < 0 ? 1 : batch_sizes[idx];
  };
  for (std::size_t i = 0; i < num_batch_dim; i++) {
    self_size = get_batch_dim(self_batch_sizes, i);
    other_size = get_batch_dim(other_batch_sizes, i);

    TORCH_CHECK(
        self_size == 1 || other_size == 1 || self_size == other_size,
        "At trailing dimension ", i, ", expected for dimensions ",
        "to either match or have one of them equal one, but got ", self_size,
        " and ", other_size, " instead!");

    sizes.push_back(std::max(self_size, other_size));
  }

  // Keep track of the inner dimensions of matmul to validate op is valid.
  std::pair<int64_t, int64_t> inner_sizes;
  if (self_matrix_sizes.size() == 1 && other_matrix_sizes.size() == 1) {
    // Dot-Product -- scalar output, so no dimensions inserted
    inner_sizes = std::make_pair(self_matrix_sizes[0], other_matrix_sizes[0]);
  } else if (self_matrix_sizes.size() == 1 && other_matrix_sizes.size() == 2) {
    // Vector-Matrix product (m) @ (m, n) -> (n)
    inner_sizes = std::make_pair(self_matrix_sizes[0], other_matrix_sizes[0]);

    sizes.push_back(other_matrix_sizes[1]);
  } else if (self_matrix_sizes.size() == 2 && other_matrix_sizes.size() == 1) {
    // Matrix-Vector product (m, n) @ (n) -> (m)
    inner_sizes = std::make_pair(self_matrix_sizes[1], other_matrix_sizes[0]);

    sizes.push_back(self_matrix_sizes[0]);
  } else if (self_matrix_sizes.size() == 2 && other_matrix_sizes.size() == 2) {
    // Matrix-Matrix product (m, n) @ (n, o) -> (m, o)
    inner_sizes = std::make_pair(self_matrix_sizes[1], other_matrix_sizes[0]);

    sizes.push_back(self_matrix_sizes[0]);
    sizes.push_back(other_matrix_sizes[1]);
  } else {
    // By this time, self_matrix_sizes and other_matrix_sizes should have at
    // most 2 dims, so if this is executed something has gone wrong...
    TORCH_CHECK(false, "Invalid matmul shape combination!");
  }

  TORCH_CHECK(
      inner_sizes.first == inner_sizes.second, "Inner dimension of matrix (",
      inner_sizes.first, ") does not ", "match (", inner_sizes.second, ")!");

  return {Shape(self.scalar_type(), sizes)};
}

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

std::vector<Shape>
compute_shape_reshape(const at::Tensor& self, at::IntArrayRef shape) {
  // Make a copy of the desired output shape.
  std::vector<int64_t> sizes(shape.begin(), shape.end());

  // Product of all sizes in input shape is the number of entries in tensor.
  int64_t num_entries = 1;
  for (int64_t i : self.sizes().vec()) {
    num_entries *= i;
  }

  // Validate the number of entries in the desired shape. If there is a wildcard
  // dimension, we need to find it now in order to populate it.
  long wildcard_idx = -1;
  int64_t num_concrete_entries = 1;
  for (std::size_t idx = 0; idx < sizes.size(); idx++) {
    if (sizes[idx] != -1) {
      num_concrete_entries *= sizes[idx];
    } else {
      TORCH_CHECK(wildcard_idx == -1, "only one dimension can be inferred");
      wildcard_idx = idx;
    }
  }

  if (wildcard_idx == -1) {
    // No wildcard, the shape should already be known.
    TORCH_CHECK(
        num_entries == num_concrete_entries, "shape `[", sizes,
        "]` is invalid for input of size ", num_concrete_entries);
  } else {
    // There is one dimension which is not explicitly declared -- we need to
    // infer.
    TORCH_CHECK(
        num_entries % num_concrete_entries == 0, "shape `[", sizes,
        "]` is invalid for input of size ", num_concrete_entries);

    sizes[wildcard_idx] = num_entries / num_concrete_entries;
  }

  return {Shape(self.scalar_type(), sizes)};
}

std::vector<Shape> compute_shape_rsub(
    const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  // Since other is scalar, the result will match tensor shape.
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape>
compute_shape_select(const at::Tensor& self, int64_t dim, int64_t index) {
  auto original_shape = self.sizes().vec();
  std::vector<int64_t> sizes(original_shape.begin(), original_shape.end());

  TORCH_CHECK(
      dim < (int64_t)sizes.size(), "Dimension ", dim,
      " is out of bounds for tensor with ", sizes.size(), " dimensions!");
  TORCH_CHECK(
      index < sizes[dim], "Index ", index,
      " is out of bounds for dimension of size ", sizes[dim]);
  sizes.erase(sizes.begin() + dim);

  return {Shape(self.scalar_type(), sizes)};
}

std::vector<Shape> compute_shape_slice(
    const at::Tensor& self, int64_t dim, c10::optional<int64_t> start,
    c10::optional<int64_t> end, int64_t step) {
  auto original_shape = self.sizes().vec();
  std::vector<int64_t> sizes(original_shape.begin(), original_shape.end());

  int64_t dim_size = sizes[dim];

  // Index may be negative, so we must normalize it.
  int64_t start_norm = normalize_index(start.value(), dim_size);
  int64_t end_norm = normalize_index(end.value(), dim_size);

  if (start_norm >= end_norm || start_norm >= dim_size || end_norm <= 0) {
    // Slice is out of bounds, nothing in range.
    sizes[dim] = 0;
  } else {
    // Clamp upper and lower bound to valid indices.
    start_norm = std::max((int64_t)0, start_norm);
    end_norm = std::min(dim_size, end_norm);

    // Final size is determined by step and interval size.
    sizes[dim] = std::ceil((double)(end_norm - start_norm) / (double)step);
  }

  return {Shape(self.scalar_type(), sizes)};
}

std::vector<Shape> compute_shape_softmax(
    const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {Shape(dtype.value(), self.sizes().vec())};
  }
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape>
compute_shape_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  auto original_shape = self.sizes().vec();
  std::vector<int64_t> sizes{original_shape.begin(), original_shape.end()};

  // Index may be negative, so we must normalize it. We create new variables
  // instead of replacing the existing ones so that in the case of an error,
  // the original values can be printed out.
  int64_t dim0_norm = normalize_index(dim0, sizes.size());
  int64_t dim1_norm = normalize_index(dim1, sizes.size());

  // Verify dimensions are valid.
  TORCH_CHECK(
      0 <= dim0_norm && dim0_norm < (int64_t)sizes.size(), "dim0 has value ",
      dim0, ", but there are only ", sizes.size(), " tensor dimensions");
  TORCH_CHECK(
      0 <= dim1_norm && dim1_norm < (int64_t)sizes.size(), "dim1 has value ",
      dim1, ", but there are only ", sizes.size(), " tensor dimensions");

  // Swap shapes at dimensions.
  std::swap(sizes[dim0_norm], sizes[dim1_norm]);

  return {Shape(self.scalar_type(), sizes)};
}

} // namespace lazy
} // namespace torch
