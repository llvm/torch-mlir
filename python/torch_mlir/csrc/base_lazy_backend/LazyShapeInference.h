//===- LazyShapeInference.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <vector>

namespace torch {
namespace lazy {

// clang-format off

TORCH_API std::vector<Shape> compute_shape___and__(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<Shape> compute_shape__shape_as_tensor(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape__unsafe_view(const at::Tensor & self, at::IntArrayRef size);
TORCH_API std::vector<Shape> compute_shape_abs(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_adaptive_avg_pool2d(const at::Tensor & self, at::IntArrayRef output_size);
TORCH_API std::vector<Shape> compute_shape_add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<Shape> compute_shape_add_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<Shape> compute_shape_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled);
TORCH_API std::vector<Shape> compute_shape_bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator);
TORCH_API std::vector<Shape> compute_shape_bernoulli_(at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator);
TORCH_API std::vector<Shape> compute_shape_bernoulli_(at::Tensor & self, double p, c10::optional<at::Generator> generator);
TORCH_API std::vector<Shape> compute_shape_bincount(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength);
TORCH_API std::vector<Shape> compute_shape_broadcast_to(const at::Tensor & self, at::IntArrayRef size);
TORCH_API std::vector<Shape> compute_shape_bucketize(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right);
TORCH_API std::vector<Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
TORCH_API std::vector<Shape> compute_shape_conv2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
TORCH_API std::vector<Shape> compute_shape_div(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_div_(at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_dropout(const at::Tensor & input, double p, bool train);
TORCH_API std::vector<Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
TORCH_API std::vector<Shape> compute_shape_expand_as(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<Shape> compute_shape_flatten(const at::Tensor & self, int64_t start_dim, int64_t end_dim);
TORCH_API std::vector<Shape> compute_shape_floor_divide(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_fmod(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_full_like(const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<Shape> compute_shape_hardswish(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val);
TORCH_API std::vector<Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
TORCH_API std::vector<Shape> compute_shape_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enable);
TORCH_API std::vector<Shape> compute_shape_log_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape_logsumexp(const at::Tensor & self, at::IntArrayRef dim, bool keepdim);
TORCH_API std::vector<Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
TORCH_API std::vector<Shape> compute_shape_masked_fill_(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
TORCH_API std::vector<Shape> compute_shape_masked_select(const at::Tensor & self, const at::Tensor & mask);
TORCH_API std::vector<Shape> compute_shape_matmul(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<Shape> compute_shape_max(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_max_pool2d(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
TORCH_API std::vector<Shape> compute_shape_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape_mul(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_mul_(at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps);
TORCH_API std::vector<Shape> compute_shape_new_ones(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
TORCH_API std::vector<Shape> compute_shape_new_zeros(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
TORCH_API std::vector<Shape> compute_shape_rand_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<Shape> compute_shape_relu(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_relu_(at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);
TORCH_API std::vector<Shape> compute_shape_reshape(const at::Tensor & self, at::IntArrayRef shape);
TORCH_API std::vector<Shape> compute_shape_rsub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<Shape> compute_shape_select(const at::Tensor & self, int64_t dim, int64_t index);
TORCH_API std::vector<Shape> compute_shape_slice(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step);
TORCH_API std::vector<Shape> compute_shape_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape_square(const at::Tensor & self);
TORCH_API std::vector<Shape> compute_shape_std(const at::Tensor & self, bool unbiased);
TORCH_API std::vector<Shape> compute_shape_sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<Shape> compute_shape_sub_(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<Shape> compute_shape_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<Shape> compute_shape_transpose(const at::Tensor & self, int64_t dim0, int64_t dim1);
TORCH_API std::vector<Shape> compute_shape_type_as(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<Shape> compute_shape_var(const at::Tensor & self, bool unbiased);

// clang-format on

} // namespace lazy
} // namespace torch
