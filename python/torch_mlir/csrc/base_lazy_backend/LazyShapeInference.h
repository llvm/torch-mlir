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

TORCH_API std::vector<torch::lazy::Shape> compute_shape___and__(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape__reshape_alias(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_abs(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_arange_out(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli(const at::Tensor & self, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bernoulli_functional(const at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bincount(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_bucketize(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_constant_pad_nd(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_convolution_overrideable(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_div(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_flip(const at::Tensor & self, at::IntArrayRef dims);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_fmod(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardswish(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logical_or(const at::Tensor & self, const at::Tensor & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_logsumexp(const at::Tensor & self, at::IntArrayRef dim, bool keepdim);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_masked_select(const at::Tensor & self, const at::Tensor & mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_max(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_mul(const at::Tensor & self, const at::Scalar & other);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout(const at::Tensor & input, double p, c10::optional<bool> train);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_new_empty(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_relu(const at::Tensor & self);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_repeat(const at::Tensor & self, at::IntArrayRef repeats);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_resize_functional(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_uniform_functional(const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator);
TORCH_API std::vector<torch::lazy::Shape> compute_shape_zero_functional(const at::Tensor & self);


// clang-format on

} // namespace lazy
} // namespace torch
