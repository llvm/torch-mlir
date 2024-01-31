//===- LazyShapeInference.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <ATen/ATen.h>
#include <ATen/ops/where.h>
#include <c10/util/Optional.h>
#include <cmath>

#include "generated/shape_inference.h"
#include "utils/exception.h"

namespace torch {
namespace lazy {

// TODO(henrytu): Upstream these shape inference functions to PyTorch in the
// future.

std::vector<torch::lazy::Shape> compute_shape_add(const at::Tensor &self,
                                                  const at::Scalar &other,
                                                  const at::Scalar &alpha) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_sub(const at::Tensor &self,
                                                  const at::Scalar &other,
                                                  const at::Scalar &alpha) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_div(const at::Tensor &self,
                                                  const at::Scalar &other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape__make_per_channel_quantized_tensor(const at::Tensor &self,
                                                 const at::Tensor &scale,
                                                 const at::Tensor &zero_point,
                                                 int64_t axis) {
  if (self.scalar_type() == at::kChar)
    return {Shape(at::kQInt8, self.sizes().vec())};
  if (self.scalar_type() == at::kByte)
    return {Shape(at::kQUInt8, self.sizes().vec())};
  if (self.scalar_type() == at::kInt)
    return {Shape(at::kQInt32, self.sizes().vec())};
  assert(false);
}

std::vector<torch::lazy::Shape> compute_shape__make_per_tensor_quantized_tensor(
    const at::Tensor &self, double scale, int64_t zero_point) {
  if (self.scalar_type() == at::kChar)
    return {Shape(at::kQInt8, self.sizes().vec())};
  if (self.scalar_type() == at::kByte)
    return {Shape(at::kQUInt8, self.sizes().vec())};
  if (self.scalar_type() == at::kInt)
    return {Shape(at::kQInt32, self.sizes().vec())};
  assert(false);
}

std::vector<torch::lazy::Shape> compute_shape_int_repr(const at::Tensor &self) {
  if (self.scalar_type() == at::kQInt8)
    return {Shape(at::kChar, self.sizes().vec())};
  if (self.scalar_type() == at::kQUInt8)
    return {Shape(at::kByte, self.sizes().vec())};
  if (self.scalar_type() == at::kQInt32)
    return {Shape(at::kInt, self.sizes().vec())};
  assert(false);
}

std::vector<torch::lazy::Shape>
compute_shape_dequantize(const at::Tensor &self) {
  return {Shape(at::kFloat, self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_quantize_per_tensor(const at::Tensor &self, double scale,
                                  int64_t zero_point, at::ScalarType dtype) {
  return {Shape(dtype, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_isinf(const at::Tensor &self) {
  return {Shape(at::kBool, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_quantize_per_channel(
    const at::Tensor &self, const at::Tensor &scales,
    const at::Tensor &zero_points, int64_t axis, at::ScalarType dtype) {
  return {Shape(dtype, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_max_pool3d_with_indices(
    const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto in_sizes = self.sizes().vec();
  std::vector<int64_t> dhw(3, 0);
  std::vector<int64_t> paddings = padding.vec();
  std::vector<int64_t> ksizes = kernel_size.vec();
  std::vector<int64_t> dilations = dilation.vec();
  std::vector<int64_t> strides = stride.vec();
  TORCH_CHECK(in_sizes.size() == 5, "max_pool3d requires 5D inputs, but got ",
              in_sizes);
  TORCH_CHECK(kernel_size.size() == 3 && stride.size() == 3 &&
                  padding.size() == 3 && dilation.size() == 3,
              "max_pool3d requires 3D operands, but got ", kernel_size, stride,
              padding, dilation);
  int64_t batch = in_sizes[0];
  int64_t channel = in_sizes[1]; // NCDHW
  // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
  for (auto i = 0UL; i < 3; ++i) {
    double out_size = (in_sizes[2 + i] + 2 * paddings[i] -
                       dilations[i] * (ksizes[i] - 1) - 1) /
                          (double)strides[i] +
                      1;
    if (ceil_mode)
      dhw[i] = (int64_t)std::ceil(out_size);
    else
      dhw[i] = (int64_t)std::floor(out_size);
  }
  auto out_sizes = {batch, channel, dhw[0], dhw[1], dhw[2]};
  // `with_indices` returns output and index Tensor
  return {Shape(self.scalar_type(), out_sizes), Shape(at::kLong, out_sizes)};
}

std::vector<torch::lazy::Shape> compute_shape_max_pool3d_with_indices_backward(
    const at::Tensor &grad_output, const at::Tensor &self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor &indices) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_mse_loss_backward(const at::Tensor &grad_output,
                                const at::Tensor &self,
                                const at::Tensor &target, int64_t reduction) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_mul(const at::Tensor &self,
                                                  const at::Scalar &other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_var(const at::Tensor &self, at::OptionalIntArrayRef dim,
                  const c10::optional<at::Scalar> &correction, bool keepdim) {
  // Result of variance is scalar tensor.
  return {Shape(self.scalar_type(), {})};
}

std::vector<torch::lazy::Shape>
compute_shape_nan_to_num(const at::Tensor &self, c10::optional<double> nan,
                         c10::optional<double> posinf,
                         c10::optional<double> neginf) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_hardtanh(const at::Tensor &self, const at::Scalar &min_val,
                       const at::Scalar &max_val) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_hardtanh_backward(
    const at::Tensor &grad_output, const at::Tensor &self,
    const at::Scalar &min_val, const at::Scalar &max_val) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_where(const at::Tensor &condition,
                                                    const at::Tensor &self,
                                                    const at::Tensor &other) {
  // There are cases like  -
  // torch.aten.where.self %42, %arg17, %37 : !torch.vtensor<[15,10],i1>,
  // !torch.vtensor<[],f32>, !torch.vtensor<[15,10],f32>.
  // So the result tensor would the biggest of all the three operands.
  auto condition_meta = at::native::empty_strided_meta_symint(
      condition.sym_sizes(), condition.sym_strides(),
      /*dtype=*/c10::make_optional(condition.scalar_type()),
      /*layout=*/c10::make_optional(condition.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(), self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto other_meta = at::native::empty_strided_meta_symint(
      other.sym_sizes(), other.sym_strides(),
      /*dtype=*/c10::make_optional(other.scalar_type()),
      /*layout=*/c10::make_optional(other.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto out_meta = at::where(condition_meta, self_meta, other_meta);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_bucketize(const at::Tensor &self, const at::Tensor &boundaries,
                        bool out_int32, bool right) {
  auto dtype = out_int32 ? at::kInt : at::kLong;
  return {Shape(dtype, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_copy(const at::Tensor &self,
                                                   const at::Tensor &src,
                                                   bool non_blocking) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_floor_divide(const at::Tensor &self, const at::Tensor &other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_fmod(const at::Tensor &self,
                                                   const at::Scalar &other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_native_group_norm(
    const at::Tensor &input, const c10::optional<at::Tensor> &weight,
    const c10::optional<at::Tensor> &bias, int64_t N, int64_t C, int64_t HxW,
    int64_t group, double eps) {

  TORCH_CHECK(input.sizes().size() >= 2,
              "Input tensor must have at least batch and channel dimensions!");
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each group per N.
  shapes.emplace_back(at::get_default_dtype_as_scalartype(),
                      std::vector<int64_t>{N, group});

  shapes.emplace_back(at::get_default_dtype_as_scalartype(),
                      std::vector<int64_t>{N, group});
  return shapes;
}

std::vector<torch::lazy::Shape>
compute_shape_im2col(const at::Tensor &self, at::IntArrayRef kernel_size,
                     at::IntArrayRef dilation, at::IntArrayRef padding,
                     at::IntArrayRef stride) {

  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(), self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);

  auto out_meta = at::im2col(self_meta, kernel_size, dilation, padding, stride);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_native_group_norm_backward(
    const at::Tensor &grad_out, const at::Tensor &input, const at::Tensor &mean,
    const at::Tensor &rstd, const c10::optional<at::Tensor> &weight, int64_t N,
    int64_t C, int64_t HxW, int64_t group, ::std::array<bool, 3> output_mask) {

  TORCH_CHECK(input.sizes().size() >= 2,
              "Input tensor must have at least batch and channel dimensions!");
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  int64_t num_features = input.size(1);

  // `weight` and `bias` are vectors of length C (number of channels)`
  shapes.emplace_back(at::get_default_dtype_as_scalartype(),
                      std::vector<int64_t>{num_features});
  shapes.emplace_back(at::get_default_dtype_as_scalartype(),
                      std::vector<int64_t>{num_features});

  return shapes;
}
std::vector<torch::lazy::Shape>
compute_shape_remainder(const at::Tensor &self, const at::Scalar &other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_reflection_pad2d(const at::Tensor &self,
                               at::IntArrayRef padding) {
  std::vector<int64_t> paddings = padding.vec();
  std::vector<int64_t> in_sizes = self.sizes().vec();
  auto num_dims = in_sizes.size();

  TORCH_CHECK(padding.size() == 4);
  TORCH_CHECK(num_dims >= 2);

  auto vdim = num_dims - 2;
  auto hdim = num_dims - 1;
  auto padding_left = padding[0];
  auto padding_right = padding[1];
  auto padding_top = padding[2];
  auto padding_bottom = padding[3];
  TORCH_CHECK(padding_left < in_sizes[hdim]);
  TORCH_CHECK(padding_right < in_sizes[hdim]);
  TORCH_CHECK(padding_top < in_sizes[vdim]);
  TORCH_CHECK(padding_bottom < in_sizes[vdim]);

  std::vector<int64_t> out_sizes(in_sizes);
  out_sizes[hdim] += padding_left + padding_right;
  out_sizes[vdim] += padding_top + padding_bottom;

  return {Shape(self.scalar_type(), out_sizes)};
}

std::vector<torch::lazy::Shape>
compute_shape_uniform(const at::Tensor &self, double from, double to,
                      c10::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_normal_functional(const at::Tensor &self, double mean, double std,
                                c10::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_multinomial(const at::Tensor &self, int64_t num_samples,
                          bool replacement,
                          c10::optional<at::Generator> generator) {
  // Input tensor can be either 1D or 2D. The last dim of output
  // should be 'num_samples'. So the output shape can be either
  // [num_samples] or [m, num_samples].
  // Output type can only be long tensor.
  auto ishape = self.sizes().vec();
  ishape.back() = num_samples;
  return {Shape(at::kLong, ishape)};
}

std::vector<torch::lazy::Shape>
compute_shape_eye(int64_t n, c10::optional<at::ScalarType> dtype,
                  c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device,
                  c10::optional<bool> pin_memory) {
  auto out_meta =
      at::eye(n, dtype, layout, c10::Device(c10::kMeta), pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_eye(int64_t n, int64_t m, c10::optional<at::ScalarType> dtype,
                  c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device,
                  c10::optional<bool> pin_memory) {
  auto out_meta =
      at::eye(n, m, dtype, layout, c10::Device(c10::kMeta), pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_arange(const at::Scalar &end, c10::optional<at::ScalarType> dtype,
                     c10::optional<at::Layout> layout,
                     c10::optional<at::Device> device,
                     c10::optional<bool> pin_memory) {
  auto out_meta =
      at::arange(end, dtype, layout, c10::Device(c10::kMeta), pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_arange(
    const at::Scalar &start, const at::Scalar &end,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto out_meta = at::arange(start, end, dtype, layout, c10::Device(c10::kMeta),
                             pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_arange(
    const at::Scalar &start, const at::Scalar &end, const at::Scalar &step,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto out_meta = at::arange(start, end, step, dtype, layout,
                             c10::Device(c10::kMeta), pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_full(
    at::IntArrayRef size, const at::Scalar &fill_value,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                   c10::optional<at::Layout> layout,
                   c10::optional<at::Device> device,
                   c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_zeros(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                    c10::optional<at::Layout> layout,
                    c10::optional<at::Device> device,
                    c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                    c10::optional<at::Layout> layout,
                    c10::optional<at::Device> device,
                    c10::optional<bool> pin_memory,
                    c10::optional<at::MemoryFormat> memory_format) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape> compute_shape_empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape> compute_shape_fill(const at::Tensor &self,
                                                   const at::Scalar &value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_fill(const at::Tensor &self,
                                                   const at::Tensor &value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_randn(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                    c10::optional<at::Layout> layout,
                    c10::optional<at::Device> device,
                    c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape> compute_shape_randint(
    int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape> compute_shape_randint(
    int64_t low, int64_t high, at::IntArrayRef size,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  return {
      Shape(dtype.value_or(at::get_default_dtype_as_scalartype()), size.vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_resize(const at::Tensor &self, at::IntArrayRef size,
                     c10::optional<at::MemoryFormat> memory_format) {
  return {Shape(self.scalar_type(), size.vec())};
}

std::vector<torch::lazy::Shape>
compute_shape_bernoulli(const at::Tensor &self, const at::Tensor &p,
                        c10::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_scalar_tensor(
    const at::Scalar &s, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  return {Shape(dtype.value_or(s.type()), c10::ArrayRef<int64_t>{})};
}

std::vector<torch::lazy::Shape> compute_shape_roll(const at::Tensor &self,
                                                   at::IntArrayRef shifts,
                                                   at::IntArrayRef dims) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_linspace(
    const at::Scalar &start, const at::Scalar &end, int64_t steps,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto out_meta = at::linspace(start, end, steps, dtype, layout,
                               c10::Device(c10::kMeta), pin_memory);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

} // namespace lazy
} // namespace torch
