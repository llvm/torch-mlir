//===- aten_mlir_type.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// Structured similarly to code from git@github.com:pytorch/xla.git

#pragma once

#include <ATen/Tensor.h>

namespace torch_mlir {

// Base ATEN Type class where the MLIR specific overrides should be defined.
class ATenMLIRType {
public:
  static void InitializeAtenBindings();

  //////////////////////////////////////////////////////////////////////////////
  // ATEN API overrides in alphabetical order.
  // Note: The C++ signatures must match the ones listed within the following
  // pytorch folder file:
  //   build/aten/src/ATen/RegistrationDeclarations.h
  /////////////////////////////////////////////////////////////////////////////
  // The static method definitions here have multiple uses. Each function
  // signature here will override the default implementation provided by
  // aten_mlir_type_defaults.h. Most of these overrides are used to construct
  // a small internal IR that can be used for different purposes. Primarily,
  // in this code, the IR will be converted to MLIR. As such there is a often
  // a 1:1 correspondance between code here and operations in the ATen MLIR
  // dialect.

  // This file is parsed by gen_aten_dialect.py to generate
  // aten_mlir_type_defaults.*, including the appropriate bindings in that
  // file for all pytorch methods.

  static at::Tensor _adaptive_avg_pool2d(const at::Tensor &self,
                                         at::IntArrayRef output_size);

  static at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor &grad_output,
                                                  const at::Tensor &self);

  static at::Tensor add(const at::Tensor &self, const at::Tensor &other,
                        at::Scalar alpha);

  static at::Tensor &add_(at::Tensor &self, const at::Tensor &other,
                          at::Scalar alpha);

  static at::Tensor addmm(const at::Tensor &self, const at::Tensor &mat1,
                          const at::Tensor &mat2, at::Scalar beta,
                          at::Scalar alpha);

  static at::Tensor as_strided(const at::Tensor &self, at::IntArrayRef size,
                               at::IntArrayRef stride,
                               c10::optional<int64_t> storage_offset);

  static at::Tensor clone(const at::Tensor &self);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  convolution_backward_overrideable(
      const at::Tensor &grad_output, const at::Tensor &input,
      const at::Tensor &weight, at::IntArrayRef stride, at::IntArrayRef padding,
      at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
      int64_t groups, std::array<bool, 3> output_mask);

  static at::Tensor convolution_overrideable(
      const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias,
      at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
      bool transposed, at::IntArrayRef output_padding, int64_t groups);

  static at::Tensor &copy_(at::Tensor &self, const at::Tensor &src,
                           bool non_blocking);

  static at::Tensor _copy_from(const at::Tensor &self, const at::Tensor &dst,
                               bool non_blocking);

  static at::Tensor div(const at::Tensor &self, const at::Tensor &other);

  static at::Tensor &div_(at::Tensor &self, const at::Tensor &other);

  static at::Tensor div(const at::Tensor &self, at::Scalar other);

  static at::Tensor expand(const at::Tensor &self, at::IntArrayRef size,
                           bool implicit);

  static at::Tensor gather(const at::Tensor &self, int64_t dim,
                           const at::Tensor &index, bool sparse_grad);

  static at::Tensor hardtanh(const at::Tensor &self, at::Scalar min_val,
                             at::Scalar max_val);

  static at::Tensor &hardtanh_(at::Tensor &self, at::Scalar min_val,
                               at::Scalar max_val);

  static at::Tensor hardtanh_backward(const at::Tensor &grad_output,
                                      const at::Tensor &self,
                                      at::Scalar min_val, at::Scalar max_val);

  static at::Tensor _log_softmax(const at::Tensor &self, int64_t dim,
                                 bool half_to_float);

  static at::Tensor _log_softmax_backward_data(const at::Tensor &grad_output,
                                               const at::Tensor &output,
                                               int64_t dim,
                                               const at::Tensor &self);

  static std::tuple<at::Tensor, at::Tensor>
  max_pool2d_with_indices(const at::Tensor &self, at::IntArrayRef kernel_size,
                          at::IntArrayRef stride, at::IntArrayRef padding,
                          at::IntArrayRef dilation, bool ceil_mode);

  static at::Tensor max_pool2d_with_indices_backward(
      const at::Tensor &grad_output, const at::Tensor &self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
      const at::Tensor &indices);

  static at::Tensor mean(const at::Tensor &self,
                         c10::optional<at::ScalarType> dtype);

  static at::Tensor mean(const at::Tensor &self, at::IntArrayRef dim,
                         bool keepdim, c10::optional<at::ScalarType> dtype);

  static at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2);

  static at::Tensor mul(const at::Tensor &self, const at::Tensor &other);

  static at::Tensor &mul_(at::Tensor &self, const at::Tensor &other);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  native_batch_norm(const at::Tensor &input, const at::Tensor &weight,
                    const at::Tensor &bias, const at::Tensor &running_mean,
                    const at::Tensor &running_var, bool training,
                    double momentum, double eps);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  native_batch_norm_backward(const at::Tensor &grad_out,
                             const at::Tensor &input, const at::Tensor &weight,
                             const at::Tensor &running_mean,
                             const at::Tensor &running_var,
                             const at::Tensor &save_mean,
                             const at::Tensor &save_invstd, bool train,
                             double eps, std::array<bool, 3> output_mask);

  static at::Tensor neg(const at::Tensor &self);

  static std::tuple<at::Tensor, at::Tensor>
  nll_loss2d_forward(const at::Tensor &self, const at::Tensor &target,
                     const at::Tensor &weight, int64_t reduction,
                     int64_t ignore_index);

  static at::Tensor nll_loss2d_backward(const at::Tensor &grad_output,
                                        const at::Tensor &self,
                                        const at::Tensor &target,
                                        const at::Tensor &weight,
                                        int64_t reduction, int64_t ignore_index,
                                        const at::Tensor &total_weight);

  static std::tuple<at::Tensor, at::Tensor>
  nll_loss_forward(const at::Tensor &self, const at::Tensor &target,
                   const at::Tensor &weight, int64_t reduction,
                   int64_t ignore_index);

  static at::Tensor nll_loss_backward(const at::Tensor &grad_output,
                                      const at::Tensor &self,
                                      const at::Tensor &target,
                                      const at::Tensor &weight,
                                      int64_t reduction, int64_t ignore_index,
                                      const at::Tensor &total_weight);

  static at::Tensor relu(const at::Tensor &self);

  static at::Tensor &relu_(at::Tensor &self);

  static int64_t size(const at::Tensor &self, int64_t dim);

  static at::Tensor squeeze(const at::Tensor &self, int64_t dim);

  static at::Tensor sub(const at::Tensor &self, const at::Tensor &other,
                        at::Scalar alpha);

  static at::Tensor &sub_(at::Tensor &self, const at::Tensor &other,
                          at::Scalar alpha);

  static at::Tensor sum(const at::Tensor &self, at::IntArrayRef dim,
                        bool keepdim, c10::optional<at::ScalarType> dtype);

  static at::Tensor t(const at::Tensor &self);

  static at::Tensor threshold_backward(const at::Tensor &grad_output,
                                       const at::Tensor &self,
                                       at::Scalar threshold);

  static at::Tensor to(const at::Tensor &self, const at::TensorOptions &options,
                       bool non_blocking, bool copy);
  static at::Tensor to(const at::Tensor &self, c10::Device device,
                       at::ScalarType dtype, bool non_blocking, bool copy);
  static at::Tensor to(const at::Tensor &self, at::ScalarType dtype,
                       bool non_blocking, bool copy);
  static at::Tensor to(const at::Tensor &self, const at::Tensor &other,
                       bool non_blocking, bool copy);

  static at::Tensor _unsafe_view(const at::Tensor &self, at::IntArrayRef size);

  static at::Tensor unsqueeze(const at::Tensor &self, int64_t dim);

  static at::Tensor view(const at::Tensor &self, at::IntArrayRef size);
};

} // namespace torch_mlir
