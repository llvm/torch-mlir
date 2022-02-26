//===- aten_ltc_mlir_type.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/aten_ltc_ts_type.cpp
//===----------------------------------------------------------------------===//

#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/result_type.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include <torch/csrc/lazy/core/tensor_impl.h>

#include "LazyNativeFunctions.h"
#include "../utils/exception.h"

namespace torch_lazy_tensors {

namespace {

void CheckSubOperandTypes(at::ScalarType type1, at::ScalarType type2) {
  CHECK(type1 != at::kBool || type2 != at::kBool)
      << "Subtraction, the `-` operator, with two bool tensors is not "
         "supported. Use the `^` or `logical_xor()` operator instead.";
  CHECK(type1 != at::kBool && type2 != at::kBool)
      << "Subtraction, the `-` operator, with a bool tensor is not "
         "supported. If you are trying to invert a mask, use the `~` or "
         "`logical_not()` operator instead.";
}

std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  torch::lazy::LazyTensor self_tensor;
  torch::lazy::LazyTensor other_tensor;
  auto self_xtensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = torch::lazy::TryGetLtcTensor(other);
    self_tensor = GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = self_xtensor;
    other_tensor = GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor>(self_tensor, other_tensor);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor> operands =
      GetBinaryOperands(torch::lazy::UnwrapNumber(self, dtype),
                        torch::lazy::UnwrapNumber(other, dtype));
  torch::lazy::LazyTensor result = bin_op(operands.first, operands.second);
  return torch::lazy::CreateAtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  torch::lazy::LazyTensor self_tensor = torch::lazy::GetLtcTensor(self);
  torch::lazy::LazyTensor result = bin_op(self_tensor, other);
  return torch::lazy::CreateAtenFromLtcTensor(result);
}

at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

at::Tensor CreateLtcTensor(const at::Tensor& tensor,
                           const c10::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    return torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  return torch::lazy::atenDeviceToBackendDevice(*device);
}

}  // namespace


at::Tensor LazyNativeFunctions::expand(const at::Tensor& self,
                                       at::IntArrayRef size, bool implicit) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::expand")
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::expand(
  //     torch::lazy::TryGetLtcTensor(self), size.vec()));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor input_tensor = torch::lazy::TryGetLtcTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor.GetDevice();
  torch::lazy::LazyTensor running_mean_tensor =
      GetOrCreateLtcTensor(running_mean, device);
  torch::lazy::LazyTensor running_var_tensor =
      GetOrCreateLtcTensor(running_var, device);
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::native_batch_norm");
  // auto outputs = lazy_tensor_aten_ops::ts_native_batch_norm(
  //     torch::lazy::TryGetLtcTensor(input), GetOrCreateLtcTensor(weight, device),
  //     GetOrCreateLtcTensor(bias, device), running_mean_tensor,
  //     running_var_tensor, training, momentum, eps);
  // return std::make_tuple(torch::lazy::CreateAtenFromLtcTensor(std::get<0>(outputs)),
  //                        torch::lazy::CreateAtenFromLtcTensor(std::get<1>(outputs)),
  //                        torch::lazy::CreateAtenFromLtcTensor(std::get<2>(outputs)));
}

// std::tuple<at::Tensor, at::Tensor, at::Tensor>
// LazyNativeFunctions::native_batch_norm_backward(
//     const at::Tensor& grad_out, const at::Tensor& input,
//     const c10::optional<at::Tensor>& weight,
//     const c10::optional<at::Tensor>& running_mean,
//     const c10::optional<at::Tensor>& running_var,
//     const c10::optional<at::Tensor>& save_mean,
//     const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
//     std::array<bool, 3> output_mask) {
//   TORCH_LAZY_FN_COUNTER("lazy::");
//   torch::lazy::LazyTensor grad_out_tensor = torch::lazy::TryGetLtcTensor(grad_out);
//   const torch::lazy::BackendDevice& device = grad_out_tensor.GetDevice();
//   torch::lazy::LazyTensor null_tensor;
//   bool running_stats = running_mean && running_mean->defined();
//   CHECK_EQ(running_var && running_var->defined(), running_stats);
//   UNIMPLEMENTED_ERROR("LazyNativeFunctions::native_batch_norm_backward");
//   // auto gradients = lazy_tensor_aten_ops::ts_native_batch_norm_backward(
//   //     torch::lazy::TryGetLtcTensor(grad_out), torch::lazy::TryGetLtcTensor(input),
//   //     GetOrCreateLtcTensor(weight, device),
//   //     running_stats ? GetOrCreateLtcTensor(running_mean, device)
//   //                   : null_tensor,
//   //     running_stats ? GetOrCreateLtcTensor(running_var, device)
//   //                   : null_tensor,
//   //     GetOrCreateLtcTensor(save_mean, device),
//   //     GetOrCreateLtcTensor(save_invstd, device), train, eps,
//   //     output_mask);
//   // at::Tensor undefined;
//   // return std::make_tuple(
//   //     output_mask[0] ? torch::lazy::CreateAtenFromLtcTensor(std::get<0>(gradients))
//   //                    : undefined,
//   //     output_mask[1] ? torch::lazy::CreateAtenFromLtcTensor(std::get<1>(gradients))
//   //                    : undefined,
//   //     output_mask[2] ? torch::lazy::CreateAtenFromLtcTensor(std::get<2>(gradients))
//   //                    : undefined);
// }

at::Tensor LazyNativeFunctions::permute(const at::Tensor& self,
                                        at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::permute");
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::permute(
  //     self_tensor, torch::lazy::ToI64Vector(dims)));
}

at::Tensor LazyNativeFunctions::repeat(const at::Tensor& self,
                                       at::IntArrayRef repeats) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::repeat");
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::repeat(
  //     torch::lazy::TryGetLtcTensor(self), torch::lazy::ToI64Vector(repeats)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::squeeze");
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::squeeze");
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self), dim));
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::t");
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::transpose(torch::lazy::TryGetLtcTensor(self), 0, 1));
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::unsqueeze");
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::unsqueeze(torch::lazy::TryGetLtcTensor(self), dim));
}

at::Tensor LazyNativeFunctions::view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensor self_tensor = torch::lazy::TryGetLtcTensor(self);
  UNIMPLEMENTED_ERROR("LazyNativeFunctions::view");
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::view(self_tensor, torch::lazy::ToI64Vector(size)));
}

void InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
