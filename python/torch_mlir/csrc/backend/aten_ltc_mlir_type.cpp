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
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include <torch/csrc/lazy/core/tensor_impl.h>

#include "../tensor_aten_ops.h"
#include "../utils/exception.h"
#include "../utils/sys_utils.h"
#include "LazyNativeFunctions.h"
#include "LazyShapeInference.h"

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

std::pair<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr>
GetBinaryOperands(const at::Tensor& self, const at::Tensor& other) {
  torch::lazy::LazyTensorPtr self_tensor;
  torch::lazy::LazyTensorPtr other_tensor;
  auto self_xtensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = torch::lazy::TryGetLtcTensor(other);
    self_tensor = GetOrCreateLtcTensor(self, other_tensor->GetDevice());
  } else {
    self_tensor = self_xtensor;
    other_tensor = GetOrCreateLtcTensor(other, self_tensor->GetDevice());
  }
  return std::pair<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr>(
      self_tensor, other_tensor);
}

template <typename B>
at::Tensor
DoBinaryOp(const at::Tensor& self, const at::Tensor& other, const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> operands =
      GetBinaryOperands(
          torch::lazy::UnwrapNumber(self, dtype),
          torch::lazy::UnwrapNumber(other, dtype));
  torch::lazy::LazyTensorPtr result = bin_op(operands.first, operands.second);
  return torch::lazy::CreateAtenFromLtcTensor(result);
}

template <typename B>
at::Tensor
DoBinaryOp(const at::Tensor& self, const at::Scalar& other, const B& bin_op) {
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::GetLtcTensor(self);
  torch::lazy::LazyTensorPtr result = bin_op(self_tensor, other);
  return torch::lazy::CreateAtenFromLtcTensor(result);
}

at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
}

at::Tensor CreateLtcTensor(
    const at::Tensor& tensor,
    const c10::optional<torch::lazy::BackendDevice>& device) {
  if (tensor.defined() && device) {
    return torch::lazy::CreateAtenFromLtcTensor(
        torch::lazy::LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

c10::optional<torch::lazy::BackendDevice>
GetLtcDevice(const c10::optional<c10::Device>& device) {
  if (!device) {
    return c10::nullopt;
  }
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  return torch::lazy::atenDeviceToBackendDevice(*device);
}

} // namespace

// at::Tensor LazyNativeFunctions::bernoulli(
//     const at::Tensor& self, c10::optional<at::Generator> generator) {
//   TORCH_LAZY_FN_COUNTER("lazy::");
//   if (generator.has_value() && generator->defined()) {
//     UNSUPPORTED_ERROR("LazyNativeFunctions::bernoulli has generator value");
//   }
//   auto self_tensor = torch::lazy::TryGetLtcTensor(self);

//   UNIMPLEMENTED_FUNCTION_ERROR();
//   // return torch::lazy::CreateAtenFromLtcTensor(
//   //     lazy_tensor_aten_ops::bernoulli(self_tensor));
// }

// at::Tensor& LazyNativeFunctions::bernoulli_(
//     at::Tensor& self, double p, c10::optional<at::Generator> generator) {
//   TORCH_LAZY_FN_COUNTER("lazy::");
//   if (generator.has_value() && generator->defined()) {
//     UNSUPPORTED_ERROR("LazyNativeFunctions::bernoulli_ has generator value");
//   }
//   auto self_tensor = torch::lazy::TryGetLtcTensor(self);

//   UNIMPLEMENTED_FUNCTION_ERROR();
//   // lazy_tensor_aten_ops::bernoulli_(self_tensor, p);
//   // return self;
// }

at::Tensor LazyNativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto lazy_tensors = torch::lazy::GetLtcTensors(tensors);
  std::vector<torch::lazy::Value> values;
  values.reserve(lazy_tensors.size());
  for (auto& tensor : lazy_tensors) {
    values.emplace_back(tensor->GetIrValue());
  }

  auto shapes = torch::lazy::compute_shape_cat(tensors, dim);
  UNIMPLEMENTED_FUNCTION_ERROR();
  // auto node =
  //     torch::lazy::MakeNode<ir::ops::Cat>(values, dim, std::move(shapes));
  // auto result = torch::lazy::CreateAtenFromLtcTensor(
  //     torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0),
  // lazy_tensors[0]->GetDevice()));
  // return result;
}

at::Tensor LazyNativeFunctions::clone(
    const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) {
  auto self_lt = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      self_lt->Create(self_lt->GetIrValue(), self_lt->GetDevice()));
}

at::Tensor LazyNativeFunctions::_copy_from(
    const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    // providing a new 'eager' value (self) for an existing lazy tensor (dst)
    static bool sync_update =
        sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    // materializing a lazy tensor (self) and copying its value into eager
    // tensor (dst)
    // detached=false lets us skip a copy in `ToTensor`, which should be safe
    // becuase we are only going to use the tensor for dst.copy_()
    CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/false);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // Copying one lazy tensor to another
    if (!dst_tensor->CurrentIrValue()) {
      // if dest is not backed by IR (e.g. result of some lazy operation),
      // then it should have at::Tensor data backing it instead
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        // both src/dst are simply backed by at::Tensor data, no IR- do a
        // straightforward copy
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        // src needs to be materialized before its result can be used for a copy
        // into dst
        // since we use the src tensor only for making a copy, we don't need to
        // detach it
        // note: it would be even more efficient if we could cause ToTensor to
        // materialize the
        // value directly into dst's buffer (that would need to be detached
        // though).
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/false));
      }
    } else {
      lazy_tensor_aten_ops::copy_(dst_tensor, self_tensor);
      auto* impl =
          dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(
    const at::Tensor& self, const at::Tensor& dst) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto dst_tensor = torch::lazy::TryGetLtcTensor(dst);
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  if (!self_tensor) {
    CHECK(dst_tensor);
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    CHECK(self_tensor);
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is a lazy tensor
    auto* dest_impl =
        dynamic_cast<torch::lazy::LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor LazyNativeFunctions::empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  const auto device_type = torch::lazy::getBackend()->EagerFallbackDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return CreateLtcTensor(x_result, GetLtcDevice(device));
}

at::Tensor LazyNativeFunctions::expand(
    const at::Tensor& self, at::IntArrayRef size, bool implicit) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::expand(
  //     torch::lazy::TryGetLtcTensor(self), size.vec()));
}

at::Tensor&
LazyNativeFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  lazy_tensor_aten_ops::fill_(self_tensor, value);
  return self;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
LazyNativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr input_tensor = torch::lazy::TryGetLtcTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  torch::lazy::LazyTensorPtr running_mean_tensor =
      GetOrCreateLtcTensor(running_mean, device);
  torch::lazy::LazyTensorPtr running_var_tensor =
      GetOrCreateLtcTensor(running_var, device);
  UNIMPLEMENTED_FUNCTION_ERROR();
  // auto outputs = lazy_tensor_aten_ops::ts_native_batch_norm(
  //     torch::lazy::TryGetLtcTensor(input), GetOrCreateLtcTensor(weight,
  // device),
  //     GetOrCreateLtcTensor(bias, device), running_mean_tensor,
  //     running_var_tensor, training, momentum, eps);
  // return
  // std::make_tuple(torch::lazy::CreateAtenFromLtcTensor(std::get<0>(outputs)),
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
//   torch::lazy::LazyTensor grad_out_tensor =
// torch::lazy::TryGetLtcTensor(grad_out);
//   const torch::lazy::BackendDevice& device = grad_out_tensor.GetDevice();
//   torch::lazy::LazyTensor null_tensor;
//   bool running_stats = running_mean && running_mean->defined();
//   CHECK_EQ(running_var && running_var->defined(), running_stats);
//   UNIMPLEMENTED_FUNCTION_ERROR();
//   // auto gradients = lazy_tensor_aten_ops::ts_native_batch_norm_backward(
//   //     torch::lazy::TryGetLtcTensor(grad_out),
// torch::lazy::TryGetLtcTensor(input),
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
//   //     output_mask[0] ?
// torch::lazy::CreateAtenFromLtcTensor(std::get<0>(gradients))
//   //                    : undefined,
//   //     output_mask[1] ?
// torch::lazy::CreateAtenFromLtcTensor(std::get<1>(gradients))
//   //                    : undefined,
//   //     output_mask[2] ?
// torch::lazy::CreateAtenFromLtcTensor(std::get<2>(gradients))
//   //                    : undefined);
// }

at::Tensor
LazyNativeFunctions::permute(const at::Tensor& self, at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::permute(
  //     self_tensor, torch::lazy::ToI64Vector(dims)));
}

at::Tensor
LazyNativeFunctions::repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::repeat(
  //     torch::lazy::TryGetLtcTensor(self),
  // torch::lazy::ToI64Vector(repeats)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self)));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::squeeze(torch::lazy::TryGetLtcTensor(self),
  // dim));
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(lazy_tensor_aten_ops::transpose(
      torch::lazy::TryGetLtcTensor(self), 0, 1));
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  UNIMPLEMENTED_FUNCTION_ERROR();
  // return torch::lazy::CreateAtenFromLtcTensor(
  //     lazy_tensor_aten_ops::unsqueeze(torch::lazy::TryGetLtcTensor(self),
  // dim));
}

at::Tensor
LazyNativeFunctions::view(const at::Tensor& self, at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_tensor_aten_ops::view(self_tensor, torch::lazy::ToI64Vector(size)));
}

void InitializeAtenBindings() {}

} // namespace torch_lazy_tensors
