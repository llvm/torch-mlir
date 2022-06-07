//===- aten_ltc_mlir_type.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_native_functions.cpp
//===----------------------------------------------------------------------===//

#include <ATen/InferSize.h>
#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/result_type.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/library.h>

#include "ATen/MetaFunctions.h"
#include <torch/csrc/lazy/core/tensor_impl.h>

#include "../utils/exception.h"
#include "../utils/sys_utils.h"
#include "generated/shape_inference.h"
#include "generated/LazyNativeFunctions.h"
#include "ops/to_copy.h"

namespace torch {
namespace lazy {

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

torch::lazy::Value MaybeExpand(
    const torch::lazy::Value& input, const torch::lazy::Shape& target_shape) {
  if (input.shape().sizes() == target_shape.sizes()) {
    return input;
  }
  return torch::lazy::MakeExpand(
      input, target_shape.sizes().vec(),
      /*is_scalar_expand=*/false);
}

std::vector<int64_t> GetExpandDimensions(
    const torch::lazy::Shape& shape, std::vector<int64_t> dimensions) {
  CHECK_GE(dimensions.size(), shape.dim()) << shape;
  int64_t base = dimensions.size() - shape.dim();
  for (size_t i = 0; i < shape.dim(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.size(i);
    }
  }
  return dimensions;
}

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src) {
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = torch::lazy::MakeCast(
          src->GetIrValue(), input->dtype(), src->dtype());
    }
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    auto input_shape = input->shape();
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    input->UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

torch::lazy::LazyTensorPtr create_view(
    const torch::lazy::LazyTensorPtr& input,
    c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input->shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(),
      at::infer_size(output_size, input_shape.numel()));
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input->CreateViewTensor(std::move(view_info));
}

torch::lazy::ViewInfo CreateAsStridedViewInfo(
    const torch::lazy::Shape& input_shape, std::vector<int64_t> size,
    std::vector<int64_t> stride, c10::optional<int64_t> storage_offset) {
  torch::lazy::Shape result_shape =
      torch::lazy::Shape(input_shape.scalar_type(), size);
  torch::lazy::AsStridedInfo as_strided_info;
  as_strided_info.stride = std::move(stride);
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return torch::lazy::ViewInfo(
      torch::lazy::ViewInfo::Type::kAsStrided, std::move(result_shape),
      input_shape, std::move(as_strided_info));
}

torch::lazy::LazyTensorPtr lazy_narrow(
    const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
    int64_t length) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim());
  torch::lazy::Shape narrow_shape = input_shape;
  narrow_shape.set_size(dim, length);

  torch::lazy::ViewInfo::Type view_type =
      (input_shape.Get().numel() == narrow_shape.numel())
          ? torch::lazy::ViewInfo::Type::kReshape
          : torch::lazy::ViewInfo::Type::kNarrow;
  torch::lazy::ViewInfo view_info(
      view_type, std::move(narrow_shape), input_shape);
  view_info.indices[dim] =
      torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, start);
  return input->CreateViewTensor(std::move(view_info));
}

torch::lazy::LazyTensorPtr lazy_view(
    const torch::lazy::LazyTensorPtr& input,
    c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input->shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(),
      at::infer_size(output_size, input_shape.numel()));
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input->CreateViewTensor(std::move(view_info));
}

torch::lazy::LazyTensorPtr lazy_select(
    const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t index) {
  auto shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, shape.Get().dim());
  torch::lazy::LazyTensorPtr result = lazy_narrow(input, dim, index, 1);
  auto new_dims = torch::lazy::DropDimensions(shape.Get().sizes(), {dim});
  return lazy_view(result, new_dims);
}

torch::lazy::LazyTensorPtr lazy_slice(
    const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
    int64_t end, int64_t step) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim());
  start =
      torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, start);
  end = torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  torch::lazy::SelectInfo select = {dim, start, end, step};
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kSelect, input_shape, select);
  return input->CreateViewTensor(std::move(view_info));
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
//   //     torch::lazy::bernoulli(self_tensor));
// }

// at::Tensor& LazyNativeFunctions::bernoulli_(
//     at::Tensor& self, double p, c10::optional<at::Generator> generator) {
//   TORCH_LAZY_FN_COUNTER("lazy::");
//   if (generator.has_value() && generator->defined()) {
//     UNSUPPORTED_ERROR("LazyNativeFunctions::bernoulli_ has generator value");
//   }
//   auto self_tensor = torch::lazy::TryGetLtcTensor(self);

//   UNIMPLEMENTED_FUNCTION_ERROR();
//   // torch::lazy::bernoulli_(self_tensor, p);
//   // return self;
// }

at::Tensor LazyNativeFunctions::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);
  auto xsize = torch::lazy::ToI64Vector(size);
  auto xstride = torch::lazy::ToI64Vector(stride);
  if (!torch::lazy::StrideIsSupported(xstride)) {
    UNIMPLEMENTED_FUNCTION_ERROR();
  }
  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(CreateAsStridedViewInfo(
          self_tensor->shape(), std::move(xsize), std::move(xstride),
          storage_offset)));
}

const at::Tensor& LazyNativeFunctions::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);
  auto xsize = torch::lazy::ToI64Vector(size);
  auto xstride = torch::lazy::ToI64Vector(stride);
  if (!torch::lazy::StrideIsSupported(xstride)) {
    UNIMPLEMENTED_FUNCTION_ERROR();
  }
  if (self_tensor->data()->view == nullptr) {
    self_tensor->SetIrValue(torch::lazy::MakeAsStrided(
        self_tensor->GetIrValue(), std::move(xsize), std::move(xstride),
        storage_offset.value_or(0)));
  } else {
    auto input_shape = self_tensor->shape();
    self_tensor->SetSubView(CreateAsStridedViewInfo(
        input_shape, std::move(xsize), std::move(xstride), storage_offset));
  }
  return self;
}

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
      copy_(dst_tensor, self_tensor);
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

at::Tensor LazyNativeFunctions::_to_copy(
    const at::Tensor& self, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory, bool non_blocking,
    c10::optional<at::MemoryFormat> memory_format) {
  PRINT_FUNCTION();
  auto options = self.options();
  if (dtype) {
    // I put each of these setters in a conditional instead of doing `self.options().dtype(dtype).layout(layout)...
    // because calling .dtype(nullopt) on an options() that already has dtype appears to wipe it
    options = options.dtype(dtype);
  }
  if (layout) {
    options = options.layout(layout);
  }
  if (memory_format) {
    options = options.memory_format(memory_format);
  }
  if (pin_memory) {
    // TODO(whc) can we honor 'pin_memory' in some/all cases?
    options = options.pinned_memory(pin_memory);
    TORCH_WARN_ONCE("Pinned memory used in lazy _to_copy, check if the "
                    "behavior is as intended");
  }

  TORCH_LAZY_FN_COUNTER("lazy::");
  auto lazy_self = torch::lazy::TryGetLtcTensor(self);
  if (!lazy_self && device && device->type() == c10::kLazy) {
    // Case 1: eager->lazy (we create a new lazy tensor)

    auto eager_tensor =
        self.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
    lazy_self = torch::lazy::GetOrCreateLtcTensor(
        eager_tensor, torch::lazy::atenDeviceToBackendDevice(*device));
    return torch::lazy::CreateAtenFromLtcTensor(lazy_self);
  } else if (device && device->type() != c10::kLazy) {
    // Case 2: lazy->eager (forces a graph break since we are materializing a tensor)

    TORCH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    options = options.device(device);
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
    return moved_eager_tensor;
  } else if (
      device && device->type() == c10::kLazy && device->has_index() &&
      device->index() != self.device().index()) {
    // Case 3: lazy:0 -> lazy:1

    // TODO(whc) what do we actually want to do here?
    //   option 1: materialize, move eager tensor, create new lazy tensor
    //     - this should be our default, as it is what would happen before we implemented _to_copy
    //     - actually combines case 1 + case 2
    //   option 2: support multiple devices inside one lazy/TS executor (case 4)
    //     - but: we may have other assumptions that there is just one device per executor? so don't take this lightly

    TORCH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    // we move the eager tensor to the 'eager' equivalent of our lazy device
    // e.g. if our device is lazy:1, the backend maps that to cuda:1, which is what we use
    auto eager_device = c10::Device(
        torch::lazy::getBackend()->EagerFallbackDeviceType(), device->index());
    options = options.device(eager_device);
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/false, /*copy=*/true);
    lazy_self = torch::lazy::GetOrCreateLtcTensor(
        moved_eager_tensor,
        torch::lazy::atenDeviceToBackendDevice(eager_device));
    return torch::lazy::CreateAtenFromLtcTensor(lazy_self);

  } else {
    // Case 4: lazy->lazy (special case: keep the _to_copy INSIDE the lazy graph)

    // Note: captured _to_copy will be executed with real eager tensors, not lazy tensors.
    // We DO NOT want to burn 'lazy:0' as the device into this captured IR, or we will try to
    // convert an eager tensor back to a lazy one inside the torchscript executor
    // lazy:0 -> lazy:1 is handled in case3, so we can safely drop the device argument
    device = c10::nullopt;

    auto shapes = torch::lazy::compute_shape__to_copy(
        self, dtype, layout, device, pin_memory, non_blocking, memory_format);
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    auto node = torch::lazy::MakeNode<ToCopy>(
        lazy_self->GetIrValue(), dtype, layout, device, pin_memory,
        non_blocking, memory_format, std::move(shapes));

    auto result =
        torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(
            std::move(node), lazy_self->GetDevice()));
    return result;
  }
};

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

at::Tensor LazyNativeFunctions::empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return LazyNativeFunctions::as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor LazyNativeFunctions::expand(
    const at::Tensor& self, at::IntArrayRef size, bool implicit) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  auto output = torch::lazy::LazyTensor::Create(
      torch::lazy::MakeExpand(
          self_tensor->GetIrValue(),
          GetExpandDimensions(input_shape.Get(), std::move(size.vec())),
          /*is_scalar_expand=*/false),
      self_tensor->GetDevice());
  output->SetStorage(self_tensor->Storage());
  return torch::lazy::CreateAtenFromLtcTensor(output);
}

at::Tensor&
LazyNativeFunctions::fill_(at::Tensor& self, const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto self_tensor = torch::lazy::TryGetLtcTensor(self);

  torch::lazy::Value constant =
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(
          value, self_tensor->shape(), self_tensor->GetDevice());
  self_tensor->SetInPlaceIrValue(std::move(constant));
  return self;
}

at::Tensor
LazyNativeFunctions::permute(const at::Tensor& self, at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kPermute, input_shape,
      torch::lazy::GetCanonicalDimensionIndices(
          torch::lazy::ToI64Vector(dims), input_shape.Get().dim()));
  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(std::move(view_info)));
}

at::Tensor LazyNativeFunctions::select(
    const at::Tensor& self, int64_t dim, int64_t index) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(
      lazy_select(torch::lazy::TryGetLtcTensor(self), dim, index));
}

at::Tensor LazyNativeFunctions::slice(
    const at::Tensor& self, int64_t dim, c10::optional<int64_t> start,
    c10::optional<int64_t> end, int64_t step) {
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  TORCH_LAZY_FN_COUNTER("lazy::");
  return torch::lazy::CreateAtenFromLtcTensor(lazy_slice(
      torch::lazy::TryGetLtcTensor(self), dim, start_val, end_val, step));
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self) {
  return squeeze(self, -1);
}

at::Tensor LazyNativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  int64_t squeeze_dim = -1;
  if (dim != -1) {
    squeeze_dim =
        torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim());
  }
  auto output_dimensions =
      BuildSqueezedDimensions(input_shape.Get().sizes(), squeeze_dim);

  return torch::lazy::CreateAtenFromLtcTensor(
      create_view(self_tensor, output_dimensions));
}

at::Tensor LazyNativeFunctions::t(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/0, /*dim1=*/1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kPermute, input_shape, permute_dims);

  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(std::move(view_info)));
}

at::Tensor LazyNativeFunctions::transpose(
    const at::Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kPermute, input_shape, permute_dims);

  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(std::move(view_info)));
}

at::Tensor LazyNativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim() + 1);
  auto dimensions =
      BuildUnsqueezedDimensions(input_shape.Get().sizes(), squeeze_dim);
  return torch::lazy::CreateAtenFromLtcTensor(
      create_view(self_tensor, dimensions));
}

at::Tensor
LazyNativeFunctions::view(const at::Tensor& self, at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(),
      at::infer_size(torch::lazy::ToI64Vector(size), input_shape.numel()));
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(std::move(view_info)));
}

at::Tensor LazyNativeFunctions::_unsafe_view(
    const at::Tensor& self, at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  torch::lazy::LazyTensorPtr self_tensor = torch::lazy::TryGetLtcTensor(self);

  auto input_shape = self_tensor->shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(),
      at::infer_size(torch::lazy::ToI64Vector(size), input_shape.numel()));
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return torch::lazy::CreateAtenFromLtcTensor(
      self_tensor->CreateViewTensor(std::move(view_info)));
}

void InitializeAtenBindings() {}

} // namespace lazy
} // namespace torch
