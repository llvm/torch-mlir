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

#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/InferSize.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NativeFunctions.h>
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
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/library.h>

#include "generated/LazyIr.h"
#include "generated/LazyNativeFunctions.h"
#include "generated/shape_inference.h"
#include "ops/index.h"
#include "ops/ivalue.h"
#include "ops/split.h"
#include "ops/to_copy.h"
#include "ops/unbind_int.h"
#include "utils/exception.h"
#include "utils/sys_utils.h"

namespace {
at::Tensor to_meta(const at::Tensor &tensor) {
  // undefined tensors can't be converted to the meta device, since they don't
  // have sizes/strides
  if (!tensor.defined())
    return tensor;
  auto out = at::native::empty_strided_meta_symint(
      tensor.sym_sizes(), tensor.sym_strides(),
      /*dtype=*/c10::make_optional(tensor.scalar_type()),
      /*layout=*/c10::make_optional(tensor.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  // needs to handle wrapped numbers, so dtype promotion works properly.
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}

c10::optional<at::Tensor> to_meta(const c10::optional<at::Tensor> &tensor) {
  if (tensor.has_value()) {
    return to_meta(*tensor);
  }
  return c10::nullopt;
}

[[maybe_unused]] std::vector<at::Tensor> to_meta(at::ITensorListRef t_list) {
  std::vector<at::Tensor> outs;
  outs.reserve(t_list.size());
  for (const auto &tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}

c10::List<c10::optional<at::Tensor>>
to_meta(const c10::List<c10::optional<at::Tensor>> &t_list) {
  c10::List<c10::optional<at::Tensor>> outs;
  outs.reserve(t_list.size());
  for (const auto &tensor : t_list) {
    outs.push_back(to_meta(tensor));
  }
  return outs;
}
} // namespace

namespace torch {
namespace lazy {

namespace {

[[maybe_unused]] at::Tensor
CreateLtcTensor(const at::Tensor &tensor,
                const c10::optional<torch::lazy::BackendDevice> &device) {
  if (tensor.defined() && device) {
    return torch::lazy::CreateAtenFromLtcTensor(
        torch::lazy::LazyTensor::Create(tensor, *device));
  }
  return tensor;
}

[[maybe_unused]] c10::optional<torch::lazy::BackendDevice>
GetLtcDevice(const c10::optional<c10::Device> &device) {
  if (!device) {
    return c10::nullopt;
  }
  if (device->type() != at::kLazy) {
    return c10::nullopt;
  }
  return torch::lazy::atenDeviceToBackendDevice(*device);
}

torch::lazy::Value MaybeExpand(const torch::lazy::Value &input,
                               const torch::lazy::Shape &target_shape) {
  if (input.shape().sizes() == target_shape.sizes()) {
    return input;
  }
  return torch::lazy::MakeExpand(input, target_shape.sizes().vec(),
                                 /*is_scalar_expand=*/false);
}

void copy_(torch::lazy::LazyTensorPtr &input, torch::lazy::LazyTensorPtr &src) {
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = torch::lazy::MakeCast(src->GetIrValue(), input->dtype(),
                                         src->dtype());
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

} // namespace

// clone is special in LT because we make it a no-op.
// This should be safe to do, because every operator in the LT is functional.
at::Tensor
LazyNativeFunctions::clone(const at::Tensor &self,
                           c10::optional<at::MemoryFormat> memory_format) {
  auto self_lt = torch::lazy::TryGetLtcTensor(self);
  return torch::lazy::CreateAtenFromLtcTensor(
      self_lt->Create(self_lt->GetIrValue(), self_lt->GetDevice()));
}

at::Tensor LazyNativeFunctions::_copy_from(const at::Tensor &self,
                                           const at::Tensor &dst,
                                           bool non_blocking) {
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
    // because we are only going to use the tensor for dst.copy_()
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
      auto *impl =
          dynamic_cast<torch::lazy::LTCTensorImpl *>(dst.unsafeGetTensorImpl());
      impl->set_tensor(dst_tensor);
    }
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_copy_from_and_resize(const at::Tensor &self,
                                                      const at::Tensor &dst) {
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
    auto *dest_impl =
        dynamic_cast<torch::lazy::LTCTensorImpl *>(dst.unsafeGetTensorImpl());
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

at::Tensor LazyNativeFunctions::_to_copy(
    const at::Tensor &self, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory, bool non_blocking,
    c10::optional<at::MemoryFormat> memory_format) {
  PRINT_FUNCTION();
  auto options = self.options();
  if (dtype) {
    // I put each of these setters in a conditional instead of doing
    // `self.options().dtype(dtype).layout(layout)... because calling
    // .dtype(nullopt) on an options() that already has dtype appears to wipe it
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
    // See Note [Lazy Tensor Functionalization]
    // Invariant: if the functionalization key is in the exclude set, then we're
    // expected to return an ordinary tensor, which will be "lifted" into a
    // functional wrapper later.
    bool functionalize_output =
        !c10::impl::tls_local_dispatch_key_set().excluded_.has(
            c10::DispatchKey::Functionalize);
    return torch::lazy::to_lazy_tensor(
        self, options, *device, /*non_blocking=*/non_blocking,
        /*functionalize_output=*/functionalize_output);
  } else if (device && device->type() != c10::kLazy) {
    // Case 2: lazy->eager (forces a graph break since we are materializing a
    // tensor)

    TORCH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    options = options.device(device);
    auto moved_eager_tensor =
        eager_tensor.to(options, /*non_blocking=*/non_blocking, /*copy=*/true);
    return moved_eager_tensor;
  } else if (device && device->type() == c10::kLazy && device->has_index() &&
             device->index() != self.device().index()) {
    // Case 3: lazy:0 -> lazy:1

    // TODO(whc) what do we actually want to do here?
    //   option 1: materialize, move eager tensor, create new lazy tensor
    //     - this should be our default, as it is what would happen before we
    //     implemented _to_copy
    //     - actually combines case 1 + case 2
    //   option 2: support multiple devices inside one lazy/TS executor (case 4)
    //     - but: we may have other assumptions that there is just one device
    //     per executor? so don't take this lightly

    TORCH_INTERNAL_ASSERT(lazy_self);
    auto eager_tensor = lazy_self->ToTensor(/*detached=*/true);
    // we move the eager tensor to the 'eager' equivalent of our lazy device
    // e.g. if our device is lazy:1, the backend maps that to cuda:1, which is
    // what we use
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
    // Case 4: lazy->lazy (special case: keep the _to_copy INSIDE the lazy
    // graph)

    // Note: captured _to_copy will be executed with real eager tensors, not
    // lazy tensors. We DO NOT want to burn 'lazy:0' as the device into this
    // captured IR, or we will try to convert an eager tensor back to a lazy one
    // inside the torchscript executor lazy:0 -> lazy:1 is handled in case3, so
    // we can safely drop the device argument
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
}

at::Tensor LazyNativeFunctions::_unsafe_view(const at::Tensor &self,
                                             at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return LazyNativeFunctions::view_copy_symint(self,
                                               c10::fromIntArrayRefSlow(size));
}

at::Tensor LazyNativeFunctions::t(const at::Tensor &self) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  return at::functionalization::functionalize_aten_op<ATEN_OP(t)>::call(self);
}

std::vector<at::Tensor> LazyNativeFunctions::unbind_copy(const at::Tensor &self,
                                                         int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto common_device = torch::lazy::GetBackendDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);

  LazyTensorPtr lazy_self =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
  torch::lazy::NodePtr node =
      torch::lazy::ReuseNode<UnbindCopyInt>(lazy_self->GetIrValue(), dim);
  if (!node) {
    auto self_meta = to_meta(self);
    auto out_meta =
        at::compositeexplicitautogradnonfunctional::unbind_copy(self_meta, dim);

    std::vector<torch::lazy::Shape> shapes;
    for (const auto &shape : out_meta) {
      shapes.push_back(
          torch::lazy::Shape(shape.scalar_type(), shape.sizes().vec()));
    }

    if (torch::lazy::symbolicShapeEnabled()) {
      std::vector<torch::jit::IValue> inputs = {self, dim};
      const char *schema_str =
          "aten::unbind_copy.int(Tensor self, int dim=0) -> Tensor[]";
      applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }

    node = torch::lazy::MakeNode<UnbindCopyInt>(lazy_self->GetIrValue(), dim,
                                                std::move(shapes));
    CacheNode(node);
  }

  std::vector<at::Tensor> result;
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    result.push_back(
        torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(
            torch::lazy::Value(node, i), *common_device)));
  }

  return result;
}

std::vector<at::Tensor> LazyNativeFunctions::split_with_sizes_copy_symint(
    const at::Tensor &self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto common_device = torch::lazy::GetBackendDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);

  LazyTensorPtr lazy_self =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
  torch::lazy::NodePtr node = torch::lazy::ReuseNode<SplitWithSizesCopy>(
      lazy_self->GetIrValue(), GetSymIntArrayRefValue(split_sizes), dim);
  if (!node) {
    auto self_meta = to_meta(self);
    auto out_meta = at::compositeexplicitautogradnonfunctional::
        split_with_sizes_copy_symint(self_meta, split_sizes, dim);

    std::vector<torch::lazy::Shape> shapes;
    for (const auto &shape : out_meta) {
      shapes.push_back(
          torch::lazy::Shape(shape.scalar_type(), shape.sizes().vec()));
    }

    if (torch::lazy::symbolicShapeEnabled()) {
      std::vector<torch::jit::IValue> inputs = {self, split_sizes, dim};
      const char *schema_str = "aten::split_with_sizes_copy(Tensor self, "
                               "SymInt[] split_sizes, int dim=0) -> Tensor[]";
      applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }

    node = torch::lazy::MakeNode<SplitWithSizesCopy>(
        lazy_self->GetIrValue(), GetSymIntArrayRefValue(split_sizes), dim,
        std::move(shapes));
    CacheNode(node);
  }

  std::vector<at::Tensor> result;
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    result.push_back(
        torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(
            torch::lazy::Value(node, i), *common_device)));
  }

  return result;
}

std::vector<at::Tensor>
LazyNativeFunctions::split_copy_symint(const at::Tensor &self,
                                       c10::SymInt split_size, int64_t dim) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto common_device = torch::lazy::GetBackendDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);
  LazyTensorPtr lazy_self =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
  torch::lazy::NodePtr node = torch::lazy::ReuseNode<SplitCopyTensor>(
      lazy_self->GetIrValue(), GetSymIntValue(split_size), dim);
  if (!node) {
    auto self_meta = to_meta(self);
    auto out_meta =
        at::compositeexplicitautogradnonfunctional::split_copy_symint(
            self_meta, split_size, dim);

    std::vector<torch::lazy::Shape> shapes;
    for (const auto &shape : out_meta) {
      shapes.push_back(
          torch::lazy::Shape(shape.scalar_type(), shape.sizes().vec()));
    }
    const size_t num_outputs = shapes.size();

    if (torch::lazy::symbolicShapeEnabled()) {
      std::vector<torch::jit::IValue> inputs = {self, split_size, dim};
      const char *schema_str = "aten::split_copy.Tensor(Tensor self, SymInt "
                               "split_size, int dim=0) -> Tensor[]";
      applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }

    node = torch::lazy::MakeNode<SplitCopyTensor>(
        lazy_self->GetIrValue(), GetSymIntValue(split_size), dim,
        std::move(shapes), num_outputs);
    CacheNode(node);
  }

  std::vector<at::Tensor> result;
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    result.push_back(
        torch::lazy::CreateAtenFromLtcTensor(torch::lazy::LazyTensor::Create(
            torch::lazy::Value(node, i), *common_device)));
  }
  return result;
}

at::Tensor LazyNativeFunctions::index(
    const at::Tensor &self,
    const c10::List<c10::optional<at::Tensor>> &indices) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto common_device = torch::lazy::GetBackendDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);
  LazyTensorPtr lazy_self =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);

  std::vector<torch::lazy::Value> values;
  for (const auto &it : indices) {
    c10::optional<at::Tensor> tensor = it;
    LazyTensorPtr lazy_tensor =
        torch::lazy::TryGetLtcTensor(tensor.value_or(at::Tensor()));
    values.push_back(
        lazy_tensor
            ? lazy_tensor->GetIrValue()
            : torch::lazy::Value(MakeNode<IValueConstant>(c10::IValue()), 0));
  }

  auto list = MakeNode<TorchMlirOptionalTensorList>(values);

  torch::lazy::NodePtr node =
      torch::lazy::ReuseNode<IndexTensor>(lazy_self->GetIrValue(), list);

  if (!node) {
    auto self_meta = to_meta(self);
    auto indices_meta = to_meta(indices);
    auto out_meta = at::meta::index(self_meta, indices_meta);

    std::vector<torch::lazy::Shape> shapes{
        torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    if (torch::lazy::symbolicShapeEnabled()) {
      std::vector<torch::jit::IValue> inputs = {self, indices};
      const char *schema_str =
          "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor";
      applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }

    node = torch::lazy::MakeNode<IndexTensor>(lazy_self->GetIrValue(), list,
                                              std::move(shapes));
    CacheNode(node);
  }

  auto result = torch::lazy::CreateAtenFromLtcTensor(
      torch::lazy::LazyTensor::Create(std::move(node), *common_device));

  return result;
}

at::Tensor LazyNativeFunctions::index_put(
    const at::Tensor &self, const c10::List<c10::optional<at::Tensor>> &indices,
    const at::Tensor &values, bool accumulate) {
  TORCH_LAZY_FN_COUNTER("lazy::");
  auto common_device = torch::lazy::GetBackendDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);
  LazyTensorPtr lazy_self =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(self, *common_device);
  LazyTensorPtr lazy_valeus =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(values, *common_device);

  std::vector<torch::lazy::Value> indices_vector;
  for (const auto &it : indices) {
    c10::optional<at::Tensor> tensor = it;
    LazyTensorPtr lazy_tensor =
        torch::lazy::TryGetLtcTensor(tensor.value_or(at::Tensor()));
    indices_vector.push_back(
        lazy_tensor
            ? lazy_tensor->GetIrValue()
            : torch::lazy::Value(MakeNode<IValueConstant>(c10::IValue()), 0));
  }

  auto indices_list = MakeNode<TorchMlirOptionalTensorList>(indices_vector);

  torch::lazy::NodePtr node =
      torch::lazy::ReuseNode<IndexPut>(lazy_self->GetIrValue(), indices_list,
                                       lazy_valeus->GetIrValue(), accumulate);

  if (!node) {
    auto self_meta = to_meta(self);
    auto indices_meta = to_meta(indices);
    auto values_meta = to_meta(values);

    auto out_meta = at::compositeexplicitautograd::index_put(
        self_meta, indices_meta, values_meta, accumulate);

    std::vector<torch::lazy::Shape> shapes{
        torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
    TORCH_INTERNAL_ASSERT(shapes.size() == 1);
    if (torch::lazy::symbolicShapeEnabled()) {
      std::vector<torch::jit::IValue> inputs = {self, indices, values};
      const char *schema_str =
          "aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool "
          "accumulate=False) -> Tensor";
      applySymbolicShapesOnLT(schema_str, inputs, shapes);
    }

    node = torch::lazy::MakeNode<IndexPut>(
        lazy_self->GetIrValue(), indices_list, lazy_valeus->GetIrValue(),
        accumulate, std::move(shapes));
    CacheNode(node);
  }

  auto result = torch::lazy::CreateAtenFromLtcTensor(
      torch::lazy::LazyTensor::Create(std::move(node), *common_device));

  return result;
}

// This is needed by the torch.tensor constructor.
// LazyTensor always opts into functionalization.
// "lifting" a tensor for functionalization means wrapping it in a
// FunctionalTensorWrapper object.
at::Tensor LazyNativeFunctions::lift(const at::Tensor &tensor) {
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return at::functionalization::impl::to_functional_tensor(tensor);
}

at::Tensor LazyNativeFunctions::lift_fresh(const at::Tensor &tensor) {
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return at::functionalization::impl::to_functional_tensor(tensor);
}

// All of the below ops correspond to CompositeExplicitAutograd kernels from
// core that call into view operators internally. These are all composite ops
// that LTC can technically re-use / get for free, but we need to
// "functionalize" them to remove the view ops before we can use them.
at::Tensor LazyNativeFunctions::block_diag(at::TensorList tensors) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      block_diag)>::call(tensors);
}
at::Tensor LazyNativeFunctions::new_empty_strided_symint(
    const at::Tensor &self, c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  if (!device || device->type() == c10::DeviceType::Lazy) {
    return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
        new_empty_strided)>::call(self, size, stride, dtype, layout, device,
                                  pin_memory);
  }
  // For cases when device != lazy, for example:
  // lazy_tensor.new_empty_strided(..., "cpu") we need to avoid explicit
  // functionalization. To do that we create regular cpu tensors.
  at::Tensor t = at::empty_symint(
      size, (dtype ? dtype : c10::optional<at::ScalarType>(self.scalar_type())),
      (layout ? layout : c10::optional<at::Layout>(self.layout())), device,
      pin_memory, c10::nullopt);
  return t.as_strided_symint(size, stride, /*storage_offset=*/0);
}

at::Tensor LazyNativeFunctions::narrow_copy_symint(const at::Tensor &self,
                                                   int64_t dim,
                                                   c10::SymInt start,
                                                   c10::SymInt length) {
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      narrow_copy)>::call(self, dim, start, length);
}
at::Tensor LazyNativeFunctions::pixel_shuffle(const at::Tensor &self,
                                              int64_t upscale_factor) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_shuffle)>::call(self, upscale_factor);
}
at::Tensor LazyNativeFunctions::pixel_unshuffle(const at::Tensor &self,
                                                int64_t downscale_factor) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_unshuffle)>::call(self, downscale_factor);
}
at::Tensor LazyNativeFunctions::select_backward(const at::Tensor &grad_output,
                                                at::IntArrayRef input_sizes,
                                                int64_t dim, int64_t index) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      select_backward)>::call(grad_output, input_sizes, dim, index);
}
at::Tensor LazyNativeFunctions::slice_backward_symint(
    const at::Tensor &grad_output, at::SymIntArrayRef input_sizes, int64_t dim,
    c10::SymInt start, c10::SymInt end, c10::SymInt step) {
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      slice_backward)>::call(grad_output, input_sizes, dim, start, end, step);
}
at::Tensor LazyNativeFunctions::diagonal_backward(const at::Tensor &grad_output,
                                                  at::IntArrayRef input_sizes,
                                                  int64_t offset, int64_t dim1,
                                                  int64_t dim2) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      diagonal_backward)>::call(grad_output, input_sizes, offset, dim1, dim2);
}
at::Tensor LazyNativeFunctions::_trilinear(
    const at::Tensor &i1, const at::Tensor &i2, const at::Tensor &i3,
    at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3,
    at::IntArrayRef sumdim, int64_t unroll_dim) {
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      _trilinear)>::call(i1, i2, i3, expand1, expand2, expand3, sumdim,
                         unroll_dim);
}
at::Tensor LazyNativeFunctions::linalg_pinv(
    const at::Tensor &self, const c10::optional<at::Tensor> &atol,
    const c10::optional<at::Tensor> &rtol, bool hermitian) {
  return at::functionalization::functionalize_aten_op<ATEN_OP2(
      linalg_pinv, atol_rtol_tensor)>::call(self, atol, rtol, hermitian);
}

// functionalize_aten_op can't handle out= ops directly.
// Instead, we can call the composite kernel from core, and copy and mutations
// back to the inputs.
at::Tensor &LazyNativeFunctions::logsumexp_out(const at::Tensor &self,
                                               at::IntArrayRef dim,
                                               bool keepdim, at::Tensor &out) {
  auto self_wrapped = at::functionalization::impl::to_functional_tensor(self);
  auto out_wrapped = at::functionalization::impl::to_functional_tensor(out);
  // directly call the composite kernel from core.
  // Make sure to re-enable functionalization first.
  auto curr_tls = c10::impl::tls_local_dispatch_key_set();
  auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
  tls_reenable_functionalize.set_included(curr_tls.included_);
  tls_reenable_functionalize.set_excluded(
      curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
  c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
  at::native::logsumexp_out(self_wrapped, dim, keepdim, out_wrapped);
  auto out_unwrapped =
      at::functionalization::impl::from_functional_tensor(out_wrapped);
  // propagate mutations back to the inputs (including resizing)
  out.resize_(out_unwrapped.sizes());
  out.copy_(out_unwrapped);
  return out;
}

void InitializeAtenBindings() {}

} // namespace lazy
} // namespace torch