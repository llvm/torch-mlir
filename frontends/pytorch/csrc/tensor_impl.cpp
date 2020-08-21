//===- tensor_impl.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "tensor_impl.h"
#include "aten_mlir_bridge.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace torch_mlir {
namespace {

thread_local c10::Device g_current_device(at::DeviceType::XLA, 0);

struct MLIRGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device device) const override {
    std::swap(g_current_device, device);
    return device;
  }

  c10::Device getDevice() const override { return g_current_device; }

  void setDevice(c10::Device device) const override {
    g_current_device = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    g_current_device = device;
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_current_device);
  }

  c10::DeviceIndex deviceCount() const noexcept override { return 0; }
};

C10_REGISTER_GUARD_IMPL(XLA, MLIRGuardImpl);

} // namespace

MLIRTensorImpl::MLIRTensorImpl(MLIRTensor tensor)
    : c10::TensorImpl(c10::XLATensorId(), GetTypeMeta(tensor),
                      bridge::MLIRDeviceToAtenDevice(tensor.GetDevice())),
      tensor_(std::move(tensor)) {}

c10::intrusive_ptr<c10::TensorImpl> MLIRTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion &version_counter,
    bool allow_tensor_metadata_change) const {
  // std::cout << "MLIRTensorImpl::" << __func__ << std::endl;
  auto impl = c10::make_intrusive<MLIRTensorImpl>(tensor_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

void MLIRTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl> &impl) {
  // std::cout << "MLIRTensorImpl::" << __func__ << std::endl;
  MLIRTensorImpl *tensor_impl = dynamic_cast<MLIRTensorImpl *>(impl.get());
  copy_tensor_metadata(
      /*src_impl=*/tensor_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  tensor_impl->tensor_.ShallowCopyTo(&tensor_);
  generation_ = 0;
}

at::IntArrayRef MLIRTensorImpl::sizes() const {
  const_cast<MLIRTensorImpl *>(this)->SetupSizeProperties();
  return c10::TensorImpl::sizes();
}

at::IntArrayRef MLIRTensorImpl::strides() const {
  const_cast<MLIRTensorImpl *>(this)->SetupSizeProperties();
  return c10::TensorImpl::strides();
}

int64_t MLIRTensorImpl::dim() const {
  const_cast<MLIRTensorImpl *>(this)->SetupSizeProperties();
  return c10::TensorImpl::dim();
}

int64_t MLIRTensorImpl::numel() const {
  const_cast<MLIRTensorImpl *>(this)->SetupSizeProperties();
  return c10::TensorImpl::numel();
}

bool MLIRTensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  assert(is_contiguous_ && "Non-contiguous storage for MLIR tensor");
  return true;
}

int64_t MLIRTensorImpl::size(int64_t d) const {
  const_cast<MLIRTensorImpl *>(this)->SetupSizeProperties();
  return c10::TensorImpl::size(d);
}

void MLIRTensorImpl::SetupSizeProperties() {
  size_t generation = tensor_.generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto sizes = tensor_.sizes();
    auto strides = tensor_.strides();

    strides_.clear();
    sizes_.clear();
    numel_ = 1;

    for (auto t : llvm::zip(sizes, strides)) {
      auto size = std::get<0>(t);
      sizes_.push_back(size);
      strides_.push_back(std::get<1>(t));
      numel_ *= size;
    }

    generation_ = generation;
  }
}

caffe2::TypeMeta MLIRTensorImpl::GetTypeMeta(const MLIRTensor &tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

c10::Device MLIRTensorImpl::GetCurrentAtenDevice() { return g_current_device; }

c10::Device MLIRTensorImpl::SetCurrentAtenDevice(c10::Device device) {
  std::swap(g_current_device, device);
  return device;
}

void MLIRTensorImpl::AtenInitialize() {}

const at::Storage &MLIRTensorImpl::storage() const {
  assert(0 && "MLIR tensors do not have storage");
}

bool MLIRTensorImpl::has_storage() const { return false; }

} // namespace torch_mlir
