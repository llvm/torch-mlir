//===- aten_mlir_bridge.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under a pytorch-style license
// See frontends/pytorch/LICENSE for license information.
//
//===----------------------------------------------------------------------===//

// Structured similarly to code from git@github.com:pytorch/xla.git

#include "aten_mlir_bridge.h"

#include <string>
#include <vector>

#include "device.h"
#include "tensor_impl.h"

namespace torch_mlir {
namespace bridge {
namespace {

class AtenMLIRDeviceMapper {
public:
  static AtenMLIRDeviceMapper *Get();

  size_t GetDeviceOrdinal(const Device &device) const {
    auto it = devices_ordinals_.find(device);
    assert(it != devices_ordinals_.end());
    return it->second;
  }

  const Device &GetDeviceFromOrdinal(size_t ordinal) const {
    return devices_.at(ordinal);
  }

private:
  AtenMLIRDeviceMapper() {
    std::vector<std::string> local_devices{"mlir:0", "mlir:1", "mlir:2"};
    for (auto &device_str : local_devices) {
      devices_.emplace_back(device_str);
      devices_ordinals_[devices_.back()] = devices_.size() - 1;
    }
  }

  std::vector<Device> devices_;
  std::map<Device, size_t> devices_ordinals_;
};

AtenMLIRDeviceMapper *AtenMLIRDeviceMapper::Get() {
  static AtenMLIRDeviceMapper *device_mapper = new AtenMLIRDeviceMapper();
  return device_mapper;
}

} // namespace

c10::optional<MLIRTensor> TryGetMLIRTensor(const at::Tensor &tensor) {
  MLIRTensorImpl *impl =
      dynamic_cast<MLIRTensorImpl *>(tensor.unsafeGetTensorImpl());
  if (impl == nullptr) {
    return c10::nullopt;
  }
  return impl->tensor();
}

MLIRTensor GetMLIRTensor(const at::Tensor &tensor) {
  auto xtensor = TryGetMLIRTensor(tensor);
  assert(xtensor && "Input tensor is not an MLIR tensor");
  return *xtensor;
}

MLIRTensor GetOrCreateMLIRTensor(const at::Tensor &tensor,
                                 const Device &device) {
  if (!tensor.defined()) {
    return MLIRTensor();
  }
  auto xtensor = TryGetMLIRTensor(tensor);
  return xtensor ? *xtensor : MLIRTensor::Create(tensor, device);
}

std::vector<at::Tensor> MLIRCreateTensorList(const at::TensorList &tensors) {

  std::vector<at::Tensor> aten_device_tensors(tensors.size());
  std::vector<MLIRTensor> device_tensors;

  std::vector<bool> to_translate(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor &tensor = tensors[i];
    if (tensor.defined()) {
      auto xtensor = TryGetMLIRTensor(tensor);
      if (xtensor) {
        to_translate[i] = true;
        device_tensors.push_back(*xtensor);
      } else {
        aten_device_tensors[i] = tensor;
      }
    }
  }

  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_device_tensors[i] =
          std::move(device_tensors[defined_pos++].ToTensor());
    }
  }
  return aten_device_tensors;
}

c10::optional<Device> GetMLIRDevice(const at::TensorList &tensors) {
  for (const auto &tensor : tensors) {
    auto device = GetMLIRDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<Device> GetMLIRDevice(const at::TensorOptions &tensor_options) {
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetMLIRDevice(tensor_options.device());
}

c10::optional<Device> GetMLIRDevice(const c10::Device &device) {
  if (device.type() != at::kXLA) {
    return c10::nullopt;
  }
  return AtenDeviceToMLIRDevice(device);
}

c10::optional<Device> GetMLIRDevice(const at::Tensor &tensor) {
  auto xtensor = TryGetMLIRTensor(tensor);
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
}

Device AtenDeviceToMLIRDevice(const c10::Device &device) {
  assert(device.type() == at::kXLA);
  int ordinal = device.has_index() ? device.index() : -1;
  if (ordinal < 0) {
    c10::Device current_device = MLIRTensorImpl::GetCurrentAtenDevice();
    if (current_device.has_index()) {
      ordinal = current_device.index();
    }
  }
  if (ordinal < 0) {
    return *GetDefaultDevice();
  }
  return AtenMLIRDeviceMapper::Get()->GetDeviceFromOrdinal(ordinal);
}

c10::Device MLIRDeviceToAtenDevice(const Device &device) {
  // TODO: define our own device and stop hijacking the xla device.
  return c10::Device(at::kXLA,
                     AtenMLIRDeviceMapper::Get()->GetDeviceOrdinal(device));
}

at::Tensor MLIRToAtenTensor(MLIRTensor device_tensor,
                            const at::TensorOptions &tensor_options) {
  if (tensor_options.has_device()) {
    assert(tensor_options.device().type() != at::kXLA);
  }

  at::Tensor tensor = device_tensor.ToTensor();

  // We need to copy the tensor since it is cached within the MLIRTensor, and
  // returning it directly might expose it to in place changes.
  return tensor.to(tensor_options, /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor AtenFromMLIRTensor(MLIRTensor device_tensor) {
  assert(!device_tensor.is_null());
  at::Tensor ret =
      at::Tensor(c10::make_intrusive<MLIRTensorImpl>(std::move(device_tensor)));
  return ret;
}

at::Tensor CreateMLIRTensor(at::Tensor tensor,
                            const c10::optional<Device> &device) {
  if (tensor.defined() && device) {
    MLIRTensor device_tensor = MLIRTensor::Create(std::move(tensor), *device);
    tensor = AtenFromMLIRTensor(device_tensor);
  }
  return tensor;
}

} // namespace bridge
} // namespace torch_mlir
