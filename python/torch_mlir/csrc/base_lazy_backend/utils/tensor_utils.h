#pragma once

#include "torch/csrc/lazy/backend/backend_device.h"
#include "torch/csrc/lazy/core/tensor.h"

#include "../ops/device_data.h"


namespace torch {
namespace lazy {

inline torch::lazy::DeviceData* device_data_cast(
    const at::Tensor& tensor, c10::optional<torch::lazy::BackendDevice> device = c10::nullopt
) {
    if (!device) {
        device = torch::lazy::GetBackendDevice(tensor);
    }
    TORCH_CHECK(device);
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(tensor, *device);
    if (lazy_tensor) {
        torch::lazy::Value param_value = lazy_tensor->GetIrValue();
        if (param_value && param_value->op() == torch::lazy::DeviceData::ClassOpKind()) {
            return dynamic_cast<torch::lazy::DeviceData*>(param_value.node.get());
        }
    }
    return nullptr;
}

}  // namespace lazy
}  // namespace torch
