#pragma once

#include "torch/csrc/lazy/backend/backend_device.h"
#include "torch/csrc/lazy/core/tensor.h"

#include "../ops/device_data.h"

namespace torch {
namespace lazy {

TORCH_API bool is_detach_copy(const torch::lazy::Node *);
TORCH_API bool is_detach_copy(const torch::lazy::Value &);

TORCH_API torch::lazy::Node *extract_non_detach_copy_node(torch::lazy::Node *);
TORCH_API const torch::lazy::Node *
extract_non_detach_copy_node(const torch::lazy::Node *);

TORCH_API torch::lazy::DeviceData *device_data_cast(torch::lazy::Node *);
TORCH_API const torch::lazy::DeviceData *
device_data_cast(const torch::lazy::Node *);
TORCH_API torch::lazy::DeviceData *
device_data_cast(const torch::lazy::Value &value);
TORCH_API torch::lazy::DeviceData *device_data_cast(
    const at::Tensor &tensor,
    c10::optional<torch::lazy::BackendDevice> device = c10::nullopt);

} // namespace lazy
} // namespace torch
