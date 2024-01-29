#include "tensor_utils.h"

#include "../generated/LazyIr.h"
#include "../mlir_node.h"

namespace torch {
namespace lazy {

bool is_detach_copy(const torch::lazy::Node *node) {
  return node && node->op() == torch::lazy::DetachCopy::ClassOpKind();
}
bool is_detach_copy(const torch::lazy::Value &value) {
  return is_detach_copy(value.node.get());
}

torch::lazy::Node *extract_non_detach_copy_node(torch::lazy::Node *node) {
  if (!node) {
    return nullptr;
  }

  torch::lazy::TorchMlirNode *mlir_node =
      dynamic_cast<torch::lazy::TorchMlirNode *>(node);
  while (mlir_node && is_detach_copy(mlir_node)) {
    mlir_node = mlir_node->mlir_node(0);
  }
  if (!mlir_node) {
    return node;
  }
  return mlir_node;
}

const torch::lazy::Node *
extract_non_detach_copy_node(const torch::lazy::Node *node) {
  if (!node) {
    return nullptr;
  }

  const torch::lazy::TorchMlirNode *mlir_node =
      dynamic_cast<const torch::lazy::TorchMlirNode *>(node);
  while (mlir_node && is_detach_copy(mlir_node)) {
    mlir_node = mlir_node->mlir_node(0);
  }
  if (!mlir_node) {
    return node;
  }
  return mlir_node;
}

torch::lazy::DeviceData *device_data_cast(torch::lazy::Node *node) {
  if (!node) {
    return nullptr;
  }
  node = extract_non_detach_copy_node(node);
  if (node && node->op() == torch::lazy::DeviceData::ClassOpKind()) {
    return dynamic_cast<torch::lazy::DeviceData *>(node);
  }
  return nullptr;
}
const torch::lazy::DeviceData *device_data_cast(const torch::lazy::Node *node) {
  if (!node) {
    return nullptr;
  }
  node = extract_non_detach_copy_node(node);
  if (node && node->op() == torch::lazy::DeviceData::ClassOpKind()) {
    return dynamic_cast<const torch::lazy::DeviceData *>(node);
  }
  return nullptr;
}
torch::lazy::DeviceData *device_data_cast(const torch::lazy::Value &value) {
  if (!value) {
    return nullptr;
  }
  return device_data_cast(value.node.get());
}

torch::lazy::DeviceData *
device_data_cast(const at::Tensor &tensor,
                 c10::optional<torch::lazy::BackendDevice> device) {
  if (!device) {
    device = torch::lazy::GetBackendDevice(tensor);
  }
  TORCH_CHECK(device);
  torch::lazy::LazyTensorPtr lazy_tensor =
      torch::lazy::GetLtcTensorOrCreateForWrappedNumber(tensor, *device);
  if (lazy_tensor) {
    return device_data_cast(lazy_tensor->GetIrValue());
  }
  return nullptr;
}

} // namespace lazy
} // namespace torch
