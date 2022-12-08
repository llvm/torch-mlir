#include "tensor_utils.h"

#include "../generated/LazyIr.h"
#include "../mlir_node.h"


namespace torch {
namespace lazy {

bool is_detach_copy(const torch::lazy::Value& value) {
    return value->op() == torch::lazy::DetachCopy::ClassOpKind();
}

torch::lazy::DeviceData* device_data_cast(const torch::lazy::Value& value) {
    if (!value) {
        return nullptr;
    }
    torch::lazy::TorchMlirNode* node = dynamic_cast<torch::lazy::TorchMlirNode*>(value.node.get());
    while(node) {
        if (node->op() == torch::lazy::DeviceData::ClassOpKind()) {
            return dynamic_cast<torch::lazy::DeviceData*>(node);
        }
        else if (node->op() == torch::lazy::DetachCopy::ClassOpKind()) {
            node = node->mlir_node(0);
        }
        else {
            break;
        }
    }
    return nullptr;
}

torch::lazy::DeviceData* device_data_cast(
    const at::Tensor& tensor, c10::optional<torch::lazy::BackendDevice> device
) {
    if (!device) {
        device = torch::lazy::GetBackendDevice(tensor);
    }
    TORCH_CHECK(device);
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::GetLtcTensorOrCreateForWrappedNumber(tensor, *device);
    if (lazy_tensor) {
        return device_data_cast(lazy_tensor->GetIrValue());
    }
    return nullptr;
}

}  // namespace lazy
}  // namespace torch
