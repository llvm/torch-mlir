#include <sstream>

#include <torch/csrc/lazy/core/ir_builder.h>

#include "../backend_impl.h"
#include "device_data.h"

namespace torch {
namespace lazy {

DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TorchMlirNode(ClassOpKind(), data->shape(),
                    /*num_outputs=*/1,
                    /*hash_seed=*/static_cast<uint32_t>(101)),
      data_(std::move(data)) {
  propagate_name();
}

void DeviceData::propagate_name() {
  if (data_ && name_ != "") {
    // Add device data name to backend data
    TorchMlirBackendData *mlir_data =
        dynamic_cast<TorchMlirBackendData *>(data_.get());
    TORCH_CHECK(mlir_data);
    auto *info =
        dynamic_cast<TorchMlirBackendData::Info *>(mlir_data->mlir_info());
    TORCH_CHECK(info);
    info->name = name_;
  }
}

void DeviceData::SetData(std::shared_ptr<BackendData> data) {
  data_ = data;
  propagate_name();
}

void DeviceData::SetName(const std::string &name) {
  name_ = name;
  propagate_name();
}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TorchMlirNode::ToString() << ", device=" << data_->device();
  if (name_ != "") {
    ss << ", name=" << name_;
  }
  return ss.str();
}

const DeviceData *DeviceData::Cast(const Node *node) {
  return NodeCast<DeviceData>(node);
}

NodePtr DeviceData::Create(std::shared_ptr<BackendData> data) {
  NodePtr node = ReuseOrMakeNode<DeviceData>(data);
  // ReuseOrMakeNode may return a reused node which has the same shape,
  // however, we need to replace the old data_ with the new one.
  // Ditching the old data_ is safe because tracing is done iteration
  // by iteration, and after we lauch the async device execution for the
  // previous iteration, data_ in DeviceData nodes are not needed anymore.
  DeviceData *device_data = static_cast<DeviceData *>(node.get());
  device_data->SetData(data);
  return node;
}

} // namespace lazy
} // namespace torch
