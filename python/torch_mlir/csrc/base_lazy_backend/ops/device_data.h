#pragma once

#include "../mlir_node.h"

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>


namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TorchMlirNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_device_data;
  }

  explicit DeviceData(std::shared_ptr<BackendData> data);

  // A DeviceData node can be reused if the shape matches,
  // but we will substitute the actual data_ pointer under
  // the hood.
  bool CanBeReused(std::shared_ptr<BackendData> data) const {
    return data_->shape() == data->shape();
  }

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  void SetData(std::shared_ptr<BackendData> data) {
    data_ = data;
  }

  static const DeviceData* Cast(const Node* node);

  // To reuse IR nodes, use this method to create DeviceData nodes
  // instead of calling the constructor directly.
  static NodePtr Create(std::shared_ptr<BackendData> data);

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch
