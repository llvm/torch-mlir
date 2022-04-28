#pragma once

#include "../mlir_node.h"

namespace torch {
namespace lazy {

class Normal : public torch::lazy::TorchMlirNode {
 public:
  Normal(const torch::lazy::Value& self, const double& mean, const double& std, std::vector<torch::lazy::Shape>&& shapes);

  std::string ToString() const override;
  torch::lazy::TorchMlirOpVector Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const override;

  double mean_;
  double std_;
};

}  // namespace lazy
}  // namespace torch
