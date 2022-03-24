//===- mlir_lowering_context.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/lazy_tensor_staging/lazy_tensor_core/lazy_tensor_core/csrc/ts_backend/ts_lowering_context.cpp
//===----------------------------------------------------------------------===//

#include <iostream>

#include "../utils/exception.h"
#include "mlir_lowering_context.h"

namespace torch {
namespace lazy {

MlirLoweringContext::MlirLoweringContext(
    const std::string& name, BackendDevice device)
    : LoweringContext(name, std::forward<BackendDevice>(device)) {}

MlirLoweringContext::MlirLoweringContext(
    const std::string& name, BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order, Util::EmissionMap emit_status)
    : LoweringContext(
          name, std::forward<BackendDevice>(device),
          std::forward<c10::ArrayRef<torch::lazy::Node*>>(post_order),
          std::forward<Util::EmissionMap>(emit_status)) {}

int MlirComputation::parameters_size() const { UNIMPLEMENTED_FUNCTION_ERROR(); }

const std::vector<torch::lazy::Shape>&
MlirComputation::parameter_shapes() const {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

const std::vector<std::string>& MlirComputation::parameter_names() const {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

const torch::lazy::Shape& MlirComputation::result_shape() const {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

// Get the shape of the result tuple component, given by index.
torch::lazy::Shape MlirLoweringContext::GetResultShape(size_t index) const {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

// Adds the given output as a component of the result tuple and returns its
// assigned position within the tuple.
size_t MlirLoweringContext::AddResult(const torch::lazy::Output& output) {
  const torch::lazy::Node* node;
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    node = output.node;

    auto post_order = Util::ComputePostOrder(node, &emit_status_);
    for (auto po_node : post_order) {
      // TODO: uncomment after lowering is implemented
      // bool ok = lowering_->Lower(node);
      // TORCH_CHECK(ok, "Failed to lower: ", node->ToString());
    }
    emitted_outputs_[output] = node;
  } else {
    node = it->second;
  }
  result_tuple_.emplace_back(node);
  return result_tuple_.size() - 1;
}

// Associates the given output with the input parameter of the given index and
// shape. Only used for the operator-by-operator execution, mostly for
// debugging purposes.
void MlirLoweringContext::AddParameter(
    const torch::lazy::Output& output, size_t index,
    const torch::lazy::Shape& shape, const std::string& name) {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

// Build the computation capturing all the operations created with the
// embedded builder (returned by the builder() API).
ComputationPtr MlirLoweringContext::Build() {
  for (const torch::lazy::Node* output : result_tuple_) {
  }
  return std::make_shared<MlirComputation>();
}

} // namespace lazy
} // namespace torch
