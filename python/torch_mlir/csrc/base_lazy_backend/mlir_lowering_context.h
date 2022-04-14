//===- mlir_lowering_context.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_lowering_context.h
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/lazy/backend/lowering_context.h>

#include "mlir-c/IR.h"
#include "mlir_node_lowering.h"

namespace torch {
namespace lazy {

class TORCH_API TorchMlirNodeLoweringInterface {
  /**
   * This interface is only needed for legacy ops, and can be removed once all
   * ops implement LtcMlirNode->lower().
   * */
public:
  TorchMlirNodeLoweringInterface() = default;

  virtual ~TorchMlirNodeLoweringInterface() = default;

  virtual bool Lower(const Node* node) = 0;

  static std::unique_ptr<TorchMlirNodeLoweringInterface>
  Create(LoweringContext* loctx);
};

class TORCH_API TorchMlirComputation : public torch::lazy::Computation {
public:
  TorchMlirComputation(
      MlirOperation func_op, MlirContext mlir_context,
      const std::shared_ptr<torch::jit::Graph>& graph);

  int parameters_size() const override;

  const std::vector<torch::lazy::Shape>& parameter_shapes() const override;

  const std::vector<std::string>& parameter_names() const override;

  const torch::lazy::Shape& result_shape() const override;

  unsigned num_results() const;

  MlirOperation func_op() const;

  std::string to_string() const;

private:
  std::vector<std::string> parameter_names_;
  std::vector<Shape> parameter_shapes_;
  Shape result_shape_;

  MlirOperation func_op_;
  MlirContext mlir_context_;
  std::shared_ptr<torch::jit::Graph> graph_;
  unsigned num_results_;
};

class TORCH_API TorchMlirLoweringContext : public torch::lazy::LoweringContext {
public:
  TorchMlirLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device);
  TorchMlirLoweringContext(
      const std::string& name, torch::lazy::BackendDevice device,
      c10::ArrayRef<torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status);

  // Get the shape of the result tuple component, given by index.
  torch::lazy::Shape GetResultShape(size_t index) const override;

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  size_t AddResult(const torch::lazy::Output& output) override;

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  void AddParameter(
      const torch::lazy::Output& output, size_t index,
      const torch::lazy::Shape& shape, const std::string& name) override;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  torch::lazy::ComputationPtr Build() override;

  // Retrieves the lowered operation for an output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const Output& output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output& output, torch::jit::Value* op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value* GetParameter(BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const;

private:
  struct Parameter {
    torch::jit::Value* param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value* op);

  void RegisterMlirDialects();

  std::shared_ptr<torch::jit::Graph> graph_;
  MlirContext mlir_context_;
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;
  std::vector<torch::jit::Value*> root_tuple_;
  OutputMap<torch::jit::Value*> emitted_outputs_;
  std::unique_ptr<TorchMlirNodeLoweringInterface> lowering_;
};

} // namespace lazy
} // namespace torch
