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

#include <unordered_map>
#include <vector>

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/lazy/backend/lowering_context.h>

#include "mlir-c/IR.h"
#include "mlir_node_lowering.h"

namespace torch {
namespace lazy {

class TORCH_API TorchMlirLoweringContext : public torch::lazy::LoweringContext {
public:
  // Describes an input/output alias as inserted by the SetUpAlias() API.
  struct InputOutputAlias {
    // Specifies the index of the aliased buffer in the result tuple.
    std::vector<int64_t> output_index;
    // Specifies the parameter containing the buffer to be aliased.
    int64_t param_number;
    // Specifies the index of the aliased buffer in the parameter
    std::vector<int64_t> param_index;
    // Specifies if the alias is a must alias or may alias.
    bool must_alias;
  };
  using InputOutputAliases = std::vector<InputOutputAlias>;

  TorchMlirLoweringContext(const std::string &name,
                           torch::lazy::BackendDevice device);
  TorchMlirLoweringContext(const std::string &name,
                           torch::lazy::BackendDevice device,
                           c10::ArrayRef<const torch::lazy::Node *> post_order,
                           torch::lazy::Util::EmissionMap emit_status);

  void Lower(const Node *node);

  // Adds a new input/output alias.
  void SetUpAlias(const std::vector<int64_t> &output_index,
                  int64_t param_number, const std::vector<int64_t> &param_index,
                  bool must_alias = false) override;

  // Check if parameter shape matches result at index.
  bool CheckResultShape(const BackendDataPtr &parameter_data,
                        size_t result_idx) override;

  // Adds the given output as a component of the result tuple and returns its
  // assigned position within the tuple.
  size_t AddResult(const torch::lazy::Output &output) override;

  // Associates the given output with the input parameter of the given index and
  // shape. Only used for the operator-by-operator execution, mostly for
  // debugging purposes.
  void AddParameter(const torch::lazy::Output &output, size_t index,
                    const torch::lazy::Shape &shape,
                    const std::string &name) override;

  // Build the computation capturing all the operations created with the
  // embedded builder (returned by the builder() API).
  torch::lazy::ComputationPtr Build() override;

  virtual torch::lazy::ComputationPtr CreateComputation(MlirModule module_op);

  // Retrieves the lowered operation for an output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value *GetOutputOp(const Output &output);

  // Assigns the given TS operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output &output, torch::jit::Value *op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  torch::jit::Value *GetParameter(BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const;

protected:
  struct Parameter {
    torch::jit::Value *param;
    size_t index = 0;
  };

  size_t AddResult(torch::jit::Value *op);

  // Creates a jit::Function from the current jit::Graph. Input and output
  // type information is patched to include shape.
  std::unique_ptr<torch::jit::Function> generate_jit_fn() const;

  void RegisterMlirDialects();

  // Holds the input/output alias information populated by the SetUpAlias() API.
  InputOutputAliases input_output_aliases_;
  std::shared_ptr<torch::jit::Graph> graph_;
  std::shared_ptr<torch::jit::GraphFunction> function_;
  MlirContext mlir_context_;
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;
  std::unordered_map<int, std::string> parameter_names_;
  std::vector<torch::jit::Value *> root_tuple_;
  OutputMap<torch::jit::Value *> emitted_outputs_;
};

class TORCH_API TorchMlirComputation : public torch::lazy::Computation {
public:
  using InputOutputAliases = TorchMlirLoweringContext::InputOutputAliases;
  using InputOutputAlias = TorchMlirLoweringContext::InputOutputAlias;

  TorchMlirComputation(MlirModule module_op, MlirContext mlir_context,
                       const std::shared_ptr<torch::jit::Graph> &graph,
                       std::unordered_map<int, std::string> parameters_map,
                       InputOutputAliases input_output_aliases);

  int parameters_size() const override;

  const std::vector<torch::lazy::Shape> &parameter_shapes() const override;

  const std::vector<std::string> &parameter_names() const override;

  const std::unordered_map<int, std::string> &parameters_map() const;

  const torch::lazy::Shape &result_shape() const override;

  std::shared_ptr<torch::jit::Graph> graph() const;

  MlirOperation func_op() const;

  MlirModule module_op() const;

  MlirContext mlir_context() const;

  virtual const std::string debug_string() const;

  virtual const std::string to_string() const override;

protected:
  size_t num_parameters_;
  MlirModule module_op_;
  MlirContext mlir_context_;
  std::shared_ptr<torch::jit::Graph> graph_;
  InputOutputAliases input_output_aliases_;
  std::unordered_map<int, std::string> parameters_map_;
  std::vector<std::string> parameter_names_;
  std::vector<Shape> parameter_shapes_;
  Shape result_shape_;
};

} // namespace lazy
} // namespace torch
