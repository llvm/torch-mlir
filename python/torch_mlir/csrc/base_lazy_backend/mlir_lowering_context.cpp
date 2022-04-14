//===- mlir_lowering_context.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_lowering_context.cpp
//===----------------------------------------------------------------------===//

#include <iostream>

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include "../../dialects/torch/importer/jit_ir/csrc/function_importer.h"
#include "../utils/debug.h"
#include "../utils/exception.h"
#include "backend_impl.h"
#include "mlir-c/Registration.h"
#include "mlir_lowering_context.h"
#include "mlir_node.h"
#include "torch-mlir-c/Registration.h"

namespace torch {
namespace lazy {

///////////////////////////////////////////////////////////////////////////////
// TorchMlir Computation
///////////////////////////////////////////////////////////////////////////////

TorchMlirComputation::TorchMlirComputation(
    MlirOperation func_op, MlirContext mlir_context,
    const std::shared_ptr<torch::jit::Graph>& graph)
    : func_op_(std::move(func_op)), mlir_context_(std::move(mlir_context)),
      graph_(graph), num_results_(graph_->outputs().size()) {

  // TODO(henrytu): Save parameter shape information.

  for (torch::jit::Value* input : graph_->inputs()) {
    parameter_names_.push_back(input->debugName());
  }
}

int TorchMlirComputation::parameters_size() const {
  return parameter_names_.size();
}

const std::vector<torch::lazy::Shape>&
TorchMlirComputation::parameter_shapes() const {
  throw std::runtime_error(
      "todo(whc) implement ts computation shapes or change interface");
  return parameter_shapes_;
}

const std::vector<std::string>& TorchMlirComputation::parameter_names() const {
  return parameter_names_;
}

const torch::lazy::Shape& TorchMlirComputation::result_shape() const {
  throw std::runtime_error(
      "todo(whc) implement ts computation shapes or change interface");
  return result_shape_;
}

unsigned TorchMlirComputation::num_results() const { return num_results_; }

MlirOperation TorchMlirComputation::func_op() const { return func_op_; }

std::string TorchMlirComputation::to_string() const {
  // Since we use the C-MLIR API, we need to use a callback to print.
  MlirStringCallback print_callback = [](MlirStringRef part, void* user_data) {
    // user_data is a void ptr to some data structure of our choice -- in this
    // case, the string stream where we'll be accumulating the strings.
    std::stringstream* ss_ptr = static_cast<std::stringstream*>(user_data);
    *ss_ptr << std::string(part.data, part.length);
  };

  std::stringstream ss;
  ss << "JIT Graph: \n"
     << graph_->toString() << "\n\n"
     << "MLIR: \n";
  mlirOperationPrint(func_op_, print_callback, &ss);

  return ss.str();
}

///////////////////////////////////////////////////////////////////////////////
// TorchMlir Lowering Context
///////////////////////////////////////////////////////////////////////////////

TorchMlirLoweringContext::TorchMlirLoweringContext(
    const std::string& name, BackendDevice device)
    : LoweringContext(name, std::forward<BackendDevice>(device)),
      graph_(std::make_shared<torch::jit::Graph>()),
      mlir_context_(mlirContextCreate()) {
  lowering_ = TorchMlirNodeLoweringInterface::Create(this);
  RegisterMlirDialects();
}

TorchMlirLoweringContext::TorchMlirLoweringContext(
    const std::string& name, BackendDevice device,
    c10::ArrayRef<torch::lazy::Node*> post_order, Util::EmissionMap emit_status)
    : LoweringContext(
          name, std::forward<BackendDevice>(device),
          std::forward<c10::ArrayRef<torch::lazy::Node*>>(post_order),
          std::forward<Util::EmissionMap>(emit_status)),
      graph_(std::make_shared<torch::jit::Graph>()),
      mlir_context_(mlirContextCreate()) {
  lowering_ = TorchMlirNodeLoweringInterface::Create(this);
  for (auto node : post_order) {
    bool ok = lowering_->Lower(node);
    CHECK(ok) << "Failed to lower: " << *node;
  }

  RegisterMlirDialects();
}

// Get the shape of the result tuple component, given by index.
torch::lazy::Shape
TorchMlirLoweringContext::GetResultShape(size_t index) const {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

size_t TorchMlirLoweringContext::AddResult(const Output& output) {
  PRINT_FUNCTION();

  return AddResult(GetOutputOp(output));
}

// Associates the given output with the input parameter of the given index and
// shape. Only used for the operator-by-operator execution, mostly for
// debugging purposes.
void TorchMlirLoweringContext::AddParameter(
    const torch::lazy::Output& output, size_t index,
    const torch::lazy::Shape& shape, const std::string& name) {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

// Build the computation capturing all the operations created with the
// embedded builder (returned by the builder() API).
ComputationPtr TorchMlirLoweringContext::Build() {
  PRINT_FUNCTION();

  for (torch::jit::Value* output : root_tuple_) {
    graph_->block()->registerOutput(output);
  }

  // Create jit::Function from jit::Graph.
  c10::QualifiedName name("graph");
  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  // IMPORTANT: We pass in a COPY of the graph into create_function, since it
  //            may get mutated in the process.
  auto jit_fn = cu->create_function(std::move(name), std::move(graph_->copy()));

  // Generate MLIR.
  MlirOperation func_op =
      torch_mlir::importJitFunctionAsFuncOp(mlir_context_, jit_fn);

  // TODO(henrytu): Inject tensor shapes into func_op
  return std::make_shared<TorchMlirComputation>(func_op, mlir_context_, graph_);
}

torch::jit::Value* TorchMlirLoweringContext::GetOutputOp(const Output& output) {
  PRINT_FUNCTION();

  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      bool ok = lowering_->Lower(node);
      TORCH_CHECK(ok, "Failed to lower: ", node->ToString());
    }
    // At this point the output better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    TORCH_CHECK(
        it != emitted_outputs_.end(),
        "No MLIR operation emitted for output: ", output.ToString());
  }
  return it->second;
}

void TorchMlirLoweringContext::AssignOutputOp(
    const Output& output, torch::jit::Value* op) {
  PRINT_FUNCTION();

  auto torch_mlir_node =
      NodeCast<TorchMlirNode>(output.node, output.node->op());
  if (!torch_mlir_node->getPythonStacktrace().empty()) {
    op->node()->s_(
        c10::Symbol::attr("source"), torch_mlir_node->getPythonStacktrace());
  }
  emitted_outputs_[output] = std::move(op);
}

torch::jit::Value* TorchMlirLoweringContext::GetParameter(BackendDataPtr data) {
  PRINT_FUNCTION();

  if (!dynamic_cast<TorchMlirBackendData*>(data.get())) {
    TORCH_CHECK(
        false,
        "Expected TorchMlirBackendData. Got some other BackendData type");
  }
  const auto mlir_data = std::static_pointer_cast<TorchMlirBackendData>(data);

  BackendData::Handle handle = mlir_data->GetHandle();
  auto it = parameters_map_.find(handle);

  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(c10::str("p", parameters_.size()));

    auto info = mlir_data->mlir_info();
    if (info->scalar.has_value()) {
      auto& scalar = info->scalar.value();
      if (scalar.isFloatingPoint()) {
        param->setType(c10::FloatType::get());
      } else if (scalar.isIntegral(true)) {
        param->setType(c10::IntType::get());
      } else {
        TORCH_CHECK(
            false, "Unhandled scalar type: ", c10::toString(scalar.type()));
      }
    }

    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(mlir_data);
  }

  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

std::shared_ptr<torch::jit::Graph> TorchMlirLoweringContext::graph() const {
  return graph_;
}

size_t TorchMlirLoweringContext::AddResult(torch::jit::Value* op) {
  PRINT_FUNCTION();
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

void TorchMlirLoweringContext::RegisterMlirDialects() {
  // https://reviews.llvm.org/D88162
  mlirRegisterAllDialects(mlir_context_);
  torchMlirRegisterAllDialects(mlir_context_);
}

} // namespace lazy
} // namespace torch
