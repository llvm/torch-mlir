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

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir-c/Transforms.h"
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/passes/refine_tuple_types.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include "backend_impl.h"
#include "jit_ir_importer/function_importer.h"
#include "mlir_lowering_context.h"
#include "mlir_node.h"
#include "utils/debug.h"
#include "utils/exception.h"
#include "utils/jit_utils.h"
#include "utils/string_utils.h"
#include "utils/sys_utils.h"

namespace torch {
namespace lazy {

///////////////////////////////////////////////////////////////////////////////
// TorchMlir Lowering Context
///////////////////////////////////////////////////////////////////////////////

TorchMlirLoweringContext::TorchMlirLoweringContext(const std::string &name,
                                                   BackendDevice device)
    : LoweringContext(name, std::forward<BackendDevice>(device)),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)),
      mlir_context_(mlirContextCreate()) {
  RegisterMlirDialects();
}

TorchMlirLoweringContext::TorchMlirLoweringContext(
    const std::string &name, BackendDevice device,
    c10::ArrayRef<const torch::lazy::Node *> post_order,
    Util::EmissionMap emit_status)
    : LoweringContext(
          name, std::forward<BackendDevice>(device),
          std::forward<c10::ArrayRef<const torch::lazy::Node *>>(post_order),
          std::forward<Util::EmissionMap>(emit_status)),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)),
      mlir_context_(mlirContextCreate()) {
  RegisterMlirDialects();

  for (auto node : post_order) {
    Lower(node);
  }
}

void TorchMlirLoweringContext::Lower(const Node *node) {
  if (auto *torch_mlir_node =
          dynamic_cast<const torch::lazy::TorchMlirNode *>(node)) {
    TorchMlirOpVector ops = torch_mlir_node->Lower(function_, this);
    CHECK(!ops.empty()) << "Failed to lower: " << *node;
    TORCH_CHECK_EQ(node->num_outputs(), ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
    }
  } else {
    throw std::runtime_error(
        "Expected torch::lazy::TorchMlirNode but could not dynamic cast");
  }
}

void TorchMlirLoweringContext::SetUpAlias(
    const std::vector<int64_t> &output_index, int64_t param_number,
    const std::vector<int64_t> &param_index, bool must_alias) {
  input_output_aliases_.push_back(
      {output_index, param_number, param_index, must_alias});
}

bool TorchMlirLoweringContext::CheckResultShape(
    const BackendDataPtr &parameter_data, size_t result_idx) {
  TORCH_CHECK(result_idx < root_tuple_.size(),
              "Tried getting result shape at index ", result_idx,
              " which is out of bounds!");

  torch::jit::Value *output = root_tuple_[result_idx];

  if (c10::TensorTypePtr tensor_type =
          output->type()->cast<c10::TensorType>()) {
    auto scalar_type = tensor_type->scalarType();
    auto sizes = tensor_type->sizes().concrete_sizes();

    // Not guaranteed to have concrete size, so we need to check it exists.
    if (scalar_type && sizes) {
      return Shape(parameter_data->shape()) ==
             Shape(scalar_type.value(), c10::ArrayRef<int64_t>(sizes.value()));
    }
  }

  return false;
}

size_t TorchMlirLoweringContext::AddResult(const Output &output) {
  PRINT_FUNCTION();

  return AddResult(GetOutputOp(output));
}

// Associates the given output with the input parameter of the given index and
// shape. Only used for the operator-by-operator execution, mostly for
// debugging purposes.
void TorchMlirLoweringContext::AddParameter(const torch::lazy::Output &output,
                                            size_t index,
                                            const torch::lazy::Shape &shape,
                                            const std::string &name) {
  UNIMPLEMENTED_FUNCTION_ERROR();
}

// Build the computation capturing all the operations created with the
// embedded builder (returned by the builder() API).
ComputationPtr TorchMlirLoweringContext::Build() {
  PRINT_FUNCTION();

  // Since we mutated the types of some nodes to insert shape information, we
  // must perform this pass to ensure tuples have up to date output types.
  torch::jit::RefineTupleTypes(graph_);

  // Insert return values into graph.
  for (torch::jit::Value *output : root_tuple_) {
    graph_->block()->registerOutput(output);
  }

  // During operations lowering JIT may insert ScalarImplicit ops which output
  // type !torch.number doesn't represent any existing MLIR type and should be
  // refined either to Torch::IntType or Torch::FloatType.
  torch::jit::ConvertScalarImplicit(graph_);

  // Generate MLIR.
  MlirOperation func_op = torch_mlir::importJitFunctionAsFuncOp(
      /*context=*/mlir_context_,
      /*function=*/generate_jit_fn().get(),
      /*getArgAttribute=*/[](int) -> MlirAttribute { return {nullptr}; },
      /*importOptions=*/{/*assumeTensorsHaveValueSemantics=*/true});

  // Convert MlirOperation to MlirModule.
  MlirLocation loc = mlirLocationUnknownGet(mlir_context_);
  MlirModule module_op = mlirModuleCreateEmpty(loc);
  MlirBlock block = mlirModuleGetBody(module_op);
  mlirBlockAppendOwnedOperation(block, func_op);

  // Apply passes to verify generated MLIR.
  auto pass_manager = mlirPassManagerCreate(mlir_context_);
  mlirPassManagerAddOwnedPass(
      pass_manager, mlirCreateVerifyBackendContractNoDecompositions());

  MlirLogicalResult result =
      mlirPassManagerRunOnOp(pass_manager, mlirModuleGetOperation(module_op));

  if (mlirLogicalResultIsFailure(result)) {
    throw std::runtime_error("MLIR verification has failed.");
  }

  return CreateComputation(module_op);
}

ComputationPtr
TorchMlirLoweringContext::CreateComputation(MlirModule module_op) {
  return std::make_shared<TorchMlirComputation>(module_op, mlir_context_,
                                                graph_, parameter_names_,
                                                input_output_aliases_);
}

torch::jit::Value *TorchMlirLoweringContext::GetOutputOp(const Output &output) {
  PRINT_FUNCTION();

  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      Lower(node);
    }
    // At this point the output better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    TORCH_CHECK(it != emitted_outputs_.end(),
                "No MLIR operation emitted for output: ", output.ToString());
  }
  return it->second;
}

void TorchMlirLoweringContext::AssignOutputOp(const Output &output,
                                              torch::jit::Value *op) {
  PRINT_FUNCTION();

  auto torch_mlir_node =
      NodeCast<TorchMlirNode>(output.node, output.node->op());

  std::vector<std::string> source_files, functions;
  std::vector<int64_t> line_numbers;
  const auto &metadata = torch_mlir_node->metadata();
  const auto &frames = metadata.frame_info;
  if (!frames.empty()) {
    static std::vector<std::string> g_roots =
        string_split(sys_util::GetEnvString("LTC_IR_DEBUG_ROOT_PATH", ""), ":");

    std::for_each(frames.rbegin(), frames.rend(),
                  [&](const torch::lazy::SourceLocation &location) {
                    functions.push_back(location.function);
                    line_numbers.push_back(location.line);

                    std::string file_name = location.file;
                    for (const std::string &root : g_roots) {
                      if (startswith(file_name, root)) {
                        // location.file starts with root, strip it off
                        file_name = file_name.substr(root.size());
                        break;
                      }
                    }
                    source_files.push_back(file_name);
                  });

    if (!source_files.empty()) {
      op->node()->ss_(c10::Symbol::attr("source_files"), source_files);
      op->node()->ss_(c10::Symbol::attr("functions"), functions);
      op->node()->is_(c10::Symbol::attr("line_numbers"), line_numbers);
    }
  }
  auto scope = ::c10::Symbol::scope(metadata.scope);
  op->node()->setScope(c10::make_intrusive<torch::jit::Scope>()->push(scope));

  emitted_outputs_[output] = std::move(op);
}

torch::jit::Value *TorchMlirLoweringContext::GetParameter(BackendDataPtr data) {
  PRINT_FUNCTION();

  if (!dynamic_cast<TorchMlirBackendData *>(data.get())) {
    TORCH_CHECK(
        false,
        "Expected TorchMlirBackendData. Got some other BackendData type");
  }
  const auto mlir_data = std::static_pointer_cast<TorchMlirBackendData>(data);

  BackendData::Handle handle = mlir_data->GetHandle();
  auto it = parameters_map_.find(handle);

  if (it == parameters_map_.end()) {
    torch::jit::Value *param =
        graph_->addInput(c10::str("p", parameters_.size()));

    auto *info =
        dynamic_cast<TorchMlirBackendData::Info *>(mlir_data->mlir_info());
    TORCH_CHECK(info, "Expected TorchMlirBackendData::Info");
    if (info->scalar.has_value()) {
      auto &scalar = info->scalar.value();
      if (scalar.isFloatingPoint()) {
        param->setType(c10::FloatType::get());
      } else if (scalar.isIntegral(true)) {
        param->setType(c10::IntType::get());
      } else {
        TORCH_CHECK(false,
                    "Unhandled scalar type: ", c10::toString(scalar.type()));
      }
    } else {
      // Save parameter shape information.
      param->setType(torch::jit::TensorType::create(
          /*scalar_type=*/data->shape().scalar_type(),
          /*device=*/c10::nullopt,
          /*sizes=*/c10::VaryingShape<int64_t>(data->shape().sizes()),
          /*strides=*/c10::VaryingShape<int64_t>(),
          /*requires_grad=*/c10::nullopt));

      if (info->name != "" && !startswith(info->name, "input")) {
        parameter_names_[parameters_.size()] = info->name;
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

size_t TorchMlirLoweringContext::AddResult(torch::jit::Value *op) {
  PRINT_FUNCTION();
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

// Sync vector of c10::Argument with type specified from parallel list of
// jit::Value. There must be a 1:1 map between elements of args and values.
std::vector<c10::Argument>
sync_argument_types(const std::vector<c10::Argument> &args,
                    c10::ArrayRef<torch::jit::Value *> values) {
  TORCH_CHECK(
      args.size() == values.size(),
      "Expected 1:1 mapping between list of c10::Argument and jit::Value! Got ",
      args.size(), ":", values.size(), " instead!");

  std::vector<c10::Argument> updated_args;
  for (unsigned i = 0; i < args.size(); i++) {
    updated_args.push_back(args[i].cloneWithType(values[i]->type()));
  }

  return updated_args;
}

std::unique_ptr<torch::jit::Function>
TorchMlirLoweringContext::generate_jit_fn() const {
  // IMPORTANT: We pass in a COPY of the graph into create_function, since it
  //            may get mutated in the process.
  auto fn = std::make_unique<torch::jit::GraphFunction>(
      c10::QualifiedName("graph"), graph_->copy(), nullptr);

  c10::FunctionSchema schema = fn->getSchema();

  // When constructing the default schema of a jit::GraphFunction, input and
  // output shapes are stripped (via call to unshapedType(...)); however,
  // since we want to have shape information in our MLIR, we'll add it back.
  std::vector<c10::Argument> arguments =
      sync_argument_types(schema.arguments(), graph_->inputs());
  std::vector<c10::Argument> returns =
      sync_argument_types(schema.returns(), graph_->outputs());

  fn->setSchema(schema.cloneWithArguments(arguments).cloneWithReturns(returns));

  return fn;
}

void TorchMlirLoweringContext::RegisterMlirDialects() {
  // https://reviews.llvm.org/D88162
  torchMlirRegisterAllDialects(mlir_context_);
}

///////////////////////////////////////////////////////////////////////////////
// TorchMlir Computation
///////////////////////////////////////////////////////////////////////////////

TorchMlirComputation::TorchMlirComputation(
    MlirModule module_op, MlirContext mlir_context,
    const std::shared_ptr<torch::jit::Graph> &graph,
    std::unordered_map<int, std::string> parameters_map,
    InputOutputAliases input_output_aliases)
    : module_op_(std::move(module_op)), mlir_context_(std::move(mlir_context)),
      graph_(graph), input_output_aliases_(input_output_aliases),
      parameters_map_(parameters_map) {

  num_parameters_ = graph_->inputs().size();

  parameter_names_.reserve(parameters_map_.size());
  for (auto kv : parameters_map_) {
    parameter_names_.emplace_back(kv.second);
  }
}

int TorchMlirComputation::parameters_size() const { return num_parameters_; }

const std::vector<torch::lazy::Shape> &
TorchMlirComputation::parameter_shapes() const {
  throw std::runtime_error(
      "todo(whc) implement ts computation shapes or change interface");
  return parameter_shapes_;
}

const std::vector<std::string> &TorchMlirComputation::parameter_names() const {
  return parameter_names_;
}

const std::unordered_map<int, std::string> &
TorchMlirComputation::parameters_map() const {
  return parameters_map_;
}

const torch::lazy::Shape &TorchMlirComputation::result_shape() const {
  throw std::runtime_error(
      "todo(whc) implement ts computation shapes or change interface");
  return result_shape_;
}

std::shared_ptr<torch::jit::Graph> TorchMlirComputation::graph() const {
  return graph_;
}

MlirOperation TorchMlirComputation::func_op() const {
  MlirBlock block = mlirModuleGetBody(module_op_);
  return mlirBlockGetFirstOperation(block);
}

MlirModule TorchMlirComputation::module_op() const { return module_op_; }

MlirContext TorchMlirComputation::mlir_context() const { return mlir_context_; }

const std::string TorchMlirComputation::debug_string() const {
  std::stringstream ss;

  // JIT Graph
  ss << "JIT Graph: \n" << graph_->toString() << "\n\n";

  // MLIR
  ss << "MLIR: \n" << to_string() << "\n";

  // Parameter names
  ss << "Parameter names:\n";
  for (auto &p : parameter_names_) {
    ss << "    " << p << "\n";
  }
  ss << "\n";

  // Input/Output Mapping
  ss << "Input/Output Alias Mapping: \n";
  for (InputOutputAlias input_output_alias : input_output_aliases_) {
    ss << "Output: " << input_output_alias.output_index
       << " -> Input param: " << input_output_alias.param_number << "\n";
  }
  ss << "\n";

  // Mark Step
  ss << "In Mark Step: " << (in_mark_step ? "true" : "false") << "\n";

  return ss.str();
}

const std::string TorchMlirComputation::to_string() const {
  // Since we use the C-MLIR API, we need to use a callback to print.
  MlirStringCallback print_callback = [](MlirStringRef part, void *user_data) {
    // user_data is a void ptr to some data structure of our choice -- in this
    // case, the string stream where we'll be accumulating the strings.
    std::stringstream *ss_ptr = static_cast<std::stringstream *>(user_data);
    *ss_ptr << std::string(part.data, part.length);
  };
  std::stringstream ss;

  // Setup flags for MLIR serialization.
  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  mlirOpPrintingFlagsEnableDebugInfo(flags, FLAGS_torch_lazy_ir_debug, false);
  mlirOperationPrintWithFlags(mlirModuleGetOperation(module_op_), flags,
                              print_callback, &ss);
  return ss.str();
}

} // namespace lazy
} // namespace torch
