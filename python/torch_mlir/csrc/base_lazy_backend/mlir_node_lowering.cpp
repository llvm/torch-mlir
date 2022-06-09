//===- mlir_node_lowering.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
// This file is adapted from pytorch/pytorch
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/lazy/ts_backend/ts_node_lowering.cpp
//===----------------------------------------------------------------------===//

#include "mlir_node_lowering.h"
#include "generated/LazyNonNativeIr.h"
#include "mlir_lowering_context.h"
#include "mlir_node.h"
#include "ops/device_data.h"

#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/permutation_util.h>

namespace torch {
namespace lazy {

TorchMlirOpVector LowerTorchMlirBuiltin(
    TorchMlirFunction function, c10::Symbol sym,
    const std::vector<c10::TypePtr> tensor_types,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
  auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
  CHECK(sv);

  TorchMlirOpVector results;
  if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
    // Op returns multiple values.
    const auto tuple_call_result = sv->asTuple({}, *function);
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      results.push_back(tuple_component_sv->getValue());
    }
  } else {
    // Op returns single value.
    results.push_back(sv->getValue());
  }

  // Insert known tensor type information.
  unsigned tensor_type_idx = 0;
  for (jit::Value* value : results) {
    if (value->type()->kind() == c10::TypeKind::TensorType) {
      TORCH_CHECK(
          tensor_type_idx < tensor_types.size(), function->graph()->toString(),
          "\nTensor corresponding to JIT SSA value %", value->debugName(),
          " corresponds to result #", tensor_type_idx, ", but we only have ",
          tensor_types.size(), " known types!");

      value->setType(tensor_types[tensor_type_idx++]);
    }
  }

  // Ensure that we use up all the known tensor type information available.
  TORCH_CHECK(
      tensor_type_idx == tensor_types.size(), tensor_type_idx,
      " known types were injected into jit::Value, but ", tensor_types.size(),
      " were provided from lazy::Node!");

  return results;
}

TorchMlirOpVector LowerTorchMlirBuiltin(
    TorchMlirFunction function, c10::Symbol sym,
    const c10::ArrayRef<Shape> result_shapes,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  std::vector<c10::TypePtr> tensor_types;

  // Generate types with fixed tensor shape information.
  for (const Shape& shape : result_shapes) {
    tensor_types.push_back(torch::jit::TensorType::create(
        /*scalar_type=*/shape.scalar_type(),
        /*device=*/c10::nullopt,
        /*sizes=*/c10::VaryingShape<int64_t>(shape.sizes()),
        /*strides=*/c10::VaryingShape<int64_t>(),
        /*requires_grad=*/c10::nullopt));
  }

  return LowerTorchMlirBuiltin(
      function, sym, tensor_types, arguments, kwarguments);
}

TorchMlirOpVector LowerBuiltin(
    const torch::lazy::Node* node, TorchMlirFunction function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTorchMlirBuiltin(
      function, node->op().op, node->shapes(), arguments, kwarguments);
}
TorchMlirOpVector LowerBuiltin(
    c10::Symbol sym, const c10::ArrayRef<Shape> result_shapes,
    TorchMlirFunction function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTorchMlirBuiltin(
      function, sym, result_shapes, arguments, kwarguments);
}
TorchMlirOpVector LowerBuiltin(
    c10::Symbol sym, const std::vector<c10::TypePtr> types,
    TorchMlirFunction function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTorchMlirBuiltin(function, sym, types, arguments, kwarguments);
}

c10::TensorType& cast_tensor_type(c10::TypePtr value_type) {
  auto tensor_type = value_type->cast<c10::TensorType>();
  TORCH_CHECK(tensor_type, "Unable to cast Value type to TensorType!");

  return *tensor_type.get();
}

c10::optional<std::vector<int64_t>>
get_tensor_type_shape(c10::TensorType& tensor_type) {
  auto& symbolic_shape = tensor_type.symbolic_sizes();
  if (!symbolic_shape.rank()) {
    return c10::nullopt;
  }

  // Get current tensor shape.
  std::vector<int64_t> dims;
  dims.resize(*symbolic_shape.rank());
  for (size_t i = 0; i < dims.size(); ++i) {
    auto shape_symbol = symbolic_shape[i];
    dims[i] = shape_symbol.is_static() ? shape_symbol.static_size() : -1;
  }

  return dims;
}

std::vector<torch::lazy::Shape> compute_shape_copy(c10::TypePtr value_type) {
  c10::TensorType& tensor_type = cast_tensor_type(value_type);

  auto maybe_dims = get_tensor_type_shape(tensor_type);
  TORCH_CHECK(maybe_dims.has_value(), "Cannot copy unranked tensor!");

  auto scalar_type = tensor_type.scalarType();
  TORCH_CHECK(
      scalar_type.has_value(), "Unable to copy due to lack of scalar type!");
  return {Shape(scalar_type.value(), maybe_dims.value())};
}

std::vector<torch::lazy::Shape> compute_shape_slice(
    c10::TypePtr value_type, int64_t dim, int64_t start, int64_t end,
    int64_t step) {
  c10::TensorType& tensor_type = cast_tensor_type(value_type);

  auto maybe_dims = get_tensor_type_shape(tensor_type);
  TORCH_CHECK(maybe_dims.has_value(), "Cannot slice unranked tensor!");

  std::vector<int64_t> dims = maybe_dims.value();
  int64_t num_dims = dims[dim];

  // Index may be negative, so we must normalize it.
  auto normalize_index = [](int64_t index, unsigned num_dims) {
    return index < 0 ? (int64_t)num_dims + index : index;
  };
  start = normalize_index(start, num_dims);
  end = normalize_index(end, num_dims);

  if (start >= end || start >= num_dims || end <= 0) {
    // Slice is out of bounds, nothing in range.
    dims[dim] = 0;
  } else {
    // Clamp upper and lower bound to valid indices.
    start = std::max((int64_t)0, start);
    end = std::min(num_dims, end);

    // Final size is determined by step and interval size.
    dims[dim] = std::ceil((double)(end - start) / (double)step);
  }

  auto scalar_type = tensor_type.scalarType();
  TORCH_CHECK(
      scalar_type.has_value(), "Unable to slice due to lack of scalar type!");
  return {Shape(scalar_type.value(), dims)};
}

torch::jit::Value*
GenerateClone(torch::jit::Value* val, TorchMlirFunction function) {
  std::vector<torch::jit::NamedValue> clone_arguments;
  clone_arguments.emplace_back(val);

  // Type of cloned value should be identical to the original one.
  TorchMlirOpVector cloned =
      LowerBuiltin(at::aten::clone, {val->type()}, function, clone_arguments);
  CHECK_EQ(cloned.size(), 1);
  return cloned.front();
}


void GenerateCopy(torch::jit::Value* destination, torch::jit::Value* source, TorchMlirFunction function) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(destination);
    arguments.emplace_back(source);
    LowerBuiltin(
        at::aten::copy_,
        c10::ArrayRef<Shape>(compute_shape_copy(source->type())), function, arguments);
}


torch::jit::Value* GenerateSlice(
    torch::jit::Value* base, int64_t dim, int64_t start, int64_t end,
    int64_t step, TorchMlirFunction function) {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(base);
  arguments.emplace_back(dim);
  arguments.emplace_back(start);
  arguments.emplace_back(end);
  arguments.emplace_back(step);

  TorchMlirOpVector selected = LowerBuiltin(
      at::aten::slice,
      c10::ArrayRef<Shape>(
          compute_shape_slice(base->type(), dim, start, end, step)),
      function,
      arguments);
  CHECK_EQ(selected.size(), 1);
  return selected.front();
}

// Node Lowerings

// Default Node Lowering
TorchMlirOpVector TorchMlirNode::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  for (const torch::lazy::Output& output : operands()) {
    arguments.emplace_back(loctx->GetOutputOp(output));
  }
  return LowerBuiltin(this, function, arguments);
}

// TorchMlir specific nodes

// Non-native nodes

TorchMlirOpVector
Cast::Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dtype);
  return LowerBuiltin(at::aten::to, shapes(), function, arguments);
}

TorchMlirOpVector DeviceData::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  auto infoptr = data_->info();
  auto deviceDataInfoPtr =
      (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
  if (GRAPH_DUMP_ENABLED) {
    LOG(ERROR) << "Lowering device data node, tensor id "
               << deviceDataInfoPtr->tensor_id << std::endl;
  }
  return {loctx->GetParameter(data_)};
}

TorchMlirOpVector Expand::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(size);
  auto expand_out = LowerBuiltin(this, function, arguments);
  if (is_scalar_expand) {
    // The aten::expand operations sets all strides to 0 when the original is
    // of rank 0. This leads to false positives when checking for internal
    // memory overlap, because at::has_internal_overlap returns
    // MemOverlap::YES when a stride is set to 0.
    CHECK_EQ(expand_out.size(), 1);
    return {GenerateClone(expand_out.front(), function)};
  }
  return expand_out;
}

TorchMlirOpVector Scalar::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  auto options =
      at::TensorOptions()
          .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
          .dtype(shape().scalar_type());
  return {loctx->graph()->insertConstant(at::scalar_tensor(value, options))};
}

// View Ops

TorchMlirOpVector AsStrided::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {

  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(size);
  arguments.emplace_back(stride);
  arguments.emplace_back(storage_offset);
  TorchMlirOpVector as_strided_out = LowerBuiltin(this, function, arguments);
  CHECK_EQ(as_strided_out.size(), 1);
  return {GenerateClone(as_strided_out.front(), function)};
}

TorchMlirOpVector AsStridedViewUpdate::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {

  torch::jit::Value* destination =
      GenerateClone(loctx->GetOutputOp(operand(0)), function);
  const torch::lazy::Output& input_op = operand(1);
  const torch::lazy::Shape& input_shape = input_op.shape();
  const auto input_dimensions = input_shape.sizes();
  std::vector<torch::jit::NamedValue> dest_arguments;
  dest_arguments.emplace_back(destination);
  dest_arguments.emplace_back(
      std::vector<int64_t>(input_dimensions.begin(), input_dimensions.end()));
  dest_arguments.emplace_back(stride);
  dest_arguments.emplace_back(storage_offset);
  TorchMlirOpVector as_strided_out =
      LowerBuiltin(at::aten::as_strided, shapes(), function, dest_arguments);
  CHECK_EQ(as_strided_out.size(), 1);
  torch::jit::Value* as_strided = as_strided_out.front();
  GenerateCopy(as_strided, loctx->GetOutputOp(input_op), function);
  return {destination};
}

TorchMlirOpVector Diagonal::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {

  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(offset);
  arguments.emplace_back(dim1);
  arguments.emplace_back(dim2);
  return LowerBuiltin(this, function, arguments);
}

TorchMlirOpVector DiagonalViewUpdate::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  // Since we promise the backends that we never generate any aliased
  // inplace update IR, therefore we clone the target first and then
  // update the clone inplace instead. Since the clone is transient,
  // it will never be aliased, and therefore it's safe.
  torch::jit::Value* destination =
      GenerateClone(loctx->GetOutputOp(operand(0)), function);

  // Replay the diagonal.
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(destination);
  arguments.emplace_back(offset);
  arguments.emplace_back(dim1);
  arguments.emplace_back(dim2);
  auto diag = LowerBuiltin(at::aten::diagonal, shapes(), function, arguments);

  // Update the replayed diagonal view with the input.
  GenerateCopy(diag.front(), loctx->GetOutputOp(operand(1)), function);

  // Destination's diag view should be updated.
  return {destination};
}

TorchMlirOpVector Narrow::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  const torch::lazy::Output& input = operand(0);
  torch::jit::Value* base = loctx->GetOutputOp(input);
  const torch::lazy::Shape& input_shape = input.shape();
  CHECK_EQ(sizes.size(), base_indices.size());
  CHECK_EQ(input_shape.dim(), base_indices.size());
  for (size_t dim = 0; dim < base_indices.size(); ++dim) {
    int64_t start = base_indices[dim];
    base = GenerateSlice(
        /*base=*/base, /*dim=*/dim, /*start=*/start,
        /*end=*/start + sizes[dim], /*step=*/1,
        /*function=*/function);
  }
  return {base};
}

TorchMlirOpVector NarrowViewUpdate::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  torch::jit::Value* dest =
      GenerateClone(loctx->GetOutputOp(operand(0)), function);
  const torch::lazy::Output& source_argument = operand(1);
  const torch::lazy::Shape& source_shape = source_argument.shape();
  CHECK_EQ(source_shape.dim(), base_indices.size());
  torch::jit::Value* base = dest;
  for (size_t dim = 0; dim < base_indices.size(); ++dim) {
    int64_t start = base_indices[dim];
    base = GenerateSlice(
        /*base=*/base, /*dim=*/dim, /*start=*/start,
        /*end=*/start + source_shape.size(dim), /*step=*/1,
        /*function=*/function);
  }
  GenerateCopy(base, loctx->GetOutputOp(source_argument), function);
  return {dest};
}

TorchMlirOpVector Permute::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dims);
  return LowerBuiltin(this, function, arguments);
}

TorchMlirOpVector Resize::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {

  std::vector<torch::jit::NamedValue> arguments;
  for (const torch::lazy::Output& output : operands()) {
    arguments.emplace_back(loctx->GetOutputOp(output));
  }
  return LowerBuiltin(this, function, arguments);
}

TorchMlirOpVector Select::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  int64_t step = torch::lazy::GetStride(start, end, stride);
  torch::jit::Value* base = loctx->GetOutputOp(operand(0));
  return {GenerateSlice(
      /*base=*/base, /*dim=*/dim,
      /*start=*/start, /*end=*/end,
      /*step=*/step, /*function=*/function)};
}

TorchMlirOpVector SelectViewUpdate::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  torch::jit::Value* dest =
      GenerateClone(loctx->GetOutputOp(operand(0)), function);
  int64_t step = torch::lazy::GetStride(start, end, stride);
  torch::jit::Value* selected = GenerateSlice(
      /*base=*/dest, /*dim=*/dim, /*start=*/start,
      /*end=*/end, /*step=*/step, /*function=*/function);
  GenerateCopy(selected, loctx->GetOutputOp(operand(1)), function);
  return {dest};
}

TorchMlirOpVector Squeeze::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  if (dim != -1) {
    arguments.emplace_back(dim);
  }
  return LowerBuiltin(this, function, arguments);
}

TorchMlirOpVector Unsqueeze::Lower(
    TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dim);
  return LowerBuiltin(this, function, arguments);
}

TorchMlirOpVector
View::Lower(TorchMlirFunction function, TorchMlirLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(output_size);
  return LowerBuiltin(at::aten::reshape, shapes(), function, arguments);
}

} // namespace lazy
} // namespace torch
