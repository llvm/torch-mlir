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
    std::shared_ptr<torch::jit::GraphFunction> function, c10::Symbol sym,
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
    std::shared_ptr<torch::jit::GraphFunction> function, c10::Symbol sym,
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

class TorchMlirNodeLowering : public TorchMlirNodeLoweringInterface {
public:
  TorchMlirNodeLowering(
      const std::string& name, torch::lazy::TorchMlirLoweringContext* loctx)
      : loctx_(loctx), function_(
                           loctx ? std::make_shared<torch::jit::GraphFunction>(
                                       name, loctx->graph(), nullptr)
                                 : nullptr) {}

  torch::lazy::TorchMlirLoweringContext* loctx() { return loctx_; }

  bool Lower(const torch::lazy::Node* node) override {
    if (auto* torch_mlir_node =
            dynamic_cast<const torch::lazy::TorchMlirNode*>(node)) {
      // First, we call the node lowering function, which exists for newly
      // codegenned or refactored nodes
      TorchMlirOpVector ops = torch_mlir_node->Lower(function_, loctx());
      if (ops.empty()) {
        // Then fall back to legacy lowering code, which should be gradually
        // removed
        ops = LowerNonCodegenOps(node);
      }
      if (ops.empty()) {
        return false;
      }
      CHECK_EQ(node->num_outputs(), ops.size());
      for (size_t i = 0; i < ops.size(); ++i) {
        loctx()->AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
      }
      return true;
    } else {
      TorchMlirOpVector ops = LowerNonCodegenOps(node);
      if (!ops.empty()) {
        CHECK_EQ(node->num_outputs(), ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
          loctx()->AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
        }
        return true;
      }
    }
    throw std::runtime_error(
        "Expected torch::lazy::TorchMlirNode but could not dynamic cast");
  }

  // TODO(whc) this is for legacy/non-codegen Ops, and after moving most ops
  // to codegen we should delete this and put all the lowering logic into Node
  // classes
  TorchMlirOpVector LowerNonCodegenOps(const torch::lazy::Node* node) {

    if (node->op().op == at::aten::as_strided) {
      return LowerAsStrided(torch::lazy::NodeCast<torch::lazy::AsStrided>(
          node, torch::lazy::OpKind(at::aten::as_strided)));
    }
    if (node->op() == *torch::lazy::ltc_as_strided_view_update) {
      return LowerAsStridedViewUpdate(
          torch::lazy::NodeCast<torch::lazy::AsStridedViewUpdate>(
              node, *torch::lazy::ltc_as_strided_view_update));
    }
    if (node->op() == *torch::lazy::ltc_cast) {
      return LowerCast(torch::lazy::NodeCast<torch::lazy::Cast>(
          node, *torch::lazy::ltc_cast));
    }
    if (node->op() == *torch::lazy::ltc_select_view_update) {
      return LowerSelectViewUpdate(
          torch::lazy::NodeCast<torch::lazy::SelectViewUpdate>(
              node, *torch::lazy::ltc_select_view_update));
    }
    if (node->op() == *torch::lazy::ltc_narrow_view_update) {
      return LowerNarrowViewUpdate(
          torch::lazy::NodeCast<torch::lazy::NarrowViewUpdate>(
              node, *torch::lazy::ltc_narrow_view_update));
    }
    if (node->op().op == at::prim::Constant) {
      return LowerScalar(torch::lazy::NodeCast<torch::lazy::Scalar>(
          node, torch::lazy::OpKind(at::prim::Constant)));
    }
    if (node->op().op == at::aten::bernoulli) {
      std::vector<torch::jit::NamedValue> arguments;
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
      return LowerBuiltin(node, arguments);
    }
    if (node->op().op == at::aten::expand) {
      return LowerExpand(torch::lazy::NodeCast<torch::lazy::Expand>(
          node, torch::lazy::OpKind(at::aten::expand)));
    }
    if (node->op().op == at::aten::narrow) {
      return LowerNarrow(torch::lazy::NodeCast<torch::lazy::Narrow>(
          node, torch::lazy::OpKind(at::aten::narrow)));
    }
    if (node->op().op == at::aten::permute) {
      return LowerPermute(torch::lazy::NodeCast<torch::lazy::Permute>(
          node, torch::lazy::OpKind(at::aten::permute)));
    }
    if (node->op().op == at::aten::select) {
      return LowerSelect(torch::lazy::NodeCast<torch::lazy::Select>(
          node, torch::lazy::OpKind(at::aten::select)));
    }
    if (node->op().op == at::aten::squeeze) {
      return LowerSqueeze(torch::lazy::NodeCast<torch::lazy::Squeeze>(
          node, torch::lazy::OpKind(at::aten::squeeze)));
    }
    if (node->op().op == at::aten::unsqueeze) {
      return LowerUnsqueeze(torch::lazy::NodeCast<torch::lazy::Unsqueeze>(
          node, torch::lazy::OpKind(at::aten::unsqueeze)));
    }
    if (node->op().op == at::aten::view) {
      return LowerView(torch::lazy::NodeCast<torch::lazy::View>(
          node, torch::lazy::OpKind(at::aten::view)));
    }
    if (node->op() == *torch::lazy::ltc_device_data) {
      const torch::lazy::DeviceData* device_data_node =
          torch::lazy::NodeCast<torch::lazy::DeviceData>(
              node, *torch::lazy::ltc_device_data);
      auto infoptr = device_data_node->data()->info();
      auto deviceDataInfoPtr =
          (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
      if (GRAPH_DUMP_ENABLED) {
        LOG(ERROR) << "Lowering device data node, tensor id "
                   << deviceDataInfoPtr->tensor_id << std::endl;
      }
      return {loctx()->GetParameter(device_data_node->data())};
    }

    std::vector<torch::jit::NamedValue> arguments;
    for (const torch::lazy::Output& output : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(output));
    }
    return LowerBuiltin(node, arguments);
  }

  TorchMlirOpVector LowerBuiltin(
      const torch::lazy::Node* node,
      const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerTorchMlirBuiltin(
        function_, node->op().op, node->shapes(), arguments, kwarguments);
  }
  TorchMlirOpVector LowerBuiltin(
      c10::Symbol sym, const c10::ArrayRef<Shape> result_shapes,
      const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerTorchMlirBuiltin(
        function_, sym, result_shapes, arguments, kwarguments);
  }
  TorchMlirOpVector LowerBuiltin(
      c10::Symbol sym, const std::vector<c10::TypePtr> types,
      const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerTorchMlirBuiltin(function_, sym, types, arguments, kwarguments);
  }

  TorchMlirOpVector LowerAsStrided(const torch::lazy::AsStrided* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size);
    arguments.emplace_back(node->stride);
    arguments.emplace_back(node->storage_offset);
    TorchMlirOpVector as_strided_out = LowerBuiltin(node, arguments);
    CHECK_EQ(as_strided_out.size(), 1);
    return {GenerateClone(as_strided_out.front())};
  }

  TorchMlirOpVector
  LowerAsStridedViewUpdate(const torch::lazy::AsStridedViewUpdate* node) {
    torch::jit::Value* destination =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const torch::lazy::Output& input_op = node->operand(1);
    const torch::lazy::Shape& input_shape = input_op.shape();
    const auto input_dimensions = input_shape.sizes();
    std::vector<torch::jit::NamedValue> dest_arguments;
    dest_arguments.emplace_back(destination);
    dest_arguments.emplace_back(
        std::vector<int64_t>(input_dimensions.begin(), input_dimensions.end()));
    dest_arguments.emplace_back(node->stride);
    dest_arguments.emplace_back(node->storage_offset);
    TorchMlirOpVector as_strided_out =
        LowerBuiltin(at::aten::as_strided, node->shapes(), dest_arguments);
    CHECK_EQ(as_strided_out.size(), 1);
    torch::jit::Value* as_strided = as_strided_out.front();
    GenerateCopy(as_strided, loctx()->GetOutputOp(input_op));
    return {destination};
  }

  TorchMlirOpVector LowerCast(const torch::lazy::Cast* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dtype);
    return LowerBuiltin(at::aten::to, node->shapes(), arguments);
  }

  TorchMlirOpVector LowerExpand(const torch::lazy::Expand* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size);
    auto expand_out = LowerBuiltin(node, arguments);
    if (node->is_scalar_expand) {
      // The aten::expand operations sets all strides to 0 when the original
      // of rank 0. This leads to false positives when checking for internal
      // memory overlap, because at::has_internal_overlap returns
      // MemOverlap::YES when a stride is set to 0.
      CHECK_EQ(expand_out.size(), 1);
      return {GenerateClone(expand_out.front())};
    }
    return expand_out;
  }

  TorchMlirOpVector LowerNarrow(const torch::lazy::Narrow* node) {
    const torch::lazy::Output& input = node->operand(0);
    torch::jit::Value* base = loctx()->GetOutputOp(input);
    const auto& base_indices = node->base_indices;
    const auto& sizes = node->sizes;
    const torch::lazy::Shape& input_shape = input.shape();
    CHECK_EQ(sizes.size(), base_indices.size());
    CHECK_EQ(input_shape.dim(), base_indices.size());
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      int64_t start = base_indices[dim];
      base = GenerateSlice(
          /*base=*/base, /*dim=*/dim, /*start=*/start,
          /*end=*/start + sizes[dim], /*step=*/1);
    }
    return {base};
  }

  TorchMlirOpVector LowerPermute(const torch::lazy::Permute* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dims);
    return LowerBuiltin(node, arguments);
  }

  TorchMlirOpVector LowerScalar(const torch::lazy::Scalar* node) {
    const at::Scalar& value = node->value;
    const torch::lazy::Shape& shape = node->shape();
    auto options =
        at::TensorOptions()
            .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
            .dtype(shape.scalar_type());
    return {
        loctx()->graph()->insertConstant(at::scalar_tensor(value, options))};
  }

  TorchMlirOpVector LowerSelect(const torch::lazy::Select* node) {
    int64_t step = torch::lazy::GetStride(node->start, node->end, node->stride);
    torch::jit::Value* base = loctx()->GetOutputOp(node->operand(0));
    return {GenerateSlice(
        /*base=*/base, /*dim=*/node->dim,
        /*start=*/node->start, /*end=*/node->end,
        /*step=*/step)};
  }

  TorchMlirOpVector LowerSqueeze(const torch::lazy::Squeeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    if (node->dim != -1) {
      arguments.push_back(node->dim);
    }
    return LowerBuiltin(node, arguments);
  }

  TorchMlirOpVector
  LowerSelectViewUpdate(const torch::lazy::SelectViewUpdate* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    int64_t step = torch::lazy::GetStride(node->start, node->end, node->stride);
    torch::jit::Value* selected = GenerateSlice(
        /*base=*/dest, /*dim=*/node->dim, /*start=*/node->start,
        /*end=*/node->end, /*step=*/step);
    GenerateCopy(selected, loctx()->GetOutputOp(node->operand(1)));
    return {dest};
  }

  TorchMlirOpVector
  LowerNarrowViewUpdate(const torch::lazy::NarrowViewUpdate* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const auto& base_indices = node->base_indices;
    const torch::lazy::Output& source_argument = node->operand(1);
    const torch::lazy::Shape& source_shape = source_argument.shape();
    CHECK_EQ(source_shape.dim(), base_indices.size());
    torch::jit::Value* base = dest;
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      int64_t start = base_indices[dim];
      base = GenerateSlice(
          /*base=*/base, /*dim=*/dim, /*start=*/start,
          /*end=*/start + source_shape.size(dim),
          /*step=*/1);
    }
    GenerateCopy(base, loctx()->GetOutputOp(source_argument));
    return {dest};
  }

  TorchMlirOpVector LowerUnsqueeze(const torch::lazy::Unsqueeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dim);
    return LowerBuiltin(node, arguments);
  }

  TorchMlirOpVector LowerView(const torch::lazy::View* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->output_size);
    return LowerBuiltin(at::aten::reshape, node->shapes(), arguments);
  }

  torch::jit::Value* GenerateClone(torch::jit::Value* val) {
    std::vector<torch::jit::NamedValue> clone_arguments;
    clone_arguments.emplace_back(val);

    // Type of cloned value should be identical to the original one.
    TorchMlirOpVector cloned =
        LowerBuiltin(at::aten::clone, {val->type()}, clone_arguments);
    CHECK_EQ(cloned.size(), 1);
    return cloned.front();
  }

  void GenerateCopy(torch::jit::Value* destination, torch::jit::Value* source) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(destination);
    arguments.emplace_back(source);
    LowerBuiltin(
        at::aten::copy_,
        c10::ArrayRef<Shape>(compute_shape_copy(source->type())), arguments);
  }

  torch::jit::Value* GenerateSlice(
      torch::jit::Value* base, int64_t dim, int64_t start, int64_t end,
      int64_t step) {
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
        arguments);
    CHECK_EQ(selected.size(), 1);
    return selected.front();
  }
  torch::lazy::TorchMlirLoweringContext* loctx_;
  std::shared_ptr<torch::jit::GraphFunction> function_;
};

std::unique_ptr<TorchMlirNodeLoweringInterface>
TorchMlirNodeLoweringInterface::Create(torch::lazy::LoweringContext* loctx) {
  return std::make_unique<TorchMlirNodeLowering>(
      "TorchMlirNodeLowering",
      static_cast<torch::lazy::TorchMlirLoweringContext*>(loctx));
}
} // namespace lazy
} // namespace torch
