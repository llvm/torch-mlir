//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "OnnxImporter.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

#include <cstdio>
#include <functional>

using namespace torch_mlir_onnx;

namespace {

std::string SanitizeNameAsIdentifier(std::string_view in) {
  std::string out;
  if (!in.empty() && !std::isalnum(in.front())) {
    out.append("_");
  }
  out.append(in);
  for (char &c : out) {
    if (c == ':' || c == '/')
      c = '_';
  }
  return out;
}

template <typename T>
void AppendDelimittedStrings(std::string &into, T &container) {
  bool first = true;
  for (auto &item : container) {
    if (first) {
      first = false;
    } else {
      into.append(", ");
    }
    into.append(item);
  }
}

inline MlirStringRef toMlirStringRef(const std::string_view &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

inline MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

inline MlirStringRef toMlirStringRef(const char *s) {
  return mlirStringRefCreate(s, std::strlen(s));
}

inline MlirNamedAttribute toMlirNamedAttribute(const char *s,
                                               MlirAttribute attr) {
  MlirContext context = mlirAttributeGetContext(attr);
  MlirIdentifier ident = mlirIdentifierGet(context, toMlirStringRef(s));
  return mlirNamedAttributeGet(ident, attr);
}

std::string getMlirAsm(MlirType t) {
  std::string result;
  mlirTypePrint(
      t,
      +[](MlirStringRef sr, void *userData) {
        std::string *s = static_cast<std::string *>(userData);
        s->append(sr.data, sr.length);
      },
      static_cast<void *>(&result));
  return result;
}

// C++ helpers to create operations.
void addToMlirOperationState(MlirOperationState &state,
                             MlirNamedAttribute namedAttr) {
  mlirOperationStateAddAttributes(&state, 1, &namedAttr);
}

void addToMlirOperationState(
    MlirOperationState &state,
    std::vector<std::pair<std::string, MlirAttribute>> &attrs) {
  for (auto &p : attrs) {
    addToMlirOperationState(state,
                            toMlirNamedAttribute(p.first.c_str(), p.second));
  }
}

void addToMlirOperationState(MlirOperationState &state, MlirRegion region) {
  mlirOperationStateAddOwnedRegions(&state, 1, &region);
}

[[maybe_unused]] void addToMlirOperationState(MlirOperationState &state,
                                              MlirValue value) {
  mlirOperationStateAddOperands(&state, 1, &value);
}

void addToMlirOperationState(MlirOperationState &state,
                             const std::vector<MlirValue> &values) {
  mlirOperationStateAddOperands(&state, values.size(), values.data());
}

void addToMlirOperationState(MlirOperationState &state, MlirType resultType) {
  mlirOperationStateAddResults(&state, 1, &resultType);
}

void addToMlirOperationState(MlirOperationState &state,
                             const std::vector<MlirType> &resultTypes) {
  mlirOperationStateAddResults(&state, resultTypes.size(), resultTypes.data());
}

[[maybe_unused]] void addToMlirOperationState(MlirOperationState &state) {}

template <typename T, typename U, typename... Ts>
void addToMlirOperationState(MlirOperationState &state, T &&t, U &&u,
                             Ts &&...ts) {
  addToMlirOperationState(state, std::forward<T>(t));
  addToMlirOperationState(state, std::forward<U>(u), std::forward<Ts>(ts)...);
}

template <typename... Ts>
MlirOperation createMlirOperation(std::string name, MlirLocation loc,
                                  Ts &&...ts) {
  MlirOperationState state = mlirOperationStateGet(toMlirStringRef(name), loc);
  addToMlirOperationState(state, std::forward<Ts>(ts)...);
  return mlirOperationCreate(&state);
}

template <typename... Ts>
MlirOperation createMlirOperationAtEnd(MlirBlock block, std::string name,
                                       MlirLocation loc, Ts &&...ts) {
  MlirOperation operation =
      createMlirOperation(name, loc, std::forward<Ts>(ts)...);
  mlirBlockInsertOwnedOperationBefore(block, mlirBlockGetTerminator(block),
                                      operation);
  return operation;
}

const onnx::AttributeProto *GetValueProto(const onnx::NodeProto &node) {

  const onnx::AttributeProto *value_proto = nullptr;
  for (auto &attr : node.attribute()) {
    if (attr.name() == "value") {
      value_proto = &attr;
      break;
    }
  }
  return value_proto;
}

} // namespace

// ---------------------------------------------------------------------------//
// ModelInfo
// ---------------------------------------------------------------------------//

ModelInfo::ModelInfo() = default;

void ModelInfo::DebugDumpProto() {
  std::string debug_string = model_proto_.DebugString();
  fprintf(stderr, "%s\n", debug_string.c_str());
}

Status ModelInfo::Initialize() {
  if (!model_proto_.has_graph()) {
    return SetError("ONNX ModelProto has no main graph");
  }
  main_graph_ = std::make_unique<GraphInfo>(*this, model_proto_.graph());
  if (failed(main_graph_->Initialize())) {
    return failure();
  }

  return success();
}

// ---------------------------------------------------------------------------//
// GraphInfo
// ---------------------------------------------------------------------------//

Status GraphInfo::Initialize() {
  // Initialize look up tables.
  for (const onnx::TensorProto &t : graph_proto_.initializer()) {
    initializer_map_.emplace(t.name(), t);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.value_info()) {
    value_info_map_.emplace(v.name(), v);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.input()) {
    declared_inputs_.emplace_back(&v);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.output()) {
    outputs_.emplace_back(&v);
  }

  // Generate the effective input map, which for old models can be a subset of
  // the input map.
  if (model_info_.config().elide_initialized_inputs) {
    // Default. Add declared inputs to the input map unless if they appear
    // as an initializer.
    for (const onnx::ValueInfoProto *it : declared_inputs_) {
      std::string_view key = it->name();
      if (initializer_map_.find(key) != initializer_map_.end()) {
        // In initializers. Skip.
        continue;
      }
      inputs_.emplace_back(it);
    }
  } else {
    // Fallback for some legacy compatibility.
    inputs_ = declared_inputs_;
    std::vector<std::string_view> illegal_keys;
    for (const onnx::ValueInfoProto *it : inputs_) {
      std::string_view key = it->name();
      if (initializer_map_.find(key) != initializer_map_.end()) {
        illegal_keys.push_back(key);
      }
    }
    if (!illegal_keys.empty()) {
      std::string error = "When not in elide_initialized_inputs=true mode, we "
                          "expect inputs to not have an initial value (got ";
      AppendDelimittedStrings(error, illegal_keys);
      error.append(")");
      return model_info_.SetError(std::move(error));
    }
  }

  // Index the inputs and outputs.
  for (auto *input : inputs_) {
    input_map_.emplace(input->name(), *input);
  }
  for (auto *output : outputs_) {
    output_map_.emplace(output->name(), *output);
  }
  return success();
}

const onnx::TypeProto *GraphInfo::FindTypeProtoForName(std::string_view name) {
  // Node outputs don't typically have type information, but shape inference
  // will associate them in the value_info. If not there, it may be a
  // graph output, which must have type information.
  {
    auto it = value_info_map_.find(name);
    if (it != value_info_map_.end()) {
      return &it->second.type();
    }
  }
  {
    auto it = output_map_.find(name);
    if (it != output_map_.end()) {
      return &it->second.type();
    }
  }

  std::string msg = "No type information associated with '";
  msg.append(name);
  msg.append("'. Run shape inference?");
  model_info_.SetError(std::move(msg));
  return nullptr;
}

// ---------------------------------------------------------------------------//
// ContextCache
// ---------------------------------------------------------------------------//

MlirType ContextCache::ConvertTypeProto(const onnx::TypeProto &tp) {
  if (tp.has_tensor_type()) {
    // Convert Tensor TypeProto.
    const onnx::TypeProto_Tensor &tt = tp.tensor_type();
    if (!tt.has_shape()) {
      std::string msg =
          "Unsupported Tensor type without shape (run shape inference?): ";
      msg.append(tt.DebugString());
      model_info_.SetError(std::move(msg));
      return {nullptr};
    }

    MlirType element_type = ConvertTensorElementType(tt.elem_type());
    if (mlirTypeIsNull(element_type)) {
      return {nullptr};
    }
    shared_dims_.clear();
    shared_dims_.reserve(6);
    for (const onnx::TensorShapeProto::Dimension &dim : tt.shape().dim()) {
      if (dim.has_dim_value()) {
        // Static.
        shared_dims_.push_back(dim.dim_value());
      } else {
        // Dynamic.
        shared_dims_.push_back(-1);
      }
    }

    return GetVtensorType(shared_dims_, element_type);
  } else {
    std::string msg = "Unsupported ONNX TypeProto: ";
    msg.append(tp.DebugString());
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
}

MlirType ContextCache::ConvertTensorElementType(int elem_type) {
  auto it = elem_type_map_.find(elem_type);
  if (it != elem_type_map_.end()) {
    return it->second;
  }

  MlirType t = {nullptr};
  switch (elem_type) {
  case onnx::TensorProto::FLOAT:
    t = mlirF32TypeGet(context_);
    break;
  case onnx::TensorProto::UINT8:
    t = mlirIntegerTypeUnsignedGet(context_, 8);
    break;
  case onnx::TensorProto::INT8:
    t = mlirIntegerTypeSignedGet(context_, 8);
    break;
  case onnx::TensorProto::UINT16:
    t = mlirIntegerTypeUnsignedGet(context_, 16);
    break;
  case onnx::TensorProto::INT16:
    t = mlirIntegerTypeSignedGet(context_, 16);
    break;
  case onnx::TensorProto::INT32:
    t = mlirIntegerTypeSignedGet(context_, 32);
    break;
  case onnx::TensorProto::UINT32:
    t = mlirIntegerTypeUnsignedGet(context_, 32);
    break;
  case onnx::TensorProto::INT64:
    t = mlirIntegerTypeSignedGet(context_, 64);
    break;
  case onnx::TensorProto::UINT64:
    t = mlirIntegerTypeUnsignedGet(context_, 64);
    break;
  case onnx::TensorProto::BOOL:
    t = mlirIntegerTypeGet(context_, 1);
    break;
  case onnx::TensorProto::FLOAT16:
    t = mlirF16TypeGet(context_);
    break;
  case onnx::TensorProto::DOUBLE:
    t = mlirF64TypeGet(context_);
    break;
  case onnx::TensorProto::COMPLEX64:
    t = mlirComplexTypeGet(mlirF32TypeGet(context_));
    break;
  case onnx::TensorProto::COMPLEX128:
    t = mlirComplexTypeGet(mlirF64TypeGet(context_));
    break;
  case onnx::TensorProto::BFLOAT16:
    t = mlirBF16TypeGet(context_);
    break;
  case onnx::TensorProto::FLOAT8E4M3FN:
    t = mlirFloat8E4M3FNTypeGet(context_);
    break;
  case onnx::TensorProto::FLOAT8E4M3FNUZ:
    t = mlirFloat8E4M3FNUZTypeGet(context_);
    break;
  case onnx::TensorProto::FLOAT8E5M2:
    t = mlirFloat8E5M2TypeGet(context_);
    break;
  case onnx::TensorProto::FLOAT8E5M2FNUZ:
    t = mlirFloat8E5M2FNUZTypeGet(context_);
    break;
  default: {
    std::string msg = "Unknown ONNX tensor element type: ";
    msg.append(std::to_string(elem_type));
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
  }

  assert(t.ptr && "did not convert type");
  elem_type_map_[elem_type] = t;
  return t;
}

MlirAttribute
ContextCache::ConvertTensorProtoToAttr(const onnx::TensorProto &tp) {
  MlirType tensor_type = ConvertTensorProtoToBuiltinType(tp);
  if (tp.has_raw_data()) {
    std::string sanitized_name = SanitizeNameAsIdentifier(tp.name());
    // Conveniently, DenseResourceElementsAttr shares the raw data
    // format. We just give it maximum numeric alignment.
    return mlirUnmanagedDenseResourceElementsAttrGet(
        tensor_type, toMlirStringRef(sanitized_name),
        const_cast<void *>(static_cast<const void *>(tp.raw_data().data())),
        tp.raw_data().size(), /*dataAlignment=*/8, /*dataIsMutable=*/false,
        /*deleter=*/nullptr, /*userData=*/nullptr);
  } else {
    switch (tp.data_type()) {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
      return mlirDenseElementsAttrFloatGet(tensor_type, tp.float_data_size(),
                                           tp.float_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
      return mlirDenseElementsAttrInt32Get(tensor_type, tp.int32_data_size(),
                                           tp.int32_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
      return mlirDenseElementsAttrInt64Get(tensor_type, tp.int64_data_size(),
                                           tp.int64_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
      return mlirDenseElementsAttrDoubleGet(tensor_type, tp.double_data_size(),
                                            tp.double_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32: {
      // Special case. See proto. Someone apparently got lazy.
      std::vector<uint32_t> stupid_conversion;
      stupid_conversion.reserve(tp.uint64_data_size());
      for (uint64_t v : tp.uint64_data())
        stupid_conversion.push_back(v);
      return mlirDenseElementsAttrUInt32Get(
          tensor_type, stupid_conversion.size(), stupid_conversion.data());
    }
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
      return mlirDenseElementsAttrUInt64Get(tensor_type, tp.uint64_data_size(),
                                            tp.uint64_data().data());
    }
  }

  std::string message =
      "Unable to convert ONNX TensorProto to MLIR attribute: ";
  message.append(tp.DebugString());
  model_info_.SetError(std::move(message));
  return {nullptr};
}

MlirType
ContextCache::ConvertTensorProtoToBuiltinType(const onnx::TensorProto &tp) {
  MlirType element_type = ConvertTensorElementType(tp.data_type());
  if (mlirTypeIsNull(element_type))
    return {nullptr};

  shared_dims_.clear();
  for (auto dim : tp.dims()) {
    shared_dims_.push_back(dim);
  }
  return mlirRankedTensorTypeGet(shared_dims_.size(), shared_dims_.data(),
                                 element_type,
                                 /*encoding=*/{nullptr});
}

MlirType
ContextCache::ConvertTensorProtoToVtensorType(const onnx::TensorProto &tp) {
  MlirType element_type = ConvertTensorElementType(tp.data_type());
  if (mlirTypeIsNull(element_type))
    return {nullptr};

  shared_dims_.clear();
  for (auto dim : tp.dims()) {
    shared_dims_.push_back(dim);
  }

  return GetVtensorType(shared_dims_, element_type);
}

MlirType ContextCache::GetVtensorType(const std::vector<int64_t> &dims,
                                      MlirType element_type) {
  std::string type_asm = "!torch.vtensor<[";
  // Add dimension list.
  bool first_dim = true;
  for (int dim : dims) {
    if (first_dim)
      first_dim = false;
    else
      type_asm.push_back(',');
    if (dim < 0)
      type_asm.push_back('?');
    else
      type_asm.append(std::to_string(dim));
  }
  type_asm.append("],");

  // Add element type.
  type_asm.append(getMlirAsm(element_type));
  type_asm.push_back('>');

  // Look in cache.
  auto found_it = asm_type_map_.find(type_asm);
  if (found_it != asm_type_map_.end()) {
    return found_it->second;
  }

  // Parse.
  MlirType t = mlirTypeParseGet(context_, toMlirStringRef(type_asm));
  if (mlirTypeIsNull(t)) {
    std::string message =
        "internal error: could not parse !torch.vtensor type: ";
    message.append(type_asm);
    model_info_.SetError(std::move(message));
    return t;
  }

  asm_type_map_[std::move(type_asm)] = t;
  return t;
}

MlirType ContextCache::GetNoneType() {
  std::string type_asm = "!torch.none";
  return mlirTypeParseGet(context_, toMlirStringRef(type_asm));
}

// ---------------------------------------------------------------------------//
// NodeImporter
// ---------------------------------------------------------------------------//

NodeImporter::NodeImporter(GraphInfo &graph_info, ContextCache &cc,
                           MlirOperation module_op)
    : graph_info_(graph_info), cc_(cc),
      context_(mlirOperationGetContext(module_op)), module_op_(module_op),
      func_op_({nullptr}), body_block_({nullptr}) {
  std::string locName = "graph:";
  locName.append(graph_info.graph_proto().name());
  default_loc_ = mlirLocationNameGet(context_, toMlirStringRef(locName),
                                     /*childLoc=*/{nullptr});
}

Status NodeImporter::DefineFunction(std::optional<std::string> name) {
  const onnx::GraphProto &p = graph_info_.graph_proto();
  MlirRegion moduleBodyRegion = mlirOperationGetRegion(module_op_, 0);
  MlirBlock moduleBody = mlirRegionGetFirstBlock(moduleBodyRegion);
  MlirAttribute nameAttr;
  if (name) {
    // Explicitly named.
    nameAttr = mlirStringAttrGet(context_, toMlirStringRef(*name));
  } else {
    // Name it according to the graph.
    nameAttr = mlirStringAttrGet(context_, toMlirStringRef(p.name()));
  }

  // Derive the FunctionType.
  std::vector<MlirType> input_types;
  std::vector<MlirLocation> input_locs;
  std::vector<MlirType> output_types;
  for (auto *input : graph_info_.inputs()) {
    MlirType t = cc_.ConvertTypeProto(input->type());
    if (mlirTypeIsNull(t)) {
      return failure();
    }
    input_types.push_back(t);
    input_locs.push_back(default_loc_);
  }
  for (auto *output : graph_info_.outputs()) {
    MlirType t = cc_.ConvertTypeProto(output->type());
    if (mlirTypeIsNull(t)) {
      return failure();
    }
    output_types.push_back(t);
  }
  MlirType ftype =
      mlirFunctionTypeGet(context_, input_types.size(), input_types.data(),
                          output_types.size(), output_types.data());

  // Create func.func.
  func_op_ = createMlirOperationAtEnd(
      moduleBody, "func.func", default_loc_, mlirRegionCreate(),
      toMlirNamedAttribute("function_type", mlirTypeAttrGet(ftype)),
      toMlirNamedAttribute("sym_name", nameAttr));

  // Add entry block.
  body_block_ = mlirBlockCreate(input_types.size(), input_types.data(),
                                input_locs.data());
  MlirRegion bodyRegion = mlirOperationGetRegion(func_op_, 0);
  mlirRegionAppendOwnedBlock(bodyRegion, body_block_);

  // Map the block args to names and store for evaluation.
  for (int i = 0, e = graph_info_.inputs().size(); i < e; ++i) {
    std::string_view name = graph_info_.inputs()[i]->name();
    MlirValue value = mlirBlockGetArgument(body_block_, i);
    nv_map_[name] = value;
  }

  PopulateGraphAttrs(func_op_);
  return success();
}

void NodeImporter::PopulateGraphAttrs(MlirOperation container_op) {
  const onnx::ModelProto &m = graph_info_.model_info().model_proto();
  MlirType i64_type = mlirIntegerTypeSignedGet(context_, 64);
  int default_opset_version = 0;
  std::unordered_map<std::string_view, MlirAttribute> opset_versions;
  // Determine model level opset versions.
  for (const onnx::OperatorSetIdProto &opset_import : m.opset_import()) {
    if (opset_import.has_domain()) {
      opset_versions[opset_import.domain()] =
          mlirIntegerAttrGet(i64_type, opset_import.version());
    } else {
      default_opset_version = opset_import.version();
    }
  }

  // Set the default domain version.
  if (default_opset_version != 0) {
    mlirOperationSetDiscardableAttributeByName(
        container_op, toMlirStringRef("torch.onnx_meta.opset_version"),
        mlirIntegerAttrGet(i64_type, default_opset_version));
  }

  // Set versions for other domains.
  if (!opset_versions.empty()) {
    std::vector<MlirNamedAttribute> version_attrs;
    for (auto it : opset_versions) {
      version_attrs.push_back(mlirNamedAttributeGet(
          mlirIdentifierGet(context_, toMlirStringRef(it.first)), it.second));
    }
    MlirAttribute dict_attr = mlirDictionaryAttrGet(
        context_, version_attrs.size(), version_attrs.data());
    mlirOperationSetDiscardableAttributeByName(
        container_op, toMlirStringRef("torch.onnx_meta.opset_versions"),
        dict_attr);
  }

  // IR version and producer.
  mlirOperationSetDiscardableAttributeByName(
      container_op, toMlirStringRef("torch.onnx_meta.ir_version"),
      mlirIntegerAttrGet(i64_type, m.ir_version()));
  mlirOperationSetDiscardableAttributeByName(
      container_op, toMlirStringRef("torch.onnx_meta.producer_name"),
      mlirStringAttrGet(context_, toMlirStringRef(m.producer_name())));
  mlirOperationSetDiscardableAttributeByName(
      container_op, toMlirStringRef("torch.onnx_meta.producer_version"),
      mlirStringAttrGet(context_, toMlirStringRef(m.producer_version())));
}

void NodeImporter::GetNone() {

  MlirLocation loc =
      mlirLocationNameGet(context_, toMlirStringRef("onnx_importer.none"),
                          /*childLoc=*/{nullptr});

  createMlirOperationAtEnd(body_block_, "torch.constant.none", loc,
                           cc_.GetNoneType());
}

Status NodeImporter::ImportAll() {
  // TODO: Consider pulling in initializers on demand since there can be so
  // much unused crap.
  for (auto it : graph_info_.initializer_map()) {
    if (failed(ImportInitializer(it.second)))
      return failure();
  }

  GetNone();

  for (auto it : graph_info_.graph_proto().node()) {
    if (failed(ImportNode(it)))
      return failure();
  }

  // Lookup the outputs, which should all be in the nv_map if the graph was
  // properly formed.
  std::vector<MlirValue> output_values;
  for (const auto *output : graph_info_.outputs()) {
    std::string_view name = output->name();
    auto found_it = nv_map_.find(name);
    if (found_it == nv_map_.end()) {
      std::string msg = "Non topologically produced ONNX graph output '";
      msg.append(name);
      msg.append("'");
      return SetError(std::move(msg));
    }
    output_values.push_back(found_it->second);
  }

  createMlirOperationAtEnd(body_block_, "func.return", default_loc_,
                           output_values);
  return success();
}

Status NodeImporter::ImportInitializer(const onnx::TensorProto &initializer,
                                       std::optional<std::string> extern_name) {
  std::string_view name = extern_name ? *extern_name : initializer.name();
  MlirLocation loc = mlirLocationNameGet(context_, toMlirStringRef(name),
                                         /*childLoc=*/{nullptr});

  MlirAttribute value_attr = cc_.ConvertTensorProtoToAttr(initializer);
  MlirType vtensor_type = cc_.ConvertTensorProtoToVtensorType(initializer);
  if (mlirAttributeIsNull(value_attr) || mlirTypeIsNull(vtensor_type))
    return failure();

  MlirOperation op = createMlirOperationAtEnd(
      body_block_, "torch.operator", loc, vtensor_type,
      toMlirNamedAttribute(
          "name",
          mlirStringAttrGet(context_, toMlirStringRef("onnx.Constant"))),
      toMlirNamedAttribute("torch.onnx.value", value_attr));
  MlirValue result = mlirOperationGetResult(op, 0);

  auto inserted = nv_map_.insert(std::make_pair(name, result));
  if (!inserted.second) {
    std::string msg = "Multiple nodes produced a value for '";
    msg.append(name);
    msg.append("', most recent from ");
    msg.append(initializer.DebugString());
    return SetError(std::move(msg));
  }

  return success();
}

Status NodeImporter::ImportNode(const onnx::NodeProto &node) {
  std::string_view op_type = node.op_type();
  // Handle special-form op types that do not go down the generic path.
  if (op_type == "Constant") {
    const onnx::AttributeProto *value_proto = GetValueProto(node);
    if (value_proto) {
      return ImportConstantNodeValueAttr(node);
    }
  }

  return ImportGeneralNode(node);
}

Status NodeImporter::ImportGeneralNode(const onnx::NodeProto &node) {
  MlirLocation loc = mlirLocationNameGet(context_, toMlirStringRef(node.name()),
                                         /*childLoc=*/{nullptr});

  // Map inputs to values.
  std::vector<MlirValue> input_values;
  for (auto &input_name : node.input()) {
    auto found_it = nv_map_.find(input_name);
    if (found_it == nv_map_.end()) {
      std::string msg = "Non topologically produced ONNX node input '";
      msg.append(input_name);
      msg.append("'");
      return SetError(std::move(msg));
    }
    input_values.push_back(found_it->second);
  }

  // Map outputs to types.
  std::vector<MlirType> output_types;
  for (auto &output_name : node.output()) {
    const onnx::TypeProto *type_proto =
        graph_info_.FindTypeProtoForName(output_name);
    if (!type_proto)
      return failure();

    MlirType t = cc_.ConvertTypeProto(*type_proto);
    if (mlirTypeIsNull(t))
      return failure();
    output_types.push_back(t);
  }

  // Derive the op name.
  std::string op_name = "onnx.";
  op_name.append(node.op_type());
  MlirAttribute op_name_attr =
      mlirStringAttrGet(context_, toMlirStringRef(op_name));

  // General attributes.
  std::vector<std::pair<std::string, MlirAttribute>> general_attributes;
  for (auto &onnx_attr : node.attribute()) {
    MlirAttribute attr = ImportGeneralAttribute(onnx_attr);
    if (mlirAttributeIsNull(attr))
      return failure();
    std::string full_name = "torch.onnx.";
    full_name.append(onnx_attr.name());
    general_attributes.push_back(std::make_pair(full_name, attr));
  }

  // Create op.
  MlirOperation op = createMlirOperationAtEnd(
      body_block_, "torch.operator", loc, output_types, input_values,
      toMlirNamedAttribute("name", op_name_attr), general_attributes);

  // Record the result values.
  for (int i = 0, e = output_types.size(); i < e; ++i) {
    MlirValue result = mlirOperationGetResult(op, i);
    std::string_view name = node.output(i);
    auto inserted = nv_map_.insert(std::make_pair(name, result));
    if (!inserted.second) {
      std::string msg = "Multiple nodes produced a value for '";
      msg.append(name);
      msg.append("', most recent from ");
      msg.append(node.DebugString());
      return SetError(std::move(msg));
    }
  }

  return success();
}

MlirAttribute
NodeImporter::ImportGeneralAttribute(const onnx::AttributeProto &onnx_attr) {
  switch (onnx_attr.type()) {
  case onnx::AttributeProto::UNDEFINED:
    SetError("'UNDEFINED' attribute type not supported");
    return {nullptr};
  case onnx::AttributeProto::FLOAT:
    return mlirFloatAttrDoubleGet(context_, mlirF32TypeGet(context_),
                                  onnx_attr.f());
  case onnx::AttributeProto::INT:
    return mlirIntegerAttrGet(mlirIntegerTypeSignedGet(context_, 64),
                              onnx_attr.i());
  case onnx::AttributeProto::STRING:
    return mlirStringAttrGet(context_, toMlirStringRef(onnx_attr.s()));
  case onnx::AttributeProto::TENSOR:
    return cc_.ConvertTensorProtoToAttr(onnx_attr.t());
  case onnx::AttributeProto::GRAPH:
    SetError("'GRAPH' attribute type not supported on this node");
    return {nullptr};
  case onnx::AttributeProto::SPARSE_TENSOR:
    SetError("'SPARSE_TENSOR' attribute type not supported on this node");
    return {nullptr};
  case onnx::AttributeProto::TYPE_PROTO:
    SetError("'TYPE_PROTO' attribute type not supported on this node");
    return {nullptr};
  case onnx::AttributeProto::FLOATS: {
    std::vector<MlirAttribute> attrs;
    for (auto f : onnx_attr.floats())
      attrs.push_back(
          mlirFloatAttrDoubleGet(context_, mlirF32TypeGet(context_), f));
    return mlirArrayAttrGet(context_, attrs.size(), attrs.data());
  }
  case onnx::AttributeProto::INTS: {
    std::vector<MlirAttribute> attrs;
    for (auto i : onnx_attr.ints())
      attrs.push_back(
          mlirIntegerAttrGet(mlirIntegerTypeSignedGet(context_, 64), i));
    return mlirArrayAttrGet(context_, attrs.size(), attrs.data());
  }
  case onnx::AttributeProto::STRINGS: {
    std::vector<MlirAttribute> attrs;
    for (auto s : onnx_attr.strings())
      attrs.push_back(mlirStringAttrGet(context_, toMlirStringRef(s)));
    return mlirArrayAttrGet(context_, attrs.size(), attrs.data());
  }
  case onnx::AttributeProto::TENSORS: {
    std::vector<MlirAttribute> attrs;
    for (auto &t : onnx_attr.tensors()) {
      MlirAttribute attr = cc_.ConvertTensorProtoToAttr(t);
      if (mlirAttributeIsNull(attr))
        return {nullptr};
      attrs.push_back(attr);
    }
    return mlirArrayAttrGet(context_, attrs.size(), attrs.data());
  }
  case onnx::AttributeProto::GRAPHS:
    SetError("'GRAPHS' attribute type not supported on this node");
    return {nullptr};
  case onnx::AttributeProto::SPARSE_TENSORS:
    SetError("'SPARSE_TENSORS' attribute type not supported on this node");
    return {nullptr};
  case onnx::AttributeProto::TYPE_PROTOS:
    SetError("'TYPE_PROTOS' attribute type not supported on this node");
    return {nullptr};
  }

  std::string msg = "Unhandled ONNX attribute type code ";
  msg.append(std::to_string(onnx_attr.type()));
  msg.append(": ");
  msg.append(onnx_attr.DebugString());
  SetError(std::move(msg));
  return {nullptr};
}

// Special case only for constants specified by value attribute (for now)
Status NodeImporter::ImportConstantNodeValueAttr(const onnx::NodeProto &node) {

  const onnx::AttributeProto *value_proto = GetValueProto(node);

  // Produce an initializer for the constant, so that it can be used in
  // combination with other ops, such as ConstantOfShape, requiring
  // constant input.

  if (value_proto->type() != onnx::AttributeProto_AttributeType_TENSOR) {
    return SetError("Constant node must have a tensor value attribute");
  }
  if (node.output_size() != 1) {
    return SetError("Constant node must have one output");
  }
  const std::string &const_name = node.output(0);
  if (failed(ImportInitializer(value_proto->t(), const_name)))
    return failure();
  graph_info_.initializer_map().emplace(const_name, value_proto->t());
  return success();
}

Status NodeImporter::GetImmediateShapeTensor(const std::string &name,
                                             std::vector<int64_t> &shape) {
  auto found_it = graph_info_.initializer_map().find(name);
  if (found_it == graph_info_.initializer_map().end()) {
    std::string message = "An immediate shape value for '";
    message.append(name);
    message.append("' was required but it is dynamically produced");
    return SetError(std::move(message));
  }

  const onnx::TensorProto &tp = found_it->second;
  shape.clear();

  // Since this is being interpreted as a shape, we only support some limited
  // types.
  size_t raw_data_size;
  switch (tp.data_type()) {
  case onnx::TensorProto::DataType::TensorProto_DataType_INT32: {
    auto *raw_data = graph_info_.GetOptionalRawData<int32_t>(tp, raw_data_size);
    if (raw_data) {
      std::copy(raw_data, raw_data + raw_data_size, std::back_inserter(shape));
    } else {
      for (auto v : tp.int32_data())
        shape.push_back(v);
    }
    return success();
  }
  case onnx::TensorProto::DataType::TensorProto_DataType_INT64: {
    auto *raw_data = graph_info_.GetOptionalRawData<int64_t>(tp, raw_data_size);
    if (raw_data) {
      std::copy(raw_data, raw_data + raw_data_size, std::back_inserter(shape));
    } else {
      for (auto v : tp.int64_data())
        shape.push_back(v);
    }
    return success();
  }
  case onnx::TensorProto::DataType::TensorProto_DataType_UINT32: {
    auto *raw_data =
        graph_info_.GetOptionalRawData<uint32_t>(tp, raw_data_size);
    if (raw_data) {
      std::copy(raw_data, raw_data + raw_data_size, std::back_inserter(shape));
    } else {
      // Stupid special case: stored in uint64.
      for (auto v : tp.uint64_data())
        shape.push_back(v);
    }
    return success();
  }
  case onnx::TensorProto::DataType::TensorProto_DataType_UINT64: {
    auto *raw_data =
        graph_info_.GetOptionalRawData<uint64_t>(tp, raw_data_size);
    if (raw_data) {
      std::copy(raw_data, raw_data + raw_data_size, std::back_inserter(shape));
    } else {
      for (auto v : tp.uint64_data())
        shape.push_back(v);
    }
    return success();
  }
  }

  {
    std::string message =
        "An immediate shape value could not be converted from TensorProto: ";
    message.append(tp.DebugString());
    return SetError(std::move(message));
  }
}

void NodeImporter::DebugDumpModule() {
  auto callback = +[](MlirStringRef sr, void *) {
    fwrite(sr.data, sizeof(char), sr.length, stderr);
  };
  mlirOperationPrint(module_op_, callback, nullptr);
}
