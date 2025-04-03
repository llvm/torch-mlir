//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "OnnxImporter.h"

#include "google/protobuf/text_format.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/attribute_binder.h"
#include "onnx/shape_inference/implementation.h"
#include "torch-mlir-c/Registration.h"
#ifndef NDEBUG
#include "onnx/checker.h"
#endif

#include <cstdio>
#include <functional>
#include <numeric>
#include <type_traits>

namespace std {
template <> struct hash<MlirType> {
  size_t operator()(const MlirType &x) const {
    return std::hash<const void *>{}(x.ptr);
  }
};
} // namespace std

using namespace torch_mlir_onnx;

namespace {

template <typename TOut, typename TIn,
          typename = std::enable_if_t<std::is_convertible<TIn, TOut>::value>>
inline std::vector<TOut> elementwiseCast(std::span<TIn> arr) {
  return std::vector<TOut>(arr.begin(), arr.end());
}

bool IsIdentifer(std::string_view s) {
  bool res = true;
  res &= !s.empty() && !std::isdigit(static_cast<unsigned char>(s[0]));
  res &= std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_';
  });
  return res;
}

std::string SanitizeNameAsIdentifier(std::string_view in) {
  std::string out;
  if (!IsIdentifer(in)) {
    out.append("_");
  }
  out.append(in);

  // Remove characters that are invalid in MLIR identifier names.
  // https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords
  for (char &c : out) {
    if (!std::isalnum(c) && c != '.')
      c = '_';
  }
  return out;
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

void addToMlirOperationState(MlirOperationState &state,
                             const std::vector<MlirRegion> &regions) {
  mlirOperationStateAddOwnedRegions(&state, regions.size(), regions.data());
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

  const onnx::AttributeProto *valueProto = nullptr;
  for (const onnx::AttributeProto &attr : node.attribute()) {
    if (attr.name() == "value") {
      valueProto = &attr;
      break;
    }
  }
  return valueProto;
}

onnx::TypeProto MakeTensorTypeProto(onnx::TensorProto_DataType elemType,
                                    const auto &shape) {
  onnx::TypeProto typeProto;
  onnx::TypeProto_Tensor *tensorTypeProto = typeProto.mutable_tensor_type();
  tensorTypeProto->set_elem_type(elemType);
  onnx::TensorShapeProto *tensorShapeProto = tensorTypeProto->mutable_shape();

  tensorShapeProto->clear_dim();
  for (int64_t d : shape) {
    onnx::TensorShapeProto_Dimension *dim = tensorShapeProto->add_dim();
    dim->set_dim_value(d);
  }
  return typeProto;
}

uint32_t CountRegions(
    const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &onnxAttrs) {
  std::vector<onnx::AttributeProto::AttributeType> types;
  types.reserve(onnxAttrs.size());
  for (const onnx::AttributeProto &attr : onnxAttrs)
    types.push_back(attr.type());
  return std::count(types.cbegin(), types.cend(), onnx::AttributeProto::GRAPH);
}

template <typename T>
std::vector<T> FillVector(uint32_t size, std::function<T()> &&genFun) {
  std::vector<T> res;
  res.reserve(size);
  for ([[maybe_unused]] uint32_t idx = 0; idx < size; ++idx) {
    res.push_back(genFun());
  }
  return res;
}

auto StringJoin(const auto &range, const auto &sep) {
  if (range.empty())
    return std::string();

  return std::accumulate(
      next(begin(range)), end(range), static_cast<std::string>(range[0]),
      [&sep](std::string result, const auto &value) {
        return std::move(result) + sep + static_cast<std::string>(value);
      });
}

size_t HashCombineRange(const auto &range) {
  if (range.empty())
    return 0;
  return std::accumulate(
      next(begin(range)), end(range), range[0], [](auto res, const auto &val) {
        return res ^ std::hash<std::remove_const_t<
                         std::remove_reference_t<decltype(val)>>>{}(val);
      });
}

FailureOr<onnx::FunctionProto> GetCDFunctionWithOpsetVersion(
    const onnx::OpSchema *op, int opsetVersion, const onnx::NodeProto &node,
    const std::span<const onnx::TypeProto *const> &inputTypes) {
  if (op->HasContextDependentFunctionWithOpsetVersion(opsetVersion)) {
    std::vector<onnx::TypeProto> inputTypesNonConst;
    inputTypesNonConst.reserve(inputTypes.size());
    for (const onnx::TypeProto *tp : inputTypes) {
      inputTypesNonConst.push_back(*tp);
    }
    onnx::FunctionBodyBuildContextImpl ctx(node, inputTypesNonConst);
    onnx::FunctionProto funcProto;
    if (!op->BuildContextDependentFunction(ctx, funcProto, opsetVersion)) {
      return failure;
    }
    return std::move(funcProto);
  }
  return failure;
}

// Helper for SpecializeFunctionAndCreateModel() that binds concrete
// values to attributes on a node in the interior of a function.
// Defaults are taken from opSchema.
//
// Relies on ONNX's attribute binder.
void BindAttributesWithDefaults(const onnx::NodeProto &callnode,
                                onnx::FunctionProto &callee,
                                const onnx::OpSchema *opSchema) {
  onnx::internal::AttributeMap map;
  for (const auto &defaultPair : opSchema->attributes()) {
    const onnx::AttributeProto &defaultValue = defaultPair.second.default_value;
    if (defaultValue.type())
      map.emplace(defaultPair.first, &defaultValue);
  }
  for (auto &attr : callnode.attribute()) {
    map[attr.name()] = &attr;
  }
  onnx::internal::AttributeBinder attrBinder(map);
  attrBinder.VisitFunction(&callee);
}

//     Helper for ModuleCache::get_operator_function() that specializes a
//     function and coverts it to a model. An ONNX function may be polymorphic,
//     parameterized over the types of its inputs and values of its attributes
//     (~= compile-time constants). We need to monomorphize it for importing
//     into MLIR. It seems like the only practical way to do this is by turning
//     it into a model:
//     - models can have types on their inputs and outputs, unlike functions
//     - ONNX provides a function to do shape inference (providing concrete
//       types for everything in the body) for models, but not for functions
//     - the rest of the code in this importer can only handle models, not
//       functions
onnx::ModelProto SpecializeFunctionAndCreateModel(
    const onnx::FunctionProto &functionProto, const onnx::OpSchema *opSchema,
    const std::string &nameToGive, int irVersion,
    const std::span<const onnx::TypeProto *const> &inputTypeProtos,
    const std::span<const onnx::TypeProto *const> &outputTypeProtos,
    const onnx::NodeProto &callerNode) {
  onnx::ModelProto modelProto;
  modelProto.mutable_opset_import()->MergeFrom(functionProto.opset_import());
  modelProto.set_ir_version(irVersion);
  onnx::GraphProto &graphProto = *modelProto.mutable_graph();

  for (int index = 0; index < functionProto.input().size(); ++index) {
    const std::string &inputName = functionProto.input()[index];
    const onnx::TypeProto *const inputTypeProto = inputTypeProtos[index];
    onnx::ValueInfoProto &inputProto = *graphProto.add_input();
    inputProto.set_name(inputName);
    inputProto.mutable_type()->CopyFrom(*inputTypeProto);
  }

  for (int index = 0; index < functionProto.output().size(); ++index) {
    const std::string &outputName = functionProto.output()[index];
    const onnx::TypeProto *const outputTypeProto = outputTypeProtos[index];
    onnx::ValueInfoProto &outputProto = *graphProto.add_output();
    outputProto.set_name(outputName);
    outputProto.mutable_type()->CopyFrom(*outputTypeProto);
  }

  onnx::FunctionProto specializedFunProto;
  specializedFunProto.CopyFrom(functionProto);
  BindAttributesWithDefaults(callerNode, specializedFunProto, opSchema);
  graphProto.mutable_node()->Add(specializedFunProto.node().begin(),
                                 specializedFunProto.node().end());

  graphProto.set_name(nameToGive);

  onnx::ShapeInferenceOptions options{/*check_type=*/true, /*error_mode=*/1,
                                      /*enable_data_propagation=*/true};
  onnx::shape_inference::InferShapes(
      modelProto, onnx::OpSchemaRegistry::Instance(), options);

  // NOTE: (from onnx_importer.py) Useful for debugging.
  //
  // onnx::checker::check_model(modelProto, /*fullCheck=*/true);

  return modelProto;
}

} // namespace

// ---------------------------------------------------------------------------//
// ModelInfo
// ---------------------------------------------------------------------------//

void ModelInfo::DebugDumpProto() {
  std::string debugString = model_proto_.DebugString();
  fprintf(stderr, "%s\n", debugString.c_str());
}

Status ModelInfo::Initialize() {
  if (!model_proto_.has_graph()) {
    return SetError("ONNX ModelProto has no main graph");
  }
  main_graph_ = std::make_unique<GraphInfo>(*this, model_proto_.graph());
  if (failed(main_graph_->Initialize())) {
    return failure;
  }

  return success;
}

// ---------------------------------------------------------------------------//
// GraphInfo
// ---------------------------------------------------------------------------//

Status GraphInfo::Initialize() {
  // Initialize look up tables.
  for (const onnx::TensorProto &t : graph_proto_.initializer()) {
    if (initializer_map_.find(t.name()) != initializer_map_.end()) {
      return model_info_.SetError("ONNX initializer name already used: " +
                                  t.name());
    }
    InitializerMapEmplace(t.name(), t);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.value_info()) {
    if (value_info_map_.find(v.name()) != value_info_map_.end()) {
      return model_info_.SetError("ONNX value_info name already used: " +
                                  v.name());
    }
    value_info_map_.emplace(v.name(), v);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.input()) {
    if (declared_input_map_.find(v.name()) != declared_input_map_.end()) {
      return model_info_.SetError("ONNX value_info name already used: " +
                                  v.name());
    }
    declared_input_map_.emplace(v.name(), v);
  }
  for (const onnx::ValueInfoProto &v : graph_proto_.output()) {
    if (output_map_.find(v.name()) != output_map_.end()) {
      return model_info_.SetError("ONNX value_info name already used: " +
                                  v.name());
    }
    output_map_.emplace(v.name(), v);
  }

  // Generate the effective input map, which for old models can be a subset of
  // the input map.
  if (is_top_level_ && model_info_.GetConfig().elide_initialized_inputs) {
    // Default. Add declared inputs to the input map unless if they appear
    // as an initializer.
    for (const auto &decIn : declared_input_map_) {
      const std::string_view &key = decIn.first;
      if (initializer_map_.find(key) != initializer_map_.end()) {
        // In initializers. Skip.
        continue;
      }
      input_map_.emplace(key, decIn.second);
    }
  } else {
    // Fallback for some legacy compatibility.
    input_map_ = declared_input_map_;
    std::vector<std::string_view> illegalKeys;
    for (const auto &input : input_map_) {
      const std::string_view &key = input.first;
      if (initializer_map_.find(key) != initializer_map_.end()) {
        illegalKeys.push_back(key);
      }
    }
    if (!illegalKeys.empty()) {
      std::string error = "When not in elide_initialized_inputs=true mode, we "
                          "expect inputs to not have an initial value (got " +
                          StringJoin(illegalKeys, ", ") + ")";
      return model_info_.SetError(std::move(error));
    }
  }

  return success;
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
  {
    auto it = declared_input_map_.find(name);
    if (it != declared_input_map_.end()) {
      return &it->second.type();
    }
  }
  {
    auto it = initializer_map_.find(name);
    if (it != initializer_map_.end()) {
      return &it->second.second;
    }
  }
  // No type information is associated, this can occur when the value is unused.
  return nullptr;
}

void GraphInfo::InitializerMapEmplace(const std::string_view &name,
                                      const onnx::TensorProto &tp) {
  initializer_map_.emplace(
      name,
      std::pair<const onnx::TensorProto &, onnx::TypeProto>(
          tp, MakeTensorTypeProto(onnx::TensorProto_DataType(tp.data_type()),
                                  tp.dims())));
}

// ---------------------------------------------------------------------------//
// ContextCache
// ---------------------------------------------------------------------------//

MlirType ContextCache::ConvertTensorElementType(int elemType) {
  auto it = elem_type_map_.find(elemType);
  if (it != elem_type_map_.end()) {
    return it->second;
  }

  MlirType t = {nullptr};
  switch (elemType) {
  case onnx::TensorProto::FLOAT:
    t = mlirF32TypeGet(context_);
    break;
  case onnx::TensorProto::UINT4:
    t = mlirIntegerTypeUnsignedGet(context_, 4);
    break;
  case onnx::TensorProto::INT4:
    t = mlirIntegerTypeSignedGet(context_, 4);
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
  case onnx::TensorProto::STRING: {
    const std::string typeAsm = "!torch.str";
    t = mlirTypeParseGet(context_, toMlirStringRef(typeAsm));
    assert(!mlirTypeIsNull(t));
    break;
  }
  default: {
    std::string msg = "Unknown ONNX tensor element type: ";
    msg.append(std::to_string(elemType));
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
  }

  assert(t.ptr && "did not convert type");
  elem_type_map_[elemType] = t;
  return t;
}

MlirType ContextCache::GetNoneType() {
  std::string typeAsm = "!torch.none";
  return mlirTypeParseGet(context_, toMlirStringRef(typeAsm));
}

MlirType ContextCache::GetListType(const std::string &elementTypeAsm) {
  auto it = list_type_map_.find(elementTypeAsm);
  if (it != list_type_map_.end()) {
    return it->second;
  }

  std::string typeAsm = "!torch.list<" + elementTypeAsm + ">";
  MlirType t = mlirTypeParseGet(context_, toMlirStringRef(typeAsm));
  if (mlirTypeIsNull(t)) {
    std::string msg =
        "Unparseable torch type (MLIR asm format bug?): " + typeAsm;
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
  list_type_map_[elementTypeAsm] = t;
  return t;
}

MlirType ContextCache::GetOptionalType(const std::string &elementTypeAsm) {
  auto it = optional_type_map_.find(elementTypeAsm);
  if (it != optional_type_map_.end()) {
    return it->second;
  }

  std::string typeAsm = "!torch.optional<" + elementTypeAsm + ">";
  MlirType t = mlirTypeParseGet(context_, toMlirStringRef(typeAsm));
  if (mlirTypeIsNull(t)) {
    std::string msg =
        "Unparseable torch type (MLIR asm format bug?): " + typeAsm;
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
  optional_type_map_[elementTypeAsm] = t;
  return t;
}

FailureOr<std::string>
ContextCache::GetListElementTypeAsm(const onnx::TypeProto &tp) {
  if (tp.has_tensor_type()) {
    const onnx::TypeProto_Tensor &tt = tp.tensor_type();
    if (tt.has_elem_type() && tt.elem_type()) {
      MlirType elementType = ConvertTensorElementType(tt.elem_type());
      assert(!mlirTypeIsNull(elementType));
      std::vector<std::string> dims;
      if (tt.has_shape()) {
        dims.reserve(tt.shape().dim_size());
        for (const onnx::TensorShapeProto::Dimension &dim : tt.shape().dim()) {
          if (dim.has_dim_value()) {
            dims.push_back(std::to_string(dim.dim_value()));
          } else {
            dims.push_back("?");
          }
        }
      }
      return "vtensor<[" + StringJoin(dims, ",") + "]," +
             getMlirAsm(elementType) + ">";
    }
  }

  std::string msg = "Unsupported list element type.";
  model_info_.SetError(std::move(msg));
  return failure;
}

FailureOr<std::string>
ContextCache::GetOptionalElementTypeAsm(const onnx::TypeProto &tp) {
  if (tp.has_tensor_type()) {
    const onnx::TypeProto_Tensor &tt = tp.tensor_type();
    if (tt.has_elem_type()) {
      MlirType elementType = ConvertTensorElementType(tt.elem_type());
      assert(!mlirTypeIsNull(elementType));
      std::vector<std::string> dims;
      assert(tt.has_shape());
      dims.reserve(tt.shape().dim_size());
      for (const onnx::TensorShapeProto::Dimension &dim : tt.shape().dim()) {
        if (dim.has_dim_value()) {
          dims.push_back(std::to_string(dim.dim_value()));
        } else {
          dims.push_back("?");
        }
      }
      return "vtensor<[" + StringJoin(dims, ",") + "]," +
             getMlirAsm(elementType) + ">";
    }
  } else if (tp.has_sequence_type()) {
    const onnx::TypeProto_Sequence &st = tp.sequence_type();
    if (st.has_elem_type()) {
      auto elementTypeAsm = GetListElementTypeAsm(st.elem_type());
      if (failed(elementTypeAsm))
        return failure;
      return "list<" + *elementTypeAsm + ">";
    }
  }

  std::string msg = "Unsupported optional element type.";
  model_info_.SetError(std::move(msg));
  return failure;
}

MlirType ContextCache::GetVtensorType(const std::vector<int64_t> &dims,
                                      MlirType elementType) {

  VTensorSign key = {dims, elementType};

  auto it = vtensor_type_map_.find(key);
  if (it != vtensor_type_map_.end()) {
    return it->second;
  }

  std::vector<std::string> strDims;
  strDims.reserve(dims.size());
  for (const int64_t &dim : dims) {
    if (dim == -1) {
      strDims.push_back("?");
    } else {
      strDims.push_back(std::to_string(dim));
    }
  }

  std::string typeAsm = "!torch.vtensor<[" + StringJoin(strDims, ",") + "]," +
                        getMlirAsm(elementType) + ">";
  MlirType t = mlirTypeParseGet(context_, toMlirStringRef(typeAsm));
  if (mlirTypeIsNull(t)) {
    std::string msg =
        "Unparseable torch type (MLIR asm format bug?): " + typeAsm;
    model_info_.SetError(std::move(msg));
    return {nullptr};
  }
  vtensor_type_map_[key] = t;
  return t;
}

MlirType
ContextCache::ConvertTensorProtoToVtensorType(const onnx::TensorProto &tp) {
  assert(tp.has_data_type());
  MlirType elementType = ConvertTensorElementType(tp.data_type());
  if (mlirTypeIsNull(elementType))
    return {nullptr};

  std::vector<int64_t> dims;
  dims.reserve(tp.dims_size());
  for (int64_t dim : tp.dims()) {
    dims.push_back(dim);
  }

  return GetVtensorType(dims, elementType);
}

MlirType
ContextCache::ConvertTensorProtoToBuiltinType(const onnx::TensorProto &tp) {
  assert(tp.has_data_type());
  MlirType elementType = ConvertTensorElementType(tp.data_type());
  if (mlirTypeIsNull(elementType))
    return {nullptr};

  std::vector<int64_t> dims;
  dims.reserve(tp.dims_size());
  for (int64_t dim : tp.dims()) {
    dims.push_back(dim);
  }

  return mlirRankedTensorTypeGet(dims.size(), dims.data(), elementType,
                                 /*encoding=*/mlirAttributeGetNull());
}

MlirType ContextCache::ConvertTypeProto(const onnx::TypeProto *ptrTp) {
  if (ptrTp == nullptr) {
    std::cerr << "WARNING: Found a node without a valid type proto. Consider "
                 "updating the opset_version of"
                 " the model and/or running the importer with the flag "
                 "'--clear-domain'.\n";
    return GetNoneType();
  }
  const onnx::TypeProto &tp = *ptrTp;
  if (tp.has_tensor_type()) {
    // Convert Tensor TypeProto.
    const onnx::TypeProto_Tensor &tt = tp.tensor_type();

    // NOTE: Python onnx_importer.py has a check here that tt.shape is not None.
    //       However this is never the case, as, when not specified, shape is an
    //       empty TensorShapeProto, thus the check can be removed.
    //
    // if (!tt.has_shape()) {
    //   std::string msg =
    //       "Unsupported Tensor type without shape (run shape inference?): ";
    //   msg.append(tp.DebugString());
    //   model_info_.SetError(std::move(msg));
    //   return {nullptr};
    // }

    assert(tt.has_elem_type());
    MlirType elementType = ConvertTensorElementType(tt.elem_type());
    if (mlirTypeIsNull(elementType)) {
      return {nullptr};
    }
    std::vector<int64_t> dims;
    if (tt.has_shape())
      for (const onnx::TensorShapeProto::Dimension &dim : tt.shape().dim()) {
        // NOTE: dynamic dimension can either be denoted by d.dim_param being
        // set or by neither d.dim_value nor d.dim_param being set. Also note
        // that d.dim_value being 0 corresponds to the protobuf default
        // when the field is not set.
        if (dim.has_dim_value()) {
          dims.push_back(dim.dim_value());
        } else {
          dims.push_back(-1);
        }
      }

    return GetVtensorType(dims, elementType);
  } else if (tp.has_sequence_type()) {
    const onnx::TypeProto_Sequence &st = tp.sequence_type();
    if (st.has_elem_type()) {
      auto elementTypeAsm = GetListElementTypeAsm(st.elem_type());
      if (failed(elementTypeAsm))
        return {nullptr};
      return GetListType(*elementTypeAsm);
    }
  } else if (tp.has_optional_type()) {
    const onnx::TypeProto_Optional &ot = tp.optional_type();
    if (ot.has_elem_type()) {
      auto elementTypeAsm = GetOptionalElementTypeAsm(ot.elem_type());
      if (failed(elementTypeAsm))
        return {nullptr};
      return GetOptionalType(*elementTypeAsm);
    }
  } else if (tp.value_case() == onnx::TypeProto::ValueCase::VALUE_NOT_SET) {
    // (sometime happens for unused function arguments)
    return GetNoneType();
  }
  std::string msg = "Unsupported ONNX TypeProto: ";
  msg.append(tp.DebugString());
  model_info_.SetError(std::move(msg));
  return {nullptr};
}

MlirAttribute
ContextCache::ConvertTensorProtoToAttr(const onnx::TensorProto &tp) {
  MlirType tensorType = ConvertTensorProtoToBuiltinType(tp);
  if (tp.has_raw_data()) {
    std::string sanitizedName = SanitizeNameAsIdentifier(tp.name());
    // Conveniently, DenseResourceElementsAttr shares the raw data
    // format. We just give it maximum numeric alignment.
    return mlirUnmanagedDenseResourceElementsAttrGet(
        tensorType, toMlirStringRef(sanitizedName),
        const_cast<void *>(static_cast<const void *>(tp.raw_data().data())),
        tp.raw_data().size(), /*dataAlignment=*/8, /*dataIsMutable=*/false,
        /*deleter=*/nullptr, /*userData=*/nullptr);
  } else {
    switch (tp.data_type()) {
    case onnx::TensorProto::DataType::TensorProto_DataType_FLOAT:
      return mlirDenseElementsAttrFloatGet(tensorType, tp.float_data_size(),
                                           tp.float_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_BOOL:
      // NOTE: At the time of writing there are no passing e2e tests that use
      // this. onnx-ml.proto documentation is not clear about how bools are
      // organized in an int32 buffer.
      return mlirDenseElementsAttrBoolGet(tensorType, tp.int32_data_size(),
                                          tp.int32_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT8: {
      // Special case. See proto.
      auto data = elementwiseCast<uint8_t>(
          std::span(tp.int32_data().data(), tp.int32_data_size()));
      return mlirDenseElementsAttrUInt8Get(tensorType, data.size(),
                                           data.data());
    }
    case onnx::TensorProto::DataType::TensorProto_DataType_INT8: {
      // Special case. See proto.
      auto data = elementwiseCast<int8_t>(
          std::span(tp.int32_data().data(), tp.int32_data_size()));
      return mlirDenseElementsAttrInt8Get(tensorType, data.size(), data.data());
    }
    case onnx::TensorProto::DataType::TensorProto_DataType_INT16: {
      // Special case. See proto.
      auto data = elementwiseCast<int16_t>(
          std::span(tp.int32_data().data(), tp.int32_data_size()));
      return mlirDenseElementsAttrInt16Get(tensorType, data.size(),
                                           data.data());
    }
    case onnx::TensorProto::DataType::TensorProto_DataType_INT32:
      return mlirDenseElementsAttrInt32Get(tensorType, tp.int32_data_size(),
                                           tp.int32_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_INT64:
      return mlirDenseElementsAttrInt64Get(tensorType, tp.int64_data_size(),
                                           tp.int64_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_DOUBLE:
      return mlirDenseElementsAttrDoubleGet(tensorType, tp.double_data_size(),
                                            tp.double_data().data());
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT32: {
      // Special case. See proto.
      auto data = elementwiseCast<uint32_t>(
          std::span(tp.uint64_data().data(), tp.uint64_data_size()));
      return mlirDenseElementsAttrUInt32Get(tensorType, data.size(),
                                            data.data());
    }
    case onnx::TensorProto::DataType::TensorProto_DataType_UINT64:
      return mlirDenseElementsAttrUInt64Get(tensorType, tp.uint64_data_size(),
                                            tp.uint64_data().data());

      // Intentionally unsupported: STRING
    }
  }

  std::string message =
      "Unable to convert ONNX TensorProto to MLIR attribute: ";
  message.append(tp.DebugString());
  model_info_.SetError(std::move(message));
  return {nullptr};
}

bool ContextCache::VTensorSign::operator==(const VTensorSign &rhs) const {
  return dims == rhs.dims && element_type.ptr == rhs.element_type.ptr;
}

std::size_t ContextCache::VTensorSignHash::operator()(
    const ContextCache::VTensorSign &val) const {
  auto h1 = HashCombineRange(val.dims);
  auto h2 = std::hash<MlirType>{}(val.element_type);
  return h1 ^ h2;
}

// ---------------------------------------------------------------------------//
// ModuleCache
// ---------------------------------------------------------------------------//
FailureOr<std::optional<MlirOperation>> ModuleCache::GetOperatorFunction(
    std::string_view opName, std::string_view opDomain, int opsetVersion,
    int irVersion, std::span<const onnx::TypeProto *const> inputTypeProtos,
    std::span<const onnx::TypeProto *const> outputTypeProtos,
    const onnx::NodeProto &callerNode, const Config &config) {

  auto allowlists = config.function_expansion_allowlists_by_domain;
  auto denylists = config.function_expansion_denylists_by_domain;

  std::string opNameStr(opName);
  std::string opDomainStr(opDomain);

  if (allowlists && !(allowlists->count(opDomainStr) &&
                      (*allowlists)[opDomainStr].count(opNameStr)))
    return std::optional<MlirOperation>();

  if (denylists.count(opDomainStr) && denylists[opDomainStr].count(opNameStr))
    return std::optional<MlirOperation>();

  const onnx::OpSchema *opSchema =
      onnx::OpSchemaRegistry::Schema(opNameStr, opsetVersion, opDomainStr);
  if (opSchema == nullptr) {
    std::cerr << "Schema not found: (" << opNameStr << ", " << opsetVersion
              << ", " << opDomainStr << ")";
    return failure;
  }

  int specificVersion;
  bool isContextDependent;
  {
    // The onnx::OpSchemaRegistry::Schema() lookup above should get the right
    // version of the operator definition, but the function body can change
    // slightly within a single operator version, as explained in
    // https://github.com/onnx/onnx/blob/093a8d335a66ea136eb1f16b3a1ce6237ee353ab/onnx/defs/schema.h#L1070-L1086
    // There also seem to be cases where a function goes from being not
    // context-dependent to context-dependent.
    auto f = [opsetVersion](int v) { return v <= opsetVersion; };

    // Non context-dependent
    auto ncdFNs = opSchema->function_opset_versions();
    std::vector<int> ncdFNsFiltered;
    std::copy_if(ncdFNs.begin(), ncdFNs.end(),
                 std::back_inserter(ncdFNsFiltered), f);
    std::optional<int> ncdFNVersion;
    if (!ncdFNsFiltered.empty())
      ncdFNVersion = std::ranges::max(ncdFNsFiltered);

    // Context-dependent
    auto cdFNs = opSchema->context_dependent_function_opset_versions();
    std::vector<int> cdFNsFiltered;
    std::copy_if(cdFNs.begin(), cdFNs.end(), std::back_inserter(cdFNsFiltered),
                 f);
    std::optional<int> cdFNVersion;
    if (!cdFNsFiltered.empty())
      cdFNVersion = std::ranges::max(cdFNsFiltered);

    if (!ncdFNVersion && !cdFNVersion)
      // No relevant function definition
      return std::optional<MlirOperation>();

    if (ncdFNVersion && (!cdFNVersion || *cdFNVersion < *ncdFNVersion)) {
      specificVersion = *ncdFNVersion;
      isContextDependent = false;
    } else {
      specificVersion = *cdFNVersion;
      isContextDependent = true;
    }
  }

  // This is both a key for memoization of function importing and also a
  // name mangling scheme, so it must include all information needed to
  // uniquely identify a function and anything it might be parameterized
  // over.
  std::string key;
  {
    std::ostringstream keyBuffer;
    std::vector<std::string> inputTypeProtosStr, outputTypeProtosStr;
    inputTypeProtosStr.reserve(inputTypeProtos.size());
    outputTypeProtosStr.reserve(outputTypeProtos.size());
    for (const onnx::TypeProto *tp : inputTypeProtos)
      inputTypeProtosStr.push_back(tp->DebugString());
    // Though output types can be inferred from input types, it does
    // not seem to be the case that there's only one legal set of
    // outputs for a given set of inputs. When attemtping to always
    // use onnx.shape_inference.infer_function_output_types instead
    // of the caller-provided types, sometimes IR verification fails
    for (const onnx::TypeProto *tp : outputTypeProtos)
      outputTypeProtosStr.push_back(tp->DebugString());

    keyBuffer << "('" << opName << "', '" << opDomain << "', " << opsetVersion
              << ", [" << StringJoin(inputTypeProtosStr, ",") << "], ["
              << StringJoin(outputTypeProtosStr, ",") << "], ";
    if (isContextDependent) {
      keyBuffer << callerNode.DebugString();
    } else {
      const google::protobuf::Descriptor *nodeDesc = callerNode.GetDescriptor();
      const google::protobuf::FieldDescriptor *attrDesc =
          nodeDesc->FindFieldByName("attribute");
      std::vector<std::string> attrStrs;
      attrStrs.reserve(callerNode.attribute_size());
      for (int idx = 0; idx < callerNode.attribute_size(); ++idx) {
        std::string attribute_str;
        google::protobuf::TextFormat::PrintFieldValueToString(
            callerNode, attrDesc, idx, &attribute_str);
        attrStrs.push_back(std::move(attribute_str));
      }
      keyBuffer << "[" << StringJoin(attrStrs, ",") << "]";
    }
    keyBuffer << ")";
    key = keyBuffer.str();
  }

  auto it = operator_function_map_.find(key);
  if (it != operator_function_map_.end()) {
    return std::optional<MlirOperation>(it->second);
  }

  onnx::ModelProto tmpModelProto;
  if (isContextDependent) {
    FailureOr<const onnx::FunctionProto> funProto =
        GetCDFunctionWithOpsetVersion(opSchema, specificVersion, callerNode,
                                      inputTypeProtos);
    if (failed(funProto)) {
      std::cerr << "Function lookup for " << opNameStr << "/" << opDomainStr
                << "/" << specificVersion << "/" << isContextDependent
                << "failed unexpectedly. This probably indicates a bug.";
      return failure;
    }
    tmpModelProto = SpecializeFunctionAndCreateModel(
        *funProto, opSchema, key, irVersion, inputTypeProtos, outputTypeProtos,
        callerNode);
  } else {

    tmpModelProto = SpecializeFunctionAndCreateModel(
        *opSchema->GetFunction(specificVersion), opSchema, key, irVersion,
        inputTypeProtos, outputTypeProtos, callerNode);
  }

  ModelInfo tmpModelInfo(std::move(tmpModelProto), config);
  Status initS = tmpModelInfo.Initialize();
  if (failed(initS))
    return failure;

  // Mark function as private so it will be thrown away after inlining
  FailureOr<NodeImporter> imp =
      NodeImporter::DefineFunction(tmpModelInfo.GetMainGraph(), m_, cc_, *this,
                                   /*is_private=*/true);
  if (failed(imp))
    return failure;

  Status importS = imp->ImportAll();
  if (failed(importS))
    return failure;

  MlirOperation &funcOp = imp->GetParentOp();
  operator_function_map_[key] = funcOp;
  return std::optional<MlirOperation>(funcOp);
}

// ---------------------------------------------------------------------------//
// NodeImporter
// ---------------------------------------------------------------------------//
NodeImporter::NodeImporter(GraphInfo &graphInfo, MlirOperation parentOp,
                           MlirBlock block, ContextCache &cc,
                           MlirOperation moduleOp, ModuleCache &moduleCache)
    : context_(mlirOperationGetContext(parentOp)), cc_(cc),
      module_op_(std::move(moduleOp)), mc_(moduleCache), graph_info_(graphInfo),
      parent_op_(std::move(parentOp)), body_block_(std::move(block)),
      empty_type_proto_(std::make_unique<onnx::TypeProto>()) {}

FailureOr<NodeImporter> NodeImporter::DefineFunction(GraphInfo &graphInfo,
                                                     MlirOperation moduleOp,
                                                     ContextCache &contextCache,
                                                     ModuleCache &moduleCache,
                                                     bool isPrivate) {

  MlirContext context = mlirOperationGetContext(moduleOp);
  std::string locName = "graph:" + graphInfo.GetGraphProto().name();
  MlirLocation defaultLoc =
      mlirLocationNameGet(context, toMlirStringRef(locName),
                          /*childLoc=*/{nullptr});

  MlirRegion moduleBodyRegion = mlirOperationGetRegion(moduleOp, 0);
  MlirBlock moduleBody = mlirRegionGetFirstBlock(moduleBodyRegion);

  std::string funcName = graphInfo.GetGraphProto().name();

  MlirAttribute funcNameAttr =
      mlirStringAttrGet(context, toMlirStringRef(funcName));

  // Derive the FunctionType.
  std::vector<MlirType> inputTypes;
  std::vector<MlirLocation> inputLocs;
  std::vector<MlirType> outputTypes;
  for (const auto &input : graphInfo.GetInputMap()) {
    MlirType t = contextCache.ConvertTypeProto(&input.second.type());
    if (mlirTypeIsNull(t)) {
      return failure;
    }
    inputTypes.push_back(t);
    inputLocs.push_back(mlirLocationNameGet(context,
                                            toMlirStringRef(input.first),
                                            /*childLoc=*/{nullptr}));
  }
  for (const auto &output : graphInfo.GetOutputMap()) {
    MlirType t = contextCache.ConvertTypeProto(&output.second.type());
    if (mlirTypeIsNull(t)) {
      return failure;
    }
    outputTypes.push_back(t);
  }
  MlirType ftype =
      mlirFunctionTypeGet(context, inputTypes.size(), inputTypes.data(),
                          outputTypes.size(), outputTypes.data());

  // Create func.func.
  MlirOperation funcOp;
  if (!isPrivate) {
    funcOp = createMlirOperationAtEnd(
        moduleBody, "func.func", defaultLoc, mlirRegionCreate(),
        toMlirNamedAttribute("function_type", mlirTypeAttrGet(ftype)),
        toMlirNamedAttribute("sym_name", funcNameAttr));
  } else {
    funcOp = createMlirOperationAtEnd(
        moduleBody, "func.func", defaultLoc, mlirRegionCreate(),
        toMlirNamedAttribute("function_type", mlirTypeAttrGet(ftype)),
        toMlirNamedAttribute("sym_name", funcNameAttr),
        toMlirNamedAttribute(
            "sym_visibility",
            mlirStringAttrGet(context, toMlirStringRef("private"))));
  }

  // Add entry block.
  MlirBlock bodyBlock =
      mlirBlockCreate(inputTypes.size(), inputTypes.data(), inputLocs.data());
  MlirRegion bodyRegion = mlirOperationGetRegion(funcOp, 0);
  mlirRegionAppendOwnedBlock(bodyRegion, bodyBlock);

  NodeImporter imp(graphInfo, funcOp, bodyBlock, contextCache, moduleOp,
                   moduleCache);

  // Map the block args to names and store for evaluation.
  {
    size_t index = 0;
    for (const auto &input : graphInfo.GetInputMap()) {
      imp.nv_map_[input.first] = mlirBlockGetArgument(bodyBlock, index);
      ++index;
    }
  }

  imp.PopulateGraphAttrs(funcOp);
  return std::move(imp);
}

void NodeImporter::PopulateGraphAttrs(MlirOperation containerOp) {
  const onnx::ModelProto &m = graph_info_.GetModelInfo().GetModelProto();
  MlirContext containerOpContext = mlirOperationGetContext(containerOp);
  MlirType i64Type = mlirIntegerTypeSignedGet(containerOpContext, 64);
  int defaultOpsetVersion = 0;
  std::unordered_map<std::string_view, MlirAttribute> opsetVersions;
  // Determine model level opset versions.
  for (const onnx::OperatorSetIdProto &opsetImport : m.opset_import()) {
    if (opsetImport.has_domain() && opsetImport.domain() != "") {
      opsetVersions[opsetImport.domain()] =
          mlirIntegerAttrGet(i64Type, opsetImport.version());
    } else {
      defaultOpsetVersion = opsetImport.version();
    }
  }

  // Set the default domain version.
  if (defaultOpsetVersion != 0) {
    mlirOperationSetDiscardableAttributeByName(
        containerOp, toMlirStringRef("torch.onnx_meta.opset_version"),
        mlirIntegerAttrGet(i64Type, defaultOpsetVersion));
  }

  // Set versions for other domains.
  if (!opsetVersions.empty()) {
    std::vector<MlirNamedAttribute> versionAttrs;
    for (const auto &it : opsetVersions) {
      versionAttrs.push_back(mlirNamedAttributeGet(
          mlirIdentifierGet(containerOpContext, toMlirStringRef(it.first)),
          it.second));
    }
    MlirAttribute dictAttr = mlirDictionaryAttrGet(
        containerOpContext, versionAttrs.size(), versionAttrs.data());
    mlirOperationSetDiscardableAttributeByName(
        containerOp, toMlirStringRef("torch.onnx_meta.opset_versions"),
        dictAttr);
  }

  // IR version and producer.
  mlirOperationSetDiscardableAttributeByName(
      containerOp, toMlirStringRef("torch.onnx_meta.ir_version"),
      mlirIntegerAttrGet(i64Type, m.ir_version()));
  mlirOperationSetDiscardableAttributeByName(
      containerOp, toMlirStringRef("torch.onnx_meta.producer_name"),
      mlirStringAttrGet(containerOpContext,
                        toMlirStringRef(m.producer_name())));
  mlirOperationSetDiscardableAttributeByName(
      containerOp, toMlirStringRef("torch.onnx_meta.producer_version"),
      mlirStringAttrGet(containerOpContext,
                        toMlirStringRef(m.producer_version())));
}

Status NodeImporter::ImportAll(bool func) {
  // NOTE: TODO from onnx_importer.py: Consider pulling in initializers on
  // demand since there can be so much unused crap.
  for (const auto &it : graph_info_.GetInitializerMap()) {
    if (failed(ImportInitializer(it.second.first)))
      return failure;
  }

  // NOTE: present in the python version. Unclear what's the purpose. It adds an
  // unused none torch constant to every MLIR output.
  [[maybe_unused]] auto none = GetNone();

  for (const onnx::NodeProto &node : graph_info_.GetGraphProto().node()) {
    if (failed(ImportNode(node)))
      return failure;
  }

  // Lookup the outputs, which should all be in the nv_map if the graph was
  // properly formed.
  std::vector<MlirValue> outputValues;
  for (const auto &output : graph_info_.GetOutputMap()) {
    std::string_view name = output.first;
    auto foundIt = nv_map_.find(name);
    if (foundIt == nv_map_.end()) {
      std::string msg = "Non topologically produced ONNX graph output '" +
                        std::string(name) + "'";
      return SetError(std::move(msg));
    }
    outputValues.push_back(foundIt->second);
  }
  if (func) {
    createMlirOperationAtEnd(body_block_, "func.return",
                             mlirLocationUnknownGet(context_), outputValues);
  } else {
    createMlirOperationAtEnd(body_block_, "torch.operator_terminator",
                             mlirLocationUnknownGet(context_), outputValues);
  }
  return success;
}

MlirValue NodeImporter::GetNone() {
  auto foundIt = nv_map_.find("");
  if (foundIt != nv_map_.end()) {
    return foundIt->second;
  }

  MlirLocation loc =
      mlirLocationNameGet(context_, toMlirStringRef("onnx_importer.none"),
                          /*childLoc=*/{nullptr});
  MlirOperation noneOp = createMlirOperationAtEnd(
      body_block_, "torch.constant.none", loc, cc_.GetNoneType());
  MlirValue none = mlirOperationGetResult(noneOp, 0);
  nv_map_[""] = none;
  return none;
}

Status NodeImporter::ImportNode(const onnx::NodeProto &node) {
  std::string_view opType = node.op_type();
  // Handle special-form op types that do not go down the generic path.
  if (opType == "Constant") {
    // Special case only for constants specified by value attribute (for now)
    const onnx::AttributeProto *valueProto = GetValueProto(node);
    if (valueProto) {
      return ImportConstantNodeValueAttr(node);
    }
  }

  return ImportGeneralNode(node);
}

Status NodeImporter::ImportGeneralNode(const onnx::NodeProto &node) {
  MlirLocation loc = mlirLocationNameGet(context_, toMlirStringRef(node.name()),
                                         /*childLoc=*/{nullptr});
  const std::string &opType = node.op_type();
  const std::string &opDomain = node.domain();

  // Map inputs to values.
  std::vector<MlirValue> inputValues;
  std::vector<const onnx::TypeProto *> inputTypeProtos;
  for (const std::string &inputName : node.input()) {
    auto foundIt = nv_map_.find(inputName);
    if (foundIt == nv_map_.end()) {
      std::string msg =
          "Non topologically produced ONNX node input '" + inputName + "'";
      return SetError(std::move(msg));
    }
    inputValues.push_back(foundIt->second);
    const onnx::TypeProto *tp = graph_info_.FindTypeProtoForName(inputName);
    // Missing optional arguments will have empty types
    inputTypeProtos.push_back(tp != nullptr ? tp : GetEmptyTypeProto());
  }

  // Map outputs to types.
  std::vector<MlirType> outputTypes;
  std::vector<const onnx::TypeProto *> outputTypeProtos;
  for (const std::string &outputName : node.output()) {
    const onnx::TypeProto *tp = graph_info_.FindTypeProtoForName(outputName);
    // Unreferenced outputs will have empty types
    outputTypeProtos.push_back(tp != nullptr ? tp : GetEmptyTypeProto());
    MlirType t = cc_.ConvertTypeProto(tp);
    if (mlirTypeIsNull(t))
      return failure;
    outputTypes.push_back(t);
  }

  int64_t opsetVersion = 0;
  for (const onnx::OperatorSetIdProto &opsetImport :
       graph_info_.GetModelInfo().GetModelProto().opset_import()) {
    if (opsetImport.domain() == opDomain) {
      opsetVersion = opsetImport.version();
    }
  }
  if (!opsetVersion) {
    std::string msg = "Op domain not found in model's opset_import: '" +
                      opDomain + "' for op: '" + opType + "'";
    return SetError(std::move(msg));
  }

  auto operatorFuncOp = mc_.GetOperatorFunction(
      opType, opDomain, opsetVersion,
      graph_info_.GetModelInfo().GetModelProto().ir_version(), inputTypeProtos,
      outputTypeProtos, node, graph_info_.GetModelInfo().GetConfig());
  if (failed(operatorFuncOp)) {
    return failure;
  }

  MlirOperation customOp;
  if (*operatorFuncOp != std::nullopt) {
    MlirAttribute symName = mlirOperationGetAttributeByName(
        **operatorFuncOp, toMlirStringRef("sym_name"));
    customOp = createMlirOperationAtEnd(
        body_block_, "func.call", loc, outputTypes, inputValues,
        toMlirNamedAttribute("callee",
                             mlirFlatSymbolRefAttrGet(
                                 context_, mlirStringAttrGetValue(symName))));
  } else {
    // Derive the op name.
    std::string opName = "onnx." + node.op_type();
    MlirAttribute opNameAttr =
        mlirStringAttrGet(context_, toMlirStringRef(opName));

    // General attributes.
    auto generalAttrs = ImportGeneralAttributes(node.attribute());
    if (failed(generalAttrs)) {
      return failure;
    }

    uint32_t numRegions = CountRegions(node.attribute());
    std::vector<MlirRegion> regions =
        FillVector(numRegions, std::function<MlirRegion()>(mlirRegionCreate));

    customOp = createMlirOperationAtEnd(
        body_block_, "torch.operator", loc, outputTypes, inputValues,
        toMlirNamedAttribute("name", opNameAttr), *generalAttrs, regions);

    if (failed(ImportRegions(node.attribute(), customOp)))
      return failure;
  }

  // Record the result values.
  for (int index = 0; index < node.output().size(); ++index) {
    MlirValue result = mlirOperationGetResult(customOp, index);
    std::string_view name = node.output()[index];
    nv_map_[name] = result;
  }

  return success;
}

FailureOr<std::vector<std::pair<std::string, MlirAttribute>>>
NodeImporter::ImportGeneralAttributes(const onnx::AttrList &attrs) {
  std::vector<std::pair<std::string, MlirAttribute>> generalAttrs;

  // Mapping of AttributeType code to one of:
  //   std::nullopt: Ignore attribute and do not output to MLIR
  //   failure: Error if an attribute of this type is present
  //   MlirAttribute: Return MLIR attribute from onnx attribute
  auto handleAttribute = [this](const onnx::AttributeProto &onnxAttr)
      -> FailureOr<std::optional<MlirAttribute>> {
    onnx::AttributeProto_AttributeType type = onnxAttr.type();
    switch (type) {
    case onnx::AttributeProto::UNDEFINED:
      return failure;
    case onnx::AttributeProto::FLOAT:
      return {mlirFloatAttrDoubleGet(context_, mlirF32TypeGet(context_),
                                     onnxAttr.f())};
    case onnx::AttributeProto::INT:
      return {mlirIntegerAttrGet(mlirIntegerTypeSignedGet(context_, 64),
                                 onnxAttr.i())};
    case onnx::AttributeProto::STRING:
      return {mlirStringAttrGet(context_, toMlirStringRef(onnxAttr.s()))};
    case onnx::AttributeProto::TENSOR: {
      MlirAttribute attr = cc_.ConvertTensorProtoToAttr(onnxAttr.t());
      if (mlirAttributeIsNull(attr))
        return failure;
      return {attr};
    }
    case onnx::AttributeProto::GRAPH:
      return {std::nullopt};
    case onnx::AttributeProto::SPARSE_TENSOR:
      return failure;
    case onnx::AttributeProto::TYPE_PROTO:
      return failure;
    case onnx::AttributeProto::FLOATS: {
      std::vector<MlirAttribute> attrs;
      for (auto f : onnxAttr.floats())
        attrs.push_back(
            mlirFloatAttrDoubleGet(context_, mlirF32TypeGet(context_), f));
      return {mlirArrayAttrGet(context_, attrs.size(), attrs.data())};
    }
    case onnx::AttributeProto::INTS: {
      std::vector<MlirAttribute> attrs;
      for (auto i : onnxAttr.ints())
        attrs.push_back(
            mlirIntegerAttrGet(mlirIntegerTypeSignedGet(context_, 64), i));
      return {mlirArrayAttrGet(context_, attrs.size(), attrs.data())};
    }
    case onnx::AttributeProto::STRINGS: {
      std::vector<MlirAttribute> attrs;
      for (auto &s : onnxAttr.strings())
        attrs.push_back(mlirStringAttrGet(context_, toMlirStringRef(s)));
      return {mlirArrayAttrGet(context_, attrs.size(), attrs.data())};
    }
    case onnx::AttributeProto::TENSORS: {
      std::vector<MlirAttribute> attrs;
      for (auto &t : onnxAttr.tensors()) {
        MlirAttribute attr = cc_.ConvertTensorProtoToAttr(t);
        if (mlirAttributeIsNull(attr))
          return failure;
        attrs.push_back(attr);
      }
      return {mlirArrayAttrGet(context_, attrs.size(), attrs.data())};
    }
    case onnx::AttributeProto::GRAPHS:
      return failure;
    case onnx::AttributeProto::SPARSE_TENSORS:
      return failure;
    case onnx::AttributeProto::TYPE_PROTOS:
      return failure;
    }

    std::string msg = "Unhandled ONNX attribute type code ";
    msg.append(std::to_string(onnxAttr.type()));
    msg.append(": ");
    msg.append(onnxAttr.DebugString());
    SetError(std::move(msg));
    return failure;
  };

  for (const onnx::AttributeProto &onnxAttr : attrs) {
    auto res = handleAttribute(onnxAttr);

    if (failed(res)) {
      // Active error.
      // Try matching attribute type ID to name for a more descriptive error
      // message.
      auto attrType = onnxAttr.type();
      std::string attrTypeName =
          onnx::AttributeProto_AttributeType_Name(attrType);
      if (attrTypeName.empty())
        attrTypeName = "UNKNOWN";
      SetError("ONNX importer does not support generic node attribute type " +
               attrTypeName + " with ID " + std::to_string(attrType) +
               ". This likely means that this is a special node which requires "
               "specific handling in the importer: " +
               onnxAttr.DebugString());
      return failure;
    } else if (*res == std::nullopt) {
      // Active skip
      continue;
    }

    std::string fullName = "torch.onnx." + onnxAttr.name();
    generalAttrs.push_back(std::make_pair(fullName, **res));
  }
  return generalAttrs;
}

Status NodeImporter::ImportRegions(
    const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &onnxAttrs,
    MlirOperation op) {
  std::vector<std::reference_wrapper<const onnx::AttributeProto>> graphAttrs;
  graphAttrs.reserve(onnxAttrs.size());
  for (const onnx::AttributeProto &attr : onnxAttrs)
    if (attr.type() == onnx::AttributeProto::GRAPH)
      graphAttrs.push_back(attr);
  std::sort(graphAttrs.begin(), graphAttrs.end(),
            [](const onnx::AttributeProto &a, const onnx::AttributeProto &b) {
              return a.name() < b.name();
            });
  for (size_t index = 0; index < graphAttrs.size(); ++index) {
    const onnx::ValueInfoList &gInput = graphAttrs[index].get().g().input();
    std::vector<MlirType> blockTypes;
    blockTypes.reserve(gInput.size());
    for (const onnx::ValueInfoProto &vi : gInput)
      blockTypes.push_back(
          cc_.ConvertTypeProto(vi.has_type() ? &vi.type() : nullptr));
    if (std::any_of(blockTypes.cbegin(), blockTypes.cend(),
                    [](const MlirType &t) { return mlirTypeIsNull(t); }))
      return failure;
    MlirRegion region = mlirOperationGetRegion(op, index);
    std::vector<MlirLocation> blockLocs =
        FillVector(blockTypes.size(), std::function<MlirLocation()>([&op]() {
                     return mlirOperationGetLocation(op);
                   }));

    mlirRegionAppendOwnedBlock(region, mlirBlockCreate(blockTypes.size(),
                                                       blockTypes.data(),
                                                       blockLocs.data()));
    MlirBlock block = mlirRegionGetFirstBlock(region);

    GraphInfo graphInfo(graph_info_.GetModelInfo(), graphAttrs[index].get().g(),
                        /*top_level=*/false);
    Status initS = graphInfo.Initialize();
    if (failed(initS))
      return failure;
    NodeImporter importer(graphInfo, op, block, cc_, module_op_, mc_);

    std::vector<std::string_view> blockNames;
    blockNames.reserve(gInput.size());
    for (const onnx::ValueInfoProto &vi : gInput)
      blockNames.push_back(vi.name());
    assert(static_cast<size_t>(mlirBlockGetNumArguments(block)) ==
           blockNames.size());
    for (size_t blockIdx = 0; blockIdx < blockNames.size(); ++blockIdx) {
      MlirValue inputValue = mlirBlockGetArgument(block, blockIdx);
      auto inserted =
          importer.nv_map_.emplace(blockNames[blockIdx], inputValue);
      if (!inserted.second) {
        std::string msg = "Block argument name collision: '";
        msg.append(blockNames[blockIdx]);
        return SetError(std::move(msg));
      }
    }
    for (const auto &pair : nv_map_) {
      auto inserted = importer.nv_map_.insert(pair);
      if (!inserted.second) {
        std::string msg = "Outer value collision in block: '";
        msg.append(pair.first);
        return SetError(std::move(msg));
      }
    }

    if (failed(importer.ImportAll(/*func=*/false)))
      return failure;
  }
  return success;
}

Status
NodeImporter::ImportInitializer(const onnx::TensorProto &initializer,
                                std::optional<std::string_view> externName) {
  // If an explicitly specified name is given, use that; otherwise, pick
  // up the name from the tensor proto itself
  std::string_view name = externName ? *externName : initializer.name();
  MlirLocation loc = mlirLocationNameGet(context_, toMlirStringRef(name),
                                         /*childLoc=*/{nullptr});

  MlirAttribute valueAttr = cc_.ConvertTensorProtoToAttr(initializer);
  MlirType vtensorType = cc_.ConvertTensorProtoToVtensorType(initializer);
  if (mlirAttributeIsNull(valueAttr) || mlirTypeIsNull(vtensorType))
    return failure;

  MlirOperation op = createMlirOperationAtEnd(
      body_block_, "torch.operator", loc, vtensorType,
      toMlirNamedAttribute(
          "name",
          mlirStringAttrGet(context_, toMlirStringRef("onnx.Constant"))),
      toMlirNamedAttribute("torch.onnx.value", valueAttr));
  MlirValue result = mlirOperationGetResult(op, 0);

  auto inserted = nv_map_.insert(std::make_pair(name, result));
  if (!inserted.second) {
    std::string msg = "Multiple nodes produced a value for '";
    msg.append(name);
    msg.append("', most recent from ");
    msg.append(initializer.DebugString());
    return SetError(std::move(msg));
  }

  return success;
}

void NodeImporter::WriteModule(std::ostream *stream, bool assumeVerified) {
  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  if (assumeVerified) {
    mlirOpPrintingFlagsAssumeVerified(flags);
  }
  auto callback = +[](MlirStringRef sr, void *s) {
    std::ostream *stream = static_cast<std::ostream *>(s);
    stream->write(sr.data, sr.length);
  };
  mlirOperationPrintWithFlags(module_op_, flags, callback,
                              static_cast<void *>(stream));
  mlirOpPrintingFlagsDestroy(flags);
  *stream << "\n";
}

Status NodeImporter::ImportConstantNodeValueAttr(const onnx::NodeProto &node) {
  const onnx::AttributeProto *valueProto = GetValueProto(node);

  // Produce an initializer for the constant, so that it can be used in
  // combination with other ops, such as ConstantOfShape, requiring
  // constant input.
  if (valueProto->type() != onnx::AttributeProto_AttributeType_TENSOR) {
    return SetError("Constant node must have a tensor value attribute");
  }
  if (node.output_size() != 1) {
    return SetError("Constant node must have one output");
  }
  std::string_view constName = node.output(0);
  if (failed(ImportInitializer(valueProto->t(), constName)))
    return failure;

  if (graph_info_.GetInitializerMap().find(constName) !=
      graph_info_.GetInitializerMap().end()) {
    return SetError("ONNX initializer name already present: " +
                    std::string(constName));
  }
  graph_info_.InitializerMapEmplace(constName, valueProto->t());
  return success;
}

OnnxImporter::MlirState::MlirState() {
  context_ = mlirContextCreateWithThreading(false);
  torchMlirRegisterAllDialects(context_);
  module_ = mlirModuleCreateEmpty(mlirLocationUnknownGet(context_));
}

OnnxImporter::MlirState::~MlirState() {
  mlirModuleDestroy(module_);
  mlirContextDestroy(context_);
}

Status OnnxImporter::Import(onnx::ModelProto &&modelProto,
                            std::ostream *outputStream, const Config config) {

  // Parse the model proto.
  ModelInfo modelInfo(std::move(modelProto), config);

  if (failed(modelInfo.Initialize())) {
    std::cerr << "error: Import failure: " << modelInfo.GetErrorMessage()
              << "\n";
    modelInfo.DebugDumpProto();
    return failure;
  }

  MlirState s;
  MlirOperation mOp = mlirModuleGetOperation(s.module_);

  ContextCache cc(modelInfo, mlirOperationGetContext(mOp));
  ModuleCache mc(mOp, cc);
  auto importer =
      NodeImporter::DefineFunction(modelInfo.GetMainGraph(), mOp, cc, mc);

  if (failed(importer)) {
    std::cerr << "error: Could not define MLIR function for graph: "
              << modelInfo.GetErrorMessage() << "\n";
    return failure;
  }
  if (failed(importer->ImportAll())) {
    std::cerr << "error: Could not import one or more graph nodes: "
              << modelInfo.GetErrorMessage() << "\n";
    return failure;
  }

  if (!config.no_verify) {
    if (!mlirOperationVerify(mOp)) {
      std::cerr << "error: Module op doesn't verify.\n";
      return failure;
    }
  }

  importer->WriteModule(outputStream, !config.no_verify);
  return success;
}
