//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

// Stand-alone ONNX -> MLIR importer.
// This library only depends on ONNX (and transitively protobuf, of course)
// and the MLIR C API. It does this to minimize its dependency surface area
// and make it possible to integrate as source code into other systems while
// retaining this implementation as the source of truth.
//
// It uses a hybrid of LLVM and Google C++ coding style, preferring the latter
// for class members/accessors because canonical protobuf coding presumes
// this kind of style.

#include "mlir-c/IR.h"
#include "onnx/onnx_pb.h"

#include "Dict.hpp"
#include "Status.hpp"

#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>

namespace onnx {
using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;
}

namespace torch_mlir_onnx {

template <typename T> using opt_ref = std::optional<std::reference_wrapper<T>>;

struct Config;
class GraphInfo;
class ModelInfo;

struct Config {
  // Disable verification prior to printing
  bool no_verify = false;

  // Ancient ONNX exporters would often add a model input for anything that
  // might be mutable, providing an initializer for it as well. More modern
  // tools tools realized this is a really bad idea for a lot of reasons.
  // We choose to assume more recent norms, even if encountering older
  // models. Setting this to False probably won't do what you want but
  // should produce interesting errors to waste your time deciphering.
  // We mainly use it as a way to document in the code that we are
  // making an assumption.
  bool elide_initialized_inputs = true;

  // Some ONNX operators are defined by ONNX functions and will be
  // automatically expanded (see get_operator_function() below) to MLIR
  // functions by the importer. This option allows allowlisting functions that
  // should be expanded. If this is None, then allowlisting is not used (all
  // functions not explicitly denylisted will be expanded).
  //
  // Since function expansion has not always been supported, the default should
  // be to use allowlisting, to avoid disruption.
  std::optional<std::unordered_map<std::string, std::set<std::string>>>
      function_expansion_allowlists_by_domain =
          std::unordered_map<std::string, std::set<std::string>>{
              // Default domain (ONNX built-in ops)
              {"", std::set<std::string>{"MeanVarianceNormalization"}}};

  // Some ONNX operators are defined by ONNX functions and will be
  // automatically expanded (see get_operator_function() below) to MLIR
  // functions by the importer. This option allows denylisting functions that
  // should not be expanded.
  std::unordered_map<std::string, std::set<std::string>>
      function_expansion_denylists_by_domain = {
          // Default domain (ONNX built-in ops)
          {"",
           {// CastLike's second input `target_type` is used only for its
            // type (T2), from which its output's type is inferred, but
            // because its value is unused, ONNX's shape inference doesn't
            // annotate the input value with a type, so looking up the
            // function by the provided input types will fail.
            "CastLike",
            // ONNX errors when trying to infer the type of the Loop op
            // within this function: "[ShapeInferenceError] Inferred shape
            // and existing shape differ in rank: (1) vs (0)"
            "Range"}}};
};

// Accounting for a GraphProto.
class GraphInfo {
public:
  using InitializerMapT =
      Dict<std::string_view,
           std::pair<const onnx::TensorProto &, onnx::TypeProto>>;

  GraphInfo(ModelInfo &modelInfo, const onnx::GraphProto &graphProto,
            bool topLevel = true)
      : model_info_(modelInfo), graph_proto_(graphProto),
        is_top_level_(topLevel) {}
  ModelInfo &GetModelInfo() { return model_info_; }
  const onnx::GraphProto &GetGraphProto() { return graph_proto_; }

  /// Post-construction, failable initialization.
  [[nodiscard]] Status Initialize();

  /// Finds a TypeProto for the given value name. If returning nullptr, then
  /// an error will have been set.
  const onnx::TypeProto *FindTypeProtoForName(std::string_view name);

  Dict<std::string_view, const onnx::ValueInfoProto &> &GetInputMap() {
    return input_map_;
  }
  const Dict<std::string_view, const onnx::ValueInfoProto &> &
  GetInputMap() const {
    return input_map_;
  }

  Dict<std::string_view, const onnx::ValueInfoProto &> &GetOutputMap() {
    return output_map_;
  }
  const Dict<std::string_view, const onnx::ValueInfoProto &> &
  GetOutputMap() const {
    return output_map_;
  }

  void InitializerMapEmplace(const std::string_view &name,
                             const onnx::TensorProto &tp);
  const InitializerMapT &GetInitializerMap() const { return initializer_map_; }

private:
  ModelInfo &model_info_;
  const onnx::GraphProto &graph_proto_;

  InitializerMapT initializer_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> value_info_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> declared_input_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> output_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> input_map_;

  bool is_top_level_;
};

/// Top-level accounting and accessors for an ONNX model.
class ModelInfo {
public:
  ModelInfo(onnx::ModelProto &&modelProto, const Config &config)
      : config_(config), model_proto_(std::move(modelProto)) {}
  Config &GetConfig() { return config_; }
  onnx::ModelProto &GetModelProto() { return model_proto_; }

  /// Post-construction, failable initialization.
  [[nodiscard]] Status Initialize();

  GraphInfo &GetMainGraph() { return *main_graph_; }
  const std::string &GetErrorMessage() { return error_message_; }

  Status SetError(std::string msg) {
    error_message_ = std::move(msg);
    return failure;
  }

  void DebugDumpProto();

private:
  Config config_;
  onnx::ModelProto model_proto_;
  std::unique_ptr<GraphInfo> main_graph_;

  std::string error_message_;
};

/// Caches per-context lookups of various things.
class ContextCache {
public:
  ContextCache(ModelInfo &modelInfo, MlirContext context)
      : model_info_(modelInfo), context_(context) {}

  /// Converts the ONNX element type code to an MlirType, returning a null type
  /// and setting an error if not possible.
  MlirType ConvertTensorElementType(int elemTypeCode);

  MlirType GetNoneType();

  MlirType GetListType(const std::string &elemTypeAsm);
  MlirType GetOptionalType(const std::string &elemTypeAsm);

  FailureOr<std::string> GetListElementTypeAsm(const onnx::TypeProto &tp);
  FailureOr<std::string> GetOptionalElementTypeAsm(const onnx::TypeProto &tp);

  /// Gets a !torch.vtensor type for the given dims and element type.
  /// Dynamic dims are represented as -1.
  /// If it was not possible to create the type, sets an error and returns
  /// the null type.
  MlirType GetVtensorType(const std::vector<int64_t> &dims, MlirType elemType);

  /// Converts the ONNX TensorProto to a !torch.vtensor type.
  MlirType ConvertTensorProtoToVtensorType(const onnx::TensorProto &tp);

  /// Converts the ONNX TensorProto to an Mlir RankedTensor type.
  MlirType ConvertTensorProtoToBuiltinType(const onnx::TensorProto &tp);

  /// Converts the TypeProto to an MlirType, returning a null type and
  /// setting an error if not possible.
  MlirType ConvertTypeProto(const onnx::TypeProto *tp);

  /// Converts an ONNX TensorProto to an MlirAttribute, returning a null
  /// attribute and setting an error if not possible.
  MlirAttribute ConvertTensorProtoToAttr(const onnx::TensorProto &tp);

private:
  ModelInfo &model_info_;
  MlirContext context_;

  std::unordered_map<int, MlirType> elem_type_map_;
  std::unordered_map<std::string, MlirType> list_type_map_;
  std::unordered_map<std::string, MlirType> optional_type_map_;

  struct VTensorSign {
    std::vector<int64_t> dims;
    MlirType element_type;

    bool operator==(const VTensorSign &rhs) const;
  };
  struct VTensorSignHash {
    std::size_t operator()(const VTensorSign &val) const;
  };
  std::unordered_map<VTensorSign, MlirType, VTensorSignHash> vtensor_type_map_;
};

class ModuleCache {
public:
  ModuleCache(MlirOperation moduleOp, ContextCache &cc)
      : cc_(cc), m_(moduleOp) {};

  /// Get or create the MLIR function corresponding to an ONNX operator.
  /// Returns failure for ONNX operators that aren't functions.
  FailureOr<std::optional<MlirOperation>>
  GetOperatorFunction(std::string_view opName, std::string_view opDomain,
                      int opsetVersion, int irVersion,
                      std::span<const onnx::TypeProto *const> inputTypeProtos,
                      std::span<const onnx::TypeProto *const> outputTypeProtos,
                      const onnx::NodeProto &callerNode, const Config &config);

private:
  ContextCache &cc_;
  MlirOperation m_;
  std::unordered_map<std::string, MlirOperation> operator_function_map_;
};

/// Imports graph nodes into MLIR.
///
/// Typically, the top level graph will be imported into a func whereas
/// dependent graphs may just be imported with references to pre-existing
/// values.
///
/// Note that ONNX requires that graphs be sorted topologically and free of
/// cycles, so we don't take any special steps to order them for dominance.
class NodeImporter {
public:
  NodeImporter(GraphInfo &graphInfo, MlirOperation parentOp, MlirBlock block,
               ContextCache &cc, MlirOperation moduleOp,
               ModuleCache &moduleCache);

  MlirOperation &GetParentOp() { return parent_op_; }

  /// Called after construction to define the function in the module. Must be
  /// called prior to importing nodes.
  [[nodiscard]] static FailureOr<NodeImporter>
  DefineFunction(GraphInfo &graphInfo, MlirOperation moduleOp,
                 ContextCache &contextCache, ModuleCache &moduleCache,
                 bool isPrivate = false);

  /// Imports all nodes topologically.
  [[nodiscard]] Status ImportAll(bool func = true);

  void WriteModule(std::ostream *stream, bool assumeVerified);

private:
  void PopulateGraphAttrs(MlirOperation containerOp);
  [[nodiscard]] Status
  ImportInitializer(const onnx::TensorProto &initializer,
                    std::optional<std::string_view> externName = std::nullopt);
  [[nodiscard]] Status ImportNode(const onnx::NodeProto &node);
  [[nodiscard]] FailureOr<std::vector<std::pair<std::string, MlirAttribute>>>
  ImportGeneralAttributes(const onnx::AttrList &attrs);

  // Special-form nodes.
  [[nodiscard]] Status ImportGeneralNode(const onnx::NodeProto &node);
  [[nodiscard]] Status ImportConstantNodeValueAttr(const onnx::NodeProto &node);

  [[nodiscard]] Status ImportRegions(
      const google::protobuf::RepeatedPtrField<onnx::AttributeProto> &onnxAttrs,
      MlirOperation op);

  [[nodiscard]] MlirValue GetNone();

  Status SetError(std::string msg) {
    return graph_info_.GetModelInfo().SetError(std::move(msg));
  }

  const onnx::TypeProto *GetEmptyTypeProto() const {
    return empty_type_proto_.get();
  }

  MlirContext context_;
  ContextCache &cc_;
  MlirOperation module_op_;
  ModuleCache &mc_;
  GraphInfo &graph_info_;
  MlirOperation parent_op_;
  MlirBlock body_block_;
  Dict<std::string_view, MlirValue> nv_map_;
  std::unique_ptr<const onnx::TypeProto> empty_type_proto_;
};

class OnnxImporter {
public:
  static Status Import(onnx::ModelProto &&modelProto,
                       std::ostream *outputStream,
                       const Config config = Config());

protected:
  struct MlirState {
    MlirState();
    ~MlirState();

    MlirContext context_;
    MlirModule module_;
  };
};

} // namespace torch_mlir_onnx
