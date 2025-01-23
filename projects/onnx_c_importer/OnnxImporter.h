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

/// A light-weight status. It only encapsulates success/failure.
/// Full error information will be set on the ModelInfo.
class Status {
public:
  static Status success(bool isSuccess = true) { return Status(isSuccess); }
  static Status failure(bool isFailure = true) { return Status(!isFailure); }

  bool is_success() { return is_success_; }

private:
  Status(bool is_success) : is_success_(is_success) {}
  bool is_success_;
};

static inline bool succeeded(Status status) { return status.is_success(); }
static inline bool failed(Status status) { return !status.is_success(); }

// (inspired by std::nullopt_t)
struct FailureT {
  enum class _Construct { _Token };

  explicit constexpr FailureT(_Construct) noexcept {}

  operator Status() const { return Status::failure(); }
};

inline constexpr FailureT failure{FailureT::_Construct::_Token};

// (inspired by std::nullopt_t)
struct SuccessT {
  enum class _Construct { _Token };

  explicit constexpr SuccessT(_Construct) noexcept {}

  operator Status() const { return Status::success(); }
};

inline constexpr SuccessT success{SuccessT::_Construct::_Token};

// (see llvm::FailureOr)
template <typename T> class [[nodiscard]] FailureOr : public std::optional<T> {
public:
  FailureOr(FailureT) : std::optional<T>() {}
  FailureOr() : FailureOr(failure) {}
  FailureOr(T &&Y) : std::optional<T>(std::forward<T>(Y)) {}
  FailureOr(const T &Y) : std::optional<T>(Y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  FailureOr(const FailureOr<U> &Other)
      : std::optional<T>(failed(Other) ? std::optional<T>()
                                       : std::optional<T>(*Other)) {}

  operator Status() const { return Status::success(has_value()); }

private:
  /// Hide the bool conversion as it easily creates confusion.
  using std::optional<T>::operator bool;
  using std::optional<T>::has_value;
};

// Accounting for a GraphProto.
class GraphInfo {
public:
  GraphInfo(ModelInfo &model_info, const onnx::GraphProto &graph_proto,
            bool top_level = true)
      : model_info_(model_info), graph_proto_(graph_proto),
        is_top_level_(top_level) {}
  ModelInfo &model_info() { return model_info_; }
  const onnx::GraphProto &graph_proto() { return graph_proto_; }

  /// Post-construction, failable initialization.
  Status Initialize();

  /// Finds a TypeProto for the given value name. If returning nullptr, then
  /// an error will have been set.
  const onnx::TypeProto *FindTypeProtoForName(std::string_view name);

  std::vector<const onnx::ValueInfoProto *> &inputs() { return inputs_; }
  const std::vector<const onnx::ValueInfoProto *> &inputs() const {
    return inputs_;
  }

  Dict<std::string_view, const onnx::ValueInfoProto &> &input_map() {
    return input_map_;
  }
  const Dict<std::string_view, const onnx::ValueInfoProto &> &
  input_map() const {
    return input_map_;
  }

  std::vector<const onnx::ValueInfoProto *> &outputs() { return outputs_; }
  const std::vector<const onnx::ValueInfoProto *> &outputs() const {
    return outputs_;
  }

  Dict<std::string_view, const onnx::ValueInfoProto &> &output_map() {
    return output_map_;
  }
  const Dict<std::string_view, const onnx::ValueInfoProto &> &
  output_map() const {
    return output_map_;
  }

  Dict<std::string_view, const onnx::TensorProto &> &initializer_map() {
    return initializer_map_;
  }
  const Dict<std::string_view, const onnx::TensorProto &> &
  initializer_map() const {
    return initializer_map_;
  }

private:
  ModelInfo &model_info_;
  const onnx::GraphProto &graph_proto_;

  Dict<std::string_view, const onnx::TensorProto &> initializer_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> value_info_map_;

  std::vector<const onnx::ValueInfoProto *> declared_inputs_;
  std::vector<const onnx::ValueInfoProto *> inputs_;
  std::vector<const onnx::ValueInfoProto *> outputs_;
  Dict<std::string_view, const onnx::ValueInfoProto &> input_map_;
  Dict<std::string_view, const onnx::ValueInfoProto &> output_map_;

  bool is_top_level_;
};

/// Top-level accounting and accessors for an ONNX model.
class ModelInfo {
public:
  ModelInfo(onnx::ModelProto &&model_proto, const Config &config)
      : config_(config), model_proto_(std::move(model_proto)) {}
  Config &config() { return config_; }
  onnx::ModelProto &model_proto() { return model_proto_; }

  /// Post-construction, failable initialization.
  Status Initialize();

  GraphInfo &main_graph() { return *main_graph_; }
  const std::string &error_message() { return error_message_; }

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
  ContextCache(ModelInfo &model_info, MlirContext context)
      : model_info_(model_info), context_(context) {}

  MlirContext context() { return context_; }

  /// Converts the ONNX element type code to an MlirType, returning a null type
  /// and setting an error if not possible.
  MlirType ConvertTensorElementType(int element_type_code);

  MlirType GetNoneType();

  MlirType GetListType(MlirType element_type);
  MlirType GetOptionalType(MlirType element_type);

  MlirType GetListElementType(const onnx::TypeProto &tp);
  MlirType GetOptionalElementType(const onnx::TypeProto &tp);

  /// Gets a !torch.vtensor type for the given dims and element type.
  /// Dynamic dims are represented as -1.
  /// If it was not possible to create the type, sets an error and returns
  /// the null type.
  MlirType GetVtensorType(const std::vector<int64_t> &dims,
                          MlirType element_type);

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
  ModuleCache(MlirOperation module_op, ContextCache &cc)
      : cc_(cc), m_(module_op) {};

  /// Get or create the MLIR function corresponding to an ONNX operator.
  /// Returns failure for ONNX operators that aren't functions.
  FailureOr<std::optional<MlirOperation>> GetOperatorFunction(
      std::string_view op_name, std::string_view op_domain, int opset_version,
      int ir_version, std::span<const onnx::TypeProto *const> input_type_protos,
      std::span<const onnx::TypeProto *const> output_type_protos,
      const onnx::NodeProto &caller_node, const Config &config);

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

  MlirOperation &ParentOp() { return parent_op_; }

  /// Called after construction to define the function in the module. Must be
  /// called prior to importing nodes.
  static FailureOr<NodeImporter> DefineFunction(GraphInfo &graphInfo,
                                                MlirOperation moduleOp,
                                                ContextCache &contextCache,
                                                ModuleCache &moduleCache,
                                                bool isPrivate = false);

  /// Imports all nodes topologically.
  Status ImportAll(bool func = true);

  void WriteModule(std::ostream *stream, bool assumeVerified);

private:
  void PopulateGraphAttrs(MlirOperation container_op);
  Status
  ImportInitializer(const onnx::TensorProto &initializer,
                    std::optional<std::string_view> extern_name = std::nullopt);
  Status ImportNode(const onnx::NodeProto &node);
  FailureOr<std::vector<std::pair<std::string, MlirAttribute>>>
  ImportGeneralAttributes(const onnx::AttrList &attrs);

  // Special-form nodes.
  Status ImportGeneralNode(const onnx::NodeProto &node);
  Status ImportConstantNodeValueAttr(const onnx::NodeProto &node);

  Status
  ImportRegions(const google::protobuf::RepeatedPtrField<onnx::AttributeProto>
                    &onnx_attrs,
                MlirOperation op);

  MlirValue GetNone();

  Status SetError(std::string msg) {
    return graph_info_.model_info().SetError(std::move(msg));
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
                       std::ostream *output_stream,
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
