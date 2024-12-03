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

#include <optional>
#include <string_view>
#include <unordered_map>

namespace torch_mlir_onnx {

struct Config;
class GraphInfo;
class ModelInfo;

struct Config {
  // Ancient ONNX exporters would often add a model input for anything that
  // might be mutable, providing an initializer for it as well. More modern
  // tools tools realized this is a really bad idea for a lot of reasons.
  // We choose to assume more recent norms, even if encountering older
  // models. Setting this to False probably won't do what you want but
  // should produce interesting errors to waste your time deciphering.
  // We mainly use it as a way to document in the code that we are
  // making an assumption.
  bool elide_initialized_inputs = true;
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

static inline Status success() { return Status::success(); }
static inline Status failure() { return Status::failure(); }
static inline bool succeeded(Status status) { return status.is_success(); }
static inline bool failed(Status status) { return !status.is_success(); }

// Accounting for a GraphProto.
class GraphInfo {
public:
  GraphInfo(ModelInfo &model_info, const onnx::GraphProto &graph_proto)
      : model_info_(model_info), graph_proto_(graph_proto) {}
  ModelInfo &model_info() { return model_info_; }
  const onnx::GraphProto &graph_proto() { return graph_proto_; }

  /// Post-construction, failable initialization.
  Status Initialize();

  /// Finds a TypeProto for the given value name. If returning nullptr, then
  /// an error will have been set.
  const onnx::TypeProto *FindTypeProtoForName(std::string_view name);

  /// Attempts to access the raw or external data of the TensorProto. If the
  /// the data is located in those positions, returns a types pointer to it
  /// and stores the number of elements to `out_size`. Otherwise, nullptr is
  /// returned (and no error is set).
  template <typename ElementType>
  const ElementType *GetOptionalRawData(const onnx::TensorProto &tp,
                                        size_t &out_size) {
    if (tp.has_raw_data()) {
      out_size = tp.raw_data().size() / sizeof(ElementType);
      return reinterpret_cast<const ElementType *>(tp.raw_data().data());
    }
    return nullptr;
  }

  std::vector<const onnx::ValueInfoProto *> &inputs() { return inputs_; }
  std::unordered_map<std::string_view, const onnx::ValueInfoProto &> &
  input_map() {
    return input_map_;
  }
  std::vector<const onnx::ValueInfoProto *> &outputs() { return outputs_; }
  std::unordered_map<std::string_view, const onnx::ValueInfoProto &> &
  output_map() {
    return output_map_;
  }

  std::unordered_map<std::string_view, const onnx::TensorProto &> &
  initializer_map() {
    return initializer_map_;
  }

private:
  ModelInfo &model_info_;
  const onnx::GraphProto &graph_proto_;

  std::unordered_map<std::string_view, const onnx::TensorProto &>
      initializer_map_;
  std::unordered_map<std::string_view, const onnx::ValueInfoProto &>
      value_info_map_;

  std::vector<const onnx::ValueInfoProto *> declared_inputs_;
  std::vector<const onnx::ValueInfoProto *> inputs_;
  std::vector<const onnx::ValueInfoProto *> outputs_;
  std::unordered_map<std::string_view, const onnx::ValueInfoProto &> input_map_;
  std::unordered_map<std::string_view, const onnx::ValueInfoProto &>
      output_map_;
};

/// Top-level accounting and accessors for an ONNX model.
class ModelInfo {
public:
  ModelInfo();
  Config &config() { return config_; }
  onnx::ModelProto &model_proto() { return model_proto_; }

  /// Post-construction, failable initialization.
  Status Initialize();

  GraphInfo &main_graph() { return *main_graph_; }
  const std::string &error_message() { return error_message_; }

  Status SetError(std::string msg) {
    error_message_ = std::move(msg);
    return failure();
  }

  void DebugDumpProto();

private:
  Config config_;
  onnx::ModelProto model_proto_;
  std::unique_ptr<GraphInfo> main_graph_;

  std::string error_message_;
};

class ContextCache {
public:
  ContextCache(ModelInfo &model_info, MlirContext context)
      : model_info_(model_info), context_(context) {}

  MlirContext context() { return context_; }

  /// Converts the TypeProto to an MlirType, returning a null type and
  /// setting an error if not possible.
  MlirType ConvertTypeProto(const onnx::TypeProto &tp);

  /// Converts the ONNX element type code to an MlirType, returning a null type
  /// and setting an error if not possible.
  MlirType ConvertTensorElementType(int element_type_code);

  /// Converts an ONNX TensorProto to an MlirAttribute, returning a null
  /// attribute and setting an error if not possible.
  MlirAttribute ConvertTensorProtoToAttr(const onnx::TensorProto &tp);

  /// Converts the ONNX TensorProto to an Mlir RankedTensor type.
  MlirType ConvertTensorProtoToBuiltinType(const onnx::TensorProto &tp);

  /// Converts the ONNX TensorProto to a !torch.vtensor type.
  MlirType ConvertTensorProtoToVtensorType(const onnx::TensorProto &tp);

  /// Gets a !torch.vtensor type for the given dims and element type.
  /// Dynamic dims are represented as -1.
  /// If it was not possible to create the type, sets an error and returns
  /// the null type.
  MlirType GetVtensorType(const std::vector<int64_t> &dims,
                          MlirType element_type);

  MlirType GetNoneType();

private:
  ModelInfo &model_info_;
  MlirContext context_;

  std::unordered_map<int, MlirType> elem_type_map_;
  std::unordered_map<std::string, MlirType> asm_type_map_;
  std::vector<int64_t> shared_dims_;
};

/// Imports graph nodes into a function.
class NodeImporter {
public:
  NodeImporter(GraphInfo &graph_info, ContextCache &cc,
               MlirOperation module_op);

  /// Called after construction to define the function in the module. Must be
  /// called prior to importing nodes.
  Status DefineFunction(std::optional<std::string> name = {});

  /// Imports all nodes topologically.
  Status ImportAll();

  void DebugDumpModule();

private:
  void PopulateGraphAttrs(MlirOperation container_op);
  Status
  ImportInitializer(const onnx::TensorProto &initializer,
                    std::optional<std::string> extern_name = std::nullopt);
  Status ImportNode(const onnx::NodeProto &node);
  MlirAttribute ImportGeneralAttribute(const onnx::AttributeProto &onnx_attr);

  // Special-form nodes.
  Status ImportGeneralNode(const onnx::NodeProto &node);
  Status ImportConstantNodeValueAttr(const onnx::NodeProto &node);

  void GetNone();

  /// Looks for an initializer for `name` and attempts to treat it as a 1D
  /// shape, filling `shape` if successful. Returns failure and sets an error
  /// if not.
  Status GetImmediateShapeTensor(const std::string &name,
                                 std::vector<int64_t> &shape);

  Status SetError(std::string msg) {
    return graph_info_.model_info().SetError(std::move(msg));
  }

  GraphInfo &graph_info_;
  ContextCache &cc_;
  MlirContext context_;
  MlirOperation module_op_;
  MlirOperation func_op_;
  MlirBlock body_block_;
  MlirLocation default_loc_;
  std::unordered_map<std::string_view, MlirValue> nv_map_;
};

} // namespace torch_mlir_onnx
