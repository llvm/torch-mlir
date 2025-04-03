//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "onnx/common/file_utils.h"
#include "onnx/onnx_pb.h"

#include "Status.hpp"

#include <optional>

namespace fs = std::filesystem;

namespace torch_mlir_onnx {

struct ExternalDataInfo {
  std::string location = "";
  std::optional<unsigned long> offset;
  std::optional<unsigned long> length;
  std::optional<std::string> checksum;
  std::optional<std::string> basepath;

  static ExternalDataInfo FromTensorProto(const onnx::TensorProto &tp) {

    static std::unordered_map<
        std::string,
        std::function<void(const std::string &, ExternalDataInfo &)>>
        handlers{
            {"location", [](const std::string &value,
                            ExternalDataInfo &edi) { edi.location = value; }},
            {"offset",
             [](const std::string &value, ExternalDataInfo &edi) {
               edi.offset = std::stoul(value);
             }},
            {"length",
             [](const std::string &value, ExternalDataInfo &edi) {
               edi.length = std::stoul(value);
             }},
            {"checksum", [](const std::string &value,
                            ExternalDataInfo &edi) { edi.checksum = value; }},
            {"basepath", [](const std::string &value, ExternalDataInfo &edi) {
               edi.basepath = value;
             }}};

    ExternalDataInfo edi;
    for (const auto &entry : tp.external_data()) {
      if (entry.has_key() && entry.has_value())
        handlers[entry.key()](entry.value(), edi);
    }
    return edi;
  }
};

/// From onnx python API:
/// Loads data from an external file for tensor. Ideally TensorProto should
/// not hold any raw data but if it does it will be ignored.
Status loadExternalDataForTensor(onnx::TensorProto &tp,
                                 const fs::path &baseDir) {
  ExternalDataInfo edi = ExternalDataInfo::FromTensorProto(tp);
  std::string externalDataFilePath =
      onnx::checker::resolve_external_data_location(baseDir.generic_string(),
                                                    edi.location, tp.name());
  std::ifstream inputStream(externalDataFilePath,
                            std::ios::in | std::ios::binary);
  if (!inputStream.is_open()) {
    std::cerr << "Cannot open external data file: " << externalDataFilePath
              << "\n";
    return failure;
  }
  // get length of file
  inputStream.seekg(0, inputStream.end);
  unsigned long length = inputStream.tellg();

  if (edi.offset != std::nullopt) {
    inputStream.seekg(*edi.offset);
    length -= inputStream.tellg();
  } else {
    inputStream.seekg(0, inputStream.beg);
  }
  if (edi.length) {
    length = *edi.length;
  }
  std::string *strPtr = tp.mutable_raw_data();
  // NOTE: could be optimizated using c++23's std::string::resize_and_overwrite
  strPtr->resize(length); // unfortunately default-inizializes bytes to '/0'
  inputStream.read(strPtr->data(), length);

  return success;
}

void visitGraph(onnx::GraphProto &gp,
                const std::function<void(onnx::TensorProto &)> &callable) {
  for (auto &t : *gp.mutable_initializer())
    callable(t);
  for (auto &node : *gp.mutable_node()) {
    for (auto &attr : *node.mutable_attribute()) {
      if (attr.has_t())
        callable(*attr.mutable_t());
      for (auto &t : *attr.mutable_tensors())
        callable(t);
      if (attr.type() == onnx::AttributeProto::GRAPH) {
        visitGraph(*attr.mutable_g(), callable);
      }
      if (attr.type() == onnx::AttributeProto::GRAPHS) {
        for (auto &graph : *attr.mutable_graphs())
          visitGraph(graph, callable);
      }
    }
  }
}

void forEachTensor(onnx::ModelProto &mp,
                   const std::function<void(onnx::TensorProto &)> &callable) {
  visitGraph(*mp.mutable_graph(), callable);
}

bool usesExternalData(const onnx::TensorProto &tp) {
  return tp.has_data_location() &&
         tp.data_location() == onnx::TensorProto::EXTERNAL;
}

/// From onnx python API:
/// Loads external tensors into the model
///     Arguments:
///        model: ModelProto to load external data to
///         baseDir: directory that contains external data
Status loadExternalDataForModel(onnx::ModelProto &mp, const fs::path &baseDir) {
  Status s = success;
  forEachTensor(mp, [&baseDir, &s](onnx::TensorProto &tp) {
    if (usesExternalData(tp)) {
      if (failed(loadExternalDataForTensor(tp, baseDir))) {
        s = failure;
        return;
      }
      // After loading raw_data from external_data, change the state of tensors
      tp.set_data_location(onnx::TensorProto::DEFAULT);
      // and remove external data
      tp.clear_external_data();
    }
  });
  return s;
}

} // namespace torch_mlir_onnx
