//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

// This main driver uses LLVM tool-making facilities and the support lib.
// The actual importer libraries, however, only depend on the C API so that
// they can be included in foreign projects more easily.

#include "OnnxImporter.h"
#include "SimpleArgParser.hpp"

#include "onnx/checker.h"
#include "onnx/common/file_utils.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

#include <fstream>
#include <iostream>

using namespace torch_mlir_onnx;

static arg<PositionalArg, std::string> inputFilenameArg("ONNX protobuf input",
                                                        "<input file>");

static arg<OptionalArg, std::string>
    outputFilenameArg("Output path (or '-' for stdout)", "-o", "-", "filename");

static arg<OptionalArg, bool>
    noVerifyArg("Disable verification prior to printing", "--no-verify", false);

static arg<OptionalArg, bool>
    dataPropArg("Toggle data propogation for onnx shape inference",
                "--data-prop", true);

static arg<OptionalArg, bool> clearDomainArg(
    "If enabled, this will clear the domain attribute from each node"
    " in the onnx graph before performing shape inference.",
    "--clear-domain", false);

static arg<OptionalArg, bool> keepTempsArg("Keep intermediate files",
                                           "--keep-temps", false);

static arg<OptionalArg, std::string> tempDirArg(
    "Pre-existing directory in which to create temporary files."
    " For example, to place temporaries under the directory \"foo/bar\""
    " specify --temp-dir=foo/bar.  \"foo/bar\" must already exist."
    " Defaults to the directory of the input file.",
    "--temp-dir", "-", "directory");

static arg<OptionalArg, std::optional<int>> opsetVersionArg(
    "Allows specification of a newer opset_version to update the model"
    " to before importing to MLIR. This can sometime assist with shape "
    "inference.",
    "--opset-version");

static arg<OptionalArg, bool> disableFunctionExpansionAllowlistArg(
    "Disable the allowlist for ONNX function expansion,"
    " allowing non-allowlisted functions to be expanded.",
    "--disable-function-expansion-allowlist", false);

// NOTE: -data-dir
//       unfortunately it's not possible to specify a different base directory
//       for external data due to limitations of the onnx::checker API

FailureOr<onnx::ModelProto> loadOnnxModel() {
  namespace fs = std::filesystem;
  // Do shape inference two ways.  First, attempt in-memory to avoid redundant
  // loading and the need for writing a temporary file somewhere.  If that
  // fails, typically because of the 2 GB protobuf size limit, try again via
  // files.  See
  // https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#shape-inference-a-large-onnx-model-2gb
  // for details about the file-based technique.

  // Make a temp dir for all the temp files we'll be generating as a side
  // effect of infering shapes.  For now, the only file is a new .onnx holding
  // the revised model with shapes.
  //
  // TODO: If the program temp_dir is None, we should be using an ephemeral
  // temp directory instead of a hard-coded path in order to avoid data races
  // by default.
  fs::path inputFile(*inputFilenameArg);
  if (!fs::exists(inputFile) || fs::is_directory(inputFile)) {
    std::cerr << "Invalid input file path: " << *inputFilenameArg << "\n";
    return failure;
  }
  fs::path inputDir = inputFile.parent_path();
  fs::path tempDir = *tempDirArg == "-" ? inputDir : fs::path(*tempDirArg);
  tempDir /= "onnx-importer-temp";
  fs::remove_all(tempDir);

  onnx::ModelProto mp;
  {
    std::ifstream inputStream(inputFilenameArg,
                              std::ios::in | std::ios::binary);
    if (!inputStream.is_open()) {
      std::cerr << "Cannot open input file: " << *inputFilenameArg << "\n";
      return failure;
    }
    if (!mp.ParseFromIstream(&inputStream)) {
      std::cerr << "Failed to parse ONNX ModelProto \n";
      return failure;
    }
  }

  if ((*opsetVersionArg).has_value()) {
    mp = onnx::version_conversion::ConvertVersion(mp, **opsetVersionArg);
  }

  if (clearDomainArg) {
    for (auto &n : *(mp.mutable_graph()->mutable_node())) {
      n.clear_domain();
    }
  }

  onnx::checker::check_model(mp);

  onnx::ShapeInferenceOptions opts;
  opts.enable_data_propagation = dataPropArg;

  const size_t MAXIMUM_PROTOBUF = 2000000000;
  // Check whether serialized size is within threshold for in-memory shape
  // inference
  if (mp.ByteSizeLong() <= MAXIMUM_PROTOBUF) {
    onnx::shape_inference::InferShapes(mp, onnx::OpSchemaRegistry::Instance(),
                                       opts);
  } else {
    // Model is too big for in-memory inference: do file-based shape inference
    // to a temp file.

    if (!fs::create_directory(tempDir)) {
      std::cerr << "Couldn't create temp directory: " << tempDir << "\n";
      return failure;
    }
    fs::path tempInferredFile = tempDir / "inferred.onnx";
    // First save intermediate to file
    {
      std::fstream output(tempInferredFile,
                          std::ios::out | std::ios::trunc | std::ios::binary);
      std::string model_string;
      mp.SerializeToString(&model_string);
      output << model_string;
    }

    onnx::shape_inference::InferShapes(tempInferredFile, tempInferredFile,
                                       onnx::OpSchemaRegistry::Instance(),
                                       opts);

    {
      std::ifstream inputStream(tempInferredFile,
                                std::ios::in | std::ios::binary);
      mp.Clear();
      if (!mp.ParseFromIstream(&inputStream)) {
        std::cerr << "Failed to parse ONNX ModelProto \n";
        return failure;
      }
    }
  }

  // Remove the inferred shape file unless asked to keep it
  if (!keepTempsArg) {
    fs::remove_all(tempDir);
  }

  return mp;
}

int main(int argc, char **argv) {
  if (argc < 1)
    return 1;
  std::span<char *> s(&(argv[1]), argc - 1);
  if (!args::ParseArgs(s.begin(), s.end()))
    return 1;

  auto model = loadOnnxModel();
  if (failed(model)) {
    return 1;
  }

  std::unique_ptr<std::ofstream> allocatedOutputStream;
  std::ostream *outputStream = nullptr;
  if (*outputFilenameArg == "-") {
    outputStream = &std::cout;
  } else {
    allocatedOutputStream =
        std::make_unique<std::ofstream>(*outputFilenameArg, std::ios::out);
    if (!*allocatedOutputStream) {
      std::cerr << "error: could not open output file " << *outputFilenameArg
                << "\n";
      return 1;
    }
    outputStream = allocatedOutputStream.get();
  }

  Config config;
  config.no_verify = noVerifyArg;
  if (disableFunctionExpansionAllowlistArg) {
    config.function_expansion_allowlists_by_domain = std::nullopt;
  }

  Status status =
      OnnxImporter::Import(std::move(model).value(), outputStream, config);
  if (failed(status))
    return 1;

  return 0;
}
