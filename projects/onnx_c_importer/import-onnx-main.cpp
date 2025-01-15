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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "OnnxImporter.h"

#include "onnx/checker.h"
#include "onnx/common/file_utils.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

#include <fstream>
#include <iostream>

using namespace llvm;
using namespace torch_mlir_onnx;

static cl::opt<std::string>
    inputFilenameArg(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<std::string>
    outputFilenameArg("o", cl::desc("Output path (or '-' for stdout)"),
                      cl::value_desc("filename"), cl::init("-"));

static cl::opt<bool>
    noVerifyArg("no-verify", cl::desc("Disable verification prior to printing"),
                cl::init(false));

static cl::opt<bool>
    dataPropArg("data-prop",
                cl::desc("Toggle data propogation for onnx shape inference"),
                cl::init(true));

static cl::opt<bool> clearDomainArg(
    "clear-domain",
    cl::desc("If enabled, this will clear the domain attribute from each node"
             " in the onnx graph before performing shape inference."),
    cl::init(false));

static cl::opt<bool> keepTempsArg("keep-temps",
                                  cl::desc("Keep intermediate files"),
                                  cl::init(false));

static cl::opt<std::string> tempDirArg(
    "temp-dir",
    cl::desc(
        "Pre-existing directory in which to create temporary files."
        " For example, to place temporaries under the directory \"foo/bar\""
        " specify --temp-dir=foo/bar.  \"foo/bar\" must already exist."
        " Defaults to the directory of the input file."),
    cl::value_desc("directory"), cl::init("-"));

// NOTE: -data-dir
//       unfortunately it's not possible to specify a different base directory
//       for external data due to limitations of the onnx::checker API

static cl::opt<int> opsetVersionArg(
    "opset-version",
    cl::desc(
        "Allows specification of a newer opset_version to update the model"
        " to before importing to MLIR. This can sometime assist with shape "
        "inference."));

static cl::opt<bool> disableFunctionExpansionAllowlistArg(
    "disable-function-expansion-allowlist",
    cl::desc("Disable the allowlist for ONNX function expansion,"
             " allowing non-allowlisted functions to be expanded."),
    cl::init(false));

// NOTE: this function may actually fail without returning failure: since
// exceptions are disabled, onnx utilities will just print errors to stderr and
// keep executing. It's unclear what the returned ModelProto will contain.
FailureOr<onnx::ModelProto> loadOnnxModel() {
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
  SmallString<512> inputFile;
  inputFile = inputFilenameArg.getValue();
  if (!sys::fs::make_absolute(inputFile)) {
    errs() << "Invalid input file path: " << inputFilenameArg << "\n";
    return llvm::failure();
  }
  if (sys::fs::is_directory(inputFile)) {
    errs() << "Input file path is a directory: " << inputFilenameArg << "\n";
    return llvm::failure();
  }
  llvm::StringRef inputDir = sys::path::parent_path(inputFile);

  SmallString<512> tempDir;
  if (tempDirArg == "-") {
    tempDir = inputDir;
  } else {
    tempDir = tempDirArg;
  }
  sys::path::append(tempDir, "onnx-importer-temp");
  if (!sys::fs::remove_directories(tempDir, /*IgnoreErrors=*/true)) {
    errs() << "Couldn't clean temp directory: " << tempDir << "\n";
    return llvm::failure();
  }
  if (!sys::fs::create_directory(tempDir)) {
    errs() << "Couldn't create temp directory: " << tempDir << "\n";
    return llvm::failure();
  }

  onnx::ModelProto mp;
  {
    std::ifstream inputStream(inputFilenameArg,
                              std::ios::in | std::ios::binary);
    if (!mp.ParseFromIstream(&inputStream)) {
      errs() << "Failed to parse ONNX ModelProto \n";
      return llvm::failure();
    }
  }

  if (opsetVersionArg.getNumOccurrences()) {
    mp = onnx::version_conversion::ConvertVersion(mp, opsetVersionArg);
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

    SmallString<512> tempInferredFile = tempDir;
    sys::path::append(tempInferredFile, "inferred.onnx");
    std::string tempInferredFileStr = std::string(tempInferredFile);
    // First save intermediate to file
    {
      std::fstream output(tempInferredFileStr,
                          std::ios::out | std::ios::trunc | std::ios::binary);
      std::string model_string;
      mp.SerializeToString(&model_string);
      output << model_string;
    }

    onnx::shape_inference::InferShapes(tempInferredFileStr, tempInferredFileStr,
                                       onnx::OpSchemaRegistry::Instance(),
                                       opts);

    {
      std::ifstream inputStream(tempInferredFileStr,
                                std::ios::in | std::ios::binary);
      mp.Clear();
      if (!mp.ParseFromIstream(&inputStream)) {
        errs() << "Failed to parse ONNX ModelProto \n";
        return llvm::failure();
      }
    }
  }

  if (!keepTempsArg) {
    if (!sys::fs::remove_directories(tempDir, /*IgnoreErrors=*/true)) {
      errs() << "Couldn't clean temp directory: " << tempDir << "\n";
      return llvm::failure();
    }
  }

  return mp;
}

int main(int argc, char **argv) {

  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "Torch-mlir ONNX import tool");

  auto model = loadOnnxModel();
  if (llvm::failed(model)) {
    return 1;
  }

  std::unique_ptr<std::ofstream> allocatedOutputStream;
  std::ostream *outputStream = nullptr;
  if (outputFilenameArg == "-") {
    outputStream = &std::cout;
  } else {
    allocatedOutputStream =
        std::make_unique<std::ofstream>(outputFilenameArg, std::ios::out);
    if (!*allocatedOutputStream) {
      errs() << "error: could not open output file " << outputFilenameArg
             << "\n";
      return 1;
    }
    outputStream = allocatedOutputStream.get();
  }

  Status status = OnnxImporter::Import(std::move(model).value(), outputStream);
  if (failed(status))
    return 1;

  return 0;
}
