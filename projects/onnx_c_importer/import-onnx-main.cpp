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
#include "llvm/Support/InitLLVM.h"

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

//  NOTE: this function may actually fail without returning failure: since
//  exceptions are disabled, onnx utilities will just print errors to stderr and
//  keep executing. It's unclear what the returned ModelProto will contain.
//  TODO: Rely on python's load_onnx_model() and expose OnnxImporter via pybind.
FailureOr<onnx::ModelProto> loadOnnxModel() {

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
    errs() << "ModelProto too big. Missing feature.\n";
    return llvm::failure();
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
