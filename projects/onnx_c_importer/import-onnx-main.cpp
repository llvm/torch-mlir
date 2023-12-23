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

#include "torch-mlir-c/Registration.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "OnnxImporter.h"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <iostream>

using namespace llvm;
using namespace torch_mlir_onnx;

struct MlirState {
  MlirState() {
    context = mlirContextCreateWithThreading(false);
    torchMlirRegisterAllDialects(context);
    module = mlirModuleCreateEmpty(mlirLocationUnknownGet(context));
  }
  ~MlirState() {
    mlirModuleDestroy(module);
    mlirContextDestroy(context);
  }

  MlirContext context;
  MlirModule module;
};

int main(int argc, char **argv) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "torch-mlir-onnx-import-c");

  // Open the input as an istream because that is what protobuf likes.
  std::unique_ptr<std::ifstream> alloced_input_stream;
  std::istream *input_stream = nullptr;
  if (inputFilename == "-") {
    errs() << "(parsing from stdin)\n";
    input_stream = &std::cin;
  } else {
    alloced_input_stream = std::make_unique<std::ifstream>(
        inputFilename, std::ios::in | std::ios::binary);
    if (!*alloced_input_stream) {
      errs() << "error: could not open input file " << inputFilename << "\n";
      return 1;
    }
    input_stream = alloced_input_stream.get();
  }

  // Parse the model proto.
  ModelInfo model_info;
  if (!model_info.model_proto().ParseFromIstream(input_stream)) {
    errs() << "Failed to parse ONNX ModelProto from " << inputFilename << "\n";
    return 2;
  }

  if (failed(model_info.Initialize())) {
    errs() << "error: Import failure: " << model_info.error_message() << "\n";
    model_info.DebugDumpProto();
    return 3;
  }
  model_info.DebugDumpProto();

  // Import.
  MlirState owned_state;
  ContextCache cc(model_info, owned_state.context);
  NodeImporter importer(model_info.main_graph(), cc,
                        mlirModuleGetOperation(owned_state.module));
  if (failed(importer.DefineFunction())) {
    errs() << "error: Could not define MLIR function for graph: "
           << model_info.error_message() << "\n";
    return 4;
  }
  if (failed(importer.ImportAll())) {
    errs() << "error: Could not import one or more graph nodes: "
           << model_info.error_message() << "\n";
    return 5;
  }
  importer.DebugDumpModule();

  return 0;
}
