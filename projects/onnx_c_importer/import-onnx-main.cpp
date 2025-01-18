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
#include <memory>

using namespace llvm;
using namespace torch_mlir_onnx;

// Encapsulates MLIR context and module management
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
  // Define command-line options
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));
  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Output filename"), cl::value_desc("filename"), cl::init("-"));

  // Initialize LLVM and parse command-line options
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "torch-mlir-onnx-import-c");

  // Open the input file stream
  std::unique_ptr<std::ifstream> allocedInputStream;
  std::istream *inputStream = nullptr;
  if (inputFilename == "-") {
    errs() << "(Parsing from stdin)\n";
    inputStream = &std::cin;
  } else {
    allocedInputStream = std::make_unique<std::ifstream>(
        inputFilename, std::ios::in | std::ios::binary);
    if (!allocedInputStream->is_open()) {
      errs() << "Error: Could not open input file: " << inputFilename << "\n";
      return EXIT_FAILURE;
    }
    inputStream = allocedInputStream.get();
  }

  // Parse the ONNX model proto
  ModelInfo modelInfo;
  if (!modelInfo.model_proto().ParseFromIstream(inputStream)) {
    errs() << "Error: Failed to parse ONNX ModelProto from " << inputFilename << "\n";
    return EXIT_FAILURE;
  }

  // Initialize model information
  if (failed(modelInfo.Initialize())) {
    errs() << "Error: Import failure: " << modelInfo.error_message() << "\n";
    modelInfo.DebugDumpProto();
    return EXIT_FAILURE;
  }
  modelInfo.DebugDumpProto();

  // Create MLIR state and context cache
  MlirState ownedState;
  ContextCache contextCache(modelInfo, ownedState.context);

  // Import the ONNX graph into MLIR
  NodeImporter importer(
      modelInfo.main_graph(), contextCache, mlirModuleGetOperation(ownedState.module));
  if (failed(importer.DefineFunction())) {
    errs() << "Error: Could not define MLIR function for graph: "
           << modelInfo.error_message() << "\n";
    return EXIT_FAILURE;
  }
  if (failed(importer.ImportAll())) {
    errs() << "Error: Could not import one or more graph nodes: "
           << modelInfo.error_message() << "\n";
    return EXIT_FAILURE;
  }

  // Dump the imported MLIR module
  importer.DebugDumpModule();

  // Optional: Save the output MLIR module to a file
  if (outputFilename != "-") {
    std::ofstream outFile(outputFilename, std::ios::out);
    if (!outFile.is_open()) {
      errs() << "Error: Could not open output file: " << outputFilename << "\n";
      return EXIT_FAILURE;
    }
    mlirOperationPrint(mlirModuleGetOperation(ownedState.module), outFile);
    outs() << "Successfully saved MLIR module to " << outputFilename << "\n";
  } else {
    outs() << "MLIR module processing complete. Output not saved to a file.\n";
  }

  return EXIT_SUCCESS;
}

