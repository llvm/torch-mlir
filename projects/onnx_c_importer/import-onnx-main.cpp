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
#include "llvm/Support/raw_ostream.h"

#include "OnnxImporter.h"

#include "onnx/onnx_pb.h"

#include <fstream>
#include <iostream>

using namespace llvm;
using namespace torch_mlir_onnx;

int main(int argc, char **argv) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "torch-mlir-onnx-import-c");

  // TODO: load_onnx_model()
  //  Open the input as an istream because that is what protobuf likes.
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
  onnx::ModelProto modelProto;

  if (!modelProto.ParseFromIstream(input_stream)) {
    errs() << "Failed to parse ONNX ModelProto \n";
    return 1;
  }

  std::unique_ptr<std::ofstream> allocatedOutputStream;
  std::ostream *outputStream = nullptr;
  if (outputFilename == "-") {
    outputStream = &std::cout;
  } else {
    allocatedOutputStream =
        std::make_unique<std::ofstream>(outputFilename, std::ios::out);
    if (!*allocatedOutputStream) {
      errs() << "error: could not open output file " << outputFilename << "\n";
      return 1;
    }
    outputStream = allocatedOutputStream.get();
  }

  Status status = OnnxImporter::Import(std::move(modelProto), outputStream);
  if (failed(status))
    return 1;

  return 0;
}
