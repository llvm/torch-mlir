//===- npcomp-opt.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "npcomp/Dialect/Basicpy/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/NumpyDialect.h"
#include "npcomp/Dialect/TCF/IR/TCFDialect.h"
#include "npcomp/Dialect/TCP/IR/TCPDialect.h"

#include "npcomp/Conversion/TCFToTCP/TCFToTCP.h"
#include "npcomp/Conversion/TCPToLinalg/TCPToLinalg.h"
#include "npcomp/E2E/E2E.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false));

int main(int argc, char **argv) {
  // TODO: Move all npcomp registration to a common helper.
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  mlir::registerDialect<mlir::NPCOMP::Basicpy::BasicpyDialect>();
  mlir::registerDialect<mlir::NPCOMP::Numpy::NumpyDialect>();
  mlir::registerDialect<mlir::NPCOMP::tcf::TCFDialect>();
  mlir::registerDialect<mlir::NPCOMP::tcp::TCPDialect>();

  using mlir::Pass; // The .inc files reference this unqualified.
#define GEN_PASS_REGISTRATION
#include "npcomp/E2E/Passes.h.inc"
  mlir::PassPipelineRegistration<>("e2e-lowering-pipeline", "E2E lowering pipeline.",
                             mlir::NPCOMP::createE2ELoweringPipeline);
  mlir::PassPipelineRegistration<>(
      "lower-to-hybrid-tensor-memref-pipeline",
      "Pipeline lowering to hybrid tensor/memref.",
      mlir::NPCOMP::createLowerToHybridTensorMemRefPipeline);
#define GEN_PASS_REGISTRATION
#include "npcomp/Conversion/Passes.h.inc"

  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  mlir::MLIRContext context;
  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (mlir::Dialect *dialect : context.getRegisteredDialects()) {
      llvm::outs() << dialect->getNamespace() << "\n";
    }
    return 0;
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects))) {
    return 1;
  }
  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
