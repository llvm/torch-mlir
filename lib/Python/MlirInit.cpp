//===- MlirInit.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp/Python/MlirInit.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/InitAll.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

namespace mlir {
namespace npcomp {
namespace python {

bool npcompMlirInitialize() {
  // Enable LLVM's signal handler to get nice stack traces.
  llvm::sys::SetOneShotPipeSignalFunction(
      llvm::sys::DefaultOneShotPipeSignalHandler);
  llvm::sys::PrintStackTraceOnErrorSignal("npcomp");

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();

  std::string program_name = "npcomp";
  std::vector<const char *> default_options = {program_name.c_str(), nullptr};
  llvm::cl::ParseCommandLineOptions(1, default_options.data());

  // Global registration.
  ::mlir::registerAllDialects();
  ::mlir::registerAllPasses();

  // Local registration.
  ::mlir::NPCOMP::registerAllDialects();
  ::mlir::NPCOMP::registerAllPasses();

  // LLVM codegen initialization.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();

  return true;
}

LogicalResult parsePassPipeline(StringRef pipeline, OpPassManager &pm,
                                raw_ostream &errorStream = llvm::errs()) {
  return ::mlir::parsePassPipeline(pipeline, pm, errorStream);
}

} // namespace python
} // namespace npcomp
} // namespace mlir
