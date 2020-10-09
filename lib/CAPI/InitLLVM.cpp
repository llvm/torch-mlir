//===- InitLLVM.cpp - C API for initializing LLVM -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "npcomp-c/InitLLVM.h"

#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"

void npcompInitializeLLVMCodegen() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::initializeLLVMPasses();
}
