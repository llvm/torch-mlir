//===- npcomp-lsp-server.cpp - MLIR Language Server -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "npcomp/InitAll.h"
#include "torch-mlir/InitAll.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::NPCOMP::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  return failed(MlirLspServerMain(argc, argv, registry));
}
