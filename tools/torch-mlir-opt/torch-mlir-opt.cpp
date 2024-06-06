//===- torch-mlir-opt.cpp - MLIR Optimizer Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
#include "stablehlo/dialect/Register.h"
#endif

using namespace mlir;

int main(int argc, char **argv) {
  mlir::torch::registerAllPasses();

  // Core Transforms
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  registerLocationSnapshotPass();
  registerLoopInvariantCodeMotionPass();
  registerPrintOpStatsPass();
  registerViewOpGraphPass();
  registerStripDebugInfoPass();
  registerSymbolDCEPass();

  // memref passes used in torch-backend-to-linalg-on-tensors-backend-pipeline
  memref::registerExpandOpsPass();
  memref::registerResolveShapedTypeResultDimsPass();

  DialectRegistry registry;
  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  mlir::torch::registerOptionalInputDialects(registry);

#ifdef TORCH_MLIR_ENABLE_STABLEHLO
  mlir::stablehlo::registerAllDialects(registry);
#endif
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
