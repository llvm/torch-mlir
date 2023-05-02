//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/ScalarLoopOpInterface.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();

  registerTransformsPasses();
  registerSCFPasses();

  // Local dialects.
  mlir::torch::TMTensor::registerPasses();

  DialectRegistry registry;
  registry.insert<
      // Local dialects
      mlir::torch::TMTensor::TMTensorDialect,
      // Upstream dialects
      mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
      mlir::func::FuncDialect, mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
