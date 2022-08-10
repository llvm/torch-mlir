//===- Registration.cpp - C Interface for MLIR Registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-c/Registration.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/InitAllPasses.h"
#include "torch-mlir/InitAll.h"

void torchMlirRegisterRequiredDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::AffineDialect, mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::tosa::TosaDialect>();
  unwrap(context)->appendDialectRegistry(registry);
}

void torchMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  mlir::torch::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->loadAllAvailableDialects();
}

void torchMlirRegisterAllPasses() {
  mlir::arith::registerArithPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::func::registerFuncPasses();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertArithToLLVMPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertLinalgToLLVMPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertMemRefToLLVMPass();
  mlir::registerLinalgPasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFPasses();
  mlir::registerSCFToControlFlowPass();
  mlir::registerTosaToArithPass();
  mlir::registerTosaToLinalgNamedPass();
  mlir::registerTosaToLinalgPass();
  mlir::tensor::registerTensorPasses();
  mlir::torch::registerAllPasses();
}
