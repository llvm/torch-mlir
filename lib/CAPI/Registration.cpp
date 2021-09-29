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
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/InitAll.h"

void torchMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  mlir::torch::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->loadAllAvailableDialects();
}

void torchMlirRegisterAllPasses() { mlir::torch::registerAllPasses(); }
