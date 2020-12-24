//===- MergeFuncs.cpp - Bufferization for TCP dialect -------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Bufferize.h"
#include "mlir/Transforms/DialectConversion.h"
#include "npcomp/Dialect/Refback/IR/RefbackDialect.h"
#include "npcomp/Dialect/Refback/IR/RefbackOps.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class RDMergeFuncsPass : public RDMergeFuncsBase<RDMergeFuncsPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    // TODO: Remove these!
    registry.insert<refback::RefbackDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    // TODO: Implement me!
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::NPCOMP::createRDMergeFuncsPass() {
  return std::make_unique<RDMergeFuncsPass>();
}
