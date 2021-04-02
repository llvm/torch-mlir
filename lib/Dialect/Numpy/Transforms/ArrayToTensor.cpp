//===- ArrayToTensor.cpp -----------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npcomp/Dialect/Numpy/IR/NumpyDialect.h"
#include "npcomp/Dialect/Numpy/IR/NumpyOps.h"
#include "npcomp/Dialect/Numpy/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;
using namespace mlir::NPCOMP::Numpy;

namespace {

class ArrayToTensorPass : public NumpyArrayToTensorBase<ArrayToTensorPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    CopyToTensorOp::getCanonicalizationPatterns(patterns, context);
    StaticInfoCastOp::getCanonicalizationPatterns(patterns, context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::NPCOMP::Numpy::createArrayToTensorPass() {
  return std::make_unique<ArrayToTensorPass>();
}
