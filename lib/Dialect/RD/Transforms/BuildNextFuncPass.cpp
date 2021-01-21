//===- BuildInitFunc.cpp - Extracts a pipeline definition ---*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>

#include "PassDetail.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "npcomp/Dialect/RD/IR/RDDatasetInterface.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::NPCOMP;

#define DEBUG_TYPE "rd-build-next-func"

namespace {

// Clones the definition function, transforming the ops used to the `[...].next` variations of the ops.
class BuildNextFunc : public RDBuildNextFuncBase<BuildNextFunc> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto* context = &getContext();
    auto pipelineDefOp = getOperation();
    auto defFuncOpt = findDefinitionFunc(pipelineDefOp);
    if (!defFuncOpt) {
      return signalPassFailure();
    }
    auto defFunc = *defFuncOpt;

    // Create the next func op.
    auto nextFuncTy = FunctionType::get(
      TypeRange(rd::IteratorType::get(context)),
      TypeRange(defFunc.getType().getResult(0)),
      context);

    OpBuilder builder(context);
    builder.setInsertionPointAfter(defFunc);
    auto buildFuncOp = builder.create<FuncOp>(pipelineDefOp.getLoc(), "next", nextFuncTy);

    // Fill in the body with a programmatic translation of the ops.
    auto* nextBody = buildFuncOp.addEntryBlock();
    auto stateValue = nextBody->getArgument(0);
    BlockAndValueMapping mapping;
    builder.setInsertionPointToStart(nextBody);
    int64_t offset = 0;
    defFunc.walk([&](Operation* op) {
      if (rd::DatasetTransform datasetOp = dyn_cast<rd::DatasetTransform>(op)) {
        auto itrPtr = builder.create<rd::IteratorIndexOp>(op->getLoc(), stateValue, offset++);
        datasetOp.buildNextOp(builder, mapping, itrPtr);
      }
      if (auto returnOp = dyn_cast<ReturnOp>(op)) {
        builder.clone(*returnOp.getOperation(), mapping);
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<rd::PipelineDefinitionOp>> mlir::NPCOMP::createBuildNextFuncPass() {
  return std::make_unique<BuildNextFunc>();
}
