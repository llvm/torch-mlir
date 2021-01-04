//===- ExtractPipelineDef.cpp - Extracts a pipeline definition ---*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <set>

#include "PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "npcomp/Dialect/RD/IR/RDDialect.h"
#include "npcomp/Dialect/RD/IR/RDOps.h"
#include "npcomp/Dialect/RD/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::NPCOMP;

namespace {
class ExtractPipelineDef : public RDExtractPipelineDefsBase<ExtractPipelineDef> {
  void runOnOperation() override {
    auto module = getOperation();
    std::set<StringRef> datasetFuncs;  // TODO: convert to more efficient data structure.
    module.walk([&](rd::MakeIteratorOp op) {
      auto symbol = op->getAttrOfType<FlatSymbolRefAttr>("ds");
      datasetFuncs.insert(symbol.getValue());
    });

    OpBuilder builder(module);
    for (auto symbol : datasetFuncs) {
      auto func = module.lookupSymbol<FuncOp>(symbol);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(func);
      auto definition = builder.create<rd::PipelineDefinitionOp>(func.getLoc());
      definition.setName(symbol);
      func->remove();
      func.setName("definition");
      auto& bodyRegion = definition.getBodyRegion();
      builder.createBlock(&bodyRegion, bodyRegion.end());
      builder.insert(func);
      builder.create<rd::PipelineDefinitionTerminatorOp>(func.getLoc());
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::NPCOMP::createExtractPipelineDefPass() {
  return std::make_unique<ExtractPipelineDef>();
}
