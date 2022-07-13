//===- EraseModuleInitializer.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class EraseModuleInitializerPass
    : public EraseModuleInitializerBase<EraseModuleInitializerPass> {
  void runOnOperation() override {
    auto walkResult = getOperation().walk([](GlobalSlotModuleInitializerOp op) {
      auto intialize =
          cast<InitializeGlobalSlotsOp>(op.getBody()->getTerminator());
      if (intialize.getNumOperands() != 0) {
        op.emitError("could not erase non-empty module initializer");
        return WalkResult::interrupt();
      }
      op.erase();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createEraseModuleInitializerPass() {
  return std::make_unique<EraseModuleInitializerPass>();
}
