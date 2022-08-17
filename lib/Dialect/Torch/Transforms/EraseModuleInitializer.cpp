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
    for (auto initializer :
         getOperation().getOps<GlobalSlotModuleInitializerOp>()) {
      auto intialize =
          cast<InitializeGlobalSlotsOp>(initializer.getBody()->getTerminator());
      if (intialize.getNumOperands() == 0) {
        initializer.erase();
      }
      // The verifier ensures there is only one GlobalSlotModuleInitializerOp.
      break;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createEraseModuleInitializerPass() {
  return std::make_unique<EraseModuleInitializerPass>();
}
