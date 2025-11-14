//===- EraseModuleInitializer.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
namespace mlir::torch::Torch {

#define GEN_PASS_DEF_ERASEMODULEINITIALIZER
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h.inc"

namespace {
class EraseModuleInitializerPass
    : public impl::EraseModuleInitializerBase<EraseModuleInitializerPass> {
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

std::unique_ptr<OperationPass<ModuleOp>> createEraseModuleInitializerPass() {
  return std::make_unique<EraseModuleInitializerPass>();
}

} // namespace mlir::torch::Torch
