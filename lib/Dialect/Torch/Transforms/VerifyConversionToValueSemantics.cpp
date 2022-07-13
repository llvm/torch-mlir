//===- VerifyConversionToValueSemantics.cpp ----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch::Torch;

static LogicalResult checkValueType(Operation *op, Value value) {
  auto isNotValueTensorType = value.getType().isa<NonValueTensorType>();
  return isNotValueTensorType
             ? op->emitError(
                   "found a non-value tensor type, this is likely due to a "
                   "missing case in the MaximizeValueSemantics pass")
             : success();
}

namespace {
class VerifyConversionToValueSemanticsPass
    : public VerifyConversionToValueSemanticsBase<
          VerifyConversionToValueSemanticsPass> {
  void runOnOperation() override {
    bool didFail = false;
    auto walkResult = getOperation().walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        if (failed(checkValueType(block->getParentOp(), arg))) {
          didFail = true;
          return WalkResult::interrupt();
        }
      }

      for (Operation &op : *block) {
        for (OpResult result : op.getResults()) {
          if (failed(checkValueType(&op, result))) {
            didFail = true;
            return WalkResult::interrupt();
          }
        }
      }

      return WalkResult::advance();
    });

    if (didFail || walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createVerifyConversionToValueSemanticsPass() {
  return std::make_unique<VerifyConversionToValueSemanticsPass>();
}
