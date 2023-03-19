//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <cstdlib>
#include <ctime>

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class InsertConvPass : public InsertConvBase<InsertConvPass> {
public:
  InsertConvPass() = default;
  void runOnOperation() override {
    llvm::SmallPtrSet<Operation *, 16> opWorklist;
    getOperation()->walk([&](Operation *op) {
      if (isa<AtenReluOp, AtenSigmoidOp>(op)) {
        if (op->getResult(0).getType().isa<ValueTensorType>()) {
          opWorklist.insert(op);
        }
      }
    });

    if (opWorklist.empty()) {
      llvm::errs() << "Not run InsertConv\n";
      return;
    }

    insertConv(&getContext(), opWorklist);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertConvPass() {
  return std::make_unique<InsertConvPass>();
}