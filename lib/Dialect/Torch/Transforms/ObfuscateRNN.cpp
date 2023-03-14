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

static void obfuscateRNN(MLIRContext *context, Operation *f) {
  // obfuscate RNN cells

  llvm::SmallPtrSet<Operation *, 16> opWorklist, filteredOpWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenMmOp>(op)) {
      opWorklist.insert(op);
    }
  });

  // filter duplicate recurrent part
  auto it = opWorklist.begin();
  auto oldShape =
      (*it)->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
  auto oldIt = it;
  bool flag = false;
  for (it++; it != opWorklist.end(); it++) {
    auto shape =
        (*it)->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
    if (flag) {
      if (shape == oldShape) {
        filteredOpWorklist.insert(*it);
      } else {
        break;
      }
    } else {
      if (shape == oldShape) {
        flag = true;
        filteredOpWorklist.insert(*oldIt);
        filteredOpWorklist.insert(*it);
      } else {
        oldShape = shape;
        oldIt = it;
      }
    }
  }

  // apply this  translation for every AtenMmOp in OpWorklit:
  // replace its first operand x with x/2+x/2
  IRRewriter rewriter(context);
  it = filteredOpWorklist.begin();
  // obfuscate too many layers cause precision error
  for (int i=0; i<2; i++) {
    Operation *op = (*it);
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    Value op0 = op->getOperand(0);
    std::vector<long> shape =
        op0.getType().cast<ValueTensorType>().getSizes().vec();
    int size = 1;
    for (auto i : shape) {
      size *= i;
    }
    std::vector<float> weightVec(size, 0.5);
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                                 rewriter.getF32Type());
    auto dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
        llvm::ArrayRef(weightVec));
    Value weight =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    Value mul1 =
        rewriter.create<AtenMulTensorOp>(loc, op0.getType(), op0, weight);
    Value mul2 =
        rewriter.create<AtenMulTensorOp>(loc, op0.getType(), op0, weight);
    Value float0 =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0));
    Value add = rewriter.create<AtenAddTensorOp>(loc, op0.getType(), mul1, mul2,
                                                 float0);
    op->setOperand(0, add);
  }
}

namespace {
class ObfuscateRNNPass : public ObfuscateRNNBase<ObfuscateRNNPass> {
public:
  ObfuscateRNNPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    obfuscateRNN(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createObfuscateRNNPass() {
  return std::make_unique<ObfuscateRNNPass>();
}