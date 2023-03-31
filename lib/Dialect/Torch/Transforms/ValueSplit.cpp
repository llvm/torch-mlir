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

static std::vector<float> createNumbers(int n) {
  // create random p1,p2,...,pn, ensure p1+p2+...+pn=1
  // random selection pos in [0, 99], calculate intervals as pi
  std::vector<int> pos;
  std::vector<float> vals;
  for (int i = 0; i < n - 1; ++i) {
    pos.push_back(std::rand() % 100);
  }
  sort(pos.begin(), pos.end());
  int pos_before = *pos.begin();
  vals.push_back(pos_before / 100.);
  for (auto it = ++pos.begin(); it != pos.end(); ++it) {
    vals.push_back((*it - pos_before) / 100.);
    pos_before = *it;
  }
  vals.push_back((100 - pos.back()) / 100.);
  return vals;
}

static Value createSplit(Location loc, IRRewriter &rewriter, std::vector<Value> valueList,
                         Value rst) {
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  std::vector<Value> mulList;
  for (auto value : valueList) {
    mulList.push_back(
        rewriter.create<AtenMulTensorOp>(loc, rst.getType(), rst, value));
  }
  auto it = mulList.begin();
  rst = *it;
  for (++it; it != mulList.end(); ++it) {
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, *it, int1);
  }
  rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
  return rst;
}

static void valueSplit(MLIRContext *context,
                       SmallPtrSet<Operation *, 16> opWorklist,
                       int number) {
  // replace input x with p1*x1+p2x2+...+pn*xn, and p1+p2+...+pn=1

  IRRewriter rewriter(context);

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    std::vector<long> empty_dim;
    rewriter.setInsertionPoint(op);
    std::vector<Value> valueList;
    std::vector<float> vals = createNumbers(number);
    for (float v : vals) {
      valueList.push_back(createTensor(rewriter, loc, context, empty_dim,
                                       std::vector<float>{v}));
    }
    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Value rst = newOp->getResult(0);
    rst = createSplit(loc, rewriter, valueList, rst);
    rewriter.replaceOp(op, rst);
  }
}

static void valueSplitRNN(MLIRContext *context,
                          SmallPtrSet<Operation *, 16> opWorklist,
                          int number) {
  // replace input x with p1*x1+p2x2+...+pn*xn, and p1+p2+...+pn=1
  // special for RNN: hidden layer in loop share the same weight
  // prerequest: all ops in opWorklist is same op in unrolling RNN loop

  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  std::vector<long> empty_dim;
  rewriter.setInsertionPoint(op);
  std::vector<Value> valueList;
  std::vector<float> vals = createNumbers(number);
  for (float v : vals) {
    valueList.push_back(createTensor(rewriter, op->getLoc(), context, empty_dim,
                                     std::vector<float>{v}));
  }

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Value rst = newOp->getResult(0);
    rst = createSplit(loc, rewriter, valueList, rst);
    rewriter.replaceOp(op, rst);
  }
}

namespace {
class ValueSplitPass : public ValueSplitBase<ValueSplitPass> {
public:
  ValueSplitPass() = default;
  ValueSplitPass(std::string net, int number) {
    this->net = net;
    this->number = number;
  }
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);
    MLIRContext *context = &getContext();

    if (opWorklist.empty()) {
      llvm::errs() << "Not run ValueSplit\n";
      return;
    }

    if (net == "") {
      valueSplit(context, opWorklist, number);
    } else if (net == "RNN") {
      valueSplitRNN(context, opWorklist, number);
    } else {
      llvm::errs() << "unsupported net: " << net << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createValueSplitPass(std::string net, int number) {
  return std::make_unique<ValueSplitPass>(net, number);
}