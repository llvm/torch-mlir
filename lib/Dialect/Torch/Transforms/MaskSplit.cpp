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

static std::vector<Value> createMasks(Location loc, IRRewriter &rewriter,
                                      MLIRContext *context, int splitNumber,
                                      std::vector<long> shape) {
  // create masks, sum of all masks is a tensor whose elements are 1

  long size = 1;
  for (auto n : shape) {
    size *= n;
  }
  std::vector<std::vector<float>> masks(splitNumber,
                                        std::vector<float>(size, 0));
  for (int i = 0; i < size; ++i) {
    masks[random() % splitNumber][i] = 1;
  }
  std::vector<Value> vals;
  for (auto mask : masks) {
    vals.push_back(createTensor(rewriter, loc, context, shape, mask));
  }
  return vals;
}

static Value createSplit(Location loc, IRRewriter &rewriter,
                         std::vector<Value> valueList, Value rst) {
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

static void maskSplit(MLIRContext *context,
                      SmallPtrSet<Operation *, 16> opWorklist,
                      int splitNumber) {
  // replace x with x*mask1+x*mask2+...x*maskn
  // special for RNN: hidden layer in loop share the same weight
  // prerequest: all ops in opWorklist is same op in unrolling RNN loop

  IRRewriter rewriter(context);

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    std::vector<long> shape =
        op->getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
    rewriter.setInsertionPoint(op);
    std::vector<Value> valueList =
        createMasks(loc, rewriter, context, splitNumber, shape);

    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Value rst = newOp->getResult(0);
    rst = createSplit(loc, rewriter, valueList, rst);
    rewriter.replaceOp(op, rst);
  }
}

static void maskSplitRNN(MLIRContext *context,
                         SmallPtrSet<Operation *, 16> opWorklist,
                         int splitNumber) {
  // replace x with x*mask1+x*mask2+...x*maskn

  // create masks
  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  std::vector<long> shape =
      op->getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
  rewriter.setInsertionPoint(op);
  std::vector<Value> valueList =
      createMasks(op->getLoc(), rewriter, context, splitNumber, shape);

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
class MaskSplitPass : public MaskSplitBase<MaskSplitPass> {
public:
  MaskSplitPass() = default;
  MaskSplitPass(std::string net, int number) {
    this->net = net;
    this->number = number;
  }
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);
    MLIRContext *context = &getContext();

    if (opWorklist.empty()) {
      llvm::errs() << "Not run MaskSplit\n";
      return;
    }

    if (net == "") {
      maskSplit(context, opWorklist, number);
    } else if (net == "RNN") {
      maskSplitRNN(context, opWorklist, number);
    } else {
      llvm::errs() << "unsupported net: " << net << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createMaskSplitPass(std::string net, int number) {
  return std::make_unique<MaskSplitPass>(net, number);
}