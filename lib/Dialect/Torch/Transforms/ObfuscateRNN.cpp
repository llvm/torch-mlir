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

static SmallPtrSet<Operation *, 16> getRNNHidenLayers(Operation *f) {
  // return pointer set of AtenMmOp which represent hidden layer of RNN

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
  return filteredOpWorklist;
}

static void maskSplit(MLIRContext *context,
                      SmallPtrSet<Operation *, 16> opWorklist, int splitNumber) {
  // replace input x with x1+x2, x1 and x2 are obtaioned by x through the
  // opposite mask
  // todo: support number split as valueSplit

  // create constant 1 and masks
  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  std::vector<long> shape =
      op->getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
  long size = 1;
  for (auto n : shape) {
    size *= n;
  }
  std::vector<float> mask(size, 0), unMask(size, 1);
  std::srand(std::time(0)); // can delete to enforce obfuscate
  for (int i = 0; i < size; i++) {
    if (std::rand() % 2) {
      mask[i] = 1;
      unMask[i] = 0;
    }
  }

  rewriter.setInsertionPoint(op);
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               rewriter.getF32Type());
  auto maskDense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(mask));
  auto unMaskDense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
      llvm::ArrayRef(unMask));
  Value maskVal = rewriter.create<ValueTensorLiteralOp>(
      op->getLoc(), resultTensorType, maskDense);
  Value unMaskVal = rewriter.create<ValueTensorLiteralOp>(
      op->getLoc(), resultTensorType, unMaskDense);
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    Value op0 = op->getOperand(0);
    Value mul1 =
        rewriter.create<AtenMulTensorOp>(loc, op0.getType(), op0, maskVal);
    Value mul2 =
        rewriter.create<AtenMulTensorOp>(loc, op0.getType(), op0, unMaskVal);
    Value add =
        rewriter.create<AtenAddTensorOp>(loc, op0.getType(), mul1, mul2, int1);
    op->setOperand(0, add);
  }
}

static void valueSplit(MLIRContext *context,
                       SmallPtrSet<Operation *, 16> opWorklist, int splitNumber) {
  // replace input x with p1*x1+p2x2+...+pn*xn, and p1+p2+...+pn=1

  // create p1,p2,...,pn
  // random selection pos in [0, 99], calculate intervals as pi
  std::vector<int> pos;
  std::vector<double> vals;
  for (int i = 0; i < splitNumber-1; ++i) {
    pos.push_back(std::rand() % 100);
  }
  sort(pos.begin(), pos.end());
  int pos_before = *pos.begin();
  vals.push_back(pos_before/100.);
  for (auto it = ++pos.begin(); it != pos.end(); ++it) {
    vals.push_back((*it - pos_before)/100.);
    pos_before = *it;
  }
  vals.push_back((100 - pos.back())/100.);
  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  std::vector<long> empty_dim;
  rewriter.setInsertionPoint(op);
  auto resultTensorType = ValueTensorType::get(
      context, llvm::ArrayRef(empty_dim), rewriter.getF64Type());
  DenseElementsAttr dense;
  std::vector<Value> valueList;
  for (double v : vals) {
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(empty_dim), rewriter.getF64Type()),
        llvm::ArrayRef(std::vector<double>{v}));
    valueList.push_back(rewriter.create<ValueTensorLiteralOp>(
        op->getLoc(), resultTensorType, dense));
  }
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    Value op0 = op->getOperand(0);
    std::vector<Value> mulList;
    for (auto value : valueList) {
      mulList.push_back(
          rewriter.create<AtenMulTensorOp>(loc, op0.getType(), op0, value));
    }
    auto it = mulList.begin();
    Value rst;
    rst = *it;
    for (++it; it != mulList.end(); ++it) {
      rst =
          rewriter.create<AtenAddTensorOp>(loc, op0.getType(), rst, *it, int1);
    }
    op->setOperand(0, rst);
  }
}

namespace {
class ObfuscateRNNPass : public ObfuscateRNNBase<ObfuscateRNNPass> {
public:
  ObfuscateRNNPass() = default;
  ObfuscateRNNPass(std::string obfuscation, int splitNumber) {
    this->obfuscation = obfuscation;
    this->splitNumber = splitNumber;
  }
  void runOnOperation() override {
    // obfuscate RNN cells
    // apply this transformation for every hidden layer:

    MLIRContext *context = &getContext();
    auto f = getOperation();

    if (obfuscation == "valueSplit") {
      valueSplit(context, getRNNHidenLayers(f), splitNumber);
    } else if (obfuscation == "maskSplit") {
      maskSplit(context, getRNNHidenLayers(f), splitNumber);
    } else if (obfuscation == "deepen") {
      // relu only support apply after positive number
      llvm::SmallPtrSet<Operation *, 16> opWorklist;
      getOperation()->walk([&](Operation *op) {
        if (isa<AtenReluOp, AtenSigmoidOp>(op)) {
          if (op->getResult(0).getType().isa<ValueTensorType>()) {
            opWorklist.insert(op);
          }
        }
      });
      insertConv(context, opWorklist);
    } else if (obfuscation == "") {
      // default obfuscation
      valueSplit(context, getRNNHidenLayers(f), splitNumber);
    } else {
      llvm::errs() << "unsupported obfuscation: " << obfuscation << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createObfuscateRNNPass(std::string obfuscation,
                                           int splitNumber) {
  return std::make_unique<ObfuscateRNNPass>(obfuscation, splitNumber);
}