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

static std::vector<long> createNewShape(std::vector<long> shapeOrigin) {
  // if dimansion more than 2, need to reshape to 2
  // for example: (1,2,3,4) -> (6,4)
  std::vector<long> shapeNew;
  int mul = 1;
  for (unsigned long i = 0; i < shapeOrigin.size() - 1; ++i) {
    mul *= shapeOrigin[i];
  }
  shapeNew.push_back(mul);
  shapeNew.push_back(shapeOrigin[shapeOrigin.size() - 1]);
  return shapeNew;
}

static std::vector<Value> createABCD(IRRewriter &rewriter, Location loc,
                                     MLIRContext *context, long N) {
  std::vector<long> shapeWeight{N, N};
  std::vector<long> shapeBias = {N};

  // generate A, B, C, D, satisfy (xA+B)C+D == x
  float *A = new float[N * N]();
  float *B = new float[N]();
  float *C;
  float *D = new float[N]();
  // srand((unsigned)time(0));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = rand() % 100 * 0.01;
    }
    B[i] = rand() % 100 * 0.01;
  }
  C = LUP_solve_inverse(A, N);
  float sum;
  for (int i = 0; i < N; i++) {
    sum = 0;
    for (int j = 0; j < N; j++) {
      sum += B[j] * C[j * N + i];
    }
    D[i] = -sum;
  }

  // weight A
  Value weightA = createTensor(rewriter, loc, context, shapeWeight,
                               std::vector<float>(A, A + N * N));
  Value biasB = createTensor(rewriter, loc, context, shapeBias,
                             std::vector<float>(B, B + N));
  Value weightC = createTensor(rewriter, loc, context, shapeWeight,
                               std::vector<float>(C, C + N * N));
  Value biasD = createTensor(rewriter, loc, context, shapeBias,
                             std::vector<float>(D, D + N));
  return std::vector<Value>{weightA, biasB, weightC, biasD};
}

static void insertLinearRNN(MLIRContext *context,
                            SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 linear layer for every op in opWorklist
  // special for RNN: hidden layer in loop share the same weight
  // prerequest: all ops in opWorklist is same op in unrolling RNN loop

  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();

  // create reusable ops
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));
  std::vector<long> shapeOrigin =
      op->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
  std::vector<long> shapeNew;
  bool needReshape = true;
  if (shapeOrigin.size() == 2) {
    needReshape = false;
    shapeNew = shapeOrigin;
  } else {
    shapeNew = createNewShape(shapeOrigin);
  }
  std::vector<Value> values = createABCD(rewriter, loc, context, shapeNew[1]);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 linear layer
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[0]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[1],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[2]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[3],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
  }
}

static void insertLinear(MLIRContext *context,
                         llvm::SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 linear layer for every op in opWorklist

  IRRewriter rewriter(context);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                                rewriter.getI64IntegerAttr(1));
    std::vector<long> shapeOrigin =
        op->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
    std::vector<long> shapeNew;
    bool needReshape = true;
    if (shapeOrigin.size() == 2) {
      needReshape = false;
      shapeNew = shapeOrigin;
    } else {
      shapeNew = createNewShape(shapeOrigin);
    }
    std::vector<Value> values = createABCD(rewriter, loc, context, shapeNew[1]);

    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 linear layer
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[0]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[1],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, values[2]);
    rst = rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, values[3],
                                           int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
  }
}

namespace {
class InsertLinearPass : public InsertLinearBase<InsertLinearPass> {
public:
  InsertLinearPass() = default;
  InsertLinearPass(std::string net) { this->net = net; }
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);
    MLIRContext *context = &getContext();

    if (opWorklist.empty()) {
      llvm::errs() << "Not run InsertLinear\n";
      return;
    }

    if (net == "") {
      // todo: opWorklist too large will cause precision error
      while (opWorklist.size() >= 3)
        opWorklist.erase(*opWorklist.begin());
      insertLinear(context, opWorklist);
    } else if (net == "RNN") {
      insertLinearRNN(context, opWorklist);
    } else {
      llvm::errs() << "unsupported net: " << net << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertLinearPass(std::string net) {
  return std::make_unique<InsertLinearPass>(net);
}