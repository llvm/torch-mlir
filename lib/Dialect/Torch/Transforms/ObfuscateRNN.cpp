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
  // todo: is this useful? If RNN is simple, no need to use it, if RNN is
  // complecated, this logic is unable to recignoize hidden layer at present

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
                      SmallPtrSet<Operation *, 16> opWorklist,
                      int splitNumber) {
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
  Value maskVal = createTensor(rewriter, op->getLoc(), context, shape, mask);
  Value unMaskVal =
      createTensor(rewriter, op->getLoc(), context, shape, unMask);
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Value rst = newOp->getResult(0);
    Value mul1 =
        rewriter.create<AtenMulTensorOp>(loc, rst.getType(), rst, maskVal);
    Value mul2 =
        rewriter.create<AtenMulTensorOp>(loc, rst.getType(), rst, unMaskVal);
    Value add =
        rewriter.create<AtenAddTensorOp>(loc, rst.getType(), mul1, mul2, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), add);
    rewriter.replaceOp(op, rst);
  }
}

static void valueSplit(MLIRContext *context,
                       SmallPtrSet<Operation *, 16> opWorklist,
                       int splitNumber) {
  // replace input x with p1*x1+p2x2+...+pn*xn, and p1+p2+...+pn=1

  // create p1,p2,...,pn
  // random selection pos in [0, 99], calculate intervals as pi
  std::vector<int> pos;
  std::vector<float> vals;
  for (int i = 0; i < splitNumber - 1; ++i) {
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
  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  std::vector<long> empty_dim;
  rewriter.setInsertionPoint(op);
  std::vector<Value> valueList;
  for (float v : vals) {
    valueList.push_back(createTensor(rewriter, op->getLoc(), context, empty_dim,
                                     std::vector<float>{v}));
  }
  Value int1 = rewriter.create<ConstantIntOp>(op->getLoc(),
                                              rewriter.getI64IntegerAttr(1));

  // split every op in opWorklist
  for (auto op : opWorklist) {
    Location loc = op->getLoc();
    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Value rst = newOp->getResult(0);
    std::vector<Value> mulList;
    for (auto value : valueList) {
      mulList.push_back(
          rewriter.create<AtenMulTensorOp>(loc, rst.getType(), rst, value));
    }
    auto it = mulList.begin();
    rst = *it;
    for (++it; it != mulList.end(); ++it) {
      rst =
          rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, *it, int1);
    }
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rewriter.replaceOp(op, rst);
  }
}

static void insertConvRNN(MLIRContext *context,
                          SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 convolutions for every op in opWorklist, with different
  // pad and dilation
  // prerequest: all ops in opWorklist have the same shape

  IRRewriter rewriter(context);
  Operation *op = *opWorklist.begin();
  rewriter.setInsertionPoint(op);
  Location loc = op->getLoc();

  // reusable ops
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
  Value listInt1_1 = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  Value listInt = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({}));
  std::vector<long> shapeOrigin =
      op->getResult(0).getType().cast<ValueTensorType>().getSizes().vec();
  std::vector<long> shapeNew(4 - shapeOrigin.size(), 1); // change dim to 4
  shapeNew.insert(shapeNew.end(), shapeOrigin.begin(), shapeOrigin.end());
  int ChannelSz = 1;
  bool needReshape = true;
  if (shapeOrigin.size() == 4) {
    ChannelSz = shapeOrigin[1];
    needReshape = false;
  }
  int kernelSz = (1 + std::rand() % 5) * 2 + 1;
  std::vector<long> shapeKernel{ChannelSz, ChannelSz, kernelSz, kernelSz};
  int dilNum = 1 + std::rand() % 5;
  int padNum = (kernelSz - 1) * dilNum / 2;
  Value intPad =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
  Value intDil =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(dilNum));
  Value listIntPad_Pad = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({intPad, intPad}));
  Value listIntDil_Dil = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({intDil, intDil}));

  // generate A, B, C, D, satisfy C(Ax+B)+D == x
  float A = (std::rand() % 100 + 1) / 100.;
  float C = 1 / A;
  float B = (std::rand() % 100);
  float D = -B * C;
  // create convolution kernel and bias
  // new kernel size is: ChannelSz  x ChannelSz  x kernelSz x kernelSz
  std::vector<float> weightVec(ChannelSz * ChannelSz * kernelSz * kernelSz, 0);
  for (int i = 0; i < ChannelSz; i++) {
    // weightVec[i][i][kernelSz/2][kernelSz/2] = A
    weightVec[((i * ChannelSz + i) * kernelSz + kernelSz / 2) * kernelSz +
              kernelSz / 2] = A;
  }
  Value weightA = createTensor(rewriter, loc, context, shapeKernel, weightVec);
  std::vector<long> shapeBias{ChannelSz};
  std::vector<float> biasVec(shapeBias[0], B);
  Value biasB = createTensor(rewriter, loc, context, shapeBias, biasVec);
  // second conv kernel
  for (int i = 0; i < ChannelSz; i++) {
    weightVec[((i * ChannelSz + i) * kernelSz + kernelSz / 2) * kernelSz +
              kernelSz / 2] = C;
  }
  Value weightC = createTensor(rewriter, loc, context, shapeKernel, weightVec);
  // create bias
  biasVec.assign(biasVec.size(), D);
  Value biasD = createTensor(rewriter, loc, context, shapeBias, biasVec);

  for (auto op : opWorklist) {
    // select a random place to insert
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);
    // reshape if dimansion less than 4, such as : (1,84) -> (1,1,1,84)
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 convolution layer
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, weightA, biasB, listInt1_1, listIntPad_Pad,
        listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, weightC, biasD, listInt1_1, listIntPad_Pad,
        listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back to origin shape
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);
    rewriter.replaceOp(op, rst);
  }
}

static void insertLinearRNN(MLIRContext *context,
                            SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 linear layer for every op in opWorklist
  // prerequest: all ops in opWorklist have the same shape

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
    int mul = 1;
    for (unsigned long i = 0; i < shapeOrigin.size() - 1; ++i) {
      mul *= shapeOrigin[i];
    }
    shapeNew.push_back(mul);
    shapeNew.push_back(shapeOrigin[shapeOrigin.size() - 1]);
  }
  int N = shapeNew[1];
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

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    // reshape if dimansion more than 4, such as : (1,2,3,4) -> (6,4)
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 linear layer
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, weightA);
    rst =
        rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, biasB, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenMmOp>(loc, rst.getType(), rst, weightC);
    rst =
        rewriter.create<AtenAddTensorOp>(loc, rst.getType(), rst, biasD, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
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
    // llvm::SmallPtrSet<Operation *, 16> opWorklist = getRNNHidenLayers(f);
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);

    if (obfuscation == "valueSplit") {
      valueSplit(context, opWorklist, splitNumber);
    } else if (obfuscation == "maskSplit") {
      maskSplit(context, opWorklist, splitNumber);
    } else if (obfuscation == "insertConv") {
      insertConvRNN(context, opWorklist);
    } else if (obfuscation == "insertLinear") {
      insertLinearRNN(context, opWorklist);
    } else if (obfuscation == "") {
      // default obfuscation
      valueSplit(context, opWorklist, splitNumber);
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