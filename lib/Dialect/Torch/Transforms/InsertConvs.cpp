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

static std::vector<Value> createABCD(IRRewriter &rewriter, Location loc,
                                     MLIRContext *context, long channelSz,
                                     long kernelSz) {
  // generate A, B, C, D, satisfy C(Ax+B)+D == x
  float A = (std::rand() % 100 + 1) / 100.;
  float C = 1 / A;
  float B = (std::rand() % 100);
  float D = -B * C;
  // create convolution kernel and bias
  // kernel size is: (channelSz, channelSz, kernelSz, kernelSz)
  // bias size is: (channelSz)
  std::vector<long> shapeKernel{channelSz, channelSz, kernelSz, kernelSz};
  std::vector<float> kernelVec(channelSz * channelSz * kernelSz * kernelSz, 0);
  for (int i = 0; i < channelSz; i++) {
    // kernelVec[i][i][kernelSz/2][kernelSz/2] = A
    kernelVec[((i * channelSz + i) * kernelSz + kernelSz / 2) * kernelSz +
              kernelSz / 2] = A;
  }
  Value kernelA = createTensor(rewriter, loc, context, shapeKernel, kernelVec);
  std::vector<long> shapeBias{channelSz};
  std::vector<float> biasVec(shapeBias[0], B);
  Value biasB = createTensor(rewriter, loc, context, shapeBias, biasVec);
  // second conv kernel and bias
  for (int i = 0; i < channelSz; i++) {
    kernelVec[((i * channelSz + i) * kernelSz + kernelSz / 2) * kernelSz +
              kernelSz / 2] = C;
  }
  Value kernelC = createTensor(rewriter, loc, context, shapeKernel, kernelVec);
  biasVec.assign(biasVec.size(), D);
  Value biasD = createTensor(rewriter, loc, context, shapeBias, biasVec);
  return std::vector<Value>{kernelA, biasB, kernelC, biasD};
}

static void insertConvRNN(MLIRContext *context,
                          SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 convolutions for every op in opWorklist, with different
  // pad and dilation
  // special for RNN: hidden layer in loop share the same weight
  // prerequest: all ops in opWorklist is same op in unrolling RNN loop

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
  std::vector<long> shapeNew(4 - shapeOrigin.size(), 1);
  shapeNew.insert(shapeNew.end(), shapeOrigin.begin(), shapeOrigin.end());
  int channelSz = 1;
  bool needReshape = true;
  if (shapeOrigin.size() == 4) {
    channelSz = shapeOrigin[1];
    needReshape = false;
  }
  int kernelSz = (1 + std::rand() % 5) * 2 + 1;
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
  std::vector<Value> values =
      createABCD(rewriter, loc, context, channelSz, kernelSz);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    // copy op, for convinience of replace use of op
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);
    // if dimansion less than 4, need to reshape to 4
    // for example: (1,84) -> (1,1,1,84)
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);
    // create 2 convolution layer
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, values[0], values[1], listInt1_1,
        listIntPad_Pad, listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, values[2], values[3], listInt1_1,
        listIntPad_Pad, listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back to origin shape
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);
    rewriter.replaceOp(op, rst);
  }
}

static void insertConv(MLIRContext *context,
                       llvm::SmallPtrSet<Operation *, 16> opWorklist) {
  // insert 2 convolutions for every op in opWorklist, with different
  // pad and dilation

  IRRewriter rewriter(context);

  for (auto op : opWorklist) {
    rewriter.setInsertionPointAfter(op);
    Operation *newOp = rewriter.clone(*op);
    Location loc = newOp->getLoc();
    Value rst = newOp->getResult(0);

    // if dimansion less than 4, need to reshape to 4
    // for example: (1,84) -> (1,1,1,84)
    std::vector<long> shapeOrigin =
        rst.getType().cast<ValueTensorType>().getSizes().vec();
    bool needReshape = false;
    std::vector<long> shapeNew(4 - shapeOrigin.size(), 1);
    shapeNew.insert(shapeNew.end(), shapeOrigin.begin(), shapeOrigin.end());
    if (shapeOrigin.size() != 4) {
      needReshape = true;
    }
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeNew, rst);

    int channelSz = shapeNew[1];
    int kernelSz = (1 + std::rand() % 5) * 2 + 1;
    std::vector<Value> values =
        createABCD(rewriter, loc, context, channelSz, kernelSz);

    // create other oprands for conv
    int dilNum = 1 + std::rand() % 5;
    int padNum = (kernelSz - 1) * dilNum / 2;
    Value intPad =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
    Value intDil =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(dilNum));
    Value listIntPad_Pad = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)),
        ValueRange({intPad, intPad}));
    Value listIntDil_Dil = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)),
        ValueRange({intDil, intDil}));
    Value int1 =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value listInt1_1 = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    Value listInt = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({}));

    // create 2 conv
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, values[0], values[1], listInt1_1,
        listIntPad_Pad, listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, values[2], values[3], listInt1_1,
        listIntPad_Pad, listIntDil_Dil, constFalse, listInt, int1);
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // reshape back to origin shape
    if (needReshape)
      rst = createReshape(rewriter, loc, context, shapeOrigin, rst);

    rewriter.replaceOp(op, rst);
  }
}

namespace {
class InsertConvPass : public InsertConvBase<InsertConvPass> {
public:
  InsertConvPass() = default;
  InsertConvPass(std::string net) { this->net = net; }
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> opWorklist = getPositiveLayers(f);
    MLIRContext *context = &getContext();

    if (opWorklist.empty()) {
      llvm::errs() << "Not run InsertConv\n";
      return;
    }

    if (net == "") {
      insertConv(context, opWorklist);
    } else if (net == "RNN") {
      insertConvRNN(context, opWorklist);
    } else {
      llvm::errs() << "unsupported net: " << net << "\n";
      return;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertConvPass(std::string net) {
  return std::make_unique<InsertConvPass>(net);
}