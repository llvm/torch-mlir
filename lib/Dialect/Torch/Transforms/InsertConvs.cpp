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

static void insertConv(MLIRContext *context, Operation *f, int number) {
  // insert invariant convolutions

  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
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

  std::srand(std::time(0));
  for (int i = 0; i < number; i++) {
    // select a random place to insert
    Operation *originOp =
        *(std::next(opWorklist.begin(), std::rand() % opWorklist.size()));
    IRRewriter rewriter(context);
    rewriter.setInsertionPointAfter(originOp);
    // copy originOp, for convinience of replace use of op
    Operation *op = rewriter.clone(*originOp);
    Location loc = op->getLoc();

    // create other oprands for conv
    int padNum = std::rand() % 10;
    Value int0 =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value int1 =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value intPad =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(padNum));
    Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value listInt1_1 = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
    Value listIntPad_Pad = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)),
        ValueRange({intPad, intPad}));
    Value listInt = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), ValueRange({}));
    // create unsqueeze if dimansion less than 4, such as : (1,84) -> (1,1,1,84)
    Value rst = op->getResult(0);
    std::vector<long> shape =
        rst.getType().cast<ValueTensorType>().getSizes().vec();
    int squeezeTime = 4 - shape.size();
    if (squeezeTime < 0) {
      continue;
    }
    for (int i = 0; i < squeezeTime; i++) {
      shape.insert(shape.begin(), 1);
      rst = rewriter.create<AtenUnsqueezeOp>(
          loc,
          ValueTensorType::get(context, llvm::ArrayRef(shape),
                               rewriter.getF32Type()),
          rst, int0);
    }
    // create unit tensor as convolution kernel
    // new kernel size is: ChannelSz  x ChannelSz  x kernelSz x kernelSz
    int ChannelSz = shape[1];
    int kernelSz = 2 * padNum + 1;
    shape[0] = ChannelSz;
    shape[2] = shape[3] = kernelSz;
    std::vector<float> unitWeightVec(ChannelSz * ChannelSz *kernelSz*kernelSz, 0);
    for (int i = 0; i < ChannelSz; i++) {
      // unitWeightVec[i][i][padNum][padNum] = 1
      unitWeightVec[((i * ChannelSz + i) * kernelSz + padNum) * kernelSz +
                    padNum] = 1;
    }
    auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                                 rewriter.getF32Type());
    auto dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
        llvm::ArrayRef(unitWeightVec));
    Value unitWeight =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    // create zero bias
    shape.erase(shape.begin() + 1, shape.end());
    std::vector<float> zeroBiasVec(shape[0], 0);
    resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                            rewriter.getF32Type());
    dense = DenseElementsAttr::get(
        RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()),
        llvm::ArrayRef(zeroBiasVec));
    Value zeroBias =
        rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
    // create conv
    rst = rewriter.create<AtenConvolutionOp>(
        loc, rst.getType(), rst, unitWeight, zeroBias, listInt1_1, listIntPad_Pad,
        listInt1_1, constFalse, listInt, int1);
    // create relu
    rst = rewriter.create<AtenReluOp>(loc, rst.getType(), rst);
    // create squeeze
    for (int i = 0; i < squeezeTime; i++) {
      shape = rst.getType().cast<ValueTensorType>().getSizes().vec();
      shape.erase(shape.begin());
      rst = rewriter.create<AtenSqueezeDimOp>(
          loc,
          ValueTensorType::get(context, llvm::ArrayRef(shape),
                               rewriter.getF32Type()),
          rst, int0);
    }

    rewriter.replaceOp(originOp, rst);
    opWorklist.erase(originOp);
    opWorklist.insert({op});
  }
}

namespace {
class InsertConvPass : public InsertConvBase<InsertConvPass> {
public:
  InsertConvPass() = default;
  InsertConvPass(int number) { this->number = number; }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertConv(context, f, number);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertConvPass(int number) {
  return std::make_unique<InsertConvPass>(number);
}