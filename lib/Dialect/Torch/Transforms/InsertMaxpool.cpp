//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <iostream>
#include <random>

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

static void insertMaxpool(MLIRContext *context, Operation *f) {
  // this demo insert a Maxpool into the network

  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  int i = 0;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      i++;
      opWorklist.insert(op);
    } else if (i == 2) {
      opWorklist.insert(op);
      i++;
    }
  });

  auto it = opWorklist.begin();
  it++;
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();
  auto shape =
      convOp.getOperand(0).getType().cast<ValueTensorType>().getSizes().vec();
  Value int0 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value int1 =
      rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));

  // parameters of maxPool2dOp
  // kernel_size
  Value list_kernel = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  // stride
  Value list_stride = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  // padding
  Value list_padding = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int0, int0}));
  // dilation
  Value list_dilation = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange({int1, int1}));
  // ceil_mode
  Value constFalse = rewriter.create<ConstantBoolOp>(loc, false);

  Value maxPool2dOp = rewriter.create<AtenMaxPool2dOp>(
      loc, convOp.getOperand(0).getType(), convOp.getOperand(0), list_kernel,
      list_stride, list_padding, list_dilation, constFalse);

  // parameters of convOp
  // weight
  auto shape_weight =
      convOp.getOperand(1).getType().cast<ValueTensorType>().getSizes().vec();
  int weightSize =
      shape_weight[0] * shape_weight[1] * shape_weight[2] * shape_weight[3];
  std::vector<float> zeroWeightVec(weightSize, 0);
  Value zeroWeight =
      Torch::createTensor(rewriter, loc, context, shape_weight, zeroWeightVec);
  // bias
  auto shape_bias =
      convOp.getOperand(2).getType().cast<ValueTensorType>().getSizes().vec();
  ;
  std::vector<float> zeroBiasVec(shape_bias[0], 0);
  Value zeroBias =
      Torch::createTensor(rewriter, loc, context, shape_bias, zeroBiasVec);
  // conv
  Value list = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(IntType::get(context)), ValueRange());
  Value zeroConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getType(), maxPool2dOp, zeroWeight, zeroBias, list_stride,
      list_padding, list_dilation, constFalse, list, int1);

  rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
      convOp, convOp.getType(), convOp.getOperand(0), convOp.getOperand(1),
      convOp.getOperand(2), convOp.getOperand(3), convOp.getOperand(4),
      convOp.getOperand(5), convOp.getOperand(6), convOp.getOperand(7),
      convOp.getOperand(8));
  it++;
  AtenReluOp reluOp = llvm::dyn_cast<AtenReluOp>(*it);
  // IRRewriter rewriter(context);
  rewriter.setInsertionPoint(reluOp);
  loc = reluOp.getLoc();
  // add
  Value float0 =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0));
  Value skip = rewriter.create<AtenAddTensorOp>(
      loc, reluOp.getType(), reluOp.getOperand(), zeroConv, float0);
  // replace
  rewriter.replaceOpWithNewOp<AtenReluOp>(reluOp, reluOp.getType(), skip);
}

namespace {
class InsertMaxpoolPass : public InsertMaxpoolBase<InsertMaxpoolPass> {
public:
  InsertMaxpoolPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertMaxpool(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertMaxpoolPass() {
  return std::make_unique<InsertMaxpoolPass>();
}