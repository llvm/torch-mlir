//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

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

static void insertSepraConv(MLIRContext *context, Operation *f) {
  // this demo insert a separable convolution

  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  f->walk([&](Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      opWorklist.insert(op);
    }
  });
  if (opWorklist.empty()) {
    llvm::errs() << "Not run InsertSkip\n";
    return;
  }
  auto it = opWorklist.begin(); it++;
  
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  auto shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();

  bool isGroup = false; //是否采用分组卷积
  int group = 1;
  // shape
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  if (isGroup) {
    group = shape[0];
    shape[1] = 1;
  }
  // kernel
  int kernelSize = shape[0] * shape[1] * shape[2] * shape[3];
  std::vector<float> oneKernelVec(kernelSize);
  if(isGroup) {
    // (in_channels,1,1,1)的卷积核，共in_channels组
    for (int i = 0; i < shape[0]; i++) oneKernelVec[i] = 1.0;
  } else {
    // (in_channels,in_channels,1,1)的卷积核，单位矩阵
    for (int i = 0; i < shape[0]; i++) oneKernelVec[i*shape[0] + i] = 1.0;
  }
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape), rewriter.getF32Type());
  auto dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()), llvm::ArrayRef(oneKernelVec));
  Value oneKernel = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // bias
  shape.erase(shape.begin() + 1, shape.end());
  std::vector<float> zeroBiasVec(shape[0], 0);
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape), rewriter.getF32Type());
  dense = DenseElementsAttr::get(
      RankedTensorType::get(llvm::ArrayRef(shape), rewriter.getF32Type()), llvm::ArrayRef(zeroBiasVec));
  Value zeroBias = rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // conv
  Value groupsOp = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(group));
  Value oneConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getOperand(0).getType(), convOp.getOperand(0), oneKernel, zeroBias, 
      convOp.getOperand(3), convOp.getOperand(4), convOp.getOperand(5), convOp.getOperand(6),
      convOp.getOperand(7), groupsOp);
  // replace
  Value newConv = rewriter.create<AtenConvolutionOp>(
    loc, convOp.getType(), oneConv, oldKernel, oldBias, convOp.getOperand(3),
    convOp.getOperand(4), convOp.getOperand(5), convOp.getOperand(6),
    convOp.getOperand(7), convOp.getOperand(8));
  rewriter.replaceOp(convOp, newConv);
}

namespace {
class InsertSepraConvPass : public InsertSepraConvBase<InsertSepraConvPass> {
public:
  InsertSepraConvPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertSepraConv(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertSepraConvPass() {
  return std::make_unique<InsertSepraConvPass>();
}
