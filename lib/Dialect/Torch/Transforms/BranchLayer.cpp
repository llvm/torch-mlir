
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

static void branchLayer(MLIRContext *context, Operation *f) {
  // this demo branch the colored layer and insert a
  // convolution into the left branch.

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

  Value convResult = convOp.getOperand(0);
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  auto resultShape = convResult.getType().cast<ValueTensorType>().getSizes().vec();

  // ******************************slice tensor******************************
  const int dim = 1;
  auto halfShape = resultShape; halfShape[dim] = resultShape[dim]/2;
  int halfLen = resultShape[dim]/2, wholeLen = resultShape[dim];
  auto newTensorType = ValueTensorType::get(context, llvm::ArrayRef(halfShape), rewriter.getF32Type());

  Value start1 = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value end1 = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(halfLen));  
  Value start2 = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(halfLen));
  Value end2 = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(wholeLen)); 
  Value step = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value kernelDim = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(dim));

  Value newResult1 = rewriter.create<AtenSliceTensorOp>(loc, newTensorType,
                  convResult, kernelDim, start1, end1, step);
  Value newResult2 = rewriter.create<AtenSliceTensorOp>(loc, newTensorType,
                  convResult, kernelDim, start2, end2, step);

  // ******************************handle tensor1******************************
  // 添加(in_channels,in_channels,1,1)的卷积核，单位矩阵
  auto newShape1 = halfShape;
  newShape1[0] = newShape1[1] = halfShape[dim];
  newShape1[2] = newShape1[3] = 1;
  // kernel
  int kernelSize1 = newShape1[0] * newShape1[1] * newShape1[2] * newShape1[3];
  std::vector<float> kernelVec1(kernelSize1, 0);
  for (int i = 0; i < newShape1[1]; i++) {
    kernelVec1[i*newShape1[1] + i] = 1.0;
  }
  auto tensorType1 = ValueTensorType::get(context, llvm::ArrayRef(newShape1), rewriter.getF32Type());                                
  auto dense1 = DenseElementsAttr::get(
    RankedTensorType::get(llvm::ArrayRef(newShape1), rewriter.getF32Type()), llvm::ArrayRef(kernelVec1));
  Value kernel1 = rewriter.create<ValueTensorLiteralOp>(loc, tensorType1, dense1);
  // bias
  newShape1.erase(newShape1.begin() + 1, newShape1.end());
  std::vector<float> zeroBiasVec1(newShape1[0], 0);
  tensorType1 = ValueTensorType::get(context, llvm::ArrayRef(newShape1), rewriter.getF32Type());
  dense1 = DenseElementsAttr::get(
    RankedTensorType::get(llvm::ArrayRef(newShape1), rewriter.getF32Type()), llvm::ArrayRef(zeroBiasVec1));
  Value bias1 = rewriter.create<ValueTensorLiteralOp>(loc, tensorType1, dense1);
  // conv
  newResult1 = rewriter.create<AtenConvolutionOp>(
      loc, newResult1.getType(), newResult1, kernel1, bias1, 
      convOp.getOperand(3), convOp.getOperand(4), convOp.getOperand(5), 
      convOp.getOperand(6), convOp.getOperand(7), convOp.getOperand(8));
  
  // ******************************cat tensors******************************
  Value catTensors = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(newTensorType), ValueRange({newResult1, newResult2}));
  newTensorType = ValueTensorType::get(context, llvm::ArrayRef(resultShape), rewriter.getF32Type());
  Value catKernelOp = rewriter.create<AtenCatOp>(loc, newTensorType, catTensors, kernelDim);
  
  // ******************************replace op******************************
  Value newConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getType(), catKernelOp, oldKernel, oldBias, convOp.getOperand(3),
      convOp.getOperand(4), convOp.getOperand(5), convOp.getOperand(6),
      convOp.getOperand(7), convOp.getOperand(8));
  rewriter.replaceOp(convOp, newConv);
}

namespace {
class BranchLayerPass : public BranchLayerBase<BranchLayerPass> {
public:
  BranchLayerPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    branchLayer(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createBranchLayerPass() {
  return std::make_unique<BranchLayerPass>();
}
