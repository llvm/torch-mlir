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

static void insertSkip(MLIRContext *context, Operation *f) {
  // this demo insert a skip for the second convolution

  llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
  f->walk([&](mlir::Operation *op) {
    if (isa<AtenConvolutionOp>(op)) {
      opWorklist.insert(op);
    }
  });
  auto it = opWorklist.begin();
  it++;
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  mlir::OpBuilder builder(convOp);
  mlir::IRRewriter rewriter(builder);
  Location loc = convOp.getLoc();

  // create a new conv with zero kernel and bias, to make sure output is the
  // same as input
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  // kernel
  // shape: (new channels, old channels, height, width)
  auto shape = oldKernel.getType().cast<ValueTensorType>().getSizes().vec();
  // todo: 不确定有没有理解错，是添加一个1x1卷积核的卷积层吗
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  int kernelSize = shape[0] * shape[1] * shape[2] * shape[3];
  std::vector<float> zeroKernelVec(kernelSize, 0);
  auto resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                               builder.getF32Type());
  auto dense = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(llvm::ArrayRef(shape), builder.getF32Type()),
      llvm::ArrayRef(zeroKernelVec));
  Value zeroKernel =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // bias
  shape.erase(shape.begin() + 1, shape.end());
  std::vector<float> zeroBiasVec(shape[0], 0);
  resultTensorType = ValueTensorType::get(context, llvm::ArrayRef(shape),
                                          builder.getF32Type());
  dense = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(llvm::ArrayRef(shape), builder.getF32Type()),
      llvm::ArrayRef(zeroBiasVec));
  Value zeroBias =
      rewriter.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  // conv
  Value zeroConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getOperand(0).getType(), convOp.getOperand(0), zeroKernel,
      zeroBias, convOp.getOperand(3), convOp.getOperand(4),
      convOp.getOperand(5), convOp.getOperand(6), convOp.getOperand(7),
      convOp.getOperand(8));
  // add
  Value float0 =
      rewriter.create<ConstantFloatOp>(loc, builder.getF64FloatAttr(0));
  Value skip = rewriter.create<AtenAddTensorOp>(
      loc, zeroConv.getType(), convOp.getOperand(0), zeroConv, float0);
  rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
      convOp, convOp.getType(), skip, oldKernel, oldBias, convOp.getOperand(3),
      convOp.getOperand(4), convOp.getOperand(5), convOp.getOperand(6),
      convOp.getOperand(7), convOp.getOperand(8));
}

namespace {
class InsertSkipPass : public InsertSkipBase<InsertSkipPass> {
public:
  InsertSkipPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    insertSkip(context, f);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertSkipPass() {
  return std::make_unique<InsertSkipPass>();
}

