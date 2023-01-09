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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ObfuscateMM : public OpRewritePattern<AtenMmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMmOp op,
                                PatternRewriter &rewriter) const override {
    // just for test
    Location loc = op.getLoc();
    Value input0 = op.getOperand(0);
    Value input1 = op.getOperand(1);
    auto resultTensorType = op.getType().cast<BaseTensorType>();
    Value add =
        rewriter.create<AtenDivTensorOp>(loc, resultTensorType, input0, input0);
    Value add1 =
        rewriter.create<AtenDivTensorOp>(loc, resultTensorType, input1, input1);
    Value result =
        rewriter.create<AtenDivTensorOp>(loc, resultTensorType, add, add1);
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTensorType,
                                                        result);
    return success();
  }
};
} // namespace

static void
widenConvLayer(MLIRContext *context,
               llvm::SmallPtrSet<mlir::Operation *, 16> &opWorklist) {
  // widen conv layer by add instruction to change weight matrix of convolution
  // layer, maybe the better way is changing the weight matrix at definition
  // place
  auto it = opWorklist.begin();
  AtenConvolutionOp conv = llvm::dyn_cast<AtenConvolutionOp>(*it);
  mlir::OpBuilder builder(conv);
  Location loc = conv.getLoc();

  // widen first convolution
  Value oldKernel = conv.getOperand(1);
  Value oldBias = conv.getOperand(2);
  auto tensorTy = oldKernel.getType().cast<ValueTensorType>();
  auto shapeNewKernel = tensorTy.getSizes().vec();
  // kernel layout is CCHW: new channels, old channels, height, width
  shapeNewKernel[0] = shapeNewKernel[0] + 2;
  auto resultTensorType = ValueTensorType::get(
      context, llvm::makeArrayRef(shapeNewKernel), tensorTy.getDtype());
  auto dense = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(llvm::makeArrayRef(shapeNewKernel),
                                  builder.getF32Type()),
      llvm::makeArrayRef(static_cast<float>(0.0)));
  Value newKernel =
      builder.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  Value intn0 = builder.create<ConstantIntOp>(
      loc, builder.getI64IntegerAttr(shapeNewKernel[0] - 2));
  Value intn1 = builder.create<ConstantIntOp>(
      loc, builder.getI64IntegerAttr(shapeNewKernel[0] - 1));
  Value int0 = builder.create<ConstantIntOp>(loc, builder.getI64IntegerAttr(0));
  Value int1 = builder.create<ConstantIntOp>(loc, builder.getI64IntegerAttr(1));
  Value constFalse = builder.create<ConstantBoolOp>(loc, false);
  std::vector<long> shapeCHW = {shapeNewKernel[1], shapeNewKernel[2],
                                shapeNewKernel[3]};
  resultTensorType = ValueTensorType::get(context, llvm::makeArrayRef(shapeCHW),
                                          tensorTy.getDtype());
  Value oldSelect = builder.create<AtenSelectIntOp>(loc, resultTensorType,
                                                    oldKernel, int0, int0);
  Value newSelect = builder.create<AtenSelectIntOp>(loc, resultTensorType,
                                                    newKernel, int0, intn0);
  builder.create<AtenCopy_Op>(loc, resultTensorType, newSelect, oldSelect,
                              constFalse);
  oldSelect = builder.create<AtenSelectIntOp>(loc, resultTensorType, oldKernel,
                                              int0, int1);
  newSelect = builder.create<AtenSelectIntOp>(loc, resultTensorType, newKernel,
                                              int0, intn1);
  builder.create<AtenCopy_Op>(loc, resultTensorType, newSelect, oldSelect,
                              constFalse);
  // bias
  std::vector<long> shapeNewBias = {shapeNewKernel[0]};
  resultTensorType = ValueTensorType::get(
      context, llvm::makeArrayRef(shapeNewBias), tensorTy.getDtype());
  dense = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get(llvm::makeArrayRef(shapeNewBias),
                                  builder.getF32Type()),
      llvm::makeArrayRef(static_cast<float>(0.0)));
  Value newBias =
      builder.create<ValueTensorLiteralOp>(loc, resultTensorType, dense);
  builder.create<AtenCopy_Op>(loc, resultTensorType, newBias, oldBias,
                              constFalse);
  auto shapeNewConv = conv.getType().cast<ValueTensorType>().getSizes().vec();
  shapeNewConv[1] = shapeNewKernel[0];
  resultTensorType = ValueTensorType::get(
      context, llvm::makeArrayRef(shapeNewConv), tensorTy.getDtype());
  Value newConv = builder.create<AtenConvolutionOp>(
      loc, resultTensorType, conv.getOperand(0), newKernel, newBias,
      conv.getOperand(3), conv.getOperand(4), conv.getOperand(5),
      conv.getOperand(6), conv.getOperand(7), conv.getOperand(8));
  mlir::IRRewriter rewriter(builder);
  rewriter.replaceOp(conv, newConv);
  // infer shape for relu and max_pool
  // todo
}

namespace {
class ObfuscateOpsPass : public ObfuscateOpsBase<ObfuscateOpsPass> {
public:
  ObfuscateOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    //    patterns.add<ObfuscateMM>(context);
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoIterationLimit;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }

    // widen convolution layer
    // this demo only widen first two convolution by adding two channels
    // copy channel 0 and channel 1 to new channels
    // get operations between two convolution
    auto f = getOperation();
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    bool flag = false;
    f.walk([&](mlir::Operation *op) {
      if (llvm::dyn_cast<AtenConvolutionOp>(op)) {
        flag = !flag;
        opWorklist.insert(op);
      } else if (flag) {
        opWorklist.insert(op);
      }
    });

//    widenConvLayer(context, opWorklist);
    auto it = opWorklist.begin();
    AtenConvolutionOp conv = llvm::dyn_cast<AtenConvolutionOp>(*it);
    mlir::OpBuilder builder(conv);
    Location loc = conv.getLoc();

    Value oldKernel = conv.getOperand(1);
    Value oldBias = conv.getOperand(2);
    auto oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
    auto oldBiasOp = oldBias.getDefiningOp<ValueTensorLiteralOp>();
    llvm::outs() << oldKernelOp.getValueAttrName() <<"\n";
    llvm::outs() << oldKernelOp.getValue().getElementType() << "\n";
    llvm::outs() << oldKernelOp.getValue().getType() << "\n";
    llvm::outs() << oldKernelOp.getValue().getNumElements() << "\n";
    for (auto i=oldBiasOp.getValue().value_begin<float>();i!=oldBiasOp.getValue().value_end<float>();++i) {
      llvm::outs() << *i << "=\n";
    }
    for (auto i : oldBiasOp.getValue().getValues<float>() ) {
      llvm::outs() << i << "=\n";
      i = 1;
    }
    // because of __elided, no data can be read
    for (auto i : oldKernelOp.getValue().getValues<float>() ) {
      llvm::outs() << i << "=\n";
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createObfuscateOpsPass() {
  return std::make_unique<ObfuscateOpsPass>();
}
