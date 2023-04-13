
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

#include "macroDef.h"

// this demo branch the colored layer and insert a
// convolution into the left branch.
static void branchLayer(MLIRContext *context, Operation *f, int layer, int branch) {
  // input test
  input_assert(branch < 2,"branch > 1 \n")
  input_assert(layer < 1,"layer > 0 \n")
  // get convolution operations
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  int convLayer = layer;
  f->walk(getConvOp(opWorklist, convLayer));
  // input test: layer
  input_assert(convLayer > 0, "layer <= max_layer(%d) \n", (layer-convLayer))

  auto it = opWorklist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  const int dim = 1;

  Value oldInput = convOp.getOperand(0);
  auto inputShape = oldInput.getShape();
  int inputChannels = inputShape[dim];
  // branch test: channels
  llvm_assert(inputChannels < 2, 
        "error: input_channels(%d) <= 1 \n", inputChannels)
  llvm_assert(inputChannels <= branch, 
        "error: input_channels(%d) <= branch(%d) \n", inputChannels, branch)
 
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);

  // ******************************slice randomly******************************
  std::vector<int> branchChannel(branch);
  //current channels, current branch, min channels, spare channels
  int tempVar[4] = {inputChannels, branch, 0, 0}; 
  srand(time(0));
  for (int i = 0; i < branch; i++) {
    tempVar[2] = tempVar[0] / tempVar[1];
    tempVar[3] = tempVar[0] % tempVar[1];
    branchChannel[i] = tempVar[2] + rand() % (tempVar[3]+1);
    tempVar[0] -= branchChannel[i];
    tempVar[1] -= 1;
  }

  // ******************************slice tensors******************************
  std::vector<decltype(inputShape)> branchShape(branch);
  std::vector<Value> branchTensorOp(branch);
  ValueTensorType branchTensorType;

  int curChannel = 0;  // current channel
  Value startOp;
  Value endOp = createIntOp(rewriter, loc, curChannel);
  Value stepOp = createIntOp(rewriter, loc, 1);
  Value dimOp = createIntOp(rewriter, loc, dim);
  
  for (int i = 0; i < branch; i++) {
    // update shape and type
    branchShape[i] = inputShape;
    branchShape[i][dim] = branchChannel[i];
    branchTensorType = getTensorType(context, branchShape[i], rewriter);
    // update slice tensor
    startOp = endOp;
    curChannel += branchChannel[i];
    endOp = createIntOp(rewriter, loc, curChannel);
    branchTensorOp[i] = rewriter.create<AtenSliceTensorOp>(
          loc, branchTensorType, oldInput, dimOp, startOp, endOp, stepOp);
  }

  // ******************************handle branch tensor******************************
  int handleWay;  // 0: nop, 1: insertSeparaConv 
  srand(time(0));
  for (int i = 0; i < branch; i++) {
    handleWay = rand() % 2;
    if (handleWay == 0) continue;
    // *****************insert a separable convolution******************************
    
    //*************************one kernel****************************
    // new kernel shape: (in_channels,in_channels,1,1)
    auto newShape = branchShape[i];
    newShape[0] = newShape[1];
    newShape[2] = newShape[3] = 1;
    int kernelSize = getKernelSize(newShape);
    // new kernel data
    std::vector<float> oneKernelVec(kernelSize, 0);
    for (int i = 0; i < newShape[1]; i++) {
      oneKernelVec[i * newShape[1] + i] = 1.0;
    }
    // new kernel
    auto kernelTensorType = getTensorType(context, newShape, rewriter);
    auto kernelDense = getDense(newShape, rewriter, oneKernelVec);
    auto oneKernel = createTensorOp(rewriter, loc, kernelTensorType, kernelDense);

    //*************************zero bias****************************
    // zero bias
    getBiasShape(newShape);
    std::vector<float> zeroBiasVec(newShape[0], 0);
    auto biasTensorType = getTensorType(context, newShape, rewriter);
    auto biasDense = getDense(newShape, rewriter, zeroBiasVec);
    auto zeroBias = createTensorOp(rewriter, loc, biasTensorType, biasDense);
    // insert new conv
    branchTensorOp[i] = rewriter.create<AtenConvolutionOp>(
        loc, branchTensorOp[i].getType(), branchTensorOp[i], 
        oneKernel, zeroBias, convParam_3to8(convOp));
  }

  // ******************************cat branch tensors****************************** 
  auto vtensorType = ValueTensorType::getWithLeastStaticInformation(context);
  auto catTensorList = rewriter.create<PrimListConstructOp>(
      loc, ListType::get(vtensorType), ValueRange(branchTensorOp));
  auto catTensorType = getTensorType(context, inputShape, rewriter);
  auto catTensorOp = rewriter.create<AtenCatOp>(loc, catTensorType, catTensorList, dimOp);

  // ******************************replace******************************
  Value newConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getType(), catTensorOp, oldKernel, oldBias, convParam_3to8(convOp));
  rewriter.replaceOp(convOp, newConv);
}

namespace {
  class BranchLayerPass : public BranchLayerBase<BranchLayerPass> {
  public:
  BranchLayerPass() = default;
  BranchLayerPass(int layer, int branch) { 
    this->layer = layer; 
    this->branch = branch; 
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto f = getOperation();
    branchLayer(context, f, this->layer, this->branch);
  }
  };
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createBranchLayerPass(int layer, int branch) {
  return std::make_unique<BranchLayerPass>(layer, branch);
}
