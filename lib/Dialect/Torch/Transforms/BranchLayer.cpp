
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
  llvm_assert(inputChannels % branch != 0, 
        "error: input_channels(%d) %% branch(%d) != 0 \n", inputChannels, branch)
 
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);

  // ******************************slice tensors******************************
  int sliceChannels = inputChannels / branch;
  std::vector<int> branchChannel(branch, sliceChannels);
  std::vector<Value> branchTensorOp(branch);
  //get shape and type
  auto branchShape = inputShape;
  branchShape[dim] = sliceChannels;
  auto branchTensorType = getTensorType(context, branchShape, rewriter);
  // get slice tensor
  int curChannel = 0;
  Value startOp;
  Value endOp = createIntOp(rewriter, loc, curChannel);
  Value stepOp = createIntOp(rewriter, loc, 1);
  Value dimOp = createIntOp(rewriter, loc, dim);
  for (int i = 0; i < branch; i++) {
    startOp = endOp;
    curChannel += branchChannel[dim];
    endOp = createIntOp(rewriter, loc, curChannel);
    branchTensorOp[i] = rewriter.create<AtenSliceTensorOp>(
          loc, branchTensorType, oldInput, dimOp, startOp, endOp, stepOp);
  }

  //****************************one kernel and zero bias*******************************
  // new kernel shape: (in_channels,in_channels,1,1)
  auto newShape = branchShape;
  newShape[0] = newShape[1];
  newShape[2] = newShape[3] = 1;
  int kernelSize = getKernelSize(newShape);
  // new kernel data
  std::vector<float> oneKernelVec(kernelSize, 0);
  for (int i = 0; i < newShape[1]; i++) {
    oneKernelVec[i*newShape[1] + i] = 1.0;
  }
  // new kernel
  auto kernelTensorType = getTensorType(context, newShape, rewriter);
  auto kernelDense = getDense(newShape, rewriter, oneKernelVec);
  auto oneKernel = createTensorOp(rewriter, loc, kernelTensorType, kernelDense);

  // zero bias
  getBiasShape(newShape);
  std::vector<float> zeroBiasVec(newShape[0], 0);
  auto biasTensorType = getTensorType(context, newShape, rewriter);
  auto biasDense = getDense(newShape, rewriter, zeroBiasVec);
  auto zeroBias = createTensorOp(rewriter, loc, biasTensorType, biasDense);

  // ******************************handle branch tensor******************************
  // handle way: 0 -> nop, 1 -> insertSeparaConv 
  int handleWay;  
  srand(time(0));
  for (int i = 0; i < branch; i++) {
    handleWay = rand() % 2;
    if (handleWay == 0) continue;
    // insert a separable convolution
    branchTensorOp[i] = rewriter.create<AtenConvolutionOp>(
        loc, branchTensorType, branchTensorOp[i], oneKernel, zeroBias, 
        convParam_3to8(convOp));
  }

  // ******************************cat branch tensors****************************** 
  auto catTensorList = rewriter.create<PrimListConstructOp>(
          loc, ListType::get(branchTensorType), ValueRange(branchTensorOp));
  auto catTensorType = getTensorType(context, inputShape, rewriter);
  auto catTensorOp = rewriter.create<AtenCatOp>(loc, catTensorType, catTensorList, dimOp);
  
  // ******************************replace******************************
  Value newConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getType(), catTensorOp, oldKernel, oldBias, convParam_3to8(convOp));
  rewriter.replaceOp(convOp, newConv);
}

#include "macroUndef.h"

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
