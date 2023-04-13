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

// widen convolution layer
// this demo widen two convolution by adding three channels
// randomly copy channel to new channels
static void widenConvLayer(MLIRContext *context, Operation *f,int layer, int number) {
  //input test
  input_assert(layer < 1,"layer > 0 \n")
  input_assert(number < 1,"number > 0 \n")
  // get operations between first two convolution(include convolutions)
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  int convLayer = layer;
  f->walk(getMiddleOps(opWorklist,convLayer));
  //input test
  input_assert(convLayer > -1, "layer < max_layer(%d) \n", (layer-convLayer))

  auto it = opWorklist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);

  //******************************widen conv1*****************************
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);
  auto oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
  auto oldBiasOp = oldBias.getDefiningOp<ValueTensorLiteralOp>();

  //******************************get random channels*****************************
  // kernel shape: out_channels, in_channels, height, width
  auto shape = oldKernel.getShape();
  std::vector<int> randomChannel(number);   //index of channel to copy
  std::vector<int> copyNumber(shape[0], 1); //number of copying every channel 
  srand(time(0));
  for (int i = 0; i < number; i++) {
    int index =  rand() % shape[0];
    randomChannel[i] = index;
    copyNumber[index] += 1;
  }

  //******************************widen kernel of conv1*****************************
  int channelSize = getChannelSize(shape);
  std::vector<float> kernelVec;
  copyValueTensor(kernelVec, oldKernelOp)
  // copy kernel
  shape[0] = shape[0] + number;
  for (auto channel : randomChannel) {
    auto begin = channel * channelSize;
    pushBackVec(kernelVec, kernelVec, begin, channelSize);
  }
  // create a constant tensor of float type
  auto kernelTensorType = getTensorType(context, shape, rewriter);
  auto kernelDense = getDense(shape, rewriter, kernelVec);
  rewriter.replaceValueTensorOp(oldKernelOp, kernelTensorType, kernelDense);

  //******************************widen bias of conv1*****************************
  shape = oldBias.getShape();
  std::vector<float> biasVec;
  copyValueTensor(biasVec, oldBiasOp)
  // copy bias
  shape[0] = shape[0] + number;
  for (auto channel : randomChannel) {
    biasVec.push_back(biasVec[channel]);
  }
  // create a constant tensor of float type
  auto biasTensorType = getTensorType(context, shape,rewriter);
  auto biasDense = getDense(shape, rewriter, biasVec);
  rewriter.replaceValueTensorOp(oldBiasOp, biasTensorType, biasDense);

  //******************************widen middle ops*****************************
  for (; it != opWorklist.end(); it = std::next(it)) {
    // the last op is the second conv, which don't need change result shape
    if (std::next(it) == opWorklist.end()) break;
    auto op = *it;
    auto opResult = op->getResult(0);
    auto tensorType = opResult.getType().dyn_cast<ValueTensorType>();
    if (tensorType) {
      shape = opResult.getShape();
      shape[1] += number;
      auto resultTensorType = getTensorType(context, shape, rewriter);
      opResult.setType(resultTensorType);
    }
  }

  //******************************widen conv2*****************************
  //only widen kernel, no need to widen bias
  convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  oldKernel = convOp.getOperand(1);
  oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
 
  //******************************widen kernel of conv2*****************************
  kernelVec.clear();
  copyValueTensor(kernelVec, oldKernelOp)
  // kernel shape: out_channels, in_channels, height, width
  shape = oldKernel.getShape();
  channelSize = getChannelSize(shape);
  // copy kernel
  int hwSize = shape[2] * shape[3];
  std::vector<float> newKernelVec;

  for (int i = 0; i < shape[0]; i++) {
    auto base = i * channelSize; 
    // update in_channel data
    for (int j = 0; j < shape[1]; j++) {
      if (copyNumber[j] == 1) continue;
      for (int k = 0; k < hwSize; k++) {
        auto index = base + j * hwSize + k;
        kernelVec[index] /= copyNumber[j];
      }
    }
    //copy in_channel data
    pushBackVec(newKernelVec, kernelVec, base, channelSize);
    for (auto channel : randomChannel) {
      auto begin = base + channel * hwSize;
      pushBackVec(newKernelVec, kernelVec, begin, hwSize);
    }
  }
  shape[1] = shape[1] + number;
  // create a constant tensor of float type
  kernelTensorType = getTensorType(context, shape, rewriter);
  kernelDense = getDense(shape, rewriter, newKernelVec);
  rewriter.replaceValueTensorOp(oldKernelOp, kernelTensorType, kernelDense);
}


namespace {
  class WidenConvLayerPass : public WidenConvLayerBase<WidenConvLayerPass> {
  public:
    WidenConvLayerPass() = default;
    WidenConvLayerPass(int layer, int number) { 
      this->layer = layer; 
      this->number = number; 
    }
    void runOnOperation() override {
      MLIRContext *context = &getContext();
      auto f = getOperation();
      widenConvLayer(context, f, this->layer, this->number);
    }
  };
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createWidenConvLayerPass(int layer, int number) {
  return std::make_unique<WidenConvLayerPass>(layer, number);
}

