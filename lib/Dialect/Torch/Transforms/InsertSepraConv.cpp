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
// this demo insert a separable convolution
static void insertSepraConv(MLIRContext *context, Operation *f, int layer) {
  // input test
  input_assert(layer < 1,"layer > 0 \n")
  // get convolution operations
  llvm::SmallPtrSet<Operation *, 16> opWorklist;
  int convLayer = layer;
  f->walk(getConvOp(opWorklist, convLayer));
  // input test
  input_assert(convLayer > 0, "layer <= max_layer(%d) \n", (layer-convLayer))

  auto it = opWorklist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(convOp);
  Location loc = convOp.getLoc();

  Value oldInput = convOp.getOperand(0);
  Value oldKernel = convOp.getOperand(1);
  Value oldBias = convOp.getOperand(2);

  //*************************kernel****************************
  // kernel shape: out_channels, in_channels, height, width
  auto shape = oldKernel.getShape();
  int isGroup = random() % 2; //use groups of convolution
  int group = 1;
  // new kernel shape
  shape[0] = shape[1];
  shape[2] = shape[3] = 1;
  if (isGroup) {
    group = shape[0];
    shape[1] = 1;
  }
  // new kernel data
  int kernelSize = getKernelSize(shape);
  std::vector<float> oneKernelVec(kernelSize);
  if(isGroup) {
    // (in_channels,1,1,1), group = in_channel
    for (int i = 0; i < shape[0]; i++) {
        oneKernelVec[i] = 1.0;
    }
  } else {
    // (in_channels,in_channels,1,1)
    for (int i = 0; i < shape[0]; i++) {
        oneKernelVec[i*shape[0] + i] = 1.0;
    } 
  }
  // new kernel
  auto kernelTensorType = getTensorType(context, shape, rewriter);
  auto kernelDense = getDense(shape, rewriter, oneKernelVec);
  auto oneKernel = createTensorOp(rewriter, loc, kernelTensorType, kernelDense);
  
  //*************************bias****************************
  // zero bias
  getBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  auto biasTensorType = getTensorType(context, shape, rewriter);
  auto biasDense = getDense(shape, rewriter, zeroBiasVec);
  auto zeroBias = createTensorOp(rewriter, loc, biasTensorType, biasDense);
  // insert new conv
  Value groupsOp = createIntOp(rewriter, loc, group);
  Value oneConv = rewriter.create<AtenConvolutionOp>(
      loc, oldInput.getType(), oldInput, oneKernel, zeroBias, 
      convParam_3to7(convOp), groupsOp);
  // replace old conv
  Value newConv = rewriter.create<AtenConvolutionOp>(
      loc, convOp.getType(), oneConv, oldKernel, oldBias, convParam_3to8(convOp));
  rewriter.replaceOp(convOp, newConv);
}


namespace {
  class InsertSepraConvPass : public InsertSepraConvBase<InsertSepraConvPass> {
  public:
    InsertSepraConvPass() = default;
    InsertSepraConvPass(int layer) {
      this->layer = layer;
    }
    void runOnOperation() override {
      MLIRContext *context = &getContext();
      auto f = getOperation();
      insertSepraConv(context, f, this->layer);
    }
  };
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createInsertSepraConvPass(int layer) {
  return std::make_unique<InsertSepraConvPass>(layer);
}
