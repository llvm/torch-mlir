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

namespace {
class ObfuscateOpsPass : public ObfuscateOpsBase<ObfuscateOpsPass> {
public:
  ObfuscateOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // widen convolution layer
    // this demo only widen first two convolution by adding two channels
    // copy channel 0 and channel 1 to new channels
    // get operations between two convolution
    auto f = getOperation();
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    bool flag = false;
    // make sure the first and last op in opWorklist is conv, and ops in the
    // middle result shape is NCHW
    f.walk([&](mlir::Operation *op) {
      //      if
      //      (op->getResult(0).getType().dyn_cast_or_null<ValueTensorType>())
      // above cause coredump, maybe mlir bug
      // so delay this judgement logic
      if (llvm::dyn_cast<AtenConvolutionOp>(op)) {
        flag = !flag;
        opWorklist.insert(op);
        //        llvm::outs() << op->getName() << "====\n";
      } else if (flag) {
        opWorklist.insert(op);
        //        llvm::outs() << op->getName() << "====\n";
      }
    });

    auto it = opWorklist.begin();
    AtenConvolutionOp conv = llvm::dyn_cast<AtenConvolutionOp>(*it);
    mlir::OpBuilder builder(conv);
    mlir::IRRewriter rewriter(builder);

    // add three channels by copy existing channels, two channel 0 and one
    // channel 1
    Value oldKernel = conv.getOperand(1);
    Value oldBias = conv.getOperand(2);
    auto oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
    auto oldBiasOp = oldBias.getDefiningOp<ValueTensorLiteralOp>();

    // widen conv bias
    // is there better way to get the tensor data?
    std::vector<float> biasVec;
    for (auto i : oldBiasOp.getValue().getValues<float>()) {
      biasVec.push_back(i);
    }
    auto tensorTy = oldBias.getType().cast<ValueTensorType>();
    auto shape = tensorTy.getSizes().vec();
    shape[0] = shape[0] + 3;
    biasVec.push_back(biasVec[0]);
    biasVec.push_back(biasVec[0]);
    biasVec.push_back(biasVec[1]);
    auto resultTensorType = ValueTensorType::get(
        context, llvm::makeArrayRef(shape), tensorTy.getDtype());
    auto dense = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(llvm::makeArrayRef(shape),
                                    builder.getF32Type()),
        llvm::makeArrayRef(biasVec));
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldBiasOp,
                                                      resultTensorType, dense);

    std::vector<float> kernelVec;
    for (auto i : oldKernelOp.getValue().getValues<float>()) {
      kernelVec.push_back(i);
    }
    tensorTy = oldKernel.getType().cast<ValueTensorType>();
    shape = tensorTy.getSizes().vec();
    // kernel layout is CCHW: new channels, old channels, height, width
    shape[0] = shape[0] + 3;
    int channelSize = shape[1] * shape[2] * shape[3];
    kernelVec.insert(kernelVec.end(), kernelVec.begin(),
                     kernelVec.begin() + channelSize);
    kernelVec.insert(kernelVec.end(), kernelVec.begin(),
                     kernelVec.begin() + channelSize);
    kernelVec.insert(kernelVec.end(), kernelVec.begin() + channelSize,
                     kernelVec.begin() + 2 * channelSize);
    resultTensorType = ValueTensorType::get(context, llvm::makeArrayRef(shape),
                                            tensorTy.getDtype());
    dense = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(llvm::makeArrayRef(shape),
                                    builder.getF32Type()),
        llvm::makeArrayRef(kernelVec));
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                      resultTensorType, dense);

    // change shape according to new channel number
    for (; it != opWorklist.end(); it = std::next(it)) {
      // the last op is the second conv, which don't need change result shape
      if (std::next(it) == opWorklist.end())
        break;
      auto op = *it;
      tensorTy = op->getResult(0).getType().dyn_cast_or_null<ValueTensorType>();
      if (tensorTy) {
        shape = tensorTy.getSizes().vec();
        shape[1] += 3;
        resultTensorType = ValueTensorType::get(
            context, llvm::makeArrayRef(shape), tensorTy.getDtype());
        op->getResult(0).setType(resultTensorType);
      }
    }

    // second conv kernel change according to first kernel
    conv = llvm::dyn_cast<AtenConvolutionOp>(*it);
    oldKernel = conv.getOperand(1);
    oldKernelOp = oldKernel.getDefiningOp<ValueTensorLiteralOp>();
    kernelVec.clear();
    for (auto i : oldKernelOp.getValue().getValues<float>()) {
      kernelVec.push_back(i);
    }
    tensorTy = oldKernel.getType().cast<ValueTensorType>();
    shape = tensorTy.getSizes().vec();
    // kernel layout is CCHW: new channels, old channels, height, width
    int hwSize = shape[2] * shape[3];
    channelSize = hwSize * shape[1];
    shape[1] = shape[1] + 3;
    std::vector<float> newKernelVec;
    for (int i = 0; i < shape[0]; i++) {
      int base = i * channelSize;
      for (int j = 0; j < hwSize; j++) {
        kernelVec[base + j] /= 3;
        kernelVec[base + hwSize + j] /= 2;
      }
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                          kernelVec.begin() + base + channelSize);
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                          kernelVec.begin() + base + hwSize);
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base,
                          kernelVec.begin() + base + hwSize);
      newKernelVec.insert(newKernelVec.end(), kernelVec.begin() + base + hwSize,
                          kernelVec.begin() + base + 2 * hwSize);
    }
    resultTensorType = ValueTensorType::get(context, llvm::makeArrayRef(shape),
                                            tensorTy.getDtype());
    dense = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(llvm::makeArrayRef(shape),
                                    builder.getF32Type()),
        llvm::makeArrayRef(newKernelVec));
    rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(oldKernelOp,
                                                      resultTensorType, dense);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createObfuscateOpsPass() {
  return std::make_unique<ObfuscateOpsPass>();
}
