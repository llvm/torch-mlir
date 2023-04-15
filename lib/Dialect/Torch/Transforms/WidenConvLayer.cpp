//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// widen two convolution layer by adding channels
// randomly copy channel to new channels
static void WidenConvLayer(MLIRContext *context, Operation *f, int layer,
                           int number) {
  // input test
  input_assert(layer < 1, "layer > 0 \n");
  input_assert(number < 1, "number > 0 \n");
  // get operations between first two convolution(include convolutions)
  OpList oplist;
  bool is_get = getConvMiddleOps(oplist, f, layer);
  if (!is_get)
    return;
  // get the first convolution
  auto it = oplist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  // init rewrite
  RewriteOp rewrite(context, convOp);
  // get tensor
  auto oldKernelTensor = rewrite.getKernelTensor();
  auto oldBiasTensor = rewrite.getBiasTensor();

  // get random channels to copy
  auto shape = rewrite.getKernelShape();
  std::vector<int> randomChannel(number);   // index of channel to copy
  std::vector<int> copyNumber(shape[0], 1); // number of copying every channel
  srand(time(0));
  for (int i = 0; i < number; i++) {
    int index = rand() % shape[0];
    randomChannel[i] = index;
    copyNumber[index] += 1;
  }

  // widen kernel of the first convolution
  int channelSize = getChannelSize(shape);
  std::vector<float> kernelVec;
  copyTensor(kernelVec, oldKernelTensor);
  // copy kernel channel
  shape[0] = shape[0] + number;
  for (auto channel : randomChannel) {
    auto begin = channel * channelSize;
    pushBackVec(kernelVec, begin, channelSize);
  }
  // replace old kernel tensor
  rewrite.replaceTensorOp(oldKernelTensor, shape, kernelVec);

  // widen bias of the first convolution
  shape = rewrite.getBiasShape();
  std::vector<float> biasVec;
  copyTensor(biasVec, oldBiasTensor);
  // copy bias
  shape[0] = shape[0] + number;
  for (auto channel : randomChannel) {
    biasVec.push_back(biasVec[channel]);
  }
  // replace old bias tensor
  rewrite.replaceTensorOp(oldBiasTensor, shape, biasVec);

  // widen middle operations between first two convolution(not include the
  // second conv)
  for (; it != oplist.end(); it = std::next(it)) {
    if (std::next(it) == oplist.end())
      break;
    auto op = *it;
    auto opResult = op->getResult(0);
    auto tensorType = opResult.getType().dyn_cast<ValueTensorType>();
    if (tensorType) {
      shape = getShape(opResult);
      shape[1] += number;
      auto resultTensorType = rewrite.getValueTensorType(shape);
      opResult.setType(resultTensorType);
    }
  }

  // widen kernel of the second convolution, only widen kernel, no need to widen
  // bias

  // get kernel information
  convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  rewrite.setConvOp(convOp);
  oldKernelTensor = rewrite.getKernelTensor();

  // widen kernel of the first convolution
  kernelVec.clear();
  copyTensor(kernelVec, oldKernelTensor);
  shape = rewrite.getKernelShape();
  channelSize = getChannelSize(shape);
  // copy kernel
  int hwSize = shape[2] * shape[3];
  std::vector<float> newKernelVec;
  // update and copy in_channels data for every out_channels
  for (int i = 0; i < shape[0]; i++) {
    auto base = i * channelSize;
    // update
    for (int j = 0; j < shape[1]; j++) {
      if (copyNumber[j] == 1)
        continue;
      for (int k = 0; k < hwSize; k++) {
        auto index = base + j * hwSize + k;
        kernelVec[index] /= copyNumber[j];
      }
    }
    // copy
    pushBackVec(newKernelVec, kernelVec, base, channelSize);
    for (auto channel : randomChannel) {
      auto begin = base + channel * hwSize;
      pushBackVec(newKernelVec, kernelVec, begin, hwSize);
    }
  }
  shape[1] = shape[1] + number;
  // replace old kernel tensor
  rewrite.replaceTensorOp(oldKernelTensor, shape, newKernelVec);
}

use_pass(WidenConvLayer, 2, int, layer, int, number);
