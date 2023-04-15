//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// this demo insert a skip for the second convolution
static void InsertSkip(MLIRContext *context, Operation *f, int layer) {
  // input test
  input_assert(layer < 1,"layer > 0 \n")
  // get operations that you need
  OpList oplist;
  bool is_get = getConvOp(oplist, f, layer);
  if (!is_get) return;
  // get convolution operations
  auto it = oplist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  // init rewrite
  RewriteOp rewrite(context, convOp);
  // get zero kernel
  auto shape = rewrite.getKernelShape();
  toStdShape(shape);
  int kernelSize = getKernelSize(shape);
  std::vector<float> zeroKernelVec(kernelSize, 0);
  Value zeroKernel = rewrite.createTensorOp(shape, zeroKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  auto zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // zero conv
  Value oldInput = rewrite.getInput();
  Value zeroConv = rewrite.createConvOp(oldInput, zeroKernel, zeroBias);
  // add zero conv
  // Value float0 = rewrite.createFloatOp(0);
  Value int1 = rewrite.createIntOp(1);
  Value skip = rewrite.createAddTensorOp(zeroConv.getType(), oldInput, zeroConv, int1);
  // replace old conv
  rewrite.replaceConvOp(skip);
}

use_pass(InsertSkip, 1, int, layer)

