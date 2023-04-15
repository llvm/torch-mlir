//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

// insert a separable convolution
static void InsertSepraConv(MLIRContext *context, Operation *f, int layer) {
  // input test
  input_assert(layer < 1, "layer > 0 \n");
  // get operations that you need
  OpList oplist;
  bool is_get = getConvOp(oplist, f, layer);
  if (!is_get)
    return;
  // get convolution operations
  auto it = oplist.begin();
  AtenConvolutionOp convOp = llvm::dyn_cast<AtenConvolutionOp>(*it);
  // init rewrite
  RewriteOp rewrite(context, convOp);

  // get one kernel
  auto shape = rewrite.getKernelShape();
  toStdShape(shape);
  int kernelSize = getKernelSize(shape);
  std::vector<float> oneKernelVec(kernelSize);
  creatOneTensor(oneKernelVec, shape[0]);
  Value oneKernel = rewrite.createTensorOp(shape, oneKernelVec);
  // get zero bias
  toBiasShape(shape);
  std::vector<float> zeroBiasVec(shape[0], 0);
  Value zeroBias = rewrite.createTensorOp(shape, zeroBiasVec);
  // insert new conv
  Value oldInput = rewrite.getInput();
  Value oneConv = rewrite.createConvOp(oldInput, oneKernel, zeroBias);
  // replace old conv
  rewrite.replaceConvOp(oneConv);
}

use_pass(InsertSepraConv, 1, int, layer)
