//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

Value mlir::torch::onnx_c::createConstantIntList(
    OpBinder binder, ConversionPatternRewriter &rewriter,
    SmallVector<int64_t> cstInput) {
  SmallVector<Value> cstValue;
  for (int64_t i : cstInput) {
    cstValue.push_back(rewriter.create<Torch::ConstantIntOp>(
        binder.getLoc(), rewriter.getI64IntegerAttr(i)));
  }
  return rewriter.create<Torch::PrimListConstructOp>(
      binder.getLoc(),
      Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
      cstValue);
}
