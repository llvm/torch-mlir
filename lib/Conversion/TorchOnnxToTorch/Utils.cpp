//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

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

Type mlir::torch::onnx_c::getQTorchTypeFromTorchIntType(Type ty) {
  Torch::ValueTensorType tty = dyn_cast<Torch::ValueTensorType>(ty);
  if (!tty)
    return nullptr;

  auto ctx = ty.getContext();
  Type dty = tty.getDtype();

  if (dty.isUnsignedInteger(8))
    dty = Torch::QUInt8Type::get(ctx);
  if (dty.isSignedInteger(8))
    dty = Torch::QInt8Type::get(ctx);
  if (dty.isSignedInteger(32))
    dty = Torch::QInt32Type::get(ctx);

  if (!dty)
    return nullptr;
  return Torch::ValueTensorType::get(ctx, tty.getOptionalSizes(), dty);
}
