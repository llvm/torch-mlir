//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

Value mlir::torch::onnx_c::createConstantIntList(
    OpBinder binder, ConversionPatternRewriter &rewriter,
    ArrayRef<int64_t> cstInput) {
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

Torch::ValueTensorType
mlir::torch::onnx_c::getQTorchTypeFromTorchIntType(Type ty) {
  Torch::ValueTensorType tty = dyn_cast<Torch::ValueTensorType>(ty);
  if (!tty)
    return nullptr;

  auto ctx = ty.getContext();
  Type dty = tty.getDtype();

  if (dty.isUnsignedInteger(8))
    dty = Torch::QUInt8Type::get(ctx);
  if (dty.isSignedInteger(8))
    dty = Torch::QInt8Type::get(ctx);
  if (dty.isSignedInteger(16))
    dty = Torch::QInt16Type::get(ctx);
  if (dty.isSignedInteger(32))
    dty = Torch::QInt32Type::get(ctx);

  if (!dty)
    return nullptr;
  return Torch::ValueTensorType::get(ctx, tty.getOptionalSizes(), dty);
}

bool mlir::torch::onnx_c::areAllElementsDistinct(SmallVector<int64_t> array) {
  int n = array.size();
  llvm::SetVector<int64_t> set;
  for (int i = 0; i < n; i++) {
    set.insert(array[i]);
  }

  // If all elements are distinct, then the size of set should be same
  // as array's size.
  return (set.size() == array.size());
}

std::optional<int64_t>
mlir::torch::onnx_c::onnxDtypeIntToTorchDtypeInt(int64_t dtypeIntOnnx) {
  // TODO: Add complete mapping.
  // Where are the ONNX and PyTorch dtype enums defined?
  // ONNX:
  //  https://github.com/shouxieai/tensorRT_Pro/blob/main/onnx/onnx-ml.proto
  // PyTorch:
  //  https://github.com/llvm/torch-mlir/blob/main/include/torch-mlir/Dialect/Torch/Utils/TorchUpstream.h#L88

  std::optional<int64_t> dtypeIntTorch =
      [dtypeIntOnnx]() -> std::optional<int64_t> {
    switch (dtypeIntOnnx) {
    case 1:
      return 6; // float
    case 2:
      return 0; // uint8
    case 3:
      return 1; // int8
    case 6:
      return 3; // int32
    case 7:
      return 4; // int64
    case 9:
      return 11; // bool
    case 10:
      return 5; // half
    case 11:
      return 7; // double
    case 16:
      return 15; // bfloat16
    default:
      return std::nullopt; // No dtype
    }
  }();

  return dtypeIntTorch;
}

LogicalResult mlir::torch::onnx_c::createTorchTransposeOp(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    int64_t dimA, int64_t dimB, Value &transposed) {
  Type transposedType;
  if (failed(getTransposedType(cast<Torch::BaseTensorType>(input.getType()),
                               dimA, dimB, transposedType)))
    return failure();
  Value cstDimA = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimA));
  Value cstDimB = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(dimB));
  transposed = rewriter.create<Torch::AtenTransposeIntOp>(
      loc, transposedType, input, cstDimA, cstDimB);
  return success();
}

LogicalResult mlir::torch::onnx_c::createTorchPermuteOp(
    OpBinder binder, ConversionPatternRewriter &rewriter, Location loc,
    Value input, SmallVector<int64_t> permuteDims, Value &permuted) {
  Type permutedType;
  if (failed(
          Torch::getPermutedType(cast<Torch::BaseTensorType>(input.getType()),
                                 permuteDims, permutedType)))
    return failure();
  Value permuteDimsList = createConstantIntList(binder, rewriter, permuteDims);
  permuted = rewriter.create<Torch::AtenPermuteOp>(loc, permutedType, input,
                                                   permuteDimsList);
  return success();
}

Value createActivationByName(ImplicitLocOpBuilder &b, StringRef name,
                             Value input) {
  if (name == "Sigmoid")
    return b.create<Torch::AtenSigmoidOp>(input.getType(), input);
  if (name == "Tanh")
    return b.create<Torch::AtenTanhOp>(input.getType(), input);
  if (name == "Relu")
    return b.create<Torch::AtenReluOp>(input.getType(), input);
  llvm_unreachable("Unsupported activation function");
}
