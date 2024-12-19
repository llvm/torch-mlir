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

Value mlir::torch::onnx_c::createActivationByName(ImplicitLocOpBuilder &b,
                                                  StringRef name, Value input) {
  if (name == "Sigmoid")
    return b.create<Torch::AtenSigmoidOp>(input.getType(), input);
  if (name == "Tanh")
    return b.create<Torch::AtenTanhOp>(input.getType(), input);
  if (name == "Relu")
    return b.create<Torch::AtenReluOp>(input.getType(), input);
  llvm_unreachable("Unsupported activation function");
}

LogicalResult mlir::torch::onnx_c::extractPerTensorQuantizationArguments(
    ConversionPatternRewriter &rewriter, Location loc, Value inScale,
    Value inZeroPoint, Value &outScale, Value &outZeroPoint) {

  auto check = [](Value v) {
    auto vTy = cast<Torch::ValueTensorType>(v.getType());
    for (auto dim : vTy.getSizes())
      if (dim != 1)
        return false;
    return true;
  };

  if (!check(inScale) || !check(inZeroPoint))
    return failure();

  Value emptyList = rewriter.create<Torch::PrimListConstructOp>(
      loc,
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      ValueRange{});
  auto extract = [&rewriter, &loc, &emptyList](Value v) {
    auto vTy = cast<Torch::ValueTensorType>(v.getType());
    if (!vTy.getSizes().empty()) {
      vTy = rewriter.getType<Torch::ValueTensorType>(ArrayRef<int64_t>({}),
                                                     vTy.getOptionalDtype());
      v = rewriter.create<Torch::AtenReshapeOp>(loc, vTy, v, emptyList);
    }

    Type extractTy = rewriter.getType<Torch::FloatType>();
    if (isa<IntegerType>(vTy.getDtype()))
      extractTy = rewriter.getType<Torch::IntType>();

    return rewriter.create<Torch::AtenItemOp>(loc, extractTy, v);
  };

  outScale = extract(inScale);
  outZeroPoint = extract(inZeroPoint);

  return success();
}

LogicalResult mlir::torch::onnx_c::createDequantizeTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input, Value scale,
    Value zeroPoint, Value &output) {
  auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
  if (!inputTy || !inputTy.hasSizes())
    return failure();

  Torch::ValueTensorType makeTensorTy = getQTorchTypeFromTorchIntType(inputTy);
  Value quantizedInput =
      rewriter.create<Torch::Aten_MakePerTensorQuantizedTensorOp>(
          loc, makeTensorTy, input, scale, zeroPoint);

  Torch::ValueTensorType resultTy = rewriter.getType<Torch::ValueTensorType>(
      inputTy.getSizes(), rewriter.getF32Type());
  output = rewriter.create<Torch::AtenDequantizeSelfOp>(loc, resultTy,
                                                        quantizedInput);
  return success();
}

// Checks the validity of pooling parameters and stores them in the respective
// vector.
LogicalResult mlir::torch::onnx_c::checkAndGetOnnxPoolingOpParameters(
    OpBinder binder, ConversionPatternRewriter &rewriter, Type resultDtype,
    std::string autoPad, int64_t spatialRank, Value &input,
    SmallVectorImpl<int64_t> &kernelSizeInts,
    SmallVectorImpl<int64_t> &strideInts, SmallVectorImpl<int64_t> &paddingInts,
    SmallVectorImpl<int64_t> &dilationInts) {
  SmallVector<int64_t> kernel, padding, strides, dilations;
  if (binder.s64IntegerArrayAttr(kernel, "kernel_shape", {}))
    return rewriter.notifyMatchFailure(binder.op, "kernel_shape bind failure");
  if (kernel.size() != static_cast<size_t>(spatialRank))
    return rewriter.notifyMatchFailure(
        binder.op, "kernel list size does not match the number of axes");
  if (binder.s64IntegerArrayAttr(padding, "pads", {}))
    return rewriter.notifyMatchFailure(binder.op, "pads bind failure");
  if (!padding.empty() &&
      padding.size() != static_cast<size_t>(2 * spatialRank))
    return rewriter.notifyMatchFailure(
        binder.op, "padding list must contain (begin,end) pair for each "
                   "spatial axis");
  if (binder.s64IntegerArrayAttr(strides, "strides", {}))
    return rewriter.notifyMatchFailure(binder.op, "strides bind failure");
  if (!strides.empty() && strides.size() != static_cast<size_t>(spatialRank))
    return rewriter.notifyMatchFailure(
        binder.op, "strides list size does not match the number of axes");
  if (binder.s64IntegerArrayAttr(dilations, "dilations", {}))
    return rewriter.notifyMatchFailure(binder.op, "dilations bind failure");

  // set default values for padding, strides, and dilations.
  if (padding.empty())
    padding.resize(spatialRank, 0);
  if (strides.empty())
    strides.resize(spatialRank, 1);
  if (dilations.empty())
    dilations.resize(spatialRank, 1);

  // Padding for the beginning and ending along each spatial axis, it can
  // take any value greater than or equal to 0. The value represent the
  // number of pixels added to the beginning and end part of the
  // corresponding axis. pads format should be as follow [x1_begin,
  // x2_begin…x1_end, x2_end,…], where xi_begin the number of pixels added
  // at the beginning of axis i and xi_end, the number of pixels added at
  // the end of axis i.
  auto inputTensorType = cast<Torch::ValueTensorType>(input.getType());
  if (autoPad != "NOTSET" && autoPad != "VALID") {
    const bool isSameLower = autoPad == "SAME_LOWER";
    ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
    padding.resize_for_overwrite(2 * spatialRank);
    for (unsigned dimIdx = 0; dimIdx < spatialRank; dimIdx++) {
      const int64_t dilatedKernelSize =
          dilations[dimIdx] * (kernel[dimIdx] - 1) + 1;
      int64_t totalPad =
          ((inputShape[dimIdx + 2] + strides[dimIdx] - 1) / strides[dimIdx] -
           1) *
              strides[dimIdx] +
          dilatedKernelSize - inputShape[dimIdx + 2];
      totalPad = totalPad >= 0 ? totalPad : 0;
      padding[dimIdx] = isSameLower ? ((totalPad + 1) / 2) : (totalPad / 2);
      padding[spatialRank + dimIdx] = totalPad - padding[dimIdx];
    }
  }

  // If the padding is symmetric we can push the padding operation to the
  // torch operator.
  if (padding.size() == static_cast<size_t>(2 * spatialRank)) {
    bool equal = true;
    for (int i = 0; i < spatialRank; ++i) {
      equal = equal && (padding[i] == padding[i + spatialRank]);
    }
    if (equal)
      padding.resize(spatialRank);
  }

  // Torch pool operators require equal padding on each size of each
  // dimension so we materialize the padding behavior explicitly and set
  // the padding to 0.
  if (padding.size() == static_cast<size_t>(2 * spatialRank)) {
    llvm::SmallVector<int64_t> shuffledPadding(spatialRank * 2);
    llvm::SmallVector<int64_t> paddedShape(inputTensorType.getSizes());
    for (int i = 0; i < spatialRank; ++i) {
      paddedShape[i + 2] += padding[i] + padding[i + spatialRank];
      shuffledPadding[2 * i] = padding[spatialRank - i - 1];
      shuffledPadding[2 * i + 1] = padding[2 * spatialRank - i - 1];
    }

    Value shuffledPaddingList =
        createConstantIntList(binder, rewriter, shuffledPadding);
    Value zero;
    if (isa<FloatType>(resultDtype)) {
      zero = rewriter.create<Torch::ConstantFloatOp>(
          binder.getLoc(), rewriter.getType<Torch::FloatType>(),
          rewriter.getF64FloatAttr(std::numeric_limits<double>::lowest()));
    } else if (isa<IntegerType>(resultDtype)) {
      zero = rewriter.create<Torch::ConstantIntOp>(
          binder.getLoc(),
          rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::lowest()));
    }

    auto paddedInputTy = rewriter.getType<Torch::ValueTensorType>(
        paddedShape, inputTensorType.getDtype());
    input = rewriter.create<Torch::AtenConstantPadNdOp>(
        binder.getLoc(), paddedInputTy, input, shuffledPaddingList, zero);
    padding.clear();
    padding.resize(spatialRank, 0);
  }

  kernelSizeInts = kernel;
  paddingInts = padding;
  dilationInts = dilations;
  return success();
}
