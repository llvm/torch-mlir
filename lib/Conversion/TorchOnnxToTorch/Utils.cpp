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
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

Value mlir::torch::onnx_c::createConstantIntList(
    OpBinder binder, ConversionPatternRewriter &rewriter,
    ArrayRef<int64_t> cstInput) {
  SmallVector<Value> cstValue;
  for (int64_t i : cstInput) {
    cstValue.push_back(Torch::ConstantIntOp::create(
        rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(i)));
  }
  return Torch::PrimListConstructOp::create(
      rewriter, binder.getLoc(),
      Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
      cstValue);
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
  Value cstDimA = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(dimA));
  Value cstDimB = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(dimB));
  transposed = Torch::AtenTransposeIntOp::create(rewriter, loc, transposedType,
                                                 input, cstDimA, cstDimB);
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
  permuted = Torch::AtenPermuteOp::create(rewriter, loc, permutedType, input,
                                          permuteDimsList);
  return success();
}

static Value createTorchFloat(OpBuilder &b, Location loc, double value) {
  return Torch::ConstantFloatOp::create(b, loc, b.getF64FloatAttr(value));
}

Value mlir::torch::onnx_c::createLeakyRelu(OpBuilder &b, Location loc,
                                           Type resultType, Value input,
                                           float alpha) {
  Value cstAlpha = createTorchFloat(b, loc, alpha);
  return Torch::AtenLeakyReluOp::create(b, loc, resultType, input, cstAlpha);
}

Value mlir::torch::onnx_c::createElu(OpBuilder &b, Location loc,
                                     Type resultType, Value input,
                                     float alpha) {
  Value cstAlpha = createTorchFloat(b, loc, alpha);
  Value cstOne = createTorchFloat(b, loc, 1.0);
  return Torch::AtenEluOp::create(b, loc, resultType, input, cstAlpha,
                                  /*scale=*/cstOne, /*input_scale=*/cstOne);
}

Value mlir::torch::onnx_c::createThresholdedRelu(OpBuilder &b, Location loc,
                                                 Type resultType, Value input,
                                                 float alpha) {
  // aten.threshold keeps x only where x > alpha, as ONNX ThresholdedRelu does.
  Value cstAlpha = createTorchFloat(b, loc, alpha);
  Value cstZero = createTorchFloat(b, loc, 0.0);
  return Torch::AtenThresholdOp::create(b, loc, resultType, input, cstAlpha,
                                        cstZero);
}

Value mlir::torch::onnx_c::createHardSigmoid(OpBuilder &b, Location loc,
                                             Type resultType, Value input,
                                             float alpha, float beta) {
  // clamp(alpha * x + beta, 0, 1)
  Value cstAlpha = createTorchFloat(b, loc, alpha);
  Value cstBeta = createTorchFloat(b, loc, beta);
  Value alphaMulX =
      Torch::AtenMulScalarOp::create(b, loc, resultType, input, cstAlpha);
  Value cstOne = createTorchFloat(b, loc, 1.0);
  Value linear = Torch::AtenAddScalarOp::create(b, loc, resultType, alphaMulX,
                                                cstBeta, /*alpha=*/cstOne);
  Value cstZero = createTorchFloat(b, loc, 0.0);
  return Torch::AtenClampOp::create(b, loc, resultType, linear, cstZero,
                                    cstOne);
}

Value mlir::torch::onnx_c::createSoftsign(OpBuilder &b, Location loc,
                                          Type resultType, Value input) {
  Value absX = Torch::AtenAbsOp::create(b, loc, resultType, input);
  Value cstOne = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(1));
  Value absXPlusOne =
      Torch::AtenAddScalarOp::create(b, loc, resultType, absX, cstOne, cstOne);
  return Torch::AtenDivTensorOp::create(b, loc, resultType, input, absXPlusOne);
}

Value mlir::torch::onnx_c::createSoftplus(OpBuilder &b, Location loc,
                                          Type resultType, Value input) {
  // beta = 1 matches the ONNX spec; threshold = 20 is PyTorch's default and
  // its tail correction (exp(-20) ~= 2e-9) is below the fp32 ulp at 20
  // (~1.9e-6), so the threshold shortcut stays spec-exact.
  Value beta = Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(1));
  Value threshold =
      Torch::ConstantIntOp::create(b, loc, b.getI64IntegerAttr(20));
  return Torch::AtenSoftplusOp::create(b, loc, resultType, input, beta,
                                       threshold);
}

FailureOr<SmallVector<RecurrentActivation>>
mlir::torch::onnx_c::resolveRecurrentActivations(
    OpBinder binder, ConversionPatternRewriter &rewriter,
    ArrayRef<std::string> names) {
  SmallVector<float> alphas, betas;
  if (binder.f32FloatArrayAttr(alphas, "activation_alpha", {}) ||
      binder.f32FloatArrayAttr(betas, "activation_beta", {}))
    return rewriter.notifyMatchFailure(
        binder.op, "activation_alpha and activation_beta must be f32 lists");
  // The next unused value of `list`, or std::nullopt once it runs out.
  size_t alphaIndex = 0, betaIndex = 0;
  auto next = [](ArrayRef<float> list, size_t &index) -> std::optional<float> {
    if (index < list.size())
      return list[index++];
    return std::nullopt;
  };

  using Kind = RecurrentActivation::Kind;
  SmallVector<RecurrentActivation> activations;
  for (const std::string &name : names) {
    std::optional<Kind> kind =
        llvm::StringSwitch<std::optional<Kind>>(name)
            .CaseLower("Relu", Kind::Relu)
            .CaseLower("Tanh", Kind::Tanh)
            .CaseLower("Sigmoid", Kind::Sigmoid)
            .CaseLower("Affine", Kind::Affine)
            .CaseLower("LeakyRelu", Kind::LeakyRelu)
            .CaseLower("ThresholdedRelu", Kind::ThresholdedRelu)
            .CaseLower("ScaledTanh", Kind::ScaledTanh)
            .CaseLower("HardSigmoid", Kind::HardSigmoid)
            .CaseLower("Elu", Kind::Elu)
            .CaseLower("Softsign", Kind::Softsign)
            .CaseLower("Softplus", Kind::Softplus)
            .Default(std::nullopt);
    if (!kind)
      return rewriter.notifyMatchFailure(
          binder.op, "unsupported activation function '" + name + "'");

    // Defaults are those of the ONNX operator of the same name.
    RecurrentActivation activation{*kind};
    switch (*kind) {
    case Kind::Relu:
    case Kind::Tanh:
    case Kind::Sigmoid:
    case Kind::Softsign:
    case Kind::Softplus:
      break;
    case Kind::LeakyRelu:
      activation.alpha = next(alphas, alphaIndex).value_or(kLeakyReluAlpha);
      break;
    case Kind::ThresholdedRelu:
      activation.alpha =
          next(alphas, alphaIndex).value_or(kThresholdedReluAlpha);
      break;
    case Kind::Elu:
      activation.alpha = next(alphas, alphaIndex).value_or(kEluAlpha);
      break;
    case Kind::HardSigmoid:
      activation.alpha = next(alphas, alphaIndex).value_or(kHardSigmoidAlpha);
      activation.beta = next(betas, betaIndex).value_or(kHardSigmoidBeta);
      break;
    case Kind::Affine:
    case Kind::ScaledTanh:
      // No ONNX operator, so no spec default; 0 as in onnxruntime.
      activation.alpha = next(alphas, alphaIndex).value_or(0.0f);
      activation.beta = next(betas, betaIndex).value_or(0.0f);
      break;
    }
    activations.push_back(activation);
  }
  return activations;
}

Value mlir::torch::onnx_c::createRecurrentActivation(
    ImplicitLocOpBuilder &b, const RecurrentActivation &activation,
    Value input) {
  Location loc = b.getLoc();
  Type type = input.getType();
  auto scale = [&](Value x, double factor) -> Value {
    Value cstFactor = createTorchFloat(b, loc, factor);
    return Torch::AtenMulScalarOp::create(b, type, x, cstFactor);
  };

  using Kind = RecurrentActivation::Kind;
  switch (activation.kind) {
  case Kind::Sigmoid:
    return Torch::AtenSigmoidOp::create(b, type, input);
  case Kind::Tanh:
    return Torch::AtenTanhOp::create(b, type, input);
  case Kind::Relu:
    return Torch::AtenReluOp::create(b, type, input);
  case Kind::Affine: {
    Value scaled = scale(input, activation.alpha);
    Value cstBeta = createTorchFloat(b, loc, activation.beta);
    Value cstOne = createTorchFloat(b, loc, 1.0);
    return Torch::AtenAddScalarOp::create(b, type, scaled, cstBeta, cstOne);
  }
  case Kind::ScaledTanh: {
    Value tanh =
        Torch::AtenTanhOp::create(b, type, scale(input, activation.beta));
    return scale(tanh, activation.alpha);
  }
  case Kind::LeakyRelu:
    return createLeakyRelu(b, loc, type, input, activation.alpha);
  case Kind::ThresholdedRelu:
    return createThresholdedRelu(b, loc, type, input, activation.alpha);
  case Kind::HardSigmoid:
    return createHardSigmoid(b, loc, type, input, activation.alpha,
                             activation.beta);
  case Kind::Elu:
    return createElu(b, loc, type, input, activation.alpha);
  case Kind::Softsign:
    return createSoftsign(b, loc, type, input);
  case Kind::Softplus:
    return createSoftplus(b, loc, type, input);
  }
  llvm_unreachable("unhandled recurrent activation kind");
}

LogicalResult mlir::torch::onnx_c::extractPerTensorQuantizationArguments(
    ConversionPatternRewriter &rewriter, Location loc, Value inScale,
    Value inZeroPoint, Value &outScale, Value &outZeroPoint) {

  auto check = [](Value v) {
    auto vTy = dyn_cast<Torch::ValueTensorType>(v.getType());
    if (!vTy || !vTy.hasSizes() || !vTy.hasDtype())
      return false;
    for (auto dim : vTy.getSizes())
      if (dim != 1)
        return false;
    return true;
  };

  // A null zero point is accepted and treated as a zero point of 0.
  if (!check(inScale) || (inZeroPoint && !check(inZeroPoint)))
    return failure();

  Value emptyList = Torch::PrimListConstructOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      ValueRange{});
  auto extract = [&rewriter, &loc, &emptyList](Value v) {
    auto vTy = cast<Torch::ValueTensorType>(v.getType());
    if (!vTy.getSizes().empty()) {
      vTy = rewriter.getType<Torch::ValueTensorType>(ArrayRef<int64_t>({}),
                                                     vTy.getOptionalDtype());
      v = Torch::AtenReshapeOp::create(rewriter, loc, vTy, v, emptyList);
    }

    Type extractTy = rewriter.getType<Torch::FloatType>();
    if (isa<IntegerType>(vTy.getDtype()))
      extractTy = rewriter.getType<Torch::IntType>();

    return Torch::AtenItemOp::create(rewriter, loc, extractTy, v);
  };

  outScale = extract(inScale);
  if (inZeroPoint)
    outZeroPoint = extract(inZeroPoint);
  else
    outZeroPoint = Torch::ConstantIntOp::create(rewriter, loc,
                                                rewriter.getI64IntegerAttr(0));

  return success();
}

namespace {
struct IntQuantParams {
  Value dtypeConst;
  Value quantMin;
  Value quantMax;
};
} // namespace

static FailureOr<IntQuantParams>
getIntQuantParams(ConversionPatternRewriter &rewriter, Location loc,
                  Torch::ValueTensorType ty) {
  if (!ty || !ty.hasSizes())
    return failure();
  if (!ty.hasDtype() || !isa<IntegerType>(ty.getDtype()))
    return failure();
  auto eTy = cast<IntegerType>(ty.getDtype());

  Value dtypeConst = Torch::getDtypeIntValueForType(rewriter, loc, eTy);
  auto [minInt, maxInt] = createQuantizationBounds(rewriter, loc, eTy);
  return IntQuantParams{dtypeConst, minInt, maxInt};
}

// `quantized_decomposed.dequantize_per_token` applies one scale and zero point
// per "token", i.e. per index of all dims but the last one. Its parameters
// must have the shape of the input with the last dim set to 1. Parameter
// token dims of size 1 are expanded to the input dims first, since the op
// does not broadcast. This checks the shapes without creating any IR. It
// fails if the layout is invalid, otherwise it returns whether the parameters
// must be expanded.
static FailureOr<bool> checkPerTokenParams(Torch::ValueTensorType inputTy,
                                           Value scale, Value zeroPoint) {
  auto scaleTy = dyn_cast<Torch::ValueTensorType>(scale.getType());
  auto zpTy = dyn_cast<Torch::ValueTensorType>(zeroPoint.getType());
  if (!inputTy || !inputTy.hasSizes() || !scaleTy || !scaleTy.hasSizes() ||
      !zpTy || !zpTy.hasSizes())
    return failure();
  ArrayRef<int64_t> shape = inputTy.getSizes();
  ArrayRef<int64_t> paramShape = scaleTy.getSizes();
  int64_t rank = shape.size();
  if (rank < 2 || shape[rank - 2] == Torch::kUnknownSize ||
      static_cast<int64_t>(paramShape.size()) != rank ||
      paramShape != zpTy.getSizes() || !inputTy.hasDtype() ||
      !isa<IntegerType>(inputTy.getDtype()))
    return failure();

  // The token dims of the parameters must be 1 (broadcast) or match those of
  // the input, and the last dim of the parameters must be 1.
  if (paramShape[rank - 1] != 1)
    return failure();
  bool needsExpand = false;
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (paramShape[i] == shape[i])
      continue;
    if (paramShape[i] != 1)
      return failure();
    needsExpand = true;
  }
  return needsExpand;
}

// Expands `scale` and `zeroPoint` to the token dims of `input` followed by 1.
// Dim rank-2 of `input` is static, so a dynamic dim is always a batch dim.
static void expandPerTokenParams(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input, Value &scale,
                                 Value &zeroPoint) {
  ArrayRef<int64_t> shape =
      cast<Torch::ValueTensorType>(input.getType()).getSizes();
  int64_t rank = shape.size();
  auto cstInt = [&](int64_t i) -> Value {
    return Torch::ConstantIntOp::create(rewriter, loc,
                                        rewriter.getI64IntegerAttr(i));
  };

  SmallVector<Value> expandSizes;
  SmallVector<int64_t> expandedShape;
  for (int64_t i = 0; i < rank - 1; ++i) {
    expandedShape.push_back(shape[i]);
    if (shape[i] != Torch::kUnknownSize)
      expandSizes.push_back(cstInt(shape[i]));
    else
      expandSizes.push_back(
          Torch::AtenSizeIntOp::create(rewriter, loc, input, cstInt(i)));
  }
  expandedShape.push_back(1);
  expandSizes.push_back(cstInt(1));

  auto expand = [&](Value param) -> Value {
    auto paramTy = cast<Torch::ValueTensorType>(param.getType());
    Value sizeList = Torch::PrimListConstructOp::create(
        rewriter, loc,
        rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
        expandSizes);
    Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
    return Torch::AtenExpandOp::create(
        rewriter, loc,
        paramTy.getWithSizesAndDtype(expandedShape, paramTy.getOptionalDtype()),
        param, sizeList, /*implicit=*/cstFalse);
  };
  scale = expand(scale);
  zeroPoint = expand(zeroPoint);
}

// Returns the result type of a dequantize op: the sizes of `inputTy` with
// element type `outFloatTy` (f32 if null).
static Torch::ValueTensorType
getDequantResultType(ConversionPatternRewriter &rewriter,
                     Torch::ValueTensorType inputTy, FloatType outFloatTy) {
  return rewriter.getType<Torch::ValueTensorType>(
      inputTy.getSizes(),
      outFloatTy ? Type(outFloatTy) : rewriter.getF32Type());
}

/// Dequantizes the integer tensor `input` to a float tensor of element type
/// `outFloatTy` (f32 if null). If `axis` is empty,
/// `quantized_decomposed.dequantize_per_tensor` is emitted, otherwise
/// `quantized_decomposed.dequantize_per_channel` along `axis`.
static LogicalResult createDequantizeImpl(ConversionPatternRewriter &rewriter,
                                          Location loc, Value input,
                                          Value scale, Value zeroPoint,
                                          std::optional<int64_t> axis,
                                          Value &output, FloatType outFloatTy) {
  auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
  FailureOr<IntQuantParams> params = getIntQuantParams(rewriter, loc, inputTy);
  if (failed(params))
    return failure();

  Value outDtype =
      outFloatTy ? Torch::getDtypeIntValueForType(rewriter, loc, outFloatTy)
                 : Torch::ConstantNoneOp::create(rewriter, loc).getResult();
  Torch::ValueTensorType resultTy =
      getDequantResultType(rewriter, inputTy, outFloatTy);

  if (!axis) {
    output = Torch::QuantizedDecomposedDequantizePerTensorOp::create(
        rewriter, loc, resultTy, input, scale, zeroPoint, params->quantMin,
        params->quantMax, params->dtypeConst, outDtype);
    return success();
  }

  Value cstAxis = Torch::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64IntegerAttr(*axis));
  output = Torch::QuantizedDecomposedDequantizePerChannelOp::create(
      rewriter, loc, resultTy, input, scale, zeroPoint, cstAxis,
      params->quantMin, params->quantMax, params->dtypeConst, outDtype);
  return success();
}

LogicalResult mlir::torch::onnx_c::createDequantizeTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input, Value scale,
    Value zeroPoint, Value &output, FloatType outFloatTy) {
  return createDequantizeImpl(rewriter, loc, input, scale, zeroPoint,
                              /*axis=*/std::nullopt, output, outFloatTy);
}

LogicalResult mlir::torch::onnx_c::createDequantizePerChannelTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    Value scales, Value zeroPoints, int64_t axis, Value &output,
    FloatType outFloatTy) {
  return createDequantizeImpl(rewriter, loc, input, scales, zeroPoints, axis,
                              output, outFloatTy);
}

LogicalResult mlir::torch::onnx_c::createDequantizePerTokenTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input, Value scale,
    Value zeroPoint, Value &output, FloatType outFloatTy) {
  auto inputTy = dyn_cast<Torch::ValueTensorType>(input.getType());
  // All checks are done before any IR is created.
  FailureOr<bool> needsExpand = checkPerTokenParams(inputTy, scale, zeroPoint);
  if (failed(needsExpand))
    return failure();
  FailureOr<IntQuantParams> params = getIntQuantParams(rewriter, loc, inputTy);
  if (failed(params))
    return failure();

  if (*needsExpand)
    expandPerTokenParams(rewriter, loc, input, scale, zeroPoint);

  // Unlike the other dequantize ops, out_dtype is not optional for per-token.
  Value outDtype = Torch::getDtypeIntValueForType(
      rewriter, loc, outFloatTy ? outFloatTy : rewriter.getF32Type());
  Torch::ValueTensorType resultTy =
      getDequantResultType(rewriter, inputTy, outFloatTy);

  output = Torch::QuantizedDecomposedDequantizePerTokenOp::create(
      rewriter, loc, resultTy, input, scale, zeroPoint, params->quantMin,
      params->quantMax, params->dtypeConst, outDtype);
  return success();
}

LogicalResult mlir::torch::onnx_c::createQuantizeTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input, Value scale,
    Value zeroPoint, Torch::ValueTensorType resultType, Value &output) {
  FailureOr<IntQuantParams> params =
      getIntQuantParams(rewriter, loc, resultType);
  if (failed(params))
    return failure();

  output = Torch::QuantizedDecomposedQuantizePerTensorOp::create(
      rewriter, loc, resultType, input, scale, zeroPoint, params->quantMin,
      params->quantMax, params->dtypeConst);
  return success();
}

std::pair<Value, Value> mlir::torch::onnx_c::createQuantizationBounds(
    ConversionPatternRewriter &rewriter, Location loc, IntegerType dtype) {
  auto bitWidth = dtype.getWidth();
  int64_t min =
      dtype.isSigned() ? APInt::getSignedMinValue(bitWidth).getSExtValue() : 0;

  int64_t max = dtype.isSigned()
                    ? APInt::getSignedMaxValue(bitWidth).getSExtValue()
                    : APInt::getMaxValue(bitWidth).getZExtValue();

  return {Torch::ConstantIntOp::create(rewriter, loc,
                                       rewriter.getI64IntegerAttr(min)),
          Torch::ConstantIntOp::create(rewriter, loc,
                                       rewriter.getI64IntegerAttr(max))};
}
