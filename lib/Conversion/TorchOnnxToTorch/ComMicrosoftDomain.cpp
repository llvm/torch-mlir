//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

namespace {

// A successful match supplies everything needed to build DQ without inspecting
// or classifying the input/qparam types again. Matching does not create IR.
struct QuantizedTensorInfo {
  Value input;
  Value scale;
  Value zeroPoint;
  IntegerType storageDtype;
  Torch::ValueTensorType floatType;
  int64_t axis;
  int64_t columns;
  int64_t scaleLength;
  int64_t zeroPointLength;
  bool scaleIsScalar;
  bool zeroPointIsScalar;

  bool isPerTensor() const { return scaleIsScalar && zeroPointIsScalar; }
};

// Return the qparam length as well as verifying it. A scalar and a singleton
// vector both have length one for the per-tensor/per-column classification.
FailureOr<int64_t> matchQParam(OpBinder binder,
                               ConversionPatternRewriter &rewriter, Value param,
                               Type dtype, int64_t columns) {
  auto type = cast<Torch::ValueTensorType>(param.getType());
  if (!type.hasDtype() || type.getDtype() != dtype || !type.hasSizes() ||
      type.getSizes().size() > 1) {
    return rewriter.notifyMatchFailure(
        binder.op, "expected scalar or rank-1 qparams with matching dtype");
  }
  int64_t length = type.getSizes().empty() ? 1 : type.getSizes()[0];
  // Treat [?] qparams as per-column when the column count is also dynamic.
  // DQ construction will assert length == columns; this does not support
  // runtime singleton broadcasting when columns > 1.
  if (length == Torch::kUnknownSize && columns != Torch::kUnknownSize) {
    return rewriter.notifyMatchFailure(
        binder.op, "requires statically known qparam length");
  }
  // Dynamic matrix dimensions are still supported: if columns is unknown,
  // DQ construction will assert columns == length for per-column qparams.
  if (length != 1 && columns != Torch::kUnknownSize && length != columns) {
    return rewriter.notifyMatchFailure(
        binder.op, "per-column qparam length must match input's last dim");
  }
  return length;
}

FailureOr<QuantizedTensorInfo>
matchQuantizedTensor(OpBinder binder, ConversionPatternRewriter &rewriter,
                     Value input, Value scale, Value zeroPoint) {
  auto inputType = cast<Torch::ValueTensorType>(input.getType());
  if (!inputType.hasSizes() || inputType.getSizes().empty() ||
      !inputType.hasDtype() ||
      !(inputType.getDtype().isSignedInteger(8) ||
        inputType.getDtype().isUnsignedInteger(8))) {
    return rewriter.notifyMatchFailure(
        binder.op, "expected ranked int8 or uint8 matrix inputs");
  }

  int64_t columns = inputType.getSizes().back();
  FailureOr<int64_t> scaleLength =
      matchQParam(binder, rewriter, scale, rewriter.getF32Type(), columns);
  if (failed(scaleLength)) {
    return failure();
  }
  int64_t zeroPointLength = 1;
  if (zeroPoint) {
    FailureOr<int64_t> length =
        matchQParam(binder, rewriter, zeroPoint, inputType.getDtype(), columns);
    if (failed(length)) {
      return failure();
    }
    zeroPointLength = *length;
  }
  if (*scaleLength != 1 && zeroPointLength != 1 &&
      *scaleLength != Torch::kUnknownSize &&
      zeroPointLength != Torch::kUnknownSize &&
      *scaleLength != zeroPointLength) {
    return rewriter.notifyMatchFailure(
        binder.op, "per-column scale and zero point lengths must match");
  }

  // The spec's per-column parameters index the last input dimension,
  // independently for A and B, including when the inputs are batched.
  int64_t axis = inputType.getSizes().size() - 1;
  auto floatType = cast<Torch::ValueTensorType>(inputType.getWithSizesAndDtype(
      inputType.getSizes(), rewriter.getF32Type()));
  return QuantizedTensorInfo{input,
                             scale,
                             zeroPoint,
                             cast<IntegerType>(inputType.getDtype()),
                             floatType,
                             axis,
                             columns,
                             *scaleLength,
                             zeroPointLength,
                             *scaleLength == 1,
                             zeroPointLength == 1};
}

// qparam_vector[i] = qparam_scalar, for i in [0, input.size(-1)).
// Per-channel DQ requires rank-1 scales/zero points. Expand a scalar qparam
// when the other qparam is per-column, preserving its runtime value.
Value expandScalarQParam(ConversionPatternRewriter &rewriter, Location loc,
                         const QuantizedTensorInfo &info, Value param,
                         Type dtype, Value axis) {
  Value size = Torch::AtenSizeIntOp::create(rewriter, loc, info.input, axis);
  Value sizes = Torch::PrimListConstructOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ListType>(rewriter.getType<Torch::IntType>()),
      SmallVector<Value>{size});
  auto expandedType = rewriter.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{info.columns}, dtype);
  Value implicit = Torch::ConstantBoolOp::create(rewriter, loc, false);
  return Torch::AtenExpandOp::create(rewriter, loc, expandedType, param, sizes,
                                     implicit);
}

// input.size(-1) == qparam.size(0), using a constant for known qparam lengths.
void assertQParamLength(ConversionPatternRewriter &rewriter, Location loc,
                        Value columns, Value param, int64_t length) {
  Value qparamLength;
  if (length == Torch::kUnknownSize) {
    Value zero = Torch::ConstantIntOp::create(rewriter, loc, 0);
    qparamLength = Torch::AtenSizeIntOp::create(rewriter, loc, param, zero);
  } else {
    qparamLength = Torch::ConstantIntOp::create(rewriter, loc, length);
  }
  Value matches =
      Torch::AtenEqIntOp::create(rewriter, loc, columns, qparamLength);
  Torch::RuntimeAssertOp::create(
      rewriter, loc, matches,
      rewriter.getStringAttr("MatMulIntegerToFloat: per-column qparam length "
                             "must match input's last dimension"));
}

Value createMatMulIntegerToFloatDequantize(ConversionPatternRewriter &rewriter,
                                           Location loc,
                                           const QuantizedTensorInfo &info) {
  // DQ(input) = (cast<f32>(input) - zero_point) * scale.
  auto [quantMin, quantMax] =
      createQuantizationBounds(rewriter, loc, info.storageDtype);
  Value dtype =
      Torch::getDtypeIntValueForType(rewriter, loc, info.storageDtype);
  Value outDtype =
      Torch::getDtypeIntValueForType(rewriter, loc, info.floatType.getDtype());
  Value scale = info.scale;
  Value zeroPoint = info.zeroPoint;

  if (info.isPerTensor()) {
    // ONNX carries scalar qparams as tensors, but the PT2E per-tensor op
    // accepts Torch scalars. Item keeps runtime qparams as SSA operands.
    scale = Torch::AtenItemOp::create(
        rewriter, loc, rewriter.getType<Torch::FloatType>(), scale);
    if (zeroPoint) {
      zeroPoint = Torch::AtenItemOp::create(
          rewriter, loc, rewriter.getType<Torch::IntType>(), zeroPoint);
    } else {
      zeroPoint = Torch::ConstantIntOp::create(rewriter, loc, 0);
    }
    return Torch::QuantizedDecomposedDequantizePerTensorOp::create(
        rewriter, loc, info.floatType, info.input, scale, zeroPoint, quantMin,
        quantMax, dtype, outDtype);
  }

  Value axis = Torch::ConstantIntOp::create(rewriter, loc, info.axis);
  if (info.columns == Torch::kUnknownSize) {
    Value columns =
        Torch::AtenSizeIntOp::create(rewriter, loc, info.input, axis);
    if (!info.scaleIsScalar) {
      assertQParamLength(rewriter, loc, columns, scale, info.scaleLength);
    }
    // Known equal vector lengths share one check. Unknown lengths must each
    // match columns, even when both qparam types are [?].
    if (!info.zeroPointIsScalar &&
        (info.zeroPointLength == Torch::kUnknownSize ||
         info.zeroPointLength != info.scaleLength)) {
      assertQParamLength(rewriter, loc, columns, zeroPoint,
                         info.zeroPointLength);
    }
  }
  if (info.scaleIsScalar) {
    scale = expandScalarQParam(rewriter, loc, info, scale,
                               info.floatType.getDtype(), axis);
  }
  if (zeroPoint) {
    if (info.zeroPointIsScalar) {
      zeroPoint = expandScalarQParam(rewriter, loc, info, zeroPoint,
                                     info.storageDtype, axis);
    }
  } else {
    // None is PT2E per-channel DQ's representation of a zero zero point.
    zeroPoint = Torch::ConstantNoneOp::create(rewriter, loc);
  }
  return Torch::QuantizedDecomposedDequantizePerChannelOp::create(
      rewriter, loc, info.floatType, info.input, scale, zeroPoint, axis,
      quantMin, quantMax, dtype, outDtype);
}

} // namespace

void mlir::torch::onnx_c::populateComMicrosoftDomain(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "FastGelu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value input, bias;
        Torch::ValueTensorType resultType;
        if (binder.getNumOperands() < 1 || binder.getNumOperands() > 2 ||
            binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(
              binder.op, "expected one input and an optional bias");
        }

        // Empty ONNX input names are imported as None.
        if (binder.getNumOperands() == 2 &&
            !isa<Torch::NoneType>(binder.op->getOperand(1).getType()) &&
            binder.tensorOperandAtIndex(bias, 1)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected tensor or None bias");
        }
        if (!resultType.hasDtype() ||
            !(resultType.getDtype().isF16() || resultType.getDtype().isBF16() ||
              resultType.getDtype().isF32() || resultType.getDtype().isF64())) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected floating-point result");
        }

        Location loc = binder.getLoc();
        Type computeType = resultType;
        if (bias) {
          // Keep the bias addition in f32 as part of FastGelu's opmath; an
          // intermediate f16/bf16 rounding changes the activation's result.
          if (resultType.getDtype().isF16() || resultType.getDtype().isBF16()) {
            auto f32 = rewriter.getF32Type();
            input = Torch::convertTensorToDtype(rewriter, loc, input, f32);
            bias = Torch::convertTensorToDtype(rewriter, loc, bias, f32);
            computeType = resultType.getWithSizesAndDtype(
                resultType.getOptionalSizes(), f32);
          }
          Value one = Torch::ConstantIntOp::create(rewriter, loc, 1);
          input = Torch::AtenAddTensorOp::create(rewriter, loc, computeType,
                                                 input, bias, one);
        }
        // FastGelu is the tanh GELU approximation, with bias added first.
        Value approximate = Torch::ConstantStrOp::create(rewriter, loc, "tanh");
        Value result = Torch::AtenGeluOp::create(rewriter, loc, computeType,
                                                 input, approximate);
        if (computeType != resultType)
          result = Torch::convertTensorToDtype(rewriter, loc, result,
                                               resultType.getDtype());
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp(
      "MatMulIntegerToFloat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value a, b, aScale, bScale, aZeroPoint, bZeroPoint, bias;
        Torch::ValueTensorType resultType;
        if (binder.getNumOperands() < 4 || binder.getNumOperands() > 7 ||
            binder.tensorOperandAtIndex(a, 0) ||
            binder.tensorOperandAtIndex(b, 1) ||
            binder.tensorOperandAtIndex(aScale, 2) ||
            binder.tensorOperandAtIndex(bScale, 3) ||
            binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "expected four required inputs and up to three optional inputs");
        }

        Value *optionalInputs[] = {&aZeroPoint, &bZeroPoint, &bias};
        // Empty ONNX input names become None without shifting later inputs,
        // so bias may be present even when either zero point is omitted.
        for (int64_t i = 4; i < binder.getNumOperands(); ++i) {
          if (isa<Torch::NoneType>(binder.op->getOperand(i).getType())) {
            continue;
          }
          if (binder.tensorOperandAtIndex(*optionalInputs[i - 4], i)) {
            return rewriter.notifyMatchFailure(
                binder.op, "expected tensor or None optional inputs");
          }
        }
        if (!resultType.hasDtype() || !resultType.getDtype().isF32()) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected float32 result");
        }
        FailureOr<QuantizedTensorInfo> aInfo =
            matchQuantizedTensor(binder, rewriter, a, aScale, aZeroPoint);
        FailureOr<QuantizedTensorInfo> bInfo =
            matchQuantizedTensor(binder, rewriter, b, bScale, bZeroPoint);
        if (failed(aInfo) || failed(bInfo)) {
          return failure();
        }
        if (bias) {
          auto biasType = cast<Torch::ValueTensorType>(bias.getType());
          if (!biasType.hasDtype() || !biasType.getDtype().isF32() ||
              !biasType.hasSizes() || biasType.getSizes().size() != 1) {
            return rewriter.notifyMatchFailure(binder.op,
                                               "expected rank-1 float32 bias");
          }
          int64_t biasLength = biasType.getSizes()[0];
          if (biasLength != Torch::kUnknownSize &&
              bInfo->columns != Torch::kUnknownSize &&
              biasLength != bInfo->columns) {
            return rewriter.notifyMatchFailure(
                binder.op, "bias length must match B's last dimension");
          }
        }

        Location loc = binder.getLoc();
        Value aFloat =
            createMatMulIntegerToFloatDequantize(rewriter, loc, *aInfo);
        Value bFloat =
            createMatMulIntegerToFloatDequantize(rewriter, loc, *bInfo);
        // A_f = DQ(A); B_f = DQ(B); Y = matmul(A_f, B_f) [+ bias].
        // The inputs are already quantized and the result is float, so there
        // is no output quantization boundary to create.
        Value result = Torch::AtenMatmulOp::create(rewriter, loc, resultType,
                                                   aFloat, bFloat);
        if (bias) {
          Value one = Torch::ConstantIntOp::create(rewriter, loc, 1);
          result = Torch::AtenAddTensorOp::create(rewriter, loc, resultType,
                                                  result, bias, one);
        }
        rewriter.replaceOp(binder.op, result);
        return success();
      });
  patterns.onOp(
      "RotaryEmbedding", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        int64_t interleaved, isPackedBatching, numHeads, rotaryEmbeddingDim;
        float scale;
        Value input, positionIds, cosCache, sinCache;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(positionIds, 1) ||
            binder.tensorOperandAtIndex(cosCache, 2) ||
            binder.tensorOperandAtIndex(sinCache, 3) ||
            binder.s64IntegerAttr(interleaved, "interleaved", 0) ||
            binder.s64IntegerAttr(isPackedBatching, "is_packed_batching", 0) ||
            binder.s64IntegerAttr(numHeads, "num_heads", 0) ||
            binder.s64IntegerAttr(rotaryEmbeddingDim, "rotary_embedding_dim",
                                  0) ||
            binder.f32FloatAttr(scale, "scale", 1.0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");
        }

        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }

        Value cstInterleaved = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(interleaved));
        Value cstIsPackedBatching = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(isPackedBatching));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstRotaryEmbeddingDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(rotaryEmbeddingDim));
        Value cstScale = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(scale));

        rewriter.replaceOpWithNewOp<Torch::OnnxVariantRotaryEmbeddingOp>(
            binder.op, resultType, input, positionIds, cosCache, sinCache,
            cstInterleaved, cstIsPackedBatching, cstNumHeads,
            cstRotaryEmbeddingDim, cstScale);
        return success();
      });
  patterns.onOp(
      "SimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        Value input, scale;
        float epsilon;
        int64_t axis, stashType;
        SmallVector<Type> resultTypes;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5f) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.s64IntegerAttr(stashType, "stash_type", 1) ||
            binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to bind inputs/attrs");

        if (resultTypes.size() != 1)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unsupported number of results");

        // Convert stash_type to torch dtype
        std::optional<int64_t> stashTypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(stashType);
        if (!stashTypeIntTorch.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for the given stash_type");
        FailureOr<Type> stashDtype = Torch::getTypeForScalarType(
            binder.op->getContext(),
            (torch_upstream::ScalarType)stashTypeIntTorch.value());
        if (failed(stashDtype))
          return failure();

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype() || !inputType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "input should have dtype and sizes");

        // Get tensor rank to normalize axis
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank || *maybeRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked or scalar input tensor");
        unsigned inputRank = *maybeRank;

        // Upcast input to stash_type for numerical stability
        Type origDtype = inputType.getOptionalDtype();
        if (*stashDtype != origDtype)
          input =
              Torch::convertTensorToDtype(rewriter, loc, input, *stashDtype);

        // Build normalized_shape: [inputShape[axis], ..., inputShape[-1]]
        axis = Torch::toPositiveDim(axis, inputRank);
        auto castInput = cast<Torch::ValueTensorType>(input.getType());
        ArrayRef<int64_t> inputShape = castInput.getSizes();
        SmallVector<Value> normalizedShapeValues;
        for (int64_t n = axis; n < static_cast<int64_t>(inputRank); n++) {
          normalizedShapeValues.push_back(Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(inputShape[n])));
        }
        Value normalizedShape = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            normalizedShapeValues);

        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));

        // Emit aten.rms_norm (in stash dtype if upcasted)
        Value output =
            Torch::AtenRmsNormOp::create(rewriter, loc, input.getType(), input,
                                         normalizedShape, scale, cstEpsilon);

        // Cast output back to original dtype if upcasted
        if (*stashDtype != origDtype)
          output =
              Torch::convertTensorToDtype(rewriter, loc, output, origDtype);

        rewriter.replaceOp(binder.op, {output});
        return success();
      });
  patterns.onOp(
      "SkipSimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        Value input, skip, gamma;
        float epsilon;
        int64_t stashType;
        SmallVector<Type> resultTypes;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(skip, 1) ||
            binder.tensorOperandAtIndex(gamma, 2) ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5f) ||
            binder.s64IntegerAttr(stashType, "stash_type", 1) ||
            binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to bind inputs/attrs");

        if (resultTypes.size() > 2)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unsupported number of results");

        // Optional bias (index 3)
        Value bias;
        bool hasBias = !binder.tensorOperandAtIndex(bias, 3);

        // Convert stash_type to torch dtype
        std::optional<int64_t> stashTypeIntTorch =
            onnxDtypeIntToTorchDtypeInt(stashType);
        if (!stashTypeIntTorch.has_value())
          return rewriter.notifyMatchFailure(
              binder.op, "unimplemented support for the given stash_type");
        FailureOr<Type> stashDtype = Torch::getTypeForScalarType(
            binder.op->getContext(),
            (torch_upstream::ScalarType)stashTypeIntTorch.value());
        if (failed(stashDtype))
          return failure();

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype() || !inputType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "input should have dtype and sizes");

        // Get tensor rank to compute last dimension
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank || *maybeRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked or scalar input tensor");
        unsigned inputRank = *maybeRank;

        Value cstOne = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));

        // Upcast inputs to stash_type for numerical stability
        Type origDtype = inputType.getOptionalDtype();
        if (*stashDtype != origDtype) {
          input =
              Torch::convertTensorToDtype(rewriter, loc, input, *stashDtype);
          skip = Torch::convertTensorToDtype(rewriter, loc, skip, *stashDtype);
          if (hasBias)
            bias =
                Torch::convertTensorToDtype(rewriter, loc, bias, *stashDtype);
        }

        // Step 1: Compute s = input + skip + bias (if present)
        Value s = Torch::AtenAddTensorOp::create(rewriter, loc, input.getType(),
                                                 input, skip, cstOne);

        if (hasBias) {
          s = Torch::AtenAddTensorOp::create(rewriter, loc, input.getType(), s,
                                             bias, cstOne);
        }

        // Build normalized_shape for last dimension: [inputShape[-1]]
        auto castInput = cast<Torch::ValueTensorType>(input.getType());
        ArrayRef<int64_t> inputShape = castInput.getSizes();
        Value cstLastDimSize = Torch::ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(inputShape[inputRank - 1]));
        Value normalizedShape = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            SmallVector<Value>{cstLastDimSize});

        // Emit aten.rms_norm (in stash dtype if upcasted)
        Value output =
            Torch::AtenRmsNormOp::create(rewriter, loc, input.getType(), s,
                                         normalizedShape, gamma, cstEpsilon);

        // Cast results back to original dtype if upcasted
        if (*stashDtype != origDtype) {
          output =
              Torch::convertTensorToDtype(rewriter, loc, output, origDtype);
          s = Torch::convertTensorToDtype(rewriter, loc, s, origDtype);
        }

        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else {
          // 2-output: (output, input_skip_bias_sum)
          rewriter.replaceOp(binder.op, {output, s});
        }
        return success();
      });
  patterns.onOp(
      "GroupQueryAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<Value> operands;
        SmallVector<Type> resultTypes;
        int64_t doRotary, kvNumHeads, localWindowSize, numHeads,
            rotaryInterleaved, smoothSoftmax;
        float scale, softcap;
        if (binder.tensorOperandsList(operands))
          return rewriter.notifyMatchFailure(binder.op,
                                             "operands bind failure");

        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        if (resultTypes.size() != 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 3 result types");

        if (binder.s64IntegerAttr(doRotary, "do_rotary") ||
            binder.s64IntegerAttr(kvNumHeads, "kv_num_heads") ||
            binder.s64IntegerAttr(localWindowSize, "local_window_size", -1) ||
            binder.s64IntegerAttr(numHeads, "num_heads") ||
            binder.s64IntegerAttr(rotaryInterleaved, "rotary_interleaved") ||
            binder.f32FloatAttr(scale, "scale") ||
            binder.s64IntegerAttr(smoothSoftmax, "smooth_softmax") ||
            binder.f32FloatAttr(softcap, "softcap"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "op attributes bind failure");

        // This lowering supports two input formats:
        // 1. Separate Q, K, V inputs (9 operands with rotary, 7 without):
        //    query, key, value, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        // 2. Packed QKV input (7 operands with rotary, 5 without):
        //    packed_qkv, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        int numOperands = operands.size();
        if (doRotary && numOperands != 7 && numOperands != 9) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected 7 operands (packed QKV) or 9 operands (separate Q, "
              "K, V) when do_rotary is enabled");
        }
        if (!doRotary && numOperands != 5 && numOperands != 7) {
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected 5 operands (packed QKV) or 7 operands (separate Q, "
              "K, V) when do_rotary is disabled");
        }
        bool isPackedQKV =
            (doRotary && numOperands == 7) || (!doRotary && numOperands == 5);

        if (kvNumHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "kv_num_heads is a required attribute and should be non-zero");

        if (localWindowSize != -1)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: local_window_size attribute is not supported, "
              "hence it should have default value equal to -1");

        if (numHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "num_heads is a required attribute and should be non-zero");

        if (smoothSoftmax > 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: smooth_softmax attribute is not supported, hence "
              "it should have a value <= 0 (disabled)");

        if (softcap != 0.0f)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: softcap attribute is not supported, "
                         "hence it should have default value equal to 0.0");

        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        Value query, key, value, pastKey, pastValue, seqlensK;
        Value cosCache, sinCache;

        if (isPackedQKV) {
          // Packed QKV mode: first operand contains Q, K, V concatenated
          Value packedQKV = operands[0];
          pastKey = operands[1];
          pastValue = operands[2];
          seqlensK = operands[3];
          if (doRotary) {
            cosCache = operands[5];
            sinCache = operands[6];
          }

          // Split packed QKV into separate Q, K, V tensors
          // packed_qkv shape: [batch, seq, q_hidden + k_hidden + v_hidden]
          // where q_hidden = num_heads * head_size
          //       k_hidden = kv_num_heads * head_size
          //       v_hidden = kv_num_heads * head_size
          Torch::ValueTensorType packedType =
              cast<Torch::ValueTensorType>(packedQKV.getType());
          if (!packedType.hasSizes() || packedType.getSizes().size() != 3)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected packed QKV input to have 3 dimensions");

          SmallVector<int64_t> packedDims{packedType.getSizes()};
          int64_t batchSize = packedDims[0];        // may be dynamic
          int64_t sequenceLength = packedDims[1];   // may be dynamic
          int64_t packedHiddenSize = packedDims[2]; // must be static

          if (packedHiddenSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected packed QKV hidden dimension (dim 2) to be static");

          // Calculate head_size from past_key shape: [batch, kv_num_heads,
          // past_seq, head_size]
          Torch::ValueTensorType pastKeyType =
              cast<Torch::ValueTensorType>(pastKey.getType());
          if (!(pastKeyType.hasSizes() && pastKeyType.getSizes().size() == 4))
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key to have 4 dimensions");

          int64_t headSize = pastKeyType.getSizes()[3];
          if (headSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key head_size (dim 3) to be static");

          int64_t qHiddenSize = numHeads * headSize;
          int64_t kvHiddenSize = kvNumHeads * headSize;

          // Validate packed hidden size
          if (packedHiddenSize != qHiddenSize + 2 * kvHiddenSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Packed QKV hidden size mismatch: expected " +
                               std::to_string(qHiddenSize + 2 * kvHiddenSize) +
                               " but got " + std::to_string(packedHiddenSize));

          Value cstOne = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(1));
          Value cstTwo = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(2));
          Value cstZero = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));
          Value cstQHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(qHiddenSize));
          Value cstQPlusKVHidden = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(qHiddenSize + kvHiddenSize));
          Value cstPackedHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(packedHiddenSize));

          // Slice Q: packed_qkv[:, :, 0:q_hidden]
          // batch and seq dimensions may be dynamic
          SmallVector<int64_t> querySizes{batchSize, sequenceLength,
                                          qHiddenSize};
          Torch::ValueTensorType queryType = Torch::ValueTensorType::get(
              context, querySizes, packedType.getOptionalDtype());
          query = Torch::AtenSliceTensorOp::create(
              rewriter, loc, queryType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstZero, /*end=*/cstQHidden,
              /*step=*/cstOne);

          // Slice K: packed_qkv[:, :, q_hidden:q_hidden+kv_hidden]
          SmallVector<int64_t> kvSizes{batchSize, sequenceLength, kvHiddenSize};
          Torch::ValueTensorType keyType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          key = Torch::AtenSliceTensorOp::create(rewriter, loc, keyType,
                                                 packedQKV,
                                                 /*dim=*/cstTwo,
                                                 /*start=*/cstQHidden,
                                                 /*end=*/cstQPlusKVHidden,
                                                 /*step=*/cstOne);

          // Slice V: packed_qkv[:, :, q_hidden+kv_hidden:]
          Torch::ValueTensorType valueType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          value = Torch::AtenSliceTensorOp::create(
              rewriter, loc, valueType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstQPlusKVHidden,
              /*end=*/cstPackedHidden,
              /*step=*/cstOne);
        } else {
          // Separate Q, K, V mode
          query = operands[0];
          key = operands[1];
          value = operands[2];
          pastKey = operands[3];
          pastValue = operands[4];
          seqlensK = operands[5];
          if (doRotary) {
            cosCache = operands[7];
            sinCache = operands[8];
          }
        }

        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!queryType.hasSizes() || queryType.getSizes().size() != 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `query` input to have 3 dimensions");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        int64_t batchSize = queryDims[0];
        int64_t sequenceLength = queryDims[1];
        int64_t hiddenSize = queryDims[2];
        if (hiddenSize == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected `query` hidden dimension (dim 2) to be static");
        int64_t headSize = hiddenSize / numHeads;

        // For dynamic dimensions, use aten.size.int to get runtime values
        Value cstZeroDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstBatchSize =
            rewriter.createOrFold<Torch::AtenSizeIntOp>(loc, query, cstZeroDim);
        Value cstOneDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstSequenceLength =
            rewriter.createOrFold<Torch::AtenSizeIntOp>(loc, query, cstOneDim);

        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstKVNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(kvNumHeads));

        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstIntZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstIntOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstDim1 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstDim2 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(2));
        Value cstInt64Dtype = Torch::ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr((int)torch_upstream::ScalarType::Long));

        Type intListType = Torch::ListType::get(Torch::IntType::get(context));

        // Convert seqlens_k to int64 unconditionally (may be si32 from ONNX).
        Torch::ValueTensorType seqLensKType =
            cast<Torch::ValueTensorType>(seqlensK.getType());
        if (seqLensKType.getOptionalDtype() &&
            seqLensKType.getOptionalDtype().isInteger(32)) {
          seqlensK = Torch::AtenToDtypeOp::create(
              rewriter, loc,
              seqLensKType.getWithSizesAndDtype(
                  seqLensKType.getOptionalSizes(),
                  rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
              seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
              /*copy=*/cstFalse, /*memory_format=*/cstNone);
        }

        // Reshape Q/K/V from [batch, seq, hidden] to [batch, heads, seq,
        // head_size]. This requires:
        // 1. Reshape to [batch, seq, heads, head_size]
        // 2. Transpose dims 1 and 2 to get [batch, heads, seq, head_size]
        // A direct reshape would incorrectly interleave heads and sequence
        // positions.

        // Unflatten query: [batch, seq, hidden] -> [batch, seq, num_heads,
        // head_size]
        SmallVector<int64_t> queryIntermediateSizesInt{
            batchSize, sequenceLength, numHeads, headSize};
        Value unflattenSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            llvm::SmallVector<Value>{cstNumHeads, cstHeadSize});
        Value qIntermediate = Torch::AtenUnflattenIntOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryIntermediateSizesInt,
                                           queryType.getOptionalDtype()),
            query, /*dim=*/cstDim2, unflattenSizesList);

        // Transpose query: [batch, seq, num_heads, head_size] -> [batch,
        // num_heads, seq, head_size]
        SmallVector<int64_t> queryFinalSizesInt{batchSize, numHeads,
                                                sequenceLength, headSize};
        Value qInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryFinalSizesInt,
                                           queryType.getOptionalDtype()),
            qIntermediate, cstDim1, cstDim2);

        // Unflatten key: [batch, seq, kv_hidden] -> [batch, seq, kv_num_heads,
        // head_size]
        SmallVector<int64_t> kvIntermediateSizesInt{batchSize, sequenceLength,
                                                    kvNumHeads, headSize};
        Value kvUnflattenSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            llvm::SmallVector<Value>{cstKVNumHeads, cstHeadSize});
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kIntermediate = Torch::AtenUnflattenIntOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvIntermediateSizesInt,
                                         keyType.getOptionalDtype()),
            key, /*dim=*/cstDim2, kvUnflattenSizesList);

        // Transpose key: [batch, seq, kv_num_heads, head_size] -> [batch,
        // kv_num_heads, seq, head_size]
        SmallVector<int64_t> kvFinalSizesInt{batchSize, kvNumHeads,
                                             sequenceLength, headSize};
        Value kInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvFinalSizesInt,
                                         keyType.getOptionalDtype()),
            kIntermediate, cstDim1, cstDim2);

        // Unflatten value: [batch, seq, kv_hidden] -> [batch, seq,
        // kv_num_heads, head_size]
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vIntermediate = Torch::AtenUnflattenIntOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvIntermediateSizesInt,
                                           valueType.getOptionalDtype()),
            value, /*dim=*/cstDim2, kvUnflattenSizesList);

        // Transpose value: [batch, seq, kv_num_heads, head_size] -> [batch,
        // kv_num_heads, seq, head_size]
        Value vInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvFinalSizesInt,
                                           valueType.getOptionalDtype()),
            vIntermediate, cstDim1, cstDim2);

        Value qRotary = qInput, kRotary = kInput;
        if (doRotary) {

          // Generating position_ids for rotary_embedding:
          //   total_seqlens = seqlens_k + 1
          //   past_seqlens = total_seqlens - sequence_length
          //   pos_ids = torch.arange(sequence_length).repeat(batch_size, 1)
          //   pos_ids = pos_ids + past_seqlens.view(-1, 1)
          //
          // This works for all cases:
          // - Pure prefill: past_seqlens = 0, positions = [0, 1, ..., seq-1]
          // - Single decode: past_seqlens = past_len, positions = [past_len]
          // - Multi-token with past: positions = [past_len, ...,
          // past_len+seq-1]
          SmallVector<int64_t> positionIdsSizeInt{batchSize, sequenceLength};
          Torch::ValueTensorType positionIdsType = Torch::ValueTensorType::get(
              context, positionIdsSizeInt,
              IntegerType::get(context, 64, IntegerType::Signed));

          Value cstInterleaved = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(rotaryInterleaved));
          Value cstFloatOne = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(1.0));

          // total_seqlens = seqlens_k + 1 (per ONNX spec, seqlens_k = total -
          // 1)
          Value totalSeqLens = Torch::AtenAddScalarOp::create(
              rewriter, loc, seqlensK.getType(), /*self=*/seqlensK,
              /*other=*/cstIntOne,
              /*alpha=*/cstIntOne);

          // past_seqlens = total_seqlens - sequence_length
          Value pastSeqLens = Torch::AtenSubScalarOp::create(
              rewriter, loc, totalSeqLens.getType(), /*self=*/totalSeqLens,
              /*other=*/cstSequenceLength, /*alpha=*/cstIntOne);

          // Create position IDs: arange(seq_len).repeat(batch, 1) +
          // past_seqlens
          Torch::ValueTensorType initPosIdsType = Torch::ValueTensorType::get(
              context, {sequenceLength},
              IntegerType::get(context, 64, IntegerType::Signed));
          Value initPosIds = Torch::AtenArangeOp::create(
              rewriter, loc, initPosIdsType, cstSequenceLength, cstInt64Dtype,
              /*layout=*/cstNone,
              /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value repeatValuesList = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              llvm::SmallVector<Value>{cstBatchSize, cstIntOne});
          Value positionIds = Torch::AtenRepeatOp::create(
              rewriter, loc, positionIdsType, initPosIds,
              /*repeats=*/repeatValuesList);

          // Unsqueeze past_seqlens: [batch] -> [batch, 1] for broadcasting
          Torch::ValueTensorType seqLensViewType = Torch::ValueTensorType::get(
              context, llvm::SmallVector<int64_t>{batchSize, 1},
              IntegerType::get(context, 64, IntegerType::Signed));
          pastSeqLens = Torch::AtenUnsqueezeOp::create(
              rewriter, loc, seqLensViewType, pastSeqLens, cstDim1);

          // Add past_seqlens to get final position IDs
          positionIds = Torch::AtenAddTensorOp::create(
              rewriter, loc, positionIdsType, positionIds, pastSeqLens,
              /*alpha=*/cstIntOne);

          // Performing RotaryEmbedding over Query and Key.
          qRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), qInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);

          kRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, kInput.getType(), kInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);
        }

        // Build present_key/present_value by padding past with zeros, then
        // scattering current K/V into the correct per-batch position.
        //
        // Why pad instead of cat? With cat(past, current), the current token
        // ends up at position past_seq_len. With variable seqlens_k, ORT places
        // the current token at seqlens_k[b] and leaves position past_seq_len as
        // zero/uninitialized. Using cat+scatter leaves a stale copy of the
        // current token at past_seq_len which doesn't match ORT's output.
        // Padding with zeros then scattering avoids this.
        //
        // constant_pad_nd pads innermost dims first: [0, 0, 0, seq_len]
        //   dim 3 (head_size): [0, 0]  -- no padding
        //   dim 2 (seq):       [0, seq_len] -- extend by seq_len on the right
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value padList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstIntZero, cstIntZero, cstIntZero,
                               cstSequenceLength});
        Value presentKey = Torch::AtenConstantPadNdOp::create(
            rewriter, loc, resultTypes[1], pastKey, padList, cstFloatZero);
        Value presentValue = Torch::AtenConstantPadNdOp::create(
            rewriter, loc, resultTypes[2], pastValue, padList, cstFloatZero);

        // Scatter current K/V into the padded buffer at position pastLen[b]+q.
        // pastLen = seqlens_k + 1 - seq_len
        Value totalSeqForScatter = Torch::AtenAddScalarOp::create(
            rewriter, loc, seqlensK.getType(), seqlensK, cstIntOne, cstIntOne);
        Value pastLen = Torch::AtenSubScalarOp::create(
            rewriter, loc, totalSeqForScatter.getType(), totalSeqForScatter,
            cstSequenceLength, cstIntOne);

        // qRange: [0, 1, ..., seqLen-1] — shared by scatter and mask.
        Torch::ValueTensorType qRangeType = Torch::ValueTensorType::get(
            context, {sequenceLength},
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value qRange = Torch::AtenArangeOp::create(
            rewriter, loc, qRangeType, cstSequenceLength, cstInt64Dtype,
            /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

        // pastLen -> [B, 1, 1, 1] via unsqueeze chain for 4D scatter index
        // broadcasting
        auto intSiType = rewriter.getIntegerType(64, /*isSigned=*/true);
        Value cstDim3 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(3));
        // [batch] -> [batch, 1]
        Torch::ValueTensorType pastLenUnsq1Type = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{batchSize, 1}, intSiType);
        Value pastLenUnsq1 = Torch::AtenUnsqueezeOp::create(
            rewriter, loc, pastLenUnsq1Type, pastLen, cstDim1);
        // [batch, 1] -> [batch, 1, 1]
        Torch::ValueTensorType pastLenUnsq2Type = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{batchSize, 1, 1}, intSiType);
        Value pastLenUnsq2 = Torch::AtenUnsqueezeOp::create(
            rewriter, loc, pastLenUnsq2Type, pastLenUnsq1, cstDim2);
        // [batch, 1, 1] -> [batch, 1, 1, 1]
        Torch::ValueTensorType pastLenView4dType = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{batchSize, 1, 1, 1}, intSiType);
        Value pastLenView4d = Torch::AtenUnsqueezeOp::create(
            rewriter, loc, pastLenView4dType, pastLenUnsq2, cstDim3);

        // qRange -> [1, 1, seq, 1] via unsqueeze chain for scatter
        // [seq] -> [1, seq]
        Torch::ValueTensorType qUnsq0Type = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{1, sequenceLength}, intSiType);
        Value qUnsq0 = Torch::AtenUnsqueezeOp::create(rewriter, loc, qUnsq0Type,
                                                      qRange, cstIntZero);
        // [1, seq] -> [1, 1, seq]
        Torch::ValueTensorType qUnsq1Type = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{1, 1, sequenceLength}, intSiType);
        Value qUnsq1 = Torch::AtenUnsqueezeOp::create(rewriter, loc, qUnsq1Type,
                                                      qUnsq0, cstIntZero);
        // [1, 1, seq] -> [1, 1, seq, 1]
        Torch::ValueTensorType scatterQViewType = Torch::ValueTensorType::get(
            context, SmallVector<int64_t>{1, 1, sequenceLength, 1}, intSiType);
        Value scatterQRangeView = Torch::AtenUnsqueezeOp::create(
            rewriter, loc, scatterQViewType, qUnsq1, cstDim3);

        // scatterIdxBase = pastLen[B,1,1,1] + qRange[1,1,seq,1]
        //               -> [B, 1, seq, 1]
        SmallVector<int64_t> scatterIdxBaseSizes{batchSize, 1, sequenceLength,
                                                 1};
        Torch::ValueTensorType scatterIdxBaseType = Torch::ValueTensorType::get(
            context, scatterIdxBaseSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterIdxBase = Torch::AtenAddTensorOp::create(
            rewriter, loc, scatterIdxBaseType, pastLenView4d, scatterQRangeView,
            cstIntOne);

        // Expand to [B, kv_heads, seq, head_size] to match current K/V shape
        SmallVector<int64_t> scatterExpandSizes{batchSize, kvNumHeads,
                                                sequenceLength, headSize};
        Torch::ValueTensorType scatterIdxType = Torch::ValueTensorType::get(
            context, scatterExpandSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterExpandSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstBatchSize, cstKVNumHeads, cstSequenceLength,
                               cstHeadSize});
        Value scatterIdx = Torch::AtenExpandOp::create(
            rewriter, loc, scatterIdxType, scatterIdxBase,
            scatterExpandSizeList, /*implicit=*/cstFalse);

        // Scatter current K/V into buffer at position pastLen[b] + q
        presentKey = Torch::AtenScatterSrcOp::create(
            rewriter, loc, resultTypes[1], presentKey, cstDim2, scatterIdx,
            kRotary);
        presentValue = Torch::AtenScatterSrcOp::create(
            rewriter, loc, resultTypes[2], presentValue, cstDim2, scatterIdx,
            vInput);

        // Generate causal attention mask.
        // With scatter, KV layout matches ORT: current at pastLen[b].
        // Simple boolean mask: k <= pastLen[b] + q
        // Mask shape: [batch, 1, seqLen, kvSeqLen] (i1).
        Value attnMask = cstNone;

        // Get the KV sequence length from presentKey shape
        Torch::ValueTensorType presentKeyType =
            cast<Torch::ValueTensorType>(presentKey.getType());
        if (presentKeyType.hasSizes() &&
            presentKeyType.getSizes().size() == 4) {
          int64_t kvSeqLen = presentKeyType.getSizes()[2];

          // Only generate mask if KV sequence length is dynamic or > 0
          // For dynamic shapes or non-trivial sequences, we need to mask
          if (kvSeqLen == Torch::kUnknownSize || kvSeqLen > 0) {
            // Get KV sequence dimension size
            Value kvSeqLenVal = Torch::AtenSizeIntOp::create(
                rewriter, loc, rewriter.getType<Torch::IntType>(), presentKey,
                cstDim2);

            // kRange: [0, 1, 2, ..., kvSeqLen-1] shape [kvSeqLen]
            Torch::ValueTensorType kRangeType = Torch::ValueTensorType::get(
                context, {kvSeqLen},
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value kRange = Torch::AtenArangeOp::create(
                rewriter, loc, kRangeType, kvSeqLenVal, cstInt64Dtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // Reshape for broadcasting:
            // pastLen: [batch] -> [batch, 1, 1]
            // qRange: [seqLen] -> [1, seqLen, 1]  (reuses qRange from above)
            // kRange: [kvSeqLen] -> [1, 1, kvSeqLen]

            // pastLen -> [batch, 1, 1] via unsqueeze chain
            // [batch] -> [batch, 1]
            Torch::ValueTensorType pastLenMaskUnsq1Type =
                Torch::ValueTensorType::get(
                    context, SmallVector<int64_t>{batchSize, 1}, intSiType);
            Value pastLenMaskUnsq1 = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, pastLenMaskUnsq1Type, pastLen, cstDim1);
            // [batch, 1] -> [batch, 1, 1]
            Torch::ValueTensorType seqlensViewType =
                Torch::ValueTensorType::get(
                    context, SmallVector<int64_t>{batchSize, 1, 1}, intSiType);
            Value pastLenView = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, seqlensViewType, pastLenMaskUnsq1, cstDim2);

            // qRange -> [1, seqLen, 1] via unsqueeze chain
            // [seqLen] -> [1, seqLen]
            Torch::ValueTensorType qMaskUnsq0Type = Torch::ValueTensorType::get(
                context, SmallVector<int64_t>{1, sequenceLength}, intSiType);
            Value qMaskUnsq0 = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, qMaskUnsq0Type, qRange, cstIntZero);
            // [1, seqLen] -> [1, seqLen, 1]
            Torch::ValueTensorType qViewType = Torch::ValueTensorType::get(
                context, SmallVector<int64_t>{1, sequenceLength, 1}, intSiType);
            Value qRangeView = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, qViewType, qMaskUnsq0, cstDim2);

            // kRange -> [1, 1, kvSeqLen] via unsqueeze chain
            // [kvSeqLen] -> [1, kvSeqLen]
            Torch::ValueTensorType kUnsq0Type = Torch::ValueTensorType::get(
                context, SmallVector<int64_t>{1, kvSeqLen}, intSiType);
            Value kUnsq0 = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, kUnsq0Type, kRange, cstIntZero);
            // [1, kvSeqLen] -> [1, 1, kvSeqLen]
            Torch::ValueTensorType kViewType = Torch::ValueTensorType::get(
                context, SmallVector<int64_t>{1, 1, kvSeqLen}, intSiType);
            Value kRangeView = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, kViewType, kUnsq0, cstIntZero);

            // Causal mask: k <= pastLen + q
            // pastLenView[batch,1,1] + qRangeView[1,seqLen,1]
            // -> [batch, seqLen, 1]
            SmallVector<int64_t> pastLenPlusQSizes{batchSize, sequenceLength,
                                                   1};
            Torch::ValueTensorType pastLenPlusQType =
                Torch::ValueTensorType::get(
                    context, pastLenPlusQSizes,
                    rewriter.getIntegerType(64, /*isSigned=*/true));
            Value pastLenPlusQ = Torch::AtenAddTensorOp::create(
                rewriter, loc, pastLenPlusQType, pastLenView, qRangeView,
                cstIntOne);

            // kRangeView[1,1,kvSeqLen] <= pastLenPlusQ[batch,seqLen,1]
            // -> [batch, seqLen, kvSeqLen]
            SmallVector<int64_t> maskBoolSizes{batchSize, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType maskBoolType = Torch::ValueTensorType::get(
                context, maskBoolSizes, rewriter.getI1Type());
            Value causalMask = Torch::AtenLeTensorOp::create(
                rewriter, loc, maskBoolType, kRangeView, pastLenPlusQ);

            // Unsqueeze to [batch, 1, seqLen, kvSeqLen] for SDPA.
            // Pass the boolean mask directly — downstream backends (e.g.
            // IREE's iree_linalg_ext.attention) handle bool-to-float
            // conversion internally.
            SmallVector<int64_t> attnMaskSizes{batchSize, 1, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType attnMaskType = Torch::ValueTensorType::get(
                context, attnMaskSizes, rewriter.getI1Type());
            attnMask = Torch::AtenUnsqueezeOp::create(
                rewriter, loc, attnMaskType, causalMask, cstDim1);
          }
        }

        // Do attention with full KV cache (past + current) and mask.
        Value cstEnableGQA = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstScale = cstNone;
        if (scale != 0.0f)
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));

        // Use presentKey/presentValue (full KV cache) for attention, not just
        // the current token's K/V. This is essential for proper KV caching.
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qRotary.getType(), qRotary, presentKey, presentValue,
            /*attn_mask=*/attnMask,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/cstFalse, cstScale,
            cstEnableGQA);

        // Reshaping the attention result from:
        //    (batch_size, num_heads, sequence_length, head_size)
        // -> (batch_size, sequence_length, hidden_size)
        // This requires:
        // 1. Transpose dims 1 and 2: [batch, num_heads, seq, head_size]
        //                         -> [batch, seq, num_heads, head_size]
        // 2. Reshape: [batch, seq, num_heads, head_size] -> [batch, seq,
        // hidden]
        // Transpose: [batch, num_heads, seq, head_size] -> [batch, seq,
        // num_heads, head_size]
        SmallVector<int64_t> attnTransposedSizes{batchSize, sequenceLength,
                                                 numHeads, headSize};
        Torch::ValueTensorType attnType =
            cast<Torch::ValueTensorType>(attention.getType());
        Value attnTransposed = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            attnType.getWithSizesAndDtype(attnTransposedSizes,
                                          attnType.getOptionalDtype()),
            attention, cstDim1, cstDim2);

        // Flatten: [batch, seq, num_heads, head_size] -> [batch, seq, hidden]
        attention = Torch::AtenFlattenUsingIntsOp::create(
            rewriter, loc, resultTypes[0], attnTransposed, cstDim2, cstDim3);

        rewriter.replaceOp(binder.op, {attention, presentKey, presentValue});
        return success();
      });
  patterns.onOp(
      "QLinearAdd", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, aScale, aZp, b, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the result of "Add".
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value alpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value c = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b, alpha);

        // Quantizing the result of "Add" operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "QLinearLeakyRelu", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        float alpha;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha"))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the LeakyRelu result.
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)alpha));
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y =
            Torch::AtenLeakyReluOp::create(rewriter, loc, yTy, x, constAlpha);

        // Quantizing the result of LeakyRelu op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearConcat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t axis;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(axis, "axis") ||
            binder.tensorResultType(resultType))
          return failure();

        SmallVector<Value> inputs, inputScales, inputZeroPoints;
        for (unsigned i = 2; i < operands.size(); i = i + 3) {
          inputs.push_back(operands[i]);
          inputScales.push_back(operands[i + 1]);
          inputZeroPoints.push_back(operands[i + 2]);
        }

        unsigned numInputs = (operands.size() - 2) / 3;
        if (!(llvm::all_equal({inputs.size(), inputScales.size(),
                               inputZeroPoints.size()}) &&
              inputs.size() == numInputs))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible number of input operands, scales and/or "
                         "zero-points");

        // Preparing the dequantized inputs.
        SmallVector<Value> dequantizedInputs;
        for (unsigned i = 0; i < numInputs; i++) {
          Value scale, zeroPoint;
          if (failed(extractPerTensorQuantizationArguments(
                  rewriter, loc, /*scale=*/inputScales[i],
                  /*zero_point=*/inputZeroPoints[i], scale, zeroPoint)))
            return rewriter.notifyMatchFailure(
                binder.op, "Incompatible scale and zero-points argument for "
                           "per-tensor quantization");

          Value dequantizedInput;
          if (failed(createDequantizeTensor(rewriter, loc, inputs[i], scale,
                                            zeroPoint,
                                            /*output=*/dequantizedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to dequantize the input tensor because of "
                           "missing sizes");

          dequantizedInputs.push_back(dequantizedInput);
        }

        // Concatenating the inputs.
        Type listElemType =
            cast<Torch::BaseTensorType>(dequantizedInputs[0].getType())
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = Torch::PrimListConstructOp::create(
            rewriter, binder.op->getLoc(), listType, dequantizedInputs);
        Value cstAxis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axis));
        auto concatTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value concat = Torch::AtenCatOp::create(rewriter, loc, concatTy,
                                                tensorList, cstAxis);

        // Quantizing the result of concatenated inputs.
        Value yScale, yZp;
        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[0],
                /*zero_point=*/operands[1], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible scale and zero-points argument for "
                         "per-tensor quantization");
        Torch::ValueTensorType yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        Value result = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, concat, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          result);
        return success();
      });
  patterns.onOp(
      "QLinearGlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperands(operands, 5) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        auto xTy = dyn_cast<Torch::ValueTensorType>(operands[0].getType());
        if (!xTy || !xTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input argument `x` to have sizes");
        ArrayRef<int64_t> inputShape = xTy.getSizes();

        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the AvgPool result.
        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        unsigned inputRank = inputShape.size();
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            Value inputDimSize =
                Torch::AtenSizeIntOp::create(rewriter, loc, x, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultShape, rewriter.getF32Type());
        Value avgpool;
        if (inputRank == 3) {
          avgpool = Torch::AtenAvgPool1dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad);
        } else if (inputRank == 4) {
          avgpool = Torch::AtenAvgPool2dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else if (inputRank == 5) {
          avgpool = Torch::AtenAvgPool3dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else {
          return failure();
        }

        // Quantizing the result of AvgPool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        avgpool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, avgpool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          avgpool);
        return success();
      });
  patterns.onOp(
      "QLinearSigmoid", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the Sigmoid result.
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y = Torch::AtenSigmoidOp::create(rewriter, loc, yTy, x);

        // Quantizing the result of Sigmoid op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Creating Onnx.AveragePool op.
        llvm::SmallVector<Value> newOperands = {x};
        llvm::SmallVector<NamedAttribute> newAttributes;
        for (auto namedAttr : binder.op->getAttrDictionary()) {
          if (namedAttr.getName().getValue().compare("name") == 0)
            continue;
          newAttributes.push_back(namedAttr);
        }

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Torch::OperatorOp::Properties properties;
        properties.name = rewriter.getStringAttr("onnx.AveragePool");
        Value averagePool =
            Torch::OperatorOp::create(rewriter, binder.getLoc(), yTy,
                                      newOperands, properties, newAttributes,
                                      binder.op->getRegions().size())
                .getResult(0);

        // Quantizing the result of AveragePool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        averagePool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, averagePool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          averagePool);
        return success();
      });
  patterns.onOp(
      "FusedMatMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        int64_t transA, transB, transBatchA, transBatchB;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.s64IntegerAttr(transBatchA, "transBatchA", 0) ||
            binder.s64IntegerAttr(transBatchB, "transBatchB", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        // Transposing the LHS argument.
        Value transposedLhs = lhs;
        if (transA) {
          // Determine the rank of lhs tensor.
          std::optional<unsigned> maybeRank = Torch::getTensorRank(lhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked lhs tensor");
          unsigned lhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), lhs,
                  /*dimA=*/lhsRank - 2, /*dimB=*/lhsRank - 1, transposedLhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for lhs");
        }

        // Transposing the RHS argument.
        Value transposedRhs = rhs;
        if (transB) {
          std::optional<unsigned> maybeRank = Torch::getTensorRank(rhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked rhs tensor");
          unsigned rhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), rhs,
                  /*dimA=*/rhsRank - 2, /*dimB=*/rhsRank - 1, transposedRhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for rhs");
        }

        // TODO: Add support for `transBatchA` and `transBatchB`
        // attribute.
        if (transBatchA || transBatchB)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: support not present for "
                         "transBatchA and transBatchB attribute");

        rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
            binder.op, resultType, transposedLhs, transposedRhs);
        return success();
      });
  patterns.onOp(
      "QLinearMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, b, aScale, aZp, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the Mul result.
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value c = Torch::AtenMulTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b);

        // Quantizing the result of Mul operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "MultiHeadAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // com.microsoft::MultiHeadAttention, lowered to
        // aten.scaled_dot_product_attention on [B, N, seq, head] operands with
        // an optional additive mask. Notation: B batch, S query seq, L kv seq,
        // P past seq, T = P + L, N num_heads, H / Hv head sizes.
        //
        //   query  : [B, S, N*H] or packed QKV [B, S, N, 3, H]
        //   key    : [B, L, N*H], [B, N, L, H] or packed KV [B, L, N, 2, H]
        //   value  : [B, L, N*Hv] or [B, N, L, Hv] (absent when packed)
        //   bias   : [N*H + N*H + N*Hv], added to the 3D / packed projections;
        //            [B, N, L, *] key/value are used as-is.
        //   key_padding_mask : [B] key lengths, [B, T] or [B, S, T]; padded
        //            keys get mask_filter_value.
        //   attention_bias   : [B|1, N|1, S, T], added to the scores.
        //   past_key / past_value : [B, N, P, H] / [B, N, P, Hv], prepended
        //            along the sequence dim to form present_key /
        //            present_value.
        //
        // Semantics follow the ORT CPU kernel: scale == 0 selects 1/sqrt(H);
        // unidirectional masks key j for query i when j > P + i, using the
        // lowest finite value of the element type.
        //
        // Past K / V are always attended (T = P + L), regardless of which
        // present outputs are requested. ORT CPU concatenates past only into
        // the present buffers that are requested; with past and fewer than 3
        // results it attends over a partial or no past (while still checking
        // the mask against P + L). That behaviour is not reproduced.
        //
        // Match failure: past_sequence_length, cache_indirection, the qk
        // output, a static 1D mask length != B (e.g. the [3 * B + 2]
        // encoding), and unidirectional with a static L != S (ORT does not
        // mask the last query row there). A 1D mask with a dynamic B or
        // length is taken as [B] key lengths and checked at runtime.
        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        const int64_t kUnknown = Torch::kUnknownSize;

        int64_t numOperands = binder.op->getNumOperands();
        if (numOperands > 10)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected at most 10 inputs");
        auto operand = [&](int64_t idx) -> Value {
          if (idx >= numOperands)
            return Value();
          Value v = binder.op->getOperand(idx);
          return isa<Torch::NoneType>(v.getType()) ? Value() : v;
        };
        Value query = operand(0), key = operand(1), value = operand(2),
              bias = operand(3), keyPaddingMask = operand(4),
              attentionBias = operand(5), pastKey = operand(6),
              pastValue = operand(7);
        if (!query)
          return rewriter.notifyMatchFailure(binder.op, "expected `query`");
        if (operand(8) || operand(9))
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: past_sequence_length and "
                         "cache_indirection are not supported");
        if (!pastKey != !pastValue)
          return rewriter.notifyMatchFailure(
              binder.op, "past_key and past_value must be both present or "
                         "both absent");

        int64_t numResults = binder.op->getNumResults();
        if (numResults < 1)
          return rewriter.notifyMatchFailure(
              binder.op, "expected at least the attention `output` result");
        if (numResults > 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: the `qk` output is not supported");
        SmallVector<Torch::ValueTensorType> resultTypes(numResults);
        for (int64_t i = 0; i < numResults; ++i)
          if (binder.tensorResultTypeAtIndex(resultTypes[i], i))
            return rewriter.notifyMatchFailure(binder.op,
                                               "result type bind failure");

        int64_t numHeads, unidirectional;
        float scale, maskFilterValue;
        if (binder.s64IntegerAttr(numHeads, "num_heads", 0) ||
            binder.s64IntegerAttr(unidirectional, "unidirectional", 0) ||
            binder.f32FloatAttr(scale, "scale", 0.0f) ||
            binder.f32FloatAttr(maskFilterValue, "mask_filter_value",
                                -10000.0f))
          return rewriter.notifyMatchFailure(binder.op,
                                             "attribute bind failure");
        if (numHeads <= 0)
          return rewriter.notifyMatchFailure(
              binder.op, "num_heads must be a positive attribute");

        // Validation. Nothing is emitted before all checks pass.
        auto rankOf = [](Value v) -> int64_t {
          auto t = dyn_cast<Torch::ValueTensorType>(v.getType());
          return t && t.hasSizes() ? t.getSizes().size() : -1;
        };
        // Unknown for an unranked type or an out-of-range `d`. Every caller
        // checks the rank first; this keeps the helper safe on its own.
        auto dimOf = [&](Value v, int64_t d) -> int64_t {
          auto t = dyn_cast<Torch::ValueTensorType>(v.getType());
          if (!t || !t.hasSizes() || d < 0 ||
              d >= static_cast<int64_t>(t.getSizes().size()))
            return kUnknown;
          return t.getSizes()[d];
        };
        auto compatible = [&](int64_t a, int64_t b) {
          return a == kUnknown || b == kUnknown || a == b;
        };

        auto queryType = dyn_cast<Torch::ValueTensorType>(query.getType());
        Type elemType = queryType ? queryType.getOptionalDtype() : Type();
        if (!elemType || !isa<mlir::FloatType>(elemType))
          return rewriter.notifyMatchFailure(
              binder.op, "expected a floating-point `query`");
        for (Value v : {key, value, bias, attentionBias, pastKey, pastValue}) {
          if (!v)
            continue;
          auto t = dyn_cast<Torch::ValueTensorType>(v.getType());
          if (!t || t.getOptionalDtype() != elemType)
            return rewriter.notifyMatchFailure(
                binder.op, "all T-typed inputs must share the query dtype");
        }
        if (keyPaddingMask) {
          auto t = dyn_cast<Torch::ValueTensorType>(keyPaddingMask.getType());
          if (!t || !isa_and_nonnull<mlir::IntegerType>(t.getOptionalDtype()))
            return rewriter.notifyMatchFailure(
                binder.op, "expected an integer key_padding_mask");
        }

        int64_t qRank = rankOf(query);
        int64_t kRank = key ? rankOf(key) : 0;
        int64_t vRank = value ? rankOf(value) : 0;
        if (qRank != 3 && qRank != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "expected 3D or packed 5D `query`");
        int64_t batch = dimOf(query, 0), seqLen = dimOf(query, 1);
        int64_t kvSeqLen, headSize, vHeadSize;
        if (qRank == 5) {
          if (key || value)
            return rewriter.notifyMatchFailure(
                binder.op, "packed QKV `query` excludes `key` and `value`");
          headSize = vHeadSize = dimOf(query, 4);
          kvSeqLen = seqLen;
          if (!compatible(dimOf(query, 2), numHeads) || dimOf(query, 3) != 3 ||
              headSize == kUnknown)
            return rewriter.notifyMatchFailure(
                binder.op, "expected packed QKV [B, S, num_heads, 3, H] with "
                           "static H");
        } else {
          int64_t hidden = dimOf(query, 2);
          if (hidden == kUnknown || hidden % numHeads != 0)
            return rewriter.notifyMatchFailure(
                binder.op, "expected a static `query` hidden size divisible "
                           "by num_heads");
          headSize = hidden / numHeads;
          if (kRank == 5 && !value) {
            kvSeqLen = dimOf(key, 1);
            vHeadSize = headSize;
            if (!compatible(dimOf(key, 2), numHeads) || dimOf(key, 3) != 2 ||
                !compatible(dimOf(key, 4), headSize))
              return rewriter.notifyMatchFailure(
                  binder.op, "expected packed KV [B, L, num_heads, 2, H] "
                             "matching the query head size");
          } else if (kRank == 3 && vRank == 3) {
            kvSeqLen = dimOf(key, 1);
            int64_t vHidden = dimOf(value, 2);
            if (!compatible(dimOf(key, 2), hidden))
              return rewriter.notifyMatchFailure(
                  binder.op, "`key` hidden size must equal `query` hidden "
                             "size");
            if (vHidden == kUnknown || vHidden % numHeads != 0)
              return rewriter.notifyMatchFailure(
                  binder.op, "expected a static `value` hidden size "
                             "divisible by num_heads");
            if (!compatible(dimOf(value, 1), kvSeqLen))
              return rewriter.notifyMatchFailure(
                  binder.op, "`key` and `value` sequence lengths differ");
            vHeadSize = vHidden / numHeads;
          } else if (kRank == 4 && vRank == 4) {
            kvSeqLen = dimOf(key, 2);
            vHeadSize = dimOf(value, 3);
            if (!compatible(dimOf(key, 1), numHeads) ||
                !compatible(dimOf(value, 1), numHeads) ||
                !compatible(dimOf(key, 3), headSize) || vHeadSize == kUnknown ||
                !compatible(dimOf(value, 2), kvSeqLen))
              return rewriter.notifyMatchFailure(
                  binder.op, "expected [B, num_heads, L, H] `key` and "
                             "[B, num_heads, L, Hv] `value`");
          } else {
            return rewriter.notifyMatchFailure(
                binder.op, "unsupported `key` / `value` layout");
          }
        }
        int64_t hidden = numHeads * headSize, vHidden = numHeads * vHeadSize;

        for (Value v : {key, value, pastKey, pastValue})
          if (v && rankOf(v) > 0 && !compatible(dimOf(v, 0), batch))
            return rewriter.notifyMatchFailure(
                binder.op, "batch size differs across inputs");

        if (bias && (rankOf(bias) != 1 ||
                     !compatible(dimOf(bias, 0), 2 * hidden + vHidden)))
          return rewriter.notifyMatchFailure(
              binder.op, "expected `bias` of shape [2 * N * H + N * Hv]");

        int64_t pastSeqLen = 0;
        if (pastKey) {
          if (rankOf(pastKey) != 4 || rankOf(pastValue) != 4 ||
              !compatible(dimOf(pastKey, 1), numHeads) ||
              !compatible(dimOf(pastValue, 1), numHeads) ||
              !compatible(dimOf(pastKey, 3), headSize) ||
              !compatible(dimOf(pastValue, 3), vHeadSize) ||
              !compatible(dimOf(pastKey, 2), dimOf(pastValue, 2)))
            return rewriter.notifyMatchFailure(
                binder.op, "expected [B, num_heads, P, H] `past_key` and "
                           "[B, num_heads, P, Hv] `past_value`");
          pastSeqLen = dimOf(pastKey, 2);
        }
        int64_t totalSeqLen = (pastSeqLen == kUnknown || kvSeqLen == kUnknown)
                                  ? kUnknown
                                  : pastSeqLen + kvSeqLen;

        if (unidirectional != 0 && seqLen != kUnknown && kvSeqLen != kUnknown &&
            seqLen != kvSeqLen)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: unidirectional requires the "
                         "key/value sequence length to equal the query's");

        int64_t maskRank = keyPaddingMask ? rankOf(keyPaddingMask) : 0;
        if (maskRank == 1) {
          // Only [B] key lengths are supported. A static length != B (e.g.
          // [3 * B + 2]) is rejected here; otherwise it is checked at runtime.
          if (!compatible(dimOf(keyPaddingMask, 0), batch))
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: a 1D key_padding_mask must be "
                           "[B] key lengths");
        } else if (maskRank == 2 || maskRank == 3) {
          if (!compatible(dimOf(keyPaddingMask, 0), batch) ||
              !compatible(dimOf(keyPaddingMask, maskRank - 1), totalSeqLen) ||
              (maskRank == 3 && !compatible(dimOf(keyPaddingMask, 1), seqLen)))
            return rewriter.notifyMatchFailure(
                binder.op, "expected key_padding_mask [B, T] or [B, S, T]");
        } else if (keyPaddingMask) {
          return rewriter.notifyMatchFailure(
              binder.op, "expected a 1D, 2D or 3D key_padding_mask");
        }
        auto compatibleOrOne = [&](int64_t a, int64_t b) {
          return a == 1 || compatible(a, b);
        };
        if (attentionBias &&
            (rankOf(attentionBias) != 4 ||
             !compatibleOrOne(dimOf(attentionBias, 0), batch) ||
             !compatibleOrOne(dimOf(attentionBias, 1), numHeads) ||
             !compatible(dimOf(attentionBias, 2), seqLen) ||
             !compatible(dimOf(attentionBias, 3), totalSeqLen)))
          return rewriter.notifyMatchFailure(
              binder.op, "expected `attention_bias` [B|1, N|1, S, T]");

        // Inputs that are each compatible with a dynamic query dimension can
        // still disagree with one another. Merge every static B, S and T so
        // such conflicts fail here rather than at runtime.
        auto merge = [&](int64_t &acc, int64_t s) {
          if (s == kUnknown || acc == s)
            return true;
          if (acc == kUnknown) {
            acc = s;
            return true;
          }
          return false;
        };
        int64_t knownBatch = batch, knownSeq = seqLen, knownTotal = totalSeqLen;
        bool consistent = true;
        for (Value v : {key, value, pastKey, pastValue})
          if (v)
            consistent &= merge(knownBatch, dimOf(v, 0));
        if (keyPaddingMask) {
          consistent &= merge(knownBatch, dimOf(keyPaddingMask, 0));
          if (maskRank >= 2)
            consistent &=
                merge(knownTotal, dimOf(keyPaddingMask, maskRank - 1));
          if (maskRank == 3)
            consistent &= merge(knownSeq, dimOf(keyPaddingMask, 1));
        }
        if (attentionBias) {
          if (dimOf(attentionBias, 0) != 1)
            consistent &= merge(knownBatch, dimOf(attentionBias, 0));
          consistent &= merge(knownSeq, dimOf(attentionBias, 2));
          consistent &= merge(knownTotal, dimOf(attentionBias, 3));
        }
        if (!consistent)
          return rewriter.notifyMatchFailure(
              binder.op, "conflicting static batch / sequence sizes across "
                         "inputs");

        // Emission.
        auto vtensor = [&](ArrayRef<int64_t> sizes, Type dtype) {
          return Torch::ValueTensorType::get(context, sizes, dtype);
        };
        auto cstInt = [&](int64_t v) -> Value {
          return Torch::ConstantIntOp::create(rewriter, loc,
                                              rewriter.getI64IntegerAttr(v));
        };
        auto cstFloat = [&](double v) -> Value {
          return Torch::ConstantFloatOp::create(rewriter, loc,
                                                rewriter.getF64FloatAttr(v));
        };
        Value cst0 = cstInt(0), cst1 = cstInt(1), cst2 = cstInt(2),
              cst3 = cstInt(3);
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Type intListType = Torch::ListType::get(Torch::IntType::get(context));
        Type i64Type = rewriter.getIntegerType(64, /*isSigned=*/true);
        Type i1Type = rewriter.getI1Type();

        // [B, seq, N * perHead] (+ bias slice) -> [B, N, seq, perHead].
        auto toHeads = [&](Value proj, int64_t seq, int64_t perHead,
                           int64_t biasOffset) -> Value {
          int64_t width = numHeads * perHead;
          if (bias) {
            Value start = cstInt(biasOffset);
            Value end = cstInt(biasOffset + width);
            Value slice = Torch::AtenSliceTensorOp::create(
                rewriter, loc, vtensor({width}, elemType), bias, cst0, start,
                end, cst1);
            proj = Torch::AtenAddTensorOp::create(rewriter, loc, proj.getType(),
                                                  proj, slice, cst1);
          }
          Value sizes = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              ValueRange{cstInt(numHeads), cstInt(perHead)});
          Value bsnh = Torch::AtenUnflattenIntOp::create(
              rewriter, loc, vtensor({batch, seq, numHeads, perHead}, elemType),
              proj, cst2, sizes);
          return Torch::AtenTransposeIntOp::create(
              rewriter, loc, vtensor({batch, numHeads, seq, perHead}, elemType),
              bsnh, cst1, cst2);
        };
        // Packed [B, seq, N, k, perHead] -> slot `idx` as [B, seq, N*perHead].
        auto unpack = [&](Value packed, int64_t idx, int64_t seq,
                          int64_t perHead) -> Value {
          Value slot = Torch::AtenSelectIntOp::create(
              rewriter, loc, vtensor({batch, seq, numHeads, perHead}, elemType),
              packed, cst3, cstInt(idx));
          return Torch::AtenFlattenUsingIntsOp::create(
              rewriter, loc,
              vtensor({batch, seq, numHeads * perHead}, elemType), slot, cst2,
              cst3);
        };

        // Packed layouts share one head size: vHeadSize == headSize there.
        Value qHeads, kHeads, vHeads;
        if (qRank == 5) {
          qHeads =
              toHeads(unpack(query, 0, seqLen, headSize), seqLen, headSize, 0);
          kHeads = toHeads(unpack(query, 1, seqLen, headSize), seqLen, headSize,
                           hidden);
          vHeads = toHeads(unpack(query, 2, seqLen, vHeadSize), seqLen,
                           vHeadSize, 2 * hidden);
        } else {
          qHeads = toHeads(query, seqLen, headSize, 0);
          if (kRank == 5) {
            kHeads = toHeads(unpack(key, 0, kvSeqLen, headSize), kvSeqLen,
                             headSize, hidden);
            vHeads = toHeads(unpack(key, 1, kvSeqLen, vHeadSize), kvSeqLen,
                             vHeadSize, 2 * hidden);
          } else if (kRank == 3) {
            kHeads = toHeads(key, kvSeqLen, headSize, hidden);
            vHeads = toHeads(value, kvSeqLen, vHeadSize, 2 * hidden);
          } else {
            kHeads = key;
            vHeads = value;
          }
        }

        auto prependPast = [&](Value past, Value cur, int64_t perHead) {
          if (!past)
            return cur;
          Type listElem =
              Torch::ValueTensorType::getWithLeastStaticInformation(context);
          Value list = Torch::PrimListConstructOp::create(
              rewriter, loc, Torch::ListType::get(listElem),
              ValueRange{past, cur});
          return Value(Torch::AtenCatOp::create(
              rewriter, loc,
              vtensor({batch, numHeads, totalSeqLen, perHead}, elemType), list,
              cst2));
        };
        Value presentKey = prependPast(pastKey, kHeads, headSize);
        Value presentValue = prependPast(pastValue, vHeads, vHeadSize);

        // Additive mask, accumulated in a broadcast-compatible 4D layout.
        Value attnMask;
        SmallVector<int64_t> maskSizes;
        auto accumulate = [&](Value contribution) {
          auto type = dyn_cast<Torch::ValueTensorType>(contribution.getType());
          assert(type && type.hasSizes() &&
                 "mask contributions must be ranked value tensors");
          ArrayRef<int64_t> sizes = type.getSizes();
          if (!attnMask) {
            attnMask = contribution;
            maskSizes.assign(sizes.begin(), sizes.end());
            return;
          }
          for (auto [acc, s] : llvm::zip_equal(maskSizes, sizes)) {
            // Broadcast merge: a 1 takes the other size; an unknown takes a
            // known size other than 1; a known size > 1 is kept when the other
            // is unknown, which must then be 1 or equal at runtime. Static
            // conflicts are rejected during validation.
            assert((acc == kUnknown || s == kUnknown || acc == 1 || s == 1 ||
                    acc == s) &&
                   "mask contributions must be broadcast-compatible");
            if (acc == 1 || (acc == kUnknown && s != 1))
              acc = s;
          }
          attnMask = Torch::AtenAddTensorOp::create(
              rewriter, loc, vtensor(maskSizes, elemType), attnMask,
              contribution, cst1);
        };
        // bool -> {0, fill} in the element type.
        auto toAdditive = [&](Value cond, double fill) -> Value {
          Value f = Torch::convertTensorToDtype(rewriter, loc, cond, elemType);
          return Torch::AtenMulScalarOp::create(rewriter, loc, f.getType(), f,
                                                cstFloat(fill));
        };
        auto unsqueeze = [&](Value t, int64_t dim) -> Value {
          auto type = dyn_cast<Torch::ValueTensorType>(t.getType());
          assert(type && type.hasSizes() &&
                 "unsqueeze expects a ranked value tensor");
          SmallVector<int64_t> sizes(type.getSizes());
          sizes.insert(sizes.begin() + dim, 1);
          return Torch::AtenUnsqueezeOp::create(
              rewriter, loc, vtensor(sizes, type.getOptionalDtype()), t,
              cstInt(dim));
        };
        auto dimValue = [&](Value t, int64_t dim, int64_t size) -> Value {
          if (size != kUnknown)
            return cstInt(size);
          return Torch::AtenSizeIntOp::create(rewriter, loc, t, cstInt(dim));
        };
        auto arange = [&](Value end, int64_t size) -> Value {
          return Torch::AtenArangeOp::create(
              rewriter, loc, vtensor({size}, i64Type), end, cstNone, cstNone,
              cstNone, cstNone);
        };
        auto totalSeqVal = [&] { return dimValue(presentKey, 2, totalSeqLen); };

        if (unidirectional != 0) {
          // A static L != S is rejected during validation. When either is
          // dynamic, check at runtime rather than silently diverge from ORT.
          if (seqLen == kUnknown || kvSeqLen == kUnknown) {
            Value qLen = dimValue(qHeads, 2, seqLen);
            Value kvLen = dimValue(kHeads, 2, kvSeqLen);
            Value sameLength =
                Torch::AtenEqIntOp::create(rewriter, loc, qLen, kvLen);
            Torch::RuntimeAssertOp::create(
                rewriter, loc, sameLength,
                rewriter.getStringAttr(
                    "MultiHeadAttention: unidirectional requires the key/value "
                    "sequence length to equal the query's"));
          }
          double lowest =
              APFloat::getLargest(
                  cast<mlir::FloatType>(elemType).getFloatSemantics(),
                  /*Negative=*/true)
                  .convertToDouble();
          Value rows = arange(dimValue(qHeads, 2, seqLen), seqLen);
          if (pastKey)
            rows = Torch::AtenAddScalarOp::create(
                rewriter, loc, rows.getType(), rows,
                dimValue(pastKey, 2, pastSeqLen), cst1);
          Value cols = unsqueeze(arange(totalSeqVal(), totalSeqLen), 0);
          rows = unsqueeze(rows, 1);
          Value future = Torch::AtenGtTensorOp::create(
              rewriter, loc, vtensor({seqLen, totalSeqLen}, i1Type), cols,
              rows);
          accumulate(unsqueeze(unsqueeze(toAdditive(future, lowest), 0), 0));
        }

        if (keyPaddingMask) {
          Value padded;
          if (maskRank == 1) {
            int64_t maskLen = dimOf(keyPaddingMask, 0);
            if (batch == kUnknown || maskLen == kUnknown) {
              Value maskLenVal = dimValue(keyPaddingMask, 0, maskLen);
              Value batchVal = dimValue(query, 0, batch);
              Value sameLength = Torch::AtenEqIntOp::create(
                  rewriter, loc, maskLenVal, batchVal);
              Torch::RuntimeAssertOp::create(
                  rewriter, loc, sameLength,
                  rewriter.getStringAttr(
                      "MultiHeadAttention: a 1D key_padding_mask must hold "
                      "one key length per batch entry"));
            }
            Value lengths = Torch::convertTensorToDtype(
                rewriter, loc, keyPaddingMask, i64Type);
            Value positions = unsqueeze(arange(totalSeqVal(), totalSeqLen), 0);
            lengths = unsqueeze(lengths, 1);
            int64_t lengthsBatch = batch != kUnknown ? batch : maskLen;
            padded = Torch::AtenGeTensorOp::create(
                rewriter, loc, vtensor({lengthsBatch, totalSeqLen}, i1Type),
                positions, lengths);
          } else {
            auto maskType =
                dyn_cast<Torch::ValueTensorType>(keyPaddingMask.getType());
            assert(maskType && maskType.hasSizes() &&
                   "key_padding_mask is validated as a ranked value tensor");
            padded = Torch::AtenEqScalarOp::create(
                rewriter, loc, vtensor(maskType.getSizes(), i1Type),
                keyPaddingMask, cst0);
          }
          Value additive = toAdditive(padded, maskFilterValue);
          additive = unsqueeze(additive, 1);
          if (maskRank != 3)
            additive = unsqueeze(additive, 1);
          accumulate(additive);
        }

        if (attentionBias)
          accumulate(attentionBias);

        // ORT contract: scale == 0 (the default) selects 1/sqrt(H); any other
        // value, including a negative or denormal one, is used as-is.
        Value cstScale = scale == 0.0f ? cstNone : cstFloat(scale);
        // K / V are present_key / present_value: past ++ current when past is
        // given, otherwise the current projections. They feed SDPA even when
        // not returned. The op has no dropout attribute, so dropout_p is 0.
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc,
            vtensor({batch, numHeads, seqLen, vHeadSize}, elemType), qHeads,
            presentKey, presentValue, attnMask ? attnMask : cstNone,
            /*dropout_p=*/cstFloat(0.0), /*is_causal=*/cstFalse, cstScale,
            /*enable_gqa=*/cstFalse);
        Value output = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            vtensor({batch, seqLen, numHeads, vHeadSize}, elemType), attention,
            cst1, cst2);
        output = Torch::AtenFlattenUsingIntsOp::create(
            rewriter, loc, resultTypes[0], output, cst2, cst3);

        SmallVector<Value> results{output, presentKey, presentValue};
        results.resize(numResults);
        for (int64_t i = 1; i < numResults; ++i)
          if (results[i].getType() != resultTypes[i])
            results[i] = Torch::TensorStaticInfoCastOp::create(
                rewriter, loc, resultTypes[i], results[i]);
        rewriter.replaceOp(binder.op, results);
        return success();
      });
}
