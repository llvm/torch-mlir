//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h" // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h" // from @llvm-project

namespace mlir {
namespace tosa {

Value buildRescaleMultiplier(bool scale32, PatternRewriter &rewriter,
                             Operation *op, ArrayRef<int32_t> multipliers) {
  if (scale32) {
    return tosa::getConstTensor<int32_t>(
               rewriter, op, multipliers,
               {static_cast<int64_t>(multipliers.size())})
        .value();
  } else {
    SmallVector<int16_t> vec(multipliers.begin(), multipliers.end());
    return tosa::getConstTensor<int16_t>(rewriter, op, vec,
                                         {static_cast<int64_t>(vec.size())})
        .value();
  }
}

// Create a TOSA rescale op from input framework tensor, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  Value multiplier_val =
      buildRescaleMultiplier(scale32, rewriter, op, {multiplier});
  auto shift_val = tosa::getConstTensor<int8_t>(
                       rewriter, op, {static_cast<int8_t>(shift)}, {1})
                       .value();

  bool input_unsigned = input_val.getType().isUnsignedInteger();
  bool output_unsigned = output_type.isUnsignedInteger();

  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, op->getLoc(), output_type, input_val, multiplier_val, shift_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getBoolAttr(scale32), rewriter.getBoolAttr(double_round),
      rewriter.getBoolAttr(false), rewriter.getBoolAttr(input_unsigned),
      rewriter.getBoolAttr(output_unsigned));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
                          Value input_val, double input_scale,
                          int64_t input_zp) {
  // Output is always int32 type
  auto input_type = dyn_cast<mlir::ShapedType>(input_val.getType());
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
                      input_zp, 0, false, true);
}

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter &rewriter, Operation *op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type) {
  auto input_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(input_type.getElementType());
  auto output_qtype =
      dyn_cast<mlir::quant::UniformQuantizedType>(output_type.getElementType());

  double input_scale = input_qtype.getScale();

  int64_t output_zp = output_qtype.getZeroPoint();
  double output_scale = output_qtype.getScale();

  bool scale32 = isScale32(output_qtype);
  int32_t scale_width = scale32 ? 32 : 16;

  bool input_unsigned = input_qtype.isUnsignedInteger();
  bool output_unsigned = output_qtype.isUnsignedInteger();

  if (auto weight_per_tensor_qtype =
          dyn_cast<mlir::quant::UniformQuantizedType>(
              weight_type.getElementType())) {
    // Per-tensor quantization
    double weight_scale = weight_per_tensor_qtype.getScale();

    int32_t multiplier;
    int32_t shift;

    double op_tensor_scale = (input_scale * weight_scale) / output_scale;

    computeMultiplierAndShift(op_tensor_scale, multiplier, shift, scale_width);

    Value multiplier_val =
        buildRescaleMultiplier(scale32, rewriter, op, {multiplier});
    auto shift_val = tosa::getConstTensor<int8_t>(
                         rewriter, op, {static_cast<int8_t>(shift)}, {1})
                         .value();

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val, multiplier_val,
        shift_val, rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(output_zp), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(true), rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(input_unsigned),
        rewriter.getBoolAttr(output_unsigned));

    return rescale_op.getResult();

  } else if (auto weight_per_channel_qtype =
                 dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                     weight_type.getElementType())) {
    // Per-channel quantization
    SmallVector<int32_t> multiplier_arr;
    SmallVector<int8_t> shift_arr;

    SmallVector<double> weight_scale_arr(
        weight_per_channel_qtype.getScales().begin(),
        weight_per_channel_qtype.getScales().end());

    int64_t output_zp = output_qtype.getZeroPoint();
    double output_scale = output_qtype.getScale();

    for (double weight_scale : weight_scale_arr) {
      int32_t multiplier;
      int32_t shift;

      double op_channel_scale = (input_scale * weight_scale) / output_scale;

      computeMultiplierAndShift(op_channel_scale, multiplier, shift,
                                scale_width);

      multiplier_arr.push_back(multiplier);
      shift_arr.push_back(static_cast<int8_t>(shift));
    }

    Value multiplier_val =
        buildRescaleMultiplier(scale32, rewriter, op, multiplier_arr);
    auto shift_val =
        tosa::getConstTensor<int8_t>(rewriter, op, shift_arr,
                                     {static_cast<int64_t>(shift_arr.size())})
            .value();

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val, multiplier_val,
        shift_val, rewriter.getI32IntegerAttr(0),
        rewriter.getI32IntegerAttr(output_zp), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(true), rewriter.getBoolAttr(true),
        rewriter.getBoolAttr(input_unsigned),
        rewriter.getBoolAttr(output_unsigned));

    return rescale_op.getResult();

  } else {
    op->emitOpError("buildConvRescaleOp: unknown weight quantized type");
    return nullptr;
  }
}

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type) {
  return (output_element_type.getStorageTypeIntegralWidth() == 8);
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create an int8_t const tosa.mul shift tensor from an int
Value getTosaMulShiftConstTensor(PatternRewriter &rewriter, Operation *op,
                                 int32_t shift) {
  auto shiftType = RankedTensorType::get({1}, rewriter.getIntegerType(8));
  auto shiftAttr = DenseElementsAttr::get(
      shiftType, rewriter.getIntegerAttr(rewriter.getIntegerType(8), shift));

  auto constShift =
      rewriter.create<tosa::ConstOp>(op->getLoc(), shiftType, shiftAttr);

  return constShift.getResult();
}

// Create a zero constant tensor of the desired type and shape.
std::optional<Value> getZerosLikeTensor(PatternRewriter &rewriter,
                                        Operation *op, Type type) {
  RankedTensorType resultType = dyn_cast<RankedTensorType>(type);

  if (!resultType) {
    (void)rewriter.notifyMatchFailure(op, "not ranked tensor type");
    return std::nullopt;
  }

  auto resultShape = resultType.getShape();
  ShapedType zeroType =
      RankedTensorType::get(resultShape, resultType.getElementType());
  Attribute zeroAttr = rewriter.getZeroAttr(zeroType);

  return CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(), zeroType,
                                         cast<ElementsAttr>(zeroAttr))
      .getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape,
                                    std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto width = sizeof(T) * 8;
  if constexpr (std::is_same_v<T, bool>)
    width = 1;

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(width));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
    return tosa::tosaCastTensorToType(rewriter, const_op,
                                      RankedTensorType::get(shape, *dtype))
        .value();
  }
  return const_op.getResult();
}

// Template specialization for APInt
template <>
std::optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<APInt> vec,
                                           ArrayRef<int64_t> shape,
                                           std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
    return tosa::tosaCastTensorToType(rewriter, const_op,
                                      RankedTensorType::get(shape, *dtype))
        .value();
  }
  return const_op.getResult();
}

// Template specialization for float
template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape,
                                           std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
    return tosa::tosaCastTensorToType(rewriter, const_op,
                                      RankedTensorType::get(shape, *dtype))
        .value();
  }
  return const_op.getResult();
}

// Valid TOSA casting pairs according to TOSA spec:
// https://www.mlplatform.org/tosa/tosa_spec.html#_cast
// Note: currently TOSA doesn't support casting to and from I64 and F64
[[maybe_unused]] static LogicalResult checkValidityOfCast(Type src, Type dest) {
  // clang-format off
  if ((src == dest) ||
      // int32 -> *
      (src.isInteger(32) && dest.isInteger(16)) ||
      (src.isInteger(32) && dest.isInteger(8)) ||
      (src.isInteger(32) && dest.isInteger(1)) ||
      (src.isInteger(32) && dest.isF32()) ||
      (src.isInteger(32) && dest.isF16()) ||
      (src.isInteger(32) && dest.isBF16()) ||
      // int16 -> *
      (src.isInteger(16) && dest.isInteger(32)) ||
      (src.isInteger(16) && dest.isInteger(8)) ||
      (src.isInteger(16) && dest.isInteger(1)) ||
      (src.isInteger(16) && dest.isBF16()) ||
      (src.isInteger(16) && dest.isF32()) ||
      (src.isInteger(16) && dest.isF16()) ||
      // int8 -> *
      (src.isInteger(8) && dest.isInteger(32)) ||
      (src.isInteger(8) && dest.isInteger(16)) ||
      (src.isInteger(8) && dest.isInteger(1)) ||
      (src.isInteger(8) && dest.isBF16()) ||
      (src.isInteger(8) && dest.isF32()) ||
      (src.isInteger(8) && dest.isF16()) ||
      // int1 -> *
      (src.isInteger(1) && dest.isInteger(32)) ||
      (src.isInteger(1) && dest.isInteger(16)) ||
      (src.isInteger(1) && dest.isInteger(8)) ||
      // f32 -> *
      (src.isF32() && dest.isInteger(32)) ||
      (src.isF32() && dest.isInteger(16)) ||
      (src.isF32() && dest.isInteger(8)) ||
      (src.isF32() && dest.isBF16()) ||
      (src.isF32() && dest.isF16()) ||
      (src.isF32() && isa<Float8E4M3Type>(dest)) ||
      (src.isF32() && isa<Float8E5M2Type>(dest)) ||
      // f16 -> *
      (src.isF16() && dest.isInteger(32)) ||
      (src.isF16() && dest.isInteger(16)) ||
      (src.isF16() && dest.isInteger(8)) ||
      (src.isF16() && dest.isBF16()) ||
      (src.isF16() && dest.isF32()) ||
      (src.isF16() && isa<Float8E4M3Type>(dest)) ||
      (src.isF16() && isa<Float8E5M2Type>(dest)) ||
      // bf16 -> *
      (src.isBF16() && dest.isInteger(32)) ||
      (src.isBF16() && dest.isInteger(16)) ||
      (src.isBF16() && dest.isInteger(8)) ||
      (src.isBF16() && dest.isF32()) ||
      (src.isBF16() && isa<Float8E4M3Type>(dest)) ||
      (src.isBF16() && isa<Float8E5M2Type>(dest)) ||
      // fp8e4m3 -> *
      (isa<Float8E4M3Type>(src) && dest.isBF16()) ||
      (isa<Float8E4M3Type>(src) && dest.isF32()) ||
      (isa<Float8E4M3Type>(src) && dest.isF16()) ||
      // fp8e5m2 -> *
      (isa<Float8E5M2Type>(src) && dest.isBF16()) ||
      (isa<Float8E5M2Type>(src) && dest.isF32()) ||
      (isa<Float8E5M2Type>(src) && dest.isF16())) {
    return success();
  }
  // clang-format on
  return failure();
}

// Default function to create tosa.cast op. This should be called instead of
// directly calling rewriter.create<tosa::CastOp>.
std::optional<Value> tosaCastTensorToType(PatternRewriter &rewriter, Value src,
                                          TensorType destType) {
  Operation *op = src.getDefiningOp();
  TensorType srcType = dyn_cast<TensorType>(src.getType());
  Type srcElemTy = srcType.getElementType();
  Type destElemTy = dyn_cast<TensorType>(destType).getElementType();

  // Temporarily disable checkValidityOfCast as it's currently strictly
  // following TOSA spec and might cause many e2e tests to fail. This is because
  // even though there are some casting pairs that are not congruent to TOSA
  // spec, they are still permissible. TOSA validation should flag these illegal
  // constructs in a per-profile manner. This strict validity check will be
  // enabled later in a potential `--strict` mode which checks for strict
  // casting only when needed (the default value of `--strict` mode will be
  // off).
  // if (failed(checkValidityOfCast(srcElemTy, destElemTy)))
  //   return std::nullopt;

  if (srcElemTy == destElemTy)
    return src;

  if (llvm::isa<FloatType>(srcElemTy) && destElemTy.isInteger() &&
      !destElemTy.isInteger(1)) {
    // For float->int conversion, tosa.cast performs round-to-nearest.
    // PyTorch performs round-to-zero instead.
    // Generate round-to-zero conversion prior to tosa.cast to match with
    // expected torch behavior.
    auto floor = rewriter.create<tosa::FloorOp>(op->getLoc(), srcType, src);
    auto ceil = rewriter.create<tosa::CeilOp>(op->getLoc(), srcType, src);

    auto zeroValue =
        tosa::getConstTensor<float>(rewriter, op, 0, {}, srcElemTy).value();

    if (mlir::tosa::EqualizeRanks(rewriter, op->getLoc(), src, zeroValue)
            .failed())
      return std::nullopt;

    auto boolType = srcType.clone(rewriter.getIntegerType(1));
    auto isNegative = tosa::CreateOpAndInfer<tosa::GreaterOp>(
        rewriter, op->getLoc(), boolType, zeroValue, src);
    src = tosa::CreateOpAndInfer<tosa::SelectOp>(
        rewriter, op->getLoc(), srcType, isNegative, ceil, floor);
  }

  TensorType castedSrcType = srcType.clone(destElemTy);
  return rewriter.create<tosa::CastOp>(op->getLoc(), castedSrcType, src);
}

// Template instantiation
template std::optional<Value>
getConstTensor<bool>(PatternRewriter &, Operation *, ArrayRef<bool> vec,
                     ArrayRef<int64_t> shape, std::optional<Type> dtype);

template std::optional<Value>
getConstTensor<int8_t>(PatternRewriter &, Operation *, ArrayRef<int8_t> vec,
                       ArrayRef<int64_t> shape, std::optional<Type> dtype);

template std::optional<Value>
getConstTensor<int16_t>(PatternRewriter &, Operation *, ArrayRef<int16_t> vec,
                        ArrayRef<int64_t> shape, std::optional<Type> dtype);

template std::optional<Value>
getConstTensor<int32_t>(PatternRewriter &, Operation *, ArrayRef<int32_t> vec,
                        ArrayRef<int64_t> shape, std::optional<Type> dtype);

template std::optional<Value>
getConstTensor<int64_t>(PatternRewriter &, Operation *, ArrayRef<int64_t> vec,
                        ArrayRef<int64_t> shape, std::optional<Type> dtype);

LogicalResult getAvgPool2dAccType(PatternRewriter &rewriter, Value input,
                                  TypeAttr &accType) {
  auto inputTy = llvm::dyn_cast<ShapedType>(input.getType());
  if (!inputTy)
    return failure();
  auto inputETy = inputTy.getElementType();

  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy))
    inputETy = quantType.getStorageType();

  // Tosa supports FP16 and FP32 accumulator type for FP16 input. When the time
  // FP16 is supported, the accumulator type can be selected based on trade-off
  // between performance and accuracy. Set to FP32 by default.
  accType = isa<FloatType>(inputETy)
                ? mlir::TypeAttr::get(rewriter.getF32Type())
                : mlir::TypeAttr::get(rewriter.getIntegerType(32));

  return success();
}

// Get accumulator type for TOSA convolution ops
LogicalResult getConvOpsAccType(PatternRewriter &rewriter,
                                RankedTensorType inputTy,
                                RankedTensorType weightTy,
                                RankedTensorType outputTy, TypeAttr &accType) {
  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto outputElemTy = outputTy.getElementType();

  auto quantTy = dyn_cast<quant::QuantizedType>(inputElemTy);
  if (quantTy)
    inputElemTy = quantTy.getStorageType();

  // Get TOSA conv ops acc type based on input, weight, and output types
  // according to the spec:
  // https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d
  // https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
  // https://www.mlplatform.org/tosa/tosa_spec.html#_conv3d
  //
  // For undefined dtypes in TOSA like I64 and F64, acc_type will be set to the
  // output type but does not offer any guarantee on the numerical precision
  // since such cases will fail TOSA validation.
  if ((inputElemTy.isF32() && weightElemTy.isF32() && outputElemTy.isF32()) ||
      (inputElemTy.isF16() && weightElemTy.isF16() && outputElemTy.isF16()) ||
      (inputElemTy.isBF16() && weightElemTy.isBF16() &&
       outputElemTy.isBF16())) {
    accType = mlir::TypeAttr::get(rewriter.getF32Type());
  } else if (inputElemTy.isInteger(8) &&
             (weightElemTy.isInteger(8) || weightElemTy.isInteger(4)) &&
             outputElemTy.isInteger(32)) {
    accType = mlir::TypeAttr::get(rewriter.getIntegerType(32));
  } else if (inputElemTy.isInteger(16) && weightElemTy.isInteger(8) &&
             outputElemTy.isInteger(48)) {
    accType = mlir::TypeAttr::get(rewriter.getIntegerType(48));
  } else if ((isa<Float8E4M3Type>(inputElemTy) &&
              isa<Float8E4M3Type>(weightElemTy) && outputElemTy.isF16()) ||
             (isa<Float8E5M2Type>(inputElemTy) &&
              isa<Float8E5M2Type>(weightElemTy) && outputElemTy.isF16())) {
    accType = mlir::TypeAttr::get(rewriter.getF16Type());
  } else {
    accType = mlir::TypeAttr::get(outputElemTy);
  }

  return success();
}

} // namespace tosa
} // namespace mlir
