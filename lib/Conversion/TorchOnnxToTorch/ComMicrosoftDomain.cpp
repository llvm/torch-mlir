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

namespace {

/// The weight layout that the MatMulNBits attributes determine, as the
/// com.microsoft.MatMulNBits spec defines it.
struct NBitsLayout {
  /// Input features: A's last dimension.
  int64_t k;
  /// Output features: the rows of the stored weight.
  int64_t n;
  /// Width of one quantized weight value, 2, 4 or 8.
  int64_t bits;
  /// Number of consecutive values along K that share a scale and zero point.
  int64_t blockSize;

  /// ceil(K / block_size): the blocks per row of B.
  int64_t getKBlocks() const { return llvm::divideCeilSigned(k, blockSize); }
  /// K including the padding of the last block.
  int64_t getPaddedK() const { return getKBlocks() * blockSize; }
  /// 8 / bits values share a byte of B and of bit-packed zero points.
  int64_t getValuesPerByte() const { return 8 / bits; }
  /// block_size * bits / 8: the bytes per block of B.
  int64_t getBlobSize() const { return blockSize / getValuesPerByte(); }
  /// ceil(k_blocks * bits / 8): the bytes per row of bit-packed zero points.
  int64_t getZeroPointBytesPerRow() const {
    return llvm::divideCeilSigned(getKBlocks(), getValuesPerByte());
  }
};

/// What matchMatMulNBits extracts from a supported MatMulNBits op.
struct MatMulNBitsInfo {
  Value a;
  Value weight;
  Value scales;
  /// Null when the default zero point 2^(bits - 1) applies.
  Value zeroPoints;
  /// Null when there is no bias.
  Value bias;
  Torch::ValueTensorType resultType;
  /// Whether A is quantized dynamically to int8 before the matmul.
  bool quantizeA;
  NBitsLayout layout;
};

} // namespace

/// Returns whether `value` has the shape `sizes`. The MatMulNBits attributes
/// fix the shapes of B, scales, zero points and bias, so a dynamic dimension
/// does not match.
static bool hasShape(Value value, ArrayRef<int64_t> sizes) {
  auto type = cast<Torch::ValueTensorType>(value.getType());
  return type.hasSizes() && type.getSizes() == sizes;
}

/// Returns whether `value` has the per-block shape [rows, columns] or its
/// flattened form [rows * columns]. The spec gives only [rows, columns]; the
/// flattened form is accepted because onnxruntime's HQQ quantizer emits it.
static bool hasBlockShape(Value value, int64_t rows, int64_t columns) {
  return hasShape(value, {rows, columns}) || hasShape(value, {rows * columns});
}

/// Binds and verifies the operands and attributes of a MatMulNBits op. Matching
/// does not create IR.
static FailureOr<MatMulNBitsInfo>
matchMatMulNBits(OpBinder binder, ConversionPatternRewriter &rewriter) {
  Value a, weight, scales, zeroPoints, groupIndex, bias;
  Torch::ValueTensorType resultType;
  int64_t k, n, bits, blockSize, accuracyLevel, weightPrepacked;
  if (binder.getNumOperands() < 3 || binder.getNumOperands() > 6 ||
      binder.tensorOperandAtIndex(a, 0) ||
      binder.tensorOperandAtIndex(weight, 1) ||
      binder.tensorOperandAtIndex(scales, 2) ||
      binder.tensorResultType(resultType) ||
      binder.optionalS64IntegerAttr(k, "K") ||
      binder.optionalS64IntegerAttr(n, "N") ||
      binder.s64IntegerAttr(bits, "bits", 4) ||
      binder.optionalS64IntegerAttr(blockSize, "block_size") ||
      binder.s64IntegerAttr(accuracyLevel, "accuracy_level", 0) ||
      binder.s64IntegerAttr(weightPrepacked, "weight_prepacked", 0)) {
    return rewriter.notifyMatchFailure(
        binder.op, "expected A, B, scales, up to three optional inputs, "
                   "and the K, N and block_size attributes");
  }
  Value *optionalInputs[] = {&zeroPoints, &groupIndex, &bias};
  // Empty ONNX input names become None without shifting later inputs, so bias
  // may be present even when zero_points or g_idx is omitted.
  for (int64_t i = 3; i < binder.getNumOperands(); ++i) {
    if (isa<Torch::NoneType>(binder.op->getOperand(i).getType()))
      continue;
    if (binder.tensorOperandAtIndex(*optionalInputs[i - 3], i))
      return rewriter.notifyMatchFailure(
          binder.op, "expected tensor or None optional inputs");
  }

  if (!llvm::is_contained({2, 4, 8}, bits))
    return rewriter.notifyMatchFailure(binder.op, "bits must be 2, 4 or 8");
  if (blockSize < 16 || !llvm::isPowerOf2_64(blockSize))
    return rewriter.notifyMatchFailure(
        binder.op, "block_size must be a power of two of at least 16");
  if (k <= 0 || n <= 0)
    return rewriter.notifyMatchFailure(binder.op, "K and N must be positive");
  if (accuracyLevel < 0 || accuracyLevel > 4)
    return rewriter.notifyMatchFailure(binder.op,
                                       "accuracy_level must be in [0, 4]");
  if (weightPrepacked != 0)
    return rewriter.notifyMatchFailure(
        binder.op, "unsupported: weight_prepacked selects an "
                   "execution-provider specific layout of B");
  // g_idx is deprecated, and the spec defines no semantics for it.
  if (groupIndex)
    return rewriter.notifyMatchFailure(binder.op, "unsupported: g_idx input");

  NBitsLayout layout{k, n, bits, blockSize};
  auto aType = cast<Torch::ValueTensorType>(a.getType());
  if (!aType.hasDtype() || aType.getSizes().empty() ||
      !isa<Float32Type, Float16Type, BFloat16Type>(aType.getDtype()))
    return rewriter.notifyMatchFailure(
        binder.op, "expected float32, float16 or bfloat16 A");
  Type floatType = aType.getDtype();
  if (aType.getSizes().back() != k)
    return rewriter.notifyMatchFailure(
        binder.op, "expected A's last dimension to be the static K");
  if (!resultType.hasDtype() || resultType.getDtype() != floatType)
    return rewriter.notifyMatchFailure(binder.op,
                                       "expected the result dtype to match A");

  auto weightType = cast<Torch::ValueTensorType>(weight.getType());
  if (!weightType.hasDtype() || !weightType.getDtype().isUnsignedInteger(8) ||
      !hasShape(weight, {n, layout.getKBlocks(), layout.getBlobSize()}))
    return rewriter.notifyMatchFailure(
        binder.op, "expected uint8 B of shape [N, ceil(K / block_size), "
                   "block_size * bits / 8]");
  auto scalesType = cast<Torch::ValueTensorType>(scales.getType());
  if (!scalesType.hasDtype() || scalesType.getDtype() != floatType ||
      !hasBlockShape(scales, n, layout.getKBlocks()))
    return rewriter.notifyMatchFailure(binder.op,
                                       "expected scales of A's dtype and shape "
                                       "[N, ceil(K / block_size)]");
  bool unpackedZeroPoints = false;
  if (zeroPoints) {
    auto zeroPointsType = cast<Torch::ValueTensorType>(zeroPoints.getType());
    bool packed =
        zeroPointsType.hasDtype() &&
        zeroPointsType.getDtype().isUnsignedInteger(8) &&
        hasBlockShape(zeroPoints, n, layout.getZeroPointBytesPerRow());
    unpackedZeroPoints = zeroPointsType.hasDtype() &&
                         zeroPointsType.getDtype() == floatType &&
                         hasBlockShape(zeroPoints, n, layout.getKBlocks());
    if (!packed && !unpackedZeroPoints)
      return rewriter.notifyMatchFailure(
          binder.op, "expected bit-packed uint8 zero points of shape [N, "
                     "ceil(ceil(K / block_size) * bits / 8)] or zero points of "
                     "A's dtype and shape [N, ceil(K / block_size)]");
  }
  if (bias) {
    auto biasType = cast<Torch::ValueTensorType>(bias.getType());
    if (!biasType.hasDtype() || biasType.getDtype() != floatType ||
        !hasShape(bias, {n}))
      return rewriter.notifyMatchFailure(
          binder.op, "expected bias of A's dtype and shape [N]");
  }
  // onnxruntime's CPU kernel quantizes A for accuracy_level 4 only
  // (GetComputeType in matmul_nbits.cc), and its PrePack keeps A in floating
  // point for floating-point zero points, which have no int8 path.
  bool quantizeA = accuracyLevel == 4 && !unpackedZeroPoints;
  return MatMulNBitsInfo{a,    weight,     scales,    zeroPoints,
                         bias, resultType, quantizeA, layout};
}

/// Reshapes a per-block parameter accepted by hasBlockShape to [rows, columns].
static Value reshapeToBlockShape(ConversionPatternRewriter &rewriter,
                                 Location loc, Value value, int64_t rows,
                                 int64_t columns) {
  assert(hasBlockShape(value, rows, columns) &&
         "expected a [rows, columns] or [rows * columns] parameter");
  if (hasShape(value, {rows, columns}))
    return value;
  Type dtype = cast<Torch::ValueTensorType>(value.getType()).getDtype();
  return Torch::AtenReshapeOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(ArrayRef<int64_t>{rows, columns},
                                               dtype),
      value, Torch::toIntListConstruct(rewriter, loc, {rows, columns}));
}

/// Unpacks `packed`, whose ui8 elements each hold 8 / bits values of `bits`
/// bits along the last dimension, lowest bits first, into one ui8 element per
/// value. The last dimension grows by 8 / bits:
///   (packed[..., None] >> [0, bits, 2 * bits, ...]) & (2^bits - 1)
static Value unpackLastDimension(ConversionPatternRewriter &rewriter,
                                 Location loc, Value packed, int64_t bits) {
  assert(llvm::is_contained({2, 4, 8}, bits) && "expected 2, 4 or 8 bits");
  // 8-bit values already take one byte each.
  if (bits == 8)
    return packed;
  auto uint8 = rewriter.getIntegerType(8, /*isSigned=*/false);
  int64_t valuesPerByte = 8 / bits;
  SmallVector<int64_t> sizes(
      cast<Torch::ValueTensorType>(packed.getType()).getSizes());

  SmallVector<int64_t> expandedSizes(sizes);
  expandedSizes.push_back(1);
  Value minusOne = Torch::ConstantIntOp::create(rewriter, loc, -1);
  Value expanded = Torch::AtenUnsqueezeOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(expandedSizes, uint8), packed,
      minusOne);

  SmallVector<APInt> shiftValues;
  for (int64_t i = 0; i < valuesPerByte; ++i)
    shiftValues.push_back(APInt(8, i * bits));
  Value shifts = Torch::ValueTensorLiteralOp::create(
      rewriter, loc,
      DenseElementsAttr::get(RankedTensorType::get({valuesPerByte}, uint8),
                             shiftValues));
  expandedSizes.back() = valuesPerByte;
  auto valuesType =
      rewriter.getType<Torch::ValueTensorType>(expandedSizes, uint8);
  Value shifted = Torch::AtenBitwiseRightShiftTensorOp::create(
      rewriter, loc, valuesType, expanded, shifts);
  Value mask =
      Torch::ConstantIntOp::create(rewriter, loc, llvm::maxUIntN(bits));
  Value values = Torch::AtenBitwiseAndScalarOp::create(
      rewriter, loc, valuesType, shifted, mask);

  sizes.back() *= valuesPerByte;
  Value minusTwo = Torch::ConstantIntOp::create(rewriter, loc, -2);
  return Torch::AtenFlattenUsingIntsOp::create(
      rewriter, loc, rewriter.getType<Torch::ValueTensorType>(sizes, uint8),
      values, minusTwo, minusOne);
}

/// Returns the dequantized [N, paddedK] weight of MatMulNBits:
///   W[i, j] = (q[i, j] - zero_points[i, j / block_size]) *
///             scales[i, j / block_size]
/// as one dequantize_per_channel_group. Columns from K on dequantize the
/// padding of the last block; they are finite and meet zero padding in A.
/// `weight`, `scales` and `zeroPoints` satisfy the shapes checked by
/// matchMatMulNBits; a null `zeroPoints` selects the default zero point
/// 2^(bits - 1).
static Value createNBitsWeight(ConversionPatternRewriter &rewriter,
                               Location loc, const NBitsLayout &layout,
                               Value weight, Value scales, Value zeroPoints,
                               Type floatType) {
  auto uint8 = rewriter.getIntegerType(8, /*isSigned=*/false);
  int64_t n = layout.n;

  Value quantized = unpackLastDimension(rewriter, loc, weight, layout.bits);
  Value one = Torch::ConstantIntOp::create(rewriter, loc, 1);
  Value two = Torch::ConstantIntOp::create(rewriter, loc, 2);
  quantized = Torch::AtenFlattenUsingIntsOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(
          ArrayRef<int64_t>{n, layout.getPaddedK()}, uint8),
      quantized, one, two);

  scales = reshapeToBlockShape(rewriter, loc, scales, n, layout.getKBlocks());

  if (!zeroPoints) {
    int64_t defaultZeroPoint = int64_t(1) << (layout.bits - 1);
    auto zeroPointsType =
        RankedTensorType::get({n, layout.getKBlocks()}, uint8);
    zeroPoints = Torch::ValueTensorLiteralOp::create(
        rewriter, loc,
        DenseElementsAttr::get(zeroPointsType, APInt(8, defaultZeroPoint)));
  } else if (cast<Torch::ValueTensorType>(zeroPoints.getType())
                 .getDtype()
                 .isUnsignedInteger(8)) {
    zeroPoints = reshapeToBlockShape(rewriter, loc, zeroPoints, n,
                                     layout.getZeroPointBytesPerRow());
    zeroPoints = unpackLastDimension(rewriter, loc, zeroPoints, layout.bits);
    // Rows of packed zero points are padded to whole bytes.
    if (layout.getZeroPointBytesPerRow() * layout.getValuesPerByte() !=
        layout.getKBlocks()) {
      Value zero = Torch::ConstantIntOp::create(rewriter, loc, 0);
      Value end =
          Torch::ConstantIntOp::create(rewriter, loc, layout.getKBlocks());
      zeroPoints = Torch::AtenSliceTensorOp::create(
          rewriter, loc,
          rewriter.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>{n, layout.getKBlocks()}, uint8),
          zeroPoints, one, zero, end, one);
    }
  } else {
    assert(isa<mlir::FloatType>(
               cast<Torch::ValueTensorType>(zeroPoints.getType()).getDtype()) &&
           "expected unpacked floating-point zero points");
    zeroPoints =
        reshapeToBlockShape(rewriter, loc, zeroPoints, n, layout.getKBlocks());
  }

  auto [quantMin, quantMax] = createQuantizationBounds(
      rewriter, loc, rewriter.getIntegerType(layout.bits, /*isSigned=*/false));
  Value storageDtype = Torch::getDtypeIntValueForType(rewriter, loc, uint8);
  Value groupSize =
      Torch::ConstantIntOp::create(rewriter, loc, layout.blockSize);
  Value outDtype = Torch::getDtypeIntValueForType(rewriter, loc, floatType);
  return Torch::QuantizedDecomposedDequantizePerChannelGroupOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(
          ArrayRef<int64_t>{n, layout.getPaddedK()}, floatType),
      quantized, scales, zeroPoints, quantMin, quantMax, storageDtype,
      groupSize, outDtype);
}

/// Returns A', the activation that linear multiplies with the dequantized
/// weight: `a` zero padded from K to paddedK, so the padding of B's last block
/// meets zeros. With `quantizeA`, A' is also quantized to int8 and dequantized
/// again, as accuracy_level 4 permits: per block of block_size along K, with
/// the symmetric scale max(|a|) / 127 that choose_qparams_per_token picks. The
/// zero padding leaves the max of the last block unchanged.
static Value createNBitsActivation(ConversionPatternRewriter &rewriter,
                                   Location loc, const NBitsLayout &layout,
                                   Value a, bool quantizeA) {
  auto aType = cast<Torch::ValueTensorType>(a.getType());
  assert(aType.getSizes().back() == layout.k &&
         "expected A's last dimension to be K");
  Type floatType = aType.getDtype();
  SmallVector<int64_t> paddedSizes(aType.getSizes());
  paddedSizes.back() = layout.getPaddedK();
  auto paddedType =
      rewriter.getType<Torch::ValueTensorType>(paddedSizes, floatType);
  if (layout.getPaddedK() != layout.k) {
    Value padding = Torch::toIntListConstruct(
        rewriter, loc, {0, layout.getPaddedK() - layout.k});
    Value zero = Torch::ConstantFloatOp::create(rewriter, loc,
                                                rewriter.getF64FloatAttr(0.0));
    a = Torch::AtenConstantPadNdOp::create(rewriter, loc, paddedType, a,
                                           padding, zero);
  }
  if (!quantizeA)
    return a;

  SmallVector<int64_t> blockedSizes(paddedSizes);
  blockedSizes.back() = layout.getKBlocks();
  blockedSizes.push_back(layout.blockSize);
  Value minusOne = Torch::ConstantIntOp::create(rewriter, loc, -1);
  Value blocked = Torch::AtenUnflattenIntOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(blockedSizes, floatType), a,
      minusOne,
      Torch::toIntListConstruct(rewriter, loc,
                                {layout.getKBlocks(), layout.blockSize}));

  auto int8 = rewriter.getIntegerType(8, /*isSigned=*/true);
  SmallVector<int64_t> qparamSizes(blockedSizes);
  qparamSizes.back() = 1;
  // onnxruntime's CPU kernel computes the activation scales in f32, also for
  // f16 A.
  auto scalesType = rewriter.getType<Torch::ValueTensorType>(
      qparamSizes, rewriter.getF32Type());
  auto zeroPointsType = rewriter.getType<Torch::ValueTensorType>(
      qparamSizes, rewriter.getIntegerType(64, /*isSigned=*/true));
  Value storageDtype = Torch::getDtypeIntValueForType(rewriter, loc, int8);
  auto qparams = Torch::QuantizedDecomposedChooseQparamsPerTokenOp::create(
      rewriter, loc, scalesType, zeroPointsType, blocked, storageDtype);
  auto [quantMin, quantMax] = createQuantizationBounds(rewriter, loc, int8);
  Value quantized = Torch::QuantizedDecomposedQuantizePerTokenOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(blockedSizes, int8), blocked,
      qparams.getScale(), qparams.getZeroPoint(), quantMin, quantMax,
      storageDtype);
  Value outDtype = Torch::getDtypeIntValueForType(rewriter, loc, floatType);
  Value dequantized = Torch::QuantizedDecomposedDequantizePerTokenOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(blockedSizes, floatType),
      quantized, qparams.getScale(), qparams.getZeroPoint(), quantMin, quantMax,
      storageDtype, outDtype);

  Value minusTwo = Torch::ConstantIntOp::create(rewriter, loc, -2);
  return Torch::AtenFlattenUsingIntsOp::create(rewriter, loc, paddedType,
                                               dequantized, minusTwo, minusOne);
}

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
  patterns.onOp("MatMulNBits", 1,
                [](OpBinder binder, ConversionPatternRewriter &rewriter) {
                  FailureOr<MatMulNBitsInfo> info =
                      matchMatMulNBits(binder, rewriter);
                  if (failed(info))
                    return failure();
                  Location loc = binder.getLoc();
                  const NBitsLayout &layout = info->layout;
                  Type floatType = info->resultType.getDtype();

                  // Y = linear(A', W, bias) computes A' @ W^T + bias for the
                  // stored weight. Both operands span the padded K.
                  Value weight = createNBitsWeight(rewriter, loc, layout,
                                                   info->weight, info->scales,
                                                   info->zeroPoints, floatType);
                  Value activation = createNBitsActivation(
                      rewriter, loc, layout, info->a, info->quantizeA);
                  Value bias = info->bias;
                  if (!bias)
                    bias = Torch::ConstantNoneOp::create(rewriter, loc);
                  rewriter.replaceOpWithNewOp<Torch::AtenLinearOp>(
                      binder.op, info->resultType, activation, weight, bias);
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
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(skip, 1) ||
            binder.tensorOperandAtIndex(gamma, 2) ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5f) ||
            binder.s64IntegerAttr(stashType, "stash_type", 1))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to bind inputs/attrs");

        // Gather result types directly from the op rather than via
        // binder.tensorResultTypes: unused optional outputs are imported as
        // !torch.none typed results, which tensorResultTypes rejects. We must
        // accept and preserve those none-typed slots so they can be emitted as
        // torch.constant.none below.
        SmallVector<Type> resultTypes(binder.op->getResultTypes());

        // ORT's SkipSimplifiedLayerNormalization defines up to 4 outputs, in
        // order: output, mean, inv_std_var, input_skip_bias_sum. Unused
        // (optional) outputs are imported as !torch.none typed results and are
        // handled below by emitting a torch.constant.none in that slot.
        if (resultTypes.size() > 4)
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

        // Keep a stash-dtype copy of the residual sum. `s` itself is cast back
        // to origDtype below so that it can be returned as
        // input_skip_bias_sum; sStash is used for all normalization math.
        Value sStash = s;

        // inv_std_var (result 2) is only computed when the model requests it
        // with a concrete tensor type; unused outputs are !torch.none.
        bool wantInvStd =
            resultTypes.size() > 2 && !isa<Torch::NoneType>(resultTypes[2]);

        Value output;
        Value invStdStash;
        if (wantInvStd) {
          // inv_std_var = 1 / sqrt(mean(s^2, dim=-1, keepdim) + eps). This is
          // the same quantity aten.rms_norm computes internally, so compute it
          // once here and build the normalized output from it explicitly
          // (output = s * inv_std_var * gamma) instead of also emitting
          // aten.rms_norm, which would recompute it.
          auto sStashType = cast<Torch::ValueTensorType>(sStash.getType());
          Value cstTwo = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(2));
          Value sSquared = Torch::AtenPowTensorScalarOp::create(
              rewriter, loc, sStashType, sStash, cstTwo);
          Value cstLastDim = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(-1));
          Value dimList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              SmallVector<Value>{cstLastDim});
          Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
          Value cstNoneForMean = Torch::ConstantNoneOp::create(rewriter, loc);
          // Reduced type: keepdim=true keeps last dim as size 1.
          SmallVector<int64_t> reducedSizes(sStashType.getSizes());
          reducedSizes.back() = 1;
          auto reducedType =
              Torch::ValueTensorType::get(binder.op->getContext(), reducedSizes,
                                          sStashType.getOptionalDtype());
          Value meanSq = Torch::AtenMeanDimOp::create(
              rewriter, loc, reducedType, sSquared, dimList,
              /*keepdim=*/cstTrue, /*dtype=*/cstNoneForMean);
          Value meanSqEps = Torch::AtenAddScalarOp::create(
              rewriter, loc, reducedType, meanSq, cstEpsilon, cstOne);
          invStdStash =
              Torch::AtenRsqrtOp::create(rewriter, loc, reducedType, meanSqEps);

          // output = (s * inv_std_var) * gamma, in the stash dtype.
          Value normed = Torch::AtenMulTensorOp::create(
              rewriter, loc, sStashType, sStash, invStdStash);
          Value gammaStash = gamma;
          auto gammaType = cast<Torch::ValueTensorType>(gamma.getType());
          if (gammaType.getOptionalDtype() != *stashDtype)
            gammaStash =
                Torch::convertTensorToDtype(rewriter, loc, gamma, *stashDtype);
          output = Torch::AtenMulTensorOp::create(rewriter, loc, sStashType,
                                                  normed, gammaStash);
        } else {
          // Build normalized_shape for last dimension: [inputShape[-1]]
          auto castInput = cast<Torch::ValueTensorType>(input.getType());
          ArrayRef<int64_t> inputShape = castInput.getSizes();
          Value cstLastDimSize = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(inputShape[inputRank - 1]));
          Value normalizedShape = Torch::PrimListConstructOp::create(
              rewriter, loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              SmallVector<Value>{cstLastDimSize});

          // Emit aten.rms_norm (in stash dtype if upcasted)
          output =
              Torch::AtenRmsNormOp::create(rewriter, loc, input.getType(), s,
                                           normalizedShape, gamma, cstEpsilon);
        }

        // Cast results back to original dtype if upcasted
        if (*stashDtype != origDtype) {
          output =
              Torch::convertTensorToDtype(rewriter, loc, output, origDtype);
          s = Torch::convertTensorToDtype(rewriter, loc, s, origDtype);
        }

        // Emit the results positionally following the ORT output ordering:
        //   0: output
        //   1: mean
        //   2: inv_std_var
        //   3: input_skip_bias_sum
        // Any output that the model does not consume is imported as a
        // !torch.none typed result; for those we emit a torch.constant.none in
        // that slot and skip computing the corresponding value.
        auto isNoneResult = [](Type t) { return isa<Torch::NoneType>(t); };
        Value cstNone;
        auto getNone = [&]() -> Value {
          if (!cstNone)
            cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
          return cstNone;
        };

        SmallVector<Value> results;
        results.reserve(resultTypes.size());

        // Result 0: output (always present).
        results.push_back(output);

        // Result 1: mean. The simplified (RMS) variant does not subtract a
        // mean, so this output is vestigial. When requested with a concrete
        // tensor type, materialize a zeros tensor of the expected shape/dtype.
        if (resultTypes.size() > 1) {
          if (isNoneResult(resultTypes[1])) {
            results.push_back(getNone());
          } else {
            auto meanType = cast<Torch::ValueTensorType>(resultTypes[1]);
            Value dtypeInt = Torch::ConstantIntOp::create(
                rewriter, loc,
                rewriter.getI64IntegerAttr(static_cast<int64_t>(
                    Torch::getScalarTypeForType(meanType.getDtype()))));
            // Build a zeros tensor with the reduced shape. Use aten.zeros with
            // the sizes taken from the requested result type. Dynamic dims
            // (kUnknownSize) cannot be baked into a constant, so resolve them
            // at runtime from the already-materialized residual sum (sStash),
            // whose leading dims match the mean output's dims by index.
            SmallVector<Value> meanShapeValues;
            ArrayRef<int64_t> meanShape = meanType.getSizes();
            for (auto [idx, dim] : llvm::enumerate(meanShape)) {
              Value dimSize;
              if (dim == Torch::kUnknownSize) {
                Value dimIndex = Torch::ConstantIntOp::create(
                    rewriter, loc, rewriter.getI64IntegerAttr(idx));
                dimSize = Torch::AtenSizeIntOp::create(rewriter, loc, sStash,
                                                       dimIndex);
              } else {
                dimSize = Torch::ConstantIntOp::create(
                    rewriter, loc, rewriter.getI64IntegerAttr(dim));
              }
              meanShapeValues.push_back(dimSize);
            }

            Value zerosSizeList = Torch::PrimListConstructOp::create(
                rewriter, loc,
                Torch::ListType::get(
                    Torch::IntType::get(binder.op->getContext())),
                meanShapeValues);
            Value cstNoneForZeros = getNone();
            Value mean = Torch::AtenZerosOp::create(
                rewriter, loc, meanType, zerosSizeList, dtypeInt,
                /*layout=*/cstNoneForZeros, /*device=*/cstNoneForZeros,
                /*pin_memory=*/cstNoneForZeros);
            results.push_back(mean);
          }
        }

        // Result 2: inv_std_var, computed above together with the output.
        if (resultTypes.size() > 2) {
          if (!wantInvStd) {
            results.push_back(getNone());
          } else {
            auto invStdType = cast<Torch::ValueTensorType>(resultTypes[2]);
            Value invStd = invStdStash;
            if (invStdType.getOptionalDtype() != *stashDtype)
              invStd = Torch::convertTensorToDtype(
                  rewriter, loc, invStd, invStdType.getOptionalDtype());
            results.push_back(invStd);
          }
        }

        // Result 3: input_skip_bias_sum = input + skip + bias.
        if (resultTypes.size() > 3) {
          if (isNoneResult(resultTypes[3])) {
            results.push_back(getNone());
          } else {
            results.push_back(s);
          }
        }

        rewriter.replaceOp(binder.op, results);
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
        Value quantizedC;
        if (failed(createQuantizeTensor(rewriter, loc, c, cScale, cZp,
                                        resultType, quantizedC)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the result tensor `c`");
        rewriter.replaceOp(binder.op, quantizedC);
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
        Value quantizedY;
        if (failed(createQuantizeTensor(rewriter, loc, y, yScale, yZp,
                                        resultType, quantizedY)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the result tensor `y`");
        rewriter.replaceOp(binder.op, quantizedY);
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
        Value quantizedResult;
        if (failed(createQuantizeTensor(rewriter, loc, concat, yScale, yZp,
                                        resultType, quantizedResult)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the concatenated result tensor");
        rewriter.replaceOp(binder.op, quantizedResult);
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
        Value quantizedAvgpool;
        if (failed(createQuantizeTensor(rewriter, loc, avgpool, yScale, yZp,
                                        resultType, quantizedAvgpool)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the AvgPool result tensor");
        rewriter.replaceOp(binder.op, quantizedAvgpool);
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
        Value quantizedY;
        if (failed(createQuantizeTensor(rewriter, loc, y, yScale, yZp,
                                        resultType, quantizedY)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the Sigmoid result tensor");
        rewriter.replaceOp(binder.op, quantizedY);
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
        Value quantizedAveragePool;
        if (failed(createQuantizeTensor(rewriter, loc, averagePool, yScale, yZp,
                                        resultType, quantizedAveragePool)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the AveragePool result tensor");
        rewriter.replaceOp(binder.op, quantizedAveragePool);
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
        Value quantizedC;
        if (failed(createQuantizeTensor(rewriter, loc, c, cScale, cZp,
                                        resultType, quantizedC)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to quantize the result tensor `c`");
        rewriter.replaceOp(binder.op, quantizedC);
        return success();
      });

  patterns.onOp(
      "Attention", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        // com.microsoft::Attention (contrib op).
        //
        // Multi-head self attention with a fused Q/K/V projection. The op is
        // decomposed into explicit Torch ops:
        //   qkv    = input @ weights [+ bias]
        //   Q,K,V  = split(qkv, qkv_hidden_sizes, dim=2)
        //   Q,K,V  = reshape to [batch, num_heads, seq, head_size]
        //   scores = (Q @ K^T) * scale
        //   scores = scores [+ causal_mask] [+ mask_index] [+ attention_bias]
        //   probs  = softmax(scores, dim=-1)
        //   ctx    = probs @ V
        //   y      = reshape ctx back to [batch, seq, v_hidden]
        //
        // Supported inputs / attributes:
        //   inputs : input (0), weights (1), bias (2, optional),
        //            mask_index (3, optional), past (4, optional),
        //            attention_bias (5, optional)
        //   attrs  : num_heads (required), qkv_hidden_sizes, scale,
        //            unidirectional, mask_filter_value, do_rotary,
        //            rotary_embedding_dim
        //
        // Unsupported features cause a clean match failure:
        //   past_sequence_length (6, tensor form), past_present_share_buffer.
        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();

        int64_t numOperands = binder.op->getNumOperands();
        if (numOperands < 2)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected at least `input` and `weights` operands");

        // Returns the operand at `idx` if it is present and not a none value,
        // otherwise returns a null Value.
        auto optionalOperand = [&](int64_t idx) -> Value {
          if (idx >= numOperands)
            return Value();
          Value v = binder.op->getOperand(idx);
          if (isa<Torch::NoneType>(v.getType()))
            return Value();
          return v;
        };

        Value input = binder.op->getOperand(0);
        Value weights = binder.op->getOperand(1);
        Value bias = optionalOperand(2);
        Value maskIndex = optionalOperand(3);
        Value past = optionalOperand(4);
        Value attentionBias = optionalOperand(5);
        Value pastSeqLength = optionalOperand(6);

        // pastSeqLength is only needed with past_present_share_buffer (kept
        // unsupported below). When past is provided without that flag we derive
        // past_seq_len from the static shape of the `past` tensor.
        if (pastSeqLength)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: `past_sequence_length` tensor input is only "
              "required with past_present_share_buffer, which is unsupported");
        int64_t numResults = binder.op->getNumResults();
        if (numResults != 1 && numResults != 2)
          return rewriter.notifyMatchFailure(binder.op,
                                             "Expected 1 or 2 results");

        int64_t numHeads, unidirectional, doRotary, rotaryEmbeddingDim,
            pastPresentShareBuffer;
        float scale = 0.0f, maskFilterValue;
        SmallVector<int64_t> qkvHiddenSizes;
        if (binder.s64IntegerAttr(numHeads, "num_heads"))
          return rewriter.notifyMatchFailure(
              binder.op, "num_heads is a required attribute");
        if (numHeads == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "num_heads should be non-zero");
        // `scale` presence must be tracked explicitly: the spec allows an
        // explicit `scale = 0.0`, so a 0.0 value cannot be used as the "unset"
        // sentinel. binder.f32FloatAttr cannot distinguish an absent attribute
        // from a present one (it returns success in both cases and simply
        // writes the default), so query the raw op attribute directly to
        // determine presence. hasScale is true iff the model provided `scale`.
        bool hasScale =
            static_cast<bool>(binder.op->getAttr("torch.onnx.scale"));
        if (hasScale && binder.f32FloatAttr(scale, "scale"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "scale attribute bind failure");

        if (binder.s64IntegerAttr(unidirectional, "unidirectional", 0) ||
            binder.s64IntegerAttr(doRotary, "do_rotary", 0) ||
            binder.s64IntegerAttr(rotaryEmbeddingDim, "rotary_embedding_dim",
                                  0) ||
            binder.s64IntegerAttr(pastPresentShareBuffer,
                                  "past_present_share_buffer", 0) ||
            binder.f32FloatAttr(maskFilterValue, "mask_filter_value",
                                -10000.0f) ||
            binder.s64IntegerArrayAttr(qkvHiddenSizes, "qkv_hidden_sizes", {}))
          return rewriter.notifyMatchFailure(binder.op,
                                             "op attributes bind failure");

        if (pastPresentShareBuffer != 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: past_present_share_buffer is not supported");

        // Resolve the rotary embedding dimension: 0 means use head_size (after
        // head size is computed below). The spec says limited to 32, 64, 128
        // but we do not enforce that here.

        // Bind the primary (attention output) result type at index 0.
        // tensorResultType cannot be used here because it requires the op to
        // have exactly one result, whereas this op may also produce a second
        // `present` result; tensorResultTypeAtIndex handles both the 1- and
        // 2-result cases.
        Torch::ValueTensorType resultType;
        if (binder.tensorResultTypeAtIndex(resultType, 0))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");

        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());

        if (!inputType.hasSizes() || inputType.getSizes().size() != 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `input` to have 3 dimensions "
                         "[batch, seq, input_hidden]");
        SmallVector<int64_t> inputSizes{inputType.getSizes()};
        int64_t batchSize = inputSizes[0];
        int64_t sequenceLength = inputSizes[1];

        Torch::ValueTensorType weightsType =
            cast<Torch::ValueTensorType>(weights.getType());
        if (!weightsType.hasSizes() || weightsType.getSizes().size() != 2)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `weights` to have 2 dimensions "
                         "[input_hidden, qkv_hidden_sum]");
        int64_t qkvHiddenSum = weightsType.getSizes()[1];
        if (qkvHiddenSum == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `weights` last dimension to be static");

        // Resolve per-projection hidden sizes.
        int64_t qHidden, kHidden, vHidden;
        if (qkvHiddenSizes.empty()) {
          if (qkvHiddenSum % 3 != 0)
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected qkv hidden sum divisible by 3 when qkv_hidden_sizes "
                "is not provided");
          qHidden = kHidden = vHidden = qkvHiddenSum / 3;
        } else {
          if (qkvHiddenSizes.size() != 3)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected qkv_hidden_sizes to have 3 entries");
          qHidden = qkvHiddenSizes[0];
          kHidden = qkvHiddenSizes[1];
          vHidden = qkvHiddenSizes[2];
          if (qHidden + kHidden + vHidden != qkvHiddenSum)
            return rewriter.notifyMatchFailure(
                binder.op, "qkv_hidden_sizes do not sum to weights hidden dim");
        }

        if (qHidden != kHidden)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected equal Q and K hidden sizes for score computation");
        if (qHidden % numHeads != 0 || vHidden % numHeads != 0)
          return rewriter.notifyMatchFailure(
              binder.op, "Q/V hidden sizes must be divisible by num_heads");

        int64_t headSize = qHidden / numHeads;
        int64_t vHeadSize = vHidden / numHeads;

        // Common constants.
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstDim2 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(2));
        Value cstDim3 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(3));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(headSize));
        Value cstVHeadSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(vHeadSize));
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Type intListType = Torch::ListType::get(Torch::IntType::get(context));

        auto elemDtype = inputType.getOptionalDtype();

        // Step 1: fused QKV projection: qkv = input @ weights + bias.
        SmallVector<int64_t> qkvSizes{batchSize, sequenceLength, qkvHiddenSum};
        Torch::ValueTensorType qkvType =
            Torch::ValueTensorType::get(context, qkvSizes, elemDtype);
        Value qkv =
            Torch::AtenMatmulOp::create(rewriter, loc, qkvType, input, weights);
        if (bias)
          qkv = Torch::AtenAddTensorOp::create(rewriter, loc, qkvType, qkv,
                                               bias, cstOne);

        // Step 2: split qkv into Q, K, V along the hidden dimension.
        Value cstQEnd = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(qHidden));
        Value cstKEnd = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(qHidden + kHidden));
        Value cstVEnd = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(qkvHiddenSum));

        Torch::ValueTensorType qSliceType = Torch::ValueTensorType::get(
            context, ArrayRef<int64_t>{batchSize, sequenceLength, qHidden},
            elemDtype);
        Value query = Torch::AtenSliceTensorOp::create(
            rewriter, loc, qSliceType, qkv, /*dim=*/cstDim2, /*start=*/cstZero,
            /*end=*/cstQEnd, /*step=*/cstOne);
        Torch::ValueTensorType kSliceType = Torch::ValueTensorType::get(
            context, ArrayRef<int64_t>{batchSize, sequenceLength, kHidden},
            elemDtype);
        Value key = Torch::AtenSliceTensorOp::create(
            rewriter, loc, kSliceType, qkv, /*dim=*/cstDim2, /*start=*/cstQEnd,
            /*end=*/cstKEnd, /*step=*/cstOne);
        Torch::ValueTensorType vSliceType = Torch::ValueTensorType::get(
            context, ArrayRef<int64_t>{batchSize, sequenceLength, vHidden},
            elemDtype);
        Value value = Torch::AtenSliceTensorOp::create(
            rewriter, loc, vSliceType, qkv, /*dim=*/cstDim2, /*start=*/cstKEnd,
            /*end=*/cstVEnd, /*step=*/cstOne);

        // Step 3: reshape Q/K/V to [batch, num_heads, seq, head_size].
        auto reshapeToHeads = [&](Value proj, int64_t perHead) -> Value {
          Value perHeadConst =
              (perHead == headSize) ? cstHeadSize : cstVHeadSize;
          Value unflattenSizes = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              llvm::SmallVector<Value>{cstNumHeads, perHeadConst});
          Torch::ValueTensorType intermediateType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, sequenceLength, numHeads, perHead},
              elemDtype);
          Value intermediate = Torch::AtenUnflattenIntOp::create(
              rewriter, loc, intermediateType, proj, /*dim=*/cstDim2,
              unflattenSizes);
          Torch::ValueTensorType finalType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, numHeads, sequenceLength, perHead},
              elemDtype);
          return Torch::AtenTransposeIntOp::create(
              rewriter, loc, finalType, intermediate, cstOne, cstDim2);
        };
        Value qHeads = reshapeToHeads(query, headSize);
        Value kHeads = reshapeToHeads(key, headSize);
        Value vHeads = reshapeToHeads(value, vHeadSize);

        // Derive pastSeq / totalSeqLen early so both the RoPE position-offset
        // (step 3b) and the causal mask shape (step 4b) can use them. The
        // actual K/V prepend is done in step 3c below.
        int64_t pastSeq = 0;
        int64_t totalSeqLen = sequenceLength;
        if (past) {
          Torch::ValueTensorType pastPreType =
              cast<Torch::ValueTensorType>(past.getType());
          if (!pastPreType.hasSizes() || pastPreType.getSizes().size() != 5)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected `past` to have 5 dimensions "
                           "[2, batch, num_heads, past_seq, head_size]");
          int64_t ps = pastPreType.getSizes()[3];
          if (ps == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op, "dynamic past_seq dimension is not supported; "
                           "past must have a static sequence-length dimension");
          pastSeq = ps;
          totalSeqLen = pastSeq + sequenceLength;
        }

        // Step 3b: apply rotary embeddings to Q and K if do_rotary is set.
        // RoPE (rotary position embeddings) rotate pairs of elements in the
        // head dimension using sinusoidal position-dependent angles:
        //   theta_i = pos / 10000^(2i / rotary_dim)
        //   [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        //
        // The rotary_dim (number of elements to rotate) defaults to head_size
        // when the attribute is 0.
        //
        // Position indices start at pastSeq so that the freshly-rotated new
        // keys are encoded at their correct absolute positions. The cached
        // keys coming from `past` are already rotated (see step 8: `present`
        // stores the post-RoPE K), so they are consumed as-is.
        // Frequency tables are always computed in f32 for precision, then cast
        // back to the model element type before the multiply.
        if (doRotary != 0) {

          // Effective rotary dimension: default to head_size.
          int64_t rotDim =
              (rotaryEmbeddingDim == 0) ? headSize : rotaryEmbeddingDim;
          int64_t halfRot = rotDim / 2;

          Value cstHalfRot = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(halfRot));
          Value cstRotDim = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(rotDim));

          // Build theta frequencies: [1/10000^(2i/rotDim)] for i=0..halfRot-1
          // Step = 2/rotDim in log space; equiv to arange(0,rotDim,2)/rotDim
          // freqType: [halfRot] f32 (RoPE tables are always computed in

          // float to preserve precision even for f16 models).
          Type f32Dtype = Float32Type::get(context);
          Torch::ValueTensorType freqType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{halfRot}, f32Dtype);
          // base: arange(0, rotDim, 2) / rotDim
          Value cstRotDimFloat = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(rotDim)));
          // Use float scalars so arange produces f32 output matching freqType.
          Value cstFreqStart = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(0.0));
          Value cstFreqEnd = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(rotDim)));
          Value cstFreqStep = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(2.0));
          Value freqRange = Torch::AtenArangeStartStepOp::create(
              rewriter, loc, freqType, cstFreqStart, cstFreqEnd, cstFreqStep,
              cstNone, cstNone, cstNone, cstNone);
          Value freqNorm = Torch::AtenDivScalarOp::create(
              rewriter, loc, freqType, freqRange, cstRotDimFloat);
          // exponent: -freqNorm * log(10000)
          Value cstLog10k = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(std::log(10000.0)));
          Value freqExp = Torch::AtenMulScalarOp::create(
              rewriter, loc, freqType, freqNorm, cstLog10k);
          Value freqExpNeg =
              Torch::AtenNegOp::create(rewriter, loc, freqType, freqExp);
          Value invFreq =
              Torch::AtenExpOp::create(rewriter, loc, freqType, freqExpNeg);
          // invFreq shape: [halfRot] -> unsqueeze to [1, halfRot]

          // Build position indices [pastSeq..pastSeq+seq-1] shape [seq]
          // Positions start at pastSeq so that cached (already-rotated)
          // tokens are not re-rotated at wrong offsets.
          Torch::ValueTensorType posType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{sequenceLength}, f32Dtype);
          // Use float scalars so arange produces f32 output matching posType.
          Value cstPosStart = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(pastSeq)));
          Value cstPosEnd = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(totalSeqLen)));
          Value cstPosStep = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(1.0));
          Value posRange = Torch::AtenArangeStartStepOp::create(
              rewriter, loc, posType, cstPosStart, cstPosEnd, cstPosStep,
              cstNone, cstNone, cstNone, cstNone);
          // posRange shape [seq] -> [seq, 1]
          Torch::ValueTensorType posU1Type = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{sequenceLength, 1}, f32Dtype);
          Value posU1 = Torch::AtenUnsqueezeOp::create(rewriter, loc, posU1Type,
                                                       posRange, cstOne);

          // invFreq [halfRot] -> [1, halfRot]
          Torch::ValueTensorType invFreqU0Type = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{1, halfRot}, f32Dtype);
          Value invFreqU0 = Torch::AtenUnsqueezeOp::create(
              rewriter, loc, invFreqU0Type, invFreq, cstZero);

          // outer product: [seq, 1] * [1, halfRot] -> [seq, halfRot]
          Torch::ValueTensorType anglesType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{sequenceLength, halfRot}, f32Dtype);
          Value angles = Torch::AtenMulTensorOp::create(
              rewriter, loc, anglesType, posU1, invFreqU0);

          // cos/sin of angles: [seq, halfRot] in f32; cast to model dtype
          // before applying to Q/K so element-wise mul is type-compatible.
          Torch::ValueTensorType anglesElemType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{sequenceLength, halfRot}, elemDtype);
          Value cosA0 =
              Torch::AtenCosOp::create(rewriter, loc, anglesType, angles);
          Value sinA0 =
              Torch::AtenSinOp::create(rewriter, loc, anglesType, angles);
          Value dtypeInt = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(static_cast<int64_t>(
                  Torch::getScalarTypeForType(elemDtype))));
          Value cosA = Torch::AtenToDtypeOp::create(
              rewriter, loc, anglesElemType, cosA0, dtypeInt,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/cstNone);
          Value sinA = Torch::AtenToDtypeOp::create(
              rewriter, loc, anglesElemType, sinA0, dtypeInt,
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/cstNone);

          // Broadcast to [1, 1, seq, halfRot] for head broadcast
          auto addHeadDims = [&](Value t, ArrayRef<int64_t> sz) -> Value {
            Torch::ValueTensorType t1 = Torch::ValueTensorType::get(
                context, ArrayRef<int64_t>{1, sz[0], sz[1]}, elemDtype);
            Value u0 =
                Torch::AtenUnsqueezeOp::create(rewriter, loc, t1, t, cstZero);
            Torch::ValueTensorType t2 = Torch::ValueTensorType::get(
                context, ArrayRef<int64_t>{1, 1, sz[0], sz[1]}, elemDtype);
            return Torch::AtenUnsqueezeOp::create(rewriter, loc, t2, u0,
                                                  cstZero);
          };
          Value cosB = addHeadDims(cosA, {sequenceLength, halfRot});
          Value sinB = addHeadDims(sinA, {sequenceLength, halfRot});

          // Helper: apply RoPE to a [batch, heads, seq, head_size] tensor.
          auto applyRope = [&](Value x, ArrayRef<int64_t> xShape) -> Value {
            // Split x along last dim into [batch,heads,seq,halfRot] x2
            Torch::ValueTensorType halfType = Torch::ValueTensorType::get(
                context,
                ArrayRef<int64_t>{xShape[0], xShape[1], xShape[2], halfRot},
                elemDtype);
            // cstHalfRot and cstRotDim are captured from outer scope.
            Value x1 = Torch::AtenSliceTensorOp::create(rewriter, loc, halfType,
                                                        x, cstDim3, cstZero,
                                                        cstHalfRot, cstOne);
            Value x2 = Torch::AtenSliceTensorOp::create(rewriter, loc, halfType,
                                                        x, cstDim3, cstHalfRot,
                                                        cstRotDim, cstOne);
            // rotated = [x1*cos - x2*sin, x2*cos + x1*sin]
            Value r1 = Torch::AtenSubTensorOp::create(
                rewriter, loc, halfType,
                Torch::AtenMulTensorOp::create(rewriter, loc, halfType, x1,
                                               cosB),
                Torch::AtenMulTensorOp::create(rewriter, loc, halfType, x2,
                                               sinB),
                cstOne);
            Value r2 = Torch::AtenAddTensorOp::create(
                rewriter, loc, halfType,
                Torch::AtenMulTensorOp::create(rewriter, loc, halfType, x2,
                                               cosB),
                Torch::AtenMulTensorOp::create(rewriter, loc, halfType, x1,
                                               sinB),
                cstOne);
            // cat([r1, r2], dim=-1) -> [batch,heads,seq,rotDim]
            Torch::ValueTensorType rotType = Torch::ValueTensorType::get(
                context,
                ArrayRef<int64_t>{xShape[0], xShape[1], xShape[2], rotDim},
                elemDtype);
            Type halfListElem =
                cast<Torch::BaseTensorType>(r1.getType())
                    .getWithSizesAndDtype(std::nullopt, nullptr);
            Value rotList = Torch::PrimListConstructOp::create(
                rewriter, loc, Torch::ListType::get(halfListElem),
                llvm::SmallVector<Value>{r1, r2});
            Value rotated = Torch::AtenCatOp::create(rewriter, loc, rotType,
                                                     rotList, cstDim3);
            if (rotDim < xShape[3]) {
              // Concatenate unrotated tail back.
              Torch::ValueTensorType tailType = Torch::ValueTensorType::get(
                  context,
                  ArrayRef<int64_t>{xShape[0], xShape[1], xShape[2],
                                    xShape[3] - rotDim},
                  elemDtype);
              Value cstHsz = Torch::ConstantIntOp::create(
                  rewriter, loc, rewriter.getI64IntegerAttr(xShape[3]));
              Value tail = Torch::AtenSliceTensorOp::create(
                  rewriter, loc, tailType, x, cstDim3, cstRotDim, cstHsz,
                  cstOne);
              Type rotListElem =
                  cast<Torch::BaseTensorType>(rotated.getType())
                      .getWithSizesAndDtype(std::nullopt, nullptr);
              Value fullList = Torch::PrimListConstructOp::create(
                  rewriter, loc, Torch::ListType::get(rotListElem),
                  llvm::SmallVector<Value>{rotated, tail});
              Torch::ValueTensorType fullType =
                  Torch::ValueTensorType::get(context, xShape, elemDtype);
              return Torch::AtenCatOp::create(rewriter, loc, fullType, fullList,
                                              cstDim3);
            }
            return rotated;
          };

          SmallVector<int64_t> qShape{batchSize, numHeads, sequenceLength,
                                      headSize};
          qHeads = applyRope(qHeads, qShape);
          kHeads = applyRope(kHeads, qShape);
        }

        // Step 3c: prepend past K/V cache if provided.
        // past shape: (2, batch, num_heads, past_seq, head_size)
        // After prepending: kHeads/vHeads shape becomes
        //   [batch, num_heads, past_seq + seq, head_size]
        // (totalSeqLen and pastSeq were already resolved before step 3b.)
        // pastK/pastV are hoisted so they can be reused in step 8 (present
        // output) without re-emitting duplicate AtenSelectIntOp nodes.
        Value pastK, pastV;
        Torch::ValueTensorType pastKType, pastVType;
        if (past) {
          // Extract past K: select index 0 along dim 0
          pastKType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, numHeads, pastSeq, headSize},
              elemDtype);
          Value cstPastDim0 = cstZero; // dim 0 of the 5D past tensor
          pastK = Torch::AtenSelectIntOp::create(rewriter, loc, pastKType, past,
                                                 cstPastDim0, cstZero);

          // Extract past V: select index 1 along dim 0
          pastVType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, numHeads, pastSeq, vHeadSize},
              elemDtype);
          pastV = Torch::AtenSelectIntOp::create(rewriter, loc, pastVType, past,
                                                 cstPastDim0, cstOne);

          // cat([pastK, kHeads], dim=2) -> [batch, heads, total_seq, head_sz]
          Type kListElem = cast<Torch::BaseTensorType>(pastK.getType())
                               .getWithSizesAndDtype(std::nullopt, nullptr);
          Value kCatList = Torch::PrimListConstructOp::create(
              rewriter, loc, Torch::ListType::get(kListElem),
              llvm::SmallVector<Value>{pastK, kHeads});
          Torch::ValueTensorType kCatType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, numHeads, totalSeqLen, headSize},
              elemDtype);
          kHeads = Torch::AtenCatOp::create(rewriter, loc, kCatType, kCatList,
                                            cstDim2);

          Type vListElem = cast<Torch::BaseTensorType>(pastV.getType())
                               .getWithSizesAndDtype(std::nullopt, nullptr);
          Value vCatList = Torch::PrimListConstructOp::create(
              rewriter, loc, Torch::ListType::get(vListElem),
              llvm::SmallVector<Value>{pastV, vHeads});
          Torch::ValueTensorType vCatType = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{batchSize, numHeads, totalSeqLen, vHeadSize},
              elemDtype);
          vHeads = Torch::AtenCatOp::create(rewriter, loc, vCatType, vCatList,
                                            cstDim2);
        }

        // Step 4: scores = (Q @ K^T) * scale.
        // K^T: transpose last two dims -> [batch, num_heads, head_size,
        // total_seq].
        Torch::ValueTensorType kTransposedType = Torch::ValueTensorType::get(
            context,
            ArrayRef<int64_t>{batchSize, numHeads, headSize, totalSeqLen},
            elemDtype);
        Value kTransposed = Torch::AtenTransposeIntOp::create(
            rewriter, loc, kTransposedType, kHeads, cstDim2, cstDim3);

        Torch::ValueTensorType scoresType = Torch::ValueTensorType::get(
            context,
            ArrayRef<int64_t>{batchSize, numHeads, sequenceLength, totalSeqLen},
            elemDtype);
        Value scores = Torch::AtenMatmulOp::create(rewriter, loc, scoresType,
                                                   qHeads, kTransposed);

        double scaleValue =
            hasScale ? static_cast<double>(scale)
                     : 1.0 / std::sqrt(static_cast<double>(headSize));
        Value cstScale = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(scaleValue));
        scores = Torch::AtenMulScalarOp::create(rewriter, loc, scoresType,
                                                scores, cstScale);

        // Step 4b: apply additive masks/biases to the scores.
        //
        // Three optional contributions are supported, each broadcast over the
        // [batch, num_heads, seq, seq] score tensor. For self attention the
        // total_sequence_length equals the query sequence length.
        Type elemType = inputType.getDtype();

        // (a) unidirectional (causal) mask: position q may only attend to
        //     key positions k <= q. Future positions are filled with
        //     mask_filter_value. Built as strictly-upper-triangular ones
        //     scaled by mask_filter_value, added to the scores.
        //
        //     When a KV cache (`past`) is present totalSeqLen > sequenceLength.
        //     The mask shape must be [sequenceLength, totalSeqLen] and the
        //     strictly-upper-triangular diagonal must be offset by pastSeq so
        //     that query position q (0-based within the new tokens) is allowed
        //     to attend to key positions 0..pastSeq+q.
        if (unidirectional != 0) {
          Value cstSeq = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(sequenceLength));
          Value cstTotalSeq = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(totalSeqLen));
          Value maskShape = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              llvm::SmallVector<Value>{cstSeq, cstTotalSeq});
          Value cstFillOne = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(1.0));
          Torch::ValueTensorType causalType = Torch::ValueTensorType::get(
              context, ArrayRef<int64_t>{sequenceLength, totalSeqLen},
              elemType);
          Value causalOnes = Torch::AtenFullOp::create(
              rewriter, loc, causalType, maskShape, cstFillOne,
              /*dtype=*/cstNone, /*layout=*/cstNone, /*device=*/cstNone,
              /*pin_memory=*/cstNone);
          // Diagonal = pastSeq+1: position q in new tokens attends to
          // keys 0..pastSeq+q (lower triangle of the full KV sequence).
          Value cstDiag = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(static_cast<int64_t>(pastSeq) + 1));
          Value causalTriu = Torch::AtenTriuOp::create(
              rewriter, loc, causalType, causalOnes, cstDiag);
          Value cstFilter = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(maskFilterValue)));
          Value causalMask = Torch::AtenMulScalarOp::create(
              rewriter, loc, causalType, causalTriu, cstFilter);
          scores = Torch::AtenAddTensorOp::create(rewriter, loc, scoresType,
                                                  scores, causalMask, cstOne);
        }

        // (b) raw attention mask (mask_index): shape (batch, total_seq) or
        //     (batch, seq, total_seq) with value 1 to keep and 0 to mask.
        //     Converted to an additive mask: keep -> 0, mask ->
        //     mask_filter_value, then broadcast into the score tensor.
        //
        //     LIMITATION: only the *raw* keep/mask forms (2D/3D, 1=keep,
        //     0=mask) are handled here. The com.microsoft spec also defines
        //     1D right-side-padding encodings ([batch], [2*batch] and
        //     [3*batch+2]) that onnxruntime commonly emits; those ranks are
        //     rejected by the rank check below (clean match failure) and are
        //     left as future work. The 1=keep / 0=mask polarity matches ORT's
        //     raw-mask semantics.
        if (maskIndex) {
          Torch::ValueTensorType maskType =
              cast<Torch::ValueTensorType>(maskIndex.getType());
          if (!maskType.hasSizes() || (maskType.getSizes().size() != 2 &&
                                       maskType.getSizes().size() != 3))
            return rewriter.notifyMatchFailure(
                binder.op,
                "Unimplemented: only 2D (batch, total_seq) or 3D "
                "(batch, seq, total_seq) raw mask_index is supported");
          int64_t maskRank = maskType.getSizes().size();
          int64_t totalSeq = maskType.getSizes()[maskRank - 1];

          // The mask's last dimension indexes key positions, so it must match
          // the attention total sequence length (past_seq + seq). A mismatch
          // means the mask cannot be broadcast against the scores.
          if (totalSeq != Torch::kUnknownSize && totalSeq != totalSeqLen)
            return rewriter.notifyMatchFailure(
                binder.op,
                "mask_index last dimension must equal the total sequence "
                "length (past_seq + seq)");

          // Cast the mask to the score element type (no-op if it already
          // matches, so guard to avoid emitting a redundant aten.to.dtype).
          Value maskFloat = maskIndex;
          if (maskType.getOptionalDtype() != elemType)
            maskFloat =
                Torch::convertTensorToDtype(rewriter, loc, maskIndex, elemType);

          // additive = maskFloat * (-mask_filter_value) + mask_filter_value.
          // keep (1) -> 0, mask (0) -> mask_filter_value.
          Value cstNegFilter = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(-static_cast<double>(maskFilterValue)));
          Value cstFilter = Torch::ConstantFloatOp::create(
              rewriter, loc,
              rewriter.getF64FloatAttr(static_cast<double>(maskFilterValue)));
          Value additive = Torch::AtenMulScalarOp::create(
              rewriter, loc, maskFloat.getType(), maskFloat, cstNegFilter);
          additive = Torch::AtenAddScalarOp::create(
              rewriter, loc, additive.getType(), additive, cstFilter, cstOne);

          // Reshape to [batch, 1, (1|seq), total_seq] for broadcasting.
          if (maskRank == 2) {
            Torch::ValueTensorType u1Type = Torch::ValueTensorType::get(
                context, ArrayRef<int64_t>{batchSize, 1, totalSeq}, elemType);
            additive = Torch::AtenUnsqueezeOp::create(rewriter, loc, u1Type,
                                                      additive, cstOne);
            Torch::ValueTensorType u2Type = Torch::ValueTensorType::get(
                context, ArrayRef<int64_t>{batchSize, 1, 1, totalSeq},
                elemType);
            additive = Torch::AtenUnsqueezeOp::create(rewriter, loc, u2Type,
                                                      additive, cstOne);
          } else {
            Torch::ValueTensorType u1Type = Torch::ValueTensorType::get(
                context,
                ArrayRef<int64_t>{batchSize, 1, sequenceLength, totalSeq},
                elemType);
            additive = Torch::AtenUnsqueezeOp::create(rewriter, loc, u1Type,
                                                      additive, cstOne);
          }
          scores = Torch::AtenAddTensorOp::create(rewriter, loc, scoresType,
                                                  scores, additive, cstOne);
        }

        // (c) attention_bias: an additive bias with shape
        //     (batch or 1, num_heads or 1, seq, total_seq) added directly to
        //     the scores. Broadcasting handles the singleton dimensions.
        if (attentionBias) {
          // Cast to the score element type (guarded to skip a redundant
          // aten.to.dtype when the bias is already in the right dtype).
          auto attnBiasType =
              cast<Torch::ValueTensorType>(attentionBias.getType());
          Value attnBias = attentionBias;
          if (attnBiasType.getOptionalDtype() != elemType)
            attnBias = Torch::convertTensorToDtype(rewriter, loc, attentionBias,
                                                   elemType);
          scores = Torch::AtenAddTensorOp::create(rewriter, loc, scoresType,
                                                  scores, attnBias, cstOne);
        }

        // Step 5: probs = softmax(scores, dim=-1).
        Value probs = Torch::AtenSoftmaxIntOp::create(
            rewriter, loc, scoresType, scores, cstDim3, /*dtype=*/cstNone);

        // Step 6: ctx = probs @ V -> [batch, num_heads, seq, v_head_size].
        Torch::ValueTensorType ctxType = Torch::ValueTensorType::get(
            context,
            ArrayRef<int64_t>{batchSize, numHeads, sequenceLength, vHeadSize},
            elemDtype);
        Value ctx =
            Torch::AtenMatmulOp::create(rewriter, loc, ctxType, probs, vHeads);

        // Step 7: reshape ctx back to [batch, seq, v_hidden].
        Torch::ValueTensorType ctxTransposedType = Torch::ValueTensorType::get(
            context,
            ArrayRef<int64_t>{batchSize, sequenceLength, numHeads, vHeadSize},
            elemDtype);
        Value ctxTransposed = Torch::AtenTransposeIntOp::create(
            rewriter, loc, ctxTransposedType, ctx, cstOne, cstDim2);
        Value output = Torch::AtenFlattenUsingIntsOp::create(
            rewriter, loc, resultType, ctxTransposed, cstDim2, cstDim3);

        // Step 8: build present state (optional result 1).
        // present shape: (2, batch, num_heads, total_seq, head_size)
        // This is the updated KV cache = cat([past_K, new_K]) stacked on dim 0.
        if (numResults == 2) {
          // Build present K/V from the post-RoPE K/V (ORT CUDA writes `present`
          // from the rotated QKV workspace, so cached keys are stored already
          // rotated). After step 3c, kHeads/vHeads already carry the past cache
          // concatenated with the new tokens at totalSeqLen, so they can be
          // used directly - no separate past concat is needed here.
          Value presentK = kHeads;
          Value presentV = vHeads;
          // Unsqueeze K and V along dim 0 to get [1, batch, heads, seq, head]

          Torch::ValueTensorType kU0Type = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{1, batchSize, numHeads, totalSeqLen, headSize},
              elemDtype);
          Value kU0 = Torch::AtenUnsqueezeOp::create(rewriter, loc, kU0Type,
                                                     presentK, cstZero);
          Torch::ValueTensorType vU0Type = Torch::ValueTensorType::get(
              context,
              ArrayRef<int64_t>{1, batchSize, numHeads, totalSeqLen, vHeadSize},
              elemDtype);
          Value vU0 = Torch::AtenUnsqueezeOp::create(rewriter, loc, vU0Type,
                                                     presentV, cstZero);
          // cat([k_unsqueezed, v_unsqueezed], dim=0) ->
          //   [2, batch, heads, total_seq, head_size]
          Type kvListElem = cast<Torch::BaseTensorType>(kU0.getType())
                                .getWithSizesAndDtype(std::nullopt, nullptr);
          Value kvList = Torch::PrimListConstructOp::create(
              rewriter, loc, Torch::ListType::get(kvListElem),
              llvm::SmallVector<Value>{kU0, vU0});
          Torch::ValueTensorType presentResultType;
          if (binder.tensorResultTypeAtIndex(presentResultType, 1))
            return rewriter.notifyMatchFailure(
                binder.op, "present result type bind failure");
          Value present = Torch::AtenCatOp::create(
              rewriter, loc, presentResultType, kvList, cstZero);
          rewriter.replaceOp(binder.op, {output, present});
        } else {
          rewriter.replaceOp(binder.op, {output});
        }
        return success();
      });
}
