//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/TypeUtilities.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <cmath>
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

namespace {

/// Positions of the com.microsoft.GroupQueryAttention inputs.
enum class GqaInput : unsigned {
  Query,
  Key,
  Value,
  PastKey,
  PastValue,
  SeqlensK,
  TotalSequenceLength,
  CosCache,
  SinCache,
  PositionIds,
  AttentionBias,
  HeadSink,
  KScale,
  VScale,
  QNormWeight,
  KNormWeight,
};

/// What matchGroupQueryAttention extracts from a supported
/// GroupQueryAttention op. Absent optional inputs and outputs are null.
struct GqaInfo {
  // Attributes, read as the onnxruntime CPU kernel reads them.
  int64_t numHeads;
  int64_t kvNumHeads;
  float scale;
  float softcap;
  bool doRotary;
  bool rotaryInterleaved;
  bool smoothSoftmax;
  bool causal;
  /// The window of a causal local attention, or 0 for none.
  int64_t localWindowSize;
  /// Whether sliding_window_cache keeps the caches at the past capacity.
  bool windowed;
  int64_t qkOutput;
  /// 0 for a float cache, else the bits of its quantized values.
  int64_t kvCacheBitWidth;
  bool perChannel;

  Value query;
  Value key;
  Value value;
  Value pastKey;
  Value pastValue;
  Value seqlensK;
  Value totalSequenceLength;
  Value cosCache;
  Value sinCache;
  Value positionIds;
  Value attentionBias;
  Value headSink;
  Value kScale;
  Value vScale;

  Torch::ValueTensorType outputType;
  Torch::ValueTensorType presentKeyType;
  Torch::ValueTensorType presentValueType;
  Torch::ValueTensorType qkType;

  /// The float type of query, key, value and the outputs.
  Type dtype;
  int64_t headSize;
  /// Whether query holds Q, K and V, which onnxruntime detects by an absent
  /// key.
  bool packed;
  /// Whether key and value hold no tokens, so that attention reads the past
  /// cache alone.
  bool sharedKv;
};

} // namespace

/// Returns a `!torch.list<int>` of `values`.
static Value createIntList(ImplicitLocOpBuilder &b, ArrayRef<Value> values) {
  return Torch::PrimListConstructOp::create(
      b, Torch::ListType::get(Torch::IntType::get(b.getContext())), values);
}

/// Returns the type of `x` with its sizes replaced by `sizes`.
static Torch::ValueTensorType withSizes(Value x, ArrayRef<int64_t> sizes) {
  return Torch::ValueTensorType::get(
      x.getContext(), sizes,
      cast<Torch::ValueTensorType>(x.getType()).getOptionalDtype());
}

/// Views the 1-D tensor `vector` with `rank` dimensions, all of size 1 except
/// `dim`, so that it broadcasts along `dim` only.
static Value viewAlongDim(ConversionPatternRewriter &rewriter, Location loc,
                          Value vector, int64_t dim, int64_t rank) {
  SmallVector<int64_t> sizes(rank, 1);
  SmallVector<int64_t> shape(rank, 1);
  sizes[dim] = cast<Torch::ValueTensorType>(vector.getType()).getSizes()[0];
  shape[dim] = -1;
  return Torch::AtenViewOp::create(
      rewriter, loc, withSizes(vector, sizes), vector,
      Torch::toIntListConstruct(rewriter, loc, shape));
}

/// Returns [0, 1, ..., length - 1] as si64, viewed along `dim` of a
/// rank-`rank` tensor.
static Value createIndexRange(ConversionPatternRewriter &rewriter, Location loc,
                              Value length, int64_t staticLength, int64_t dim,
                              int64_t rank) {
  // arange of an int length is si64 by default.
  Value none = Torch::ConstantNoneOp::create(rewriter, loc);
  Value range = Torch::AtenArangeOp::create(
      rewriter, loc,
      rewriter.getType<Torch::ValueTensorType>(
          ArrayRef<int64_t>{staticLength},
          rewriter.getIntegerType(64, /*isSigned=*/true)),
      length, none, none, none, none);
  return viewAlongDim(rewriter, loc, range, dim, rank);
}

/// Returns `x[..., start:end]` along `dim`, whose static size is `size`.
static Value createSlice(ConversionPatternRewriter &rewriter, Location loc,
                         Value x, int64_t dim, Value start, Value end,
                         int64_t size) {
  SmallVector<int64_t> sizes(
      cast<Torch::ValueTensorType>(x.getType()).getSizes());
  sizes[dim] = size;
  return Torch::AtenSliceTensorOp::create(
      rewriter, loc, withSizes(x, sizes), x,
      Torch::ConstantIntOp::create(rewriter, loc, dim), start, end,
      Torch::ConstantIntOp::create(rewriter, loc, 1));
}

/// Returns `x` of shape [B, S, numHeads * headSize] as
/// [B, numHeads, S, headSize].
static Value splitHeads(ConversionPatternRewriter &rewriter, Location loc,
                        Value x, int64_t numHeads, int64_t headSize) {
  ImplicitLocOpBuilder b(loc, rewriter);
  ArrayRef<int64_t> sizes =
      cast<Torch::ValueTensorType>(x.getType()).getSizes();
  Value heads = Torch::AtenUnflattenIntOp::create(
      b, withSizes(x, {sizes[0], sizes[1], numHeads, headSize}), x,
      Torch::ConstantIntOp::create(b, 2),
      Torch::toIntListConstruct(rewriter, loc, {numHeads, headSize}));
  return Torch::AtenTransposeIntOp::create(
      b, withSizes(x, {sizes[0], numHeads, sizes[1], headSize}), heads,
      Torch::ConstantIntOp::create(b, 1), Torch::ConstantIntOp::create(b, 2));
}

/// Returns `x` of shape [B, numHeads, S, headSize] as
/// [B, S, numHeads * headSize].
static Value mergeHeads(ConversionPatternRewriter &rewriter, Location loc,
                        Value x) {
  ImplicitLocOpBuilder b(loc, rewriter);
  ArrayRef<int64_t> sizes =
      cast<Torch::ValueTensorType>(x.getType()).getSizes();
  Value transposed = Torch::AtenTransposeIntOp::create(
      b, withSizes(x, {sizes[0], sizes[2], sizes[1], sizes[3]}), x,
      Torch::ConstantIntOp::create(b, 1), Torch::ConstantIntOp::create(b, 2));
  return Torch::AtenFlattenUsingIntsOp::create(
      b, withSizes(x, {sizes[0], sizes[2], sizes[1] * sizes[3]}), transposed,
      Torch::ConstantIntOp::create(b, 2), Torch::ConstantIntOp::create(b, 3));
}

/// Packs pairs of 4-bit ui8 values along the last dimension into one ui8
/// each, the even element in the low nibble as the int4 cache stores them:
///   values[..., 2 * i] | (values[..., 2 * i + 1] << 4)
static Value packInt4LastDimension(ConversionPatternRewriter &rewriter,
                                   Location loc, Value values) {
  ImplicitLocOpBuilder b(loc, rewriter);
  SmallVector<int64_t> sizes(
      cast<Torch::ValueTensorType>(values.getType()).getSizes());
  sizes.back() /= 2;
  SmallVector<int64_t> pairSizes(sizes);
  pairSizes.push_back(2);
  Value pairs = Torch::AtenUnflattenIntOp::create(
      b, withSizes(values, pairSizes), values,
      Torch::ConstantIntOp::create(b, -1),
      Torch::toIntListConstruct(rewriter, loc, {sizes.back(), 2}));
  auto uint8 = b.getIntegerType(8, /*isSigned=*/false);
  Value shifts = Torch::ValueTensorLiteralOp::create(
      b, DenseElementsAttr::get(RankedTensorType::get({2}, uint8),
                                ArrayRef<APInt>{APInt(8, 0), APInt(8, 4)}));
  Value shifted = Torch::AtenBitwiseLeftShiftTensorOp::create(
      b, withSizes(values, pairSizes), pairs, shifts);
  Torch::ValueTensorType packedType = withSizes(values, sizes);
  Value lastDim = Torch::ConstantIntOp::create(b, -1);
  Value low = Torch::AtenSelectIntOp::create(
      b, packedType, shifted, lastDim, Torch::ConstantIntOp::create(b, 0));
  Value high = Torch::AtenSelectIntOp::create(
      b, packedType, shifted, lastDim, Torch::ConstantIntOp::create(b, 1));
  return Torch::AtenBitwiseOrTensorOp::create(b, packedType, low, high);
}

/// Quantizes new cache rows like onnxruntime's MlasKVQuantize: x * (1 / scale)
/// rounded half to even and clamped to the signed `bitWidth`-bit range, where
/// a zero scale counts as 1. 8-bit values are stored as si8, 4-bit values as
/// one ui8 per value, biased by 8 as in the packed cache.
static Value quantizeKvCache(ConversionPatternRewriter &rewriter, Location loc,
                             Value x, Value scale, int64_t bitWidth) {
  ImplicitLocOpBuilder b(loc, rewriter);
  x = Torch::convertTensorToDtype(rewriter, loc, x, rewriter.getF32Type());
  auto scaleType = cast<Torch::ValueTensorType>(scale.getType());
  Value isNonZero = Torch::AtenNeScalarOp::create(
      b, scaleType.getWithSizesAndDtype(scaleType.getSizes(), b.getI1Type()),
      scale, Torch::ConstantFloatOp::create(b, APFloat(0.0)));
  Value reciprocal = Torch::AtenReciprocalOp::create(b, scaleType, scale);
  Value inverseScale = Torch::AtenWhereScalarOtherOp::create(
      b, scaleType, isNonZero, reciprocal,
      Torch::ConstantFloatOp::create(b, APFloat(1.0)));
  auto xType = cast<Torch::ValueTensorType>(x.getType());
  Value scaled = Torch::AtenMulTensorOp::create(b, xType, x, inverseScale);
  Value rounded = Torch::AtenRoundOp::create(b, xType, scaled);
  double limit = int64_t(1) << (bitWidth - 1);
  Value clamped = Torch::AtenClampOp::create(
      b, xType, rounded, Torch::ConstantFloatOp::create(b, APFloat(-limit)),
      Torch::ConstantFloatOp::create(b, APFloat(limit - 1)));
  if (bitWidth == 8)
    return Torch::convertTensorToDtype(
        rewriter, loc, clamped, rewriter.getIntegerType(8, /*isSigned=*/true));
  Value biased = Torch::AtenAddScalarOp::create(
      b, xType, clamped, Torch::ConstantFloatOp::create(b, APFloat(8.0)),
      Torch::ConstantIntOp::create(b, 1));
  return Torch::convertTensorToDtype(
      rewriter, loc, biased, rewriter.getIntegerType(8, /*isSigned=*/false));
}

/// Dequantizes stored cache values to `dtype`: the stored value, less the
/// bias of 8 for 4-bit values, times the scale.
static Value dequantizeKvCache(ConversionPatternRewriter &rewriter,
                               Location loc, Value stored, Value scale,
                               int64_t bitWidth, Type dtype) {
  ImplicitLocOpBuilder b(loc, rewriter);
  Value x =
      Torch::convertTensorToDtype(rewriter, loc, stored, rewriter.getF32Type());
  auto xType = cast<Torch::ValueTensorType>(x.getType());
  if (bitWidth == 4)
    x = Torch::AtenSubScalarOp::create(
        b, xType, x, Torch::ConstantFloatOp::create(b, APFloat(8.0)),
        Torch::ConstantIntOp::create(b, 1));
  x = Torch::AtenMulTensorOp::create(b, xType, x, scale);
  if (dtype.isF32())
    return x;
  return Torch::convertTensorToDtype(rewriter, loc, x, dtype);
}

/// Returns onnxruntime's windowed-cache length for per-batch lengths `length`:
///   length - blockSize * ceil(max(length - capacity, 0) / blockSize)
/// The cache drops blockSize = capacity - local_window_size + 1 rows at a time
/// once it is full, so that it compacts only every blockSize steps.
static Value getWindowedCacheLength(ConversionPatternRewriter &rewriter,
                                    Location loc, Value length, Value capacity,
                                    Value blockSize, Value blockSizeMinusOne) {
  ImplicitLocOpBuilder b(loc, rewriter);
  auto type = cast<Torch::ValueTensorType>(length.getType());
  Value one = Torch::ConstantIntOp::create(b, 1);
  Value excess = Torch::AtenSubScalarOp::create(b, type, length, capacity, one);
  Value overflow = Torch::AtenClampMinOp::create(
      b, type, excess, Torch::ConstantIntOp::create(b, 0));
  Value rounded =
      Torch::AtenAddScalarOp::create(b, type, overflow, blockSizeMinusOne, one);
  Value blocks =
      Torch::AtenFloorDivideScalarOp::create(b, type, rounded, blockSize);
  Value dropped = Torch::AtenMulScalarOp::create(b, type, blocks, blockSize);
  return Torch::AtenSubTensorOp::create(b, type, length, dropped, one);
}

/// Returns whether `type` has a dtype and sizes compatible with `sizes`, where
/// kUnknownSize matches any size.
static bool hasCompatibleShape(Torch::ValueTensorType type,
                               ArrayRef<int64_t> sizes, Type dtype) {
  return type.hasSizes() && type.hasDtype() && type.getDtype() == dtype &&
         succeeded(verifyCompatibleShape(
             Torch::makeShapeLLVMCompatible(sizes),
             Torch::makeShapeLLVMCompatible(type.getSizes())));
}

/// Binds and verifies the inputs, outputs and attributes of a
/// GroupQueryAttention op. Matching does not create IR.
static FailureOr<GqaInfo>
matchGroupQueryAttention(OpBinder binder, ConversionPatternRewriter &rewriter) {
  Operation *op = binder.op;
  GqaInfo info;
  int64_t doRotary, rotaryInterleaved, smoothSoftmax, causal,
      slidingWindowCache;
  std::string kQuantType, vQuantType;
  if (binder.s64IntegerAttr(info.numHeads, "num_heads", 0) ||
      binder.s64IntegerAttr(info.kvNumHeads, "kv_num_heads", 0) ||
      binder.f32FloatAttr(info.scale, "scale", 0.0f) ||
      binder.f32FloatAttr(info.softcap, "softcap", 0.0f) ||
      binder.s64IntegerAttr(doRotary, "do_rotary", 0) ||
      binder.s64IntegerAttr(rotaryInterleaved, "rotary_interleaved", 0) ||
      binder.s64IntegerAttr(smoothSoftmax, "smooth_softmax", -1) ||
      binder.s64IntegerAttr(causal, "causal", 1) ||
      binder.s64IntegerAttr(info.localWindowSize, "local_window_size", -1) ||
      binder.s64IntegerAttr(slidingWindowCache, "sliding_window_cache", 0) ||
      binder.s64IntegerAttr(info.qkOutput, "qk_output", 0) ||
      binder.customOpNameStringAttr(kQuantType, "k_quant_type", "NONE") ||
      binder.customOpNameStringAttr(vQuantType, "v_quant_type", "NONE") ||
      binder.s64IntegerAttr(info.kvCacheBitWidth, "kv_cache_bit_width", 0))
    return rewriter.notifyMatchFailure(op, "op attributes bind failure");
  // onnxruntime enables these flags only for the value 1.
  info.doRotary = doRotary == 1;
  info.rotaryInterleaved = rotaryInterleaved == 1;
  info.smoothSoftmax = smoothSoftmax == 1;
  info.windowed = slidingWindowCache == 1;

  int64_t numHeads = info.numHeads;
  int64_t kvNumHeads = info.kvNumHeads;
  if (numHeads <= 0 || kvNumHeads <= 0 || numHeads % kvNumHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "expected positive num_heads and kv_num_heads, with num_heads a "
            "multiple of kv_num_heads");
  if (causal != 0 && causal != 1)
    return rewriter.notifyMatchFailure(op, "expected causal to be 0 or 1");
  info.causal = causal == 1;
  // A negative local_window_size means no window, like the default -1.
  if (info.localWindowSize == 0 || (!info.causal && info.localWindowSize != -1))
    return rewriter.notifyMatchFailure(
        op, "expected local_window_size to be -1, or nonzero with causal 1");
  info.localWindowSize = std::max<int64_t>(info.localWindowSize, 0);
  if (info.qkOutput < 0 || info.qkOutput > 2)
    return rewriter.notifyMatchFailure(op, "expected qk_output 0, 1 or 2");
  // onnxruntime reads the quantization types case-insensitively, and its CPU
  // kernel requires the same type for both sides of the cache.
  std::string quantType = StringRef(kQuantType).upper();
  if (quantType != StringRef(vQuantType).upper() ||
      !llvm::is_contained({"NONE", "PER_TENSOR", "PER_CHANNEL"}, quantType))
    return rewriter.notifyMatchFailure(
        op, "expected k_quant_type and v_quant_type to be both NONE, both "
            "PER_TENSOR or both PER_CHANNEL");
  bool quantized = quantType != "NONE";
  info.perChannel = quantType == "PER_CHANNEL";
  int64_t bitWidth = info.kvCacheBitWidth;
  if (quantized ? bitWidth != 4 && bitWidth != 8 : bitWidth != 0)
    return rewriter.notifyMatchFailure(
        op, "expected kv_cache_bit_width 4 or 8 for a quantized cache and 0 "
            "otherwise");
  if (info.windowed && info.localWindowSize == 0)
    return rewriter.notifyMatchFailure(
        op, "expected a positive local_window_size with sliding_window_cache");

  // Inputs are positional, and empty ONNX inputs import as `!torch.none`.
  if (op->getNumOperands() < 7 || op->getNumOperands() > 16 ||
      !llvm::all_of(op->getOperandTypes(), [](Type type) {
        return isa<Torch::NoneType, Torch::ValueTensorType>(type);
      }))
    return rewriter.notifyMatchFailure(op, "expected 7 to 16 tensor inputs");
  auto getInput = [&](GqaInput input) -> Value {
    unsigned index = static_cast<unsigned>(input);
    if (index >= op->getNumOperands() ||
        isa<Torch::NoneType>(op->getOperand(index).getType()))
      return {};
    return op->getOperand(index);
  };
  info.query = getInput(GqaInput::Query);
  info.key = getInput(GqaInput::Key);
  info.value = getInput(GqaInput::Value);
  info.pastKey = getInput(GqaInput::PastKey);
  info.pastValue = getInput(GqaInput::PastValue);
  info.seqlensK = getInput(GqaInput::SeqlensK);
  info.totalSequenceLength = getInput(GqaInput::TotalSequenceLength);
  info.cosCache = getInput(GqaInput::CosCache);
  info.sinCache = getInput(GqaInput::SinCache);
  info.positionIds = getInput(GqaInput::PositionIds);
  info.attentionBias = getInput(GqaInput::AttentionBias);
  info.headSink = getInput(GqaInput::HeadSink);
  info.kScale = getInput(GqaInput::KScale);
  info.vScale = getInput(GqaInput::VScale);
  if (!info.query || !info.seqlensK || !info.totalSequenceLength)
    return rewriter.notifyMatchFailure(
        op, "expected query, seqlens_k and total_sequence_length");
  if (!info.key != !info.value)
    return rewriter.notifyMatchFailure(
        op, "expected key and value to be both present or both absent");
  if (!info.pastKey != !info.pastValue)
    return rewriter.notifyMatchFailure(
        op, "expected past_key and past_value to be both present or both "
            "absent");
  if (info.doRotary && (!info.cosCache || !info.sinCache))
    return rewriter.notifyMatchFailure(
        op, "expected cos_cache and sin_cache when do_rotary is set");
  if (getInput(GqaInput::QNormWeight) || getInput(GqaInput::KNormWeight))
    return rewriter.notifyMatchFailure(
        op, "unsupported: q_norm_weight and k_norm_weight, which onnxruntime "
            "honors only on CUDA and WebGPU");
  if (bool(info.kScale) != quantized || bool(info.vScale) != quantized)
    return rewriter.notifyMatchFailure(
        op, "expected k_scale and v_scale exactly for a quantized cache");
  for (Value length : {info.seqlensK, info.totalSequenceLength}) {
    auto type = cast<Torch::ValueTensorType>(length.getType());
    if (!type.hasSizes() || !type.hasDtype() ||
        !isa<IntegerType>(type.getDtype()))
      return rewriter.notifyMatchFailure(
          op, "expected integer seqlens_k and total_sequence_length of known "
              "rank");
  }

  auto queryType = cast<Torch::ValueTensorType>(info.query.getType());
  if (!queryType.hasSizes() || queryType.getSizes().size() != 3 ||
      queryType.getSizes()[2] == Torch::kUnknownSize || !queryType.hasDtype() ||
      !isa<Float32Type, Float16Type, BFloat16Type>(queryType.getDtype()))
    return rewriter.notifyMatchFailure(
        op, "expected a float query of shape [batch, sequence, static hidden]");
  info.dtype = queryType.getDtype();
  int64_t batch = queryType.getSizes()[0];
  int64_t seqLen = queryType.getSizes()[1];
  int64_t queryHidden = queryType.getSizes()[2];
  info.packed = !info.key;
  int64_t headsInQuery = info.packed ? numHeads + 2 * kvNumHeads : numHeads;
  if (queryHidden % headsInQuery != 0)
    return rewriter.notifyMatchFailure(
        op, "expected the query hidden size to be a multiple of the heads it "
            "holds");
  info.headSize = queryHidden / headsInQuery;
  int64_t kvSeqLen = seqLen;
  if (!info.packed) {
    for (Value input : {info.key, info.value}) {
      auto type = cast<Torch::ValueTensorType>(input.getType());
      if (!type.hasSizes() || type.getSizes().size() != 3 || !type.hasDtype() ||
          type.getDtype() != info.dtype ||
          type.getSizes()[2] != kvNumHeads * info.headSize)
        return rewriter.notifyMatchFailure(
            op, "expected key and value of the query dtype and shape [batch, "
                "sequence, kv_num_heads * head_size]");
    }
    kvSeqLen = cast<Torch::ValueTensorType>(info.key.getType()).getSizes()[1];
  }
  info.sharedKv = kvSeqLen == 0;
  if (!info.sharedKv && kvSeqLen != Torch::kUnknownSize &&
      seqLen != Torch::kUnknownSize && kvSeqLen != seqLen)
    return rewriter.notifyMatchFailure(
        op, "expected key and query of the same sequence length, or an empty "
            "key");
  if (info.sharedKv && (!info.pastKey || info.windowed))
    return rewriter.notifyMatchFailure(
        op, "expected a past cache and no sliding_window_cache with an empty "
            "key");
  if (bitWidth == 4 && info.headSize % 2 != 0)
    return rewriter.notifyMatchFailure(
        op, "expected an even head size for a 4-bit cache");

  // Outputs are positional too, and omitted ones import as `!torch.none`.
  if (op->getNumResults() < 1 || op->getNumResults() > 4 ||
      !llvm::all_of(op->getResultTypes(), [](Type type) {
        return isa<Torch::NoneType, Torch::ValueTensorType>(type);
      }))
    return rewriter.notifyMatchFailure(op, "expected 1 to 4 tensor outputs");
  auto getResultType = [&](unsigned index) -> Torch::ValueTensorType {
    if (index >= op->getNumResults())
      return {};
    return dyn_cast<Torch::ValueTensorType>(op->getResult(index).getType());
  };
  info.outputType = getResultType(0);
  info.presentKeyType = getResultType(1);
  info.presentValueType = getResultType(2);
  info.qkType = getResultType(3);
  if (!info.outputType ||
      !hasCompatibleShape(info.outputType,
                          {batch, seqLen, numHeads * info.headSize},
                          info.dtype))
    return rewriter.notifyMatchFailure(
        op, "expected an output of the query dtype and shape [batch, "
            "sequence, num_heads * head_size]");
  if (!info.presentKeyType != !info.presentValueType)
    return rewriter.notifyMatchFailure(
        op, "expected present_key and present_value to be both present or "
            "both absent");
  if ((info.qkOutput != 0) != bool(info.qkType))
    return rewriter.notifyMatchFailure(
        op, "expected output_qk exactly when qk_output is set");
  if (info.qkType &&
      !hasCompatibleShape(info.qkType,
                          {batch, numHeads, seqLen, Torch::kUnknownSize},
                          info.dtype))
    return rewriter.notifyMatchFailure(
        op, "expected an output_qk of the query dtype and shape [batch, "
            "num_heads, sequence, total_sequence_length]");
  if ((quantized || info.windowed) && !info.presentKeyType)
    return rewriter.notifyMatchFailure(
        op, "expected present_key and present_value for a quantized or "
            "windowed cache");

  // Quantized caches store int8 values, or two int4 values per ui8.
  Type cacheDtype = info.dtype;
  int64_t storedHeadSize = info.headSize;
  if (quantized) {
    cacheDtype = rewriter.getIntegerType(8, /*isSigned=*/bitWidth == 8);
    storedHeadSize = info.headSize * bitWidth / 8;
  }
  auto isCache = [&](Torch::ValueTensorType type) {
    // Unpacking 4-bit values needs the static packed size.
    return hasCompatibleShape(
               type, {batch, kvNumHeads, Torch::kUnknownSize, storedHeadSize},
               cacheDtype) &&
           (bitWidth != 4 || type.getSizes()[3] != Torch::kUnknownSize);
  };
  auto isCacheValue = [&](Value cache) {
    return isCache(cast<Torch::ValueTensorType>(cache.getType()));
  };
  if ((info.pastKey &&
       (!isCacheValue(info.pastKey) || !isCacheValue(info.pastValue))) ||
      (info.presentKeyType &&
       (!isCache(info.presentKeyType) || !isCache(info.presentValueType))))
    return rewriter.notifyMatchFailure(
        op, "expected past and present caches of shape [batch, kv_num_heads, "
            "sequence, head_size], int8 or packed int4 when quantized");
  // A windowed cache keeps the capacity of the past cache.
  if (info.windowed &&
      (!info.pastKey || info.qkOutput ||
       !hasCompatibleShape(
           info.presentKeyType,
           cast<Torch::ValueTensorType>(info.pastKey.getType()).getSizes(),
           cacheDtype)))
    return rewriter.notifyMatchFailure(
        op, "expected past_key and past_value of the present shape, and no "
            "qk_output, with sliding_window_cache");
  if (quantized) {
    int64_t scaleSize = info.perChannel ? kvNumHeads * info.headSize : 1;
    for (Value scale : {info.kScale, info.vScale}) {
      auto type = cast<Torch::ValueTensorType>(scale.getType());
      std::optional<int64_t> numel = Torch::getTensorNumel(scale);
      if (!type.hasDtype() || !type.getDtype().isF32() || !type.hasSizes() ||
          (numel && *numel != scaleSize))
        return rewriter.notifyMatchFailure(
            op, "expected f32 k_scale and v_scale with one element, or one per "
                "KV head and channel for PER_CHANNEL");
    }
  }
  if (info.attentionBias) {
    auto type = cast<Torch::ValueTensorType>(info.attentionBias.getType());
    if (!type.hasSizes() || type.getSizes().size() != 4 || !type.hasDtype() ||
        type.getDtype() != info.dtype ||
        !llvm::is_contained({int64_t(1), numHeads}, type.getSizes()[1]))
      return rewriter.notifyMatchFailure(
          op, "expected attention_bias of the query dtype and shape [batch or "
              "1, num_heads or 1, sequence, total_sequence_length]");
  }
  if (info.headSink &&
      !hasCompatibleShape(cast<Torch::ValueTensorType>(info.headSink.getType()),
                          {numHeads}, info.dtype))
    return rewriter.notifyMatchFailure(
        op, "expected head_sink of the query dtype and shape [num_heads]");
  if (info.positionIds) {
    auto type = cast<Torch::ValueTensorType>(info.positionIds.getType());
    if (!type.hasSizes() || type.getSizes().size() != 2)
      return rewriter.notifyMatchFailure(
          op, "expected position_ids of shape [batch, sequence]");
  }
  return info;
}

/// Returns attention of `q` over `keys` and `values` with the scores
/// computed explicitly, for what scaled_dot_product_attention cannot express:
/// softcap, a softmax sink and output_qk. `valid` masks the scores and `bias`
/// is added to them when present. Also returns the scores or probabilities
/// that qk_output selects, or null.
static std::pair<Value, Value>
createScoreAttention(ConversionPatternRewriter &rewriter, Location loc,
                     const GqaInfo &info, Value q, Value keys, Value values,
                     Value valid, Value bias) {
  ImplicitLocOpBuilder b(loc, rewriter);
  auto qType = cast<Torch::ValueTensorType>(q.getType());
  int64_t batch = qType.getSizes()[0];
  int64_t seqLen = qType.getSizes()[2];
  int64_t numHeads = info.numHeads;
  int64_t bufferLen =
      cast<Torch::ValueTensorType>(keys.getType()).getSizes()[2];
  Value none = Torch::ConstantNoneOp::create(b);
  Value one = Torch::ConstantIntOp::create(b, 1);
  // Query head h reads KV head h / groups.
  int64_t groups = numHeads / info.kvNumHeads;
  if (groups > 1) {
    auto repeat = [&](Value x) -> Value {
      return Torch::AtenRepeatInterleaveSelfIntOp::create(
          b, withSizes(x, {batch, numHeads, bufferLen, info.headSize}), x,
          Torch::ConstantIntOp::create(b, groups), one, none);
    };
    keys = repeat(keys);
    values = repeat(values);
  }
  auto scoresType = b.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{batch, numHeads, seqLen, bufferLen}, info.dtype);
  Value keysT = Torch::AtenTransposeIntOp::create(
      b, withSizes(keys, {batch, numHeads, info.headSize, bufferLen}), keys,
      Torch::ConstantIntOp::create(b, 2), Torch::ConstantIntOp::create(b, 3));
  Value scores = Torch::AtenMatmulOp::create(b, scoresType, q, keysT);
  double scale = info.scale != 0.0f
                     ? info.scale
                     : 1.0f / std::sqrt(static_cast<float>(info.headSize));
  scores = Torch::AtenMulScalarOp::create(
      b, scoresType, scores, Torch::ConstantFloatOp::create(b, APFloat(scale)));
  if (info.softcap > 0.0f) {
    Value softcap =
        Torch::ConstantFloatOp::create(b, APFloat(double(info.softcap)));
    Value scaled =
        Torch::AtenDivScalarOp::create(b, scoresType, scores, softcap);
    scores = Torch::AtenMulScalarOp::create(
        b, scoresType, Torch::AtenTanhOp::create(b, scoresType, scaled),
        softcap);
  }
  if (bias)
    scores = Torch::AtenAddTensorOp::create(b, scoresType, scores, bias, one);
  // onnxruntime reports masked scores as 0.
  Value qk;
  if (info.qkOutput == 1)
    qk = Torch::AtenWhereScalarOtherOp::create(
        b, scoresType, valid, scores,
        Torch::ConstantFloatOp::create(b, APFloat(0.0)));
  scores = Torch::AtenWhereScalarOtherOp::create(
      b, scoresType, valid, scores,
      Torch::ConstantFloatOp::create(
          b, APFloat::getInf(APFloat::IEEEdouble(), /*Negative=*/true)));

  Value lastDim = Torch::ConstantIntOp::create(b, 3);
  Value probabilities;
  if (info.headSink || info.smoothSoftmax) {
    // A sink adds exp(sink) to the softmax denominator: softmax over the
    // scores plus one column that holds the sink, without that column.
    // smooth_softmax is a zero sink.
    auto sinkType = b.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>{batch, numHeads, seqLen, 1}, info.dtype);
    Value heads = Torch::ConstantIntOp::create(b, numHeads);
    Value sinkShape =
        createIntList(b, {Torch::getTensorDimSize(rewriter, q, 0), heads,
                          Torch::getTensorDimSize(rewriter, q, 2), one});
    Value sink;
    if (info.headSink) {
      Value perHead = Torch::AtenViewOp::create(
          b, withSizes(info.headSink, {1, numHeads, 1, 1}), info.headSink,
          Torch::toIntListConstruct(rewriter, loc, {1, numHeads, 1, 1}));
      sink =
          Torch::AtenExpandOp::create(b, sinkType, perHead, sinkShape,
                                      Torch::ConstantBoolOp::create(b, false));
    } else {
      sink = Torch::AtenZerosOp::create(
          b, sinkType, sinkShape,
          Torch::getDtypeIntValueForType(rewriter, loc, info.dtype), none, none,
          none);
    }
    auto extendedType = b.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>{batch, numHeads, seqLen,
                          bufferLen == Torch::kUnknownSize ? Torch::kUnknownSize
                                                           : bufferLen + 1},
        info.dtype);
    Value tensors = Torch::PrimListConstructOp::create(
        b,
        Torch::ListType::get(
            Torch::ValueTensorType::getWithLeastStaticInformation(
                b.getContext())),
        ValueRange{scores, sink});
    Value extended =
        Torch::AtenCatOp::create(b, extendedType, tensors, lastDim);
    probabilities = Torch::AtenSoftmaxIntOp::create(b, extendedType, extended,
                                                    lastDim, none);
    probabilities = createSlice(
        rewriter, loc, probabilities, 3, Torch::ConstantIntOp::create(b, 0),
        Torch::getTensorDimSize(rewriter, keys, 2), bufferLen);
  } else {
    probabilities =
        Torch::AtenSoftmaxIntOp::create(b, scoresType, scores, lastDim, none);
  }
  if (info.qkOutput == 2)
    qk = probabilities;
  Value attention =
      Torch::AtenMatmulOp::create(b, qType, probabilities, values);
  return {attention, qk};
}

/// Lowers com.microsoft.GroupQueryAttention as the onnxruntime CPU kernel
/// computes it. Attention runs on one KV buffer per side: the present cache,
/// or for sliding_window_cache the resident rows followed by the new ones.
/// The rows of that buffer that hold no token of a batch entry are masked,
/// and the present caches zero them as onnxruntime does.
static LogicalResult
convertGroupQueryAttention(OpBinder binder,
                           ConversionPatternRewriter &rewriter) {
  FailureOr<GqaInfo> matched = matchGroupQueryAttention(binder, rewriter);
  if (failed(matched))
    return failure();
  const GqaInfo &info = *matched;
  Operation *op = binder.op;
  Location loc = binder.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  int64_t numHeads = info.numHeads;
  int64_t kvNumHeads = info.kvNumHeads;
  int64_t headSize = info.headSize;
  int64_t bitWidth = info.kvCacheBitWidth;
  auto queryType = cast<Torch::ValueTensorType>(info.query.getType());
  int64_t batch = queryType.getSizes()[0];
  int64_t seqLen = queryType.getSizes()[1];
  Type si64 = rewriter.getIntegerType(64, /*isSigned=*/true);
  Type i1 = rewriter.getI1Type();
  Value none = Torch::ConstantNoneOp::create(b);
  Value cstFalse = Torch::ConstantBoolOp::create(b, false);
  Value zero = Torch::ConstantIntOp::create(b, 0);
  Value one = Torch::ConstantIntOp::create(b, 1);
  Value batchV = Torch::getTensorDimSize(rewriter, info.query, 0);
  Value seqLenV = Torch::getTensorDimSize(rewriter, info.query, 1);
  auto createConstant = [&](int64_t value) -> Value {
    return Torch::ConstantIntOp::create(b, value);
  };

  Value query = info.query, key = info.key, value = info.value;
  if (info.packed) {
    int64_t queryEnd = numHeads * headSize;
    int64_t keyEnd = queryEnd + kvNumHeads * headSize;
    int64_t valueEnd = keyEnd + kvNumHeads * headSize;
    query = createSlice(rewriter, loc, info.query, 2, zero,
                        createConstant(queryEnd), queryEnd);
    key = createSlice(rewriter, loc, info.query, 2, createConstant(queryEnd),
                      createConstant(keyEnd), keyEnd - queryEnd);
    value = createSlice(rewriter, loc, info.query, 2, createConstant(keyEnd),
                        createConstant(valueEnd), valueEnd - keyEnd);
  }
  Value q = splitHeads(rewriter, loc, query, numHeads, headSize);
  // A shared KV cache adds no rows.
  Value k, v;
  if (!info.sharedKv) {
    k = splitHeads(rewriter, loc, key, kvNumHeads, headSize);
    v = splitHeads(rewriter, loc, value, kvNumHeads, headSize);
  }

  // seqlens_k holds the total length - 1 of each batch entry, possibly with
  // unit dimensions.
  Value seqlens = info.seqlensK;
  auto seqlensType = cast<Torch::ValueTensorType>(seqlens.getType());
  if (seqlensType.getSizes().size() != 1)
    seqlens = Torch::AtenViewOp::create(
        b, withSizes(seqlens, {batch}), seqlens,
        Torch::toIntListConstruct(rewriter, loc, {-1}));
  if (seqlensType.getDtype() != si64)
    seqlens = Torch::convertTensorToDtype(rewriter, loc, seqlens, si64);
  auto perBatchType = cast<Torch::ValueTensorType>(seqlens.getType());
  Value totalLength =
      Torch::AtenAddScalarOp::create(b, perBatchType, seqlens, one, one);
  // Tokens cached before this step. A right-padded first prompt is shorter
  // than the sequence, and onnxruntime starts it at position 0 too.
  Value pastLength = Torch::AtenClampMinOp::create(
      b, perBatchType,
      Torch::AtenSubScalarOp::create(b, perBatchType, totalLength, seqLenV,
                                     one),
      zero);

  if (info.doRotary) {
    auto positionsType = b.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>{batch, seqLen}, si64);
    Value sequencePositions =
        createIndexRange(rewriter, loc, seqLenV, seqLen, 1, 2);
    Value positions;
    if (info.positionIds) {
      // On the first prompt (sequence_length == total_sequence_length),
      // onnxruntime reads only position_ids[0, 0] and counts up from it.
      Value first = createSlice(
          rewriter, loc,
          createSlice(rewriter, loc, info.positionIds, 0, zero, one, 1), 1,
          zero, one, 1);
      Value firstPromptPositions = Torch::AtenAddTensorOp::create(
          b, withSizes(first, {1, seqLen}), first, sequencePositions, one);
      // Later, it reads the first batch * sequence_length positions in
      // order, also when position_ids is wider than the sequence.
      Value laterPositions = info.positionIds;
      if (seqLen == Torch::kUnknownSize ||
          cast<Torch::ValueTensorType>(laterPositions.getType())
                  .getSizes()[1] != seqLen) {
        Value flat = Torch::AtenFlattenUsingIntsOp::create(
            b, withSizes(laterPositions, {Torch::kUnknownSize}), laterPositions,
            zero, one);
        Value count =
            rewriter.createOrFold<Torch::AtenMulIntOp>(loc, batchV, seqLenV);
        flat = createSlice(rewriter, loc, flat, 0, zero, count,
                           Torch::kUnknownSize);
        laterPositions = Torch::AtenViewOp::create(
            b, positionsType, flat, createIntList(b, {batchV, seqLenV}));
      }
      auto totalType =
          cast<Torch::ValueTensorType>(info.totalSequenceLength.getType());
      Value isFirstPrompt = Torch::AtenEqScalarOp::create(
          b, totalType.getWithSizesAndDtype(totalType.getSizes(), i1),
          info.totalSequenceLength, seqLenV);
      positions =
          Torch::AtenWhereSelfOp::create(b, positionsType, isFirstPrompt,
                                         firstPromptPositions, laterPositions);
    } else {
      positions = Torch::AtenAddTensorOp::create(
          b, positionsType, viewAlongDim(rewriter, loc, pastLength, 0, 2),
          sequencePositions, one);
    }
    // A cache narrower than head_size / 2 rotates only the first
    // 2 * cos_cache.dim(1) values of each head.
    int64_t rotaryDim = 0;
    auto cosCacheType = cast<Torch::ValueTensorType>(info.cosCache.getType());
    if (cosCacheType.hasSizes() && cosCacheType.getSizes().size() == 2 &&
        cosCacheType.getSizes()[1] != Torch::kUnknownSize)
      rotaryDim = 2 * cosCacheType.getSizes()[1];
    Value rotaryScale = Torch::ConstantFloatOp::create(b, APFloat(1.0));
    auto rotate = [&](Value x, int64_t heads) -> Value {
      return Torch::OnnxVariantRotaryEmbeddingOp::create(
          b, x.getType(), x, positions, info.cosCache, info.sinCache,
          createConstant(info.rotaryInterleaved), /*is_packed_batching=*/zero,
          createConstant(heads), createConstant(rotaryDim), rotaryScale);
    };
    q = rotate(q, numHeads);
    if (k)
      k = rotate(k, kvNumHeads);
  }

  // Quantized caches are built from int8 values, or from 4-bit values held
  // one per ui8 and packed only for the present outputs.
  Value kScale, vScale;
  if (bitWidth) {
    SmallVector<int64_t> scaleShape{1, info.perChannel ? kvNumHeads : 1, 1,
                                    info.perChannel ? headSize : 1};
    auto viewScale = [&](Value scale) -> Value {
      return Torch::AtenViewOp::create(
          b, withSizes(scale, scaleShape), scale,
          Torch::toIntListConstruct(rewriter, loc, scaleShape));
    };
    kScale = viewScale(info.kScale);
    vScale = viewScale(info.vScale);
    if (k) {
      k = quantizeKvCache(rewriter, loc, k, kScale, bitWidth);
      v = quantizeKvCache(rewriter, loc, v, vScale, bitWidth);
    }
  }
  Type bufferDtype =
      bitWidth ? rewriter.getIntegerType(8, /*isSigned=*/bitWidth == 8)
               : info.dtype;
  Value fill =
      bitWidth ? zero : Value(Torch::ConstantFloatOp::create(b, APFloat(0.0)));

  // Attention reads one KV buffer per side. Per batch entry, rows
  // [0, attentionPast) precede the first query, and the rows from
  // attentionTotal on hold no token.
  Value keyBuffer = k, valueBuffer = v;
  int64_t bufferLen = seqLen;
  Value bufferLenV = seqLenV;
  Value attentionPast = pastLength;
  Value attentionTotal = totalLength;
  // For a windowed cache: the absolute position of buffer row 0, the buffer
  // row of each present row, and whether that present row holds a token.
  Value bufferStart, compactIndex, isResident;
  if (info.pastKey) {
    int64_t pastLen =
        cast<Torch::ValueTensorType>(info.pastKey.getType()).getSizes()[2];
    Value pastLenV = Torch::getTensorDimSize(rewriter, info.pastKey, 2);
    // The new rows, and the rows kept from the past cache. A key of dynamic
    // length may hold no tokens at runtime, which onnxruntime reads as a
    // shared KV cache: then the past keeps all total length rows.
    int64_t newRows = seqLen;
    Value newRowsV = seqLenV;
    Value keptRows = info.sharedKv ? totalLength : pastLength;
    if (k && !info.packed &&
        cast<Torch::ValueTensorType>(info.key.getType()).getSizes()[1] ==
            Torch::kUnknownSize) {
      newRows = Torch::kUnknownSize;
      newRowsV = Torch::getTensorDimSize(rewriter, info.key, 1);
      keptRows = Torch::AtenClampMinOp::create(
          b, perBatchType,
          Torch::AtenSubScalarOp::create(b, perBatchType, totalLength, newRowsV,
                                         one),
          zero);
    }
    Value writeStart = pastLength;
    Value padRows;
    if (info.windowed) {
      // The resident rows hold the latest positions and the new rows follow
      // them. The present cache drops whole blocks of the oldest rows.
      Value blockSizeMinusOne = rewriter.createOrFold<Torch::AtenSubIntOp>(
          loc, pastLenV, createConstant(info.localWindowSize));
      Value blockSize = rewriter.createOrFold<Torch::AtenAddIntOp>(
          loc, blockSizeMinusOne, one);
      Value residentBefore = getWindowedCacheLength(
          rewriter, loc, pastLength, pastLenV, blockSize, blockSizeMinusOne);
      Value residentAfter =
          getWindowedCacheLength(rewriter, loc,
                                 Torch::AtenAddScalarOp::create(
                                     b, perBatchType, pastLength, seqLenV, one),
                                 pastLenV, blockSize, blockSizeMinusOne);
      keptRows = residentBefore;
      writeStart = residentBefore;
      attentionPast = residentBefore;
      attentionTotal = Torch::AtenAddScalarOp::create(
          b, perBatchType, residentBefore, seqLenV, one);
      bufferStart = Torch::AtenSubTensorOp::create(b, perBatchType, pastLength,
                                                   residentBefore, one);
      padRows = seqLenV;
      bufferLen =
          pastLen == Torch::kUnknownSize || seqLen == Torch::kUnknownSize
              ? Torch::kUnknownSize
              : pastLen + seqLen;
      bufferLenV =
          rewriter.createOrFold<Torch::AtenAddIntOp>(loc, pastLenV, seqLenV);
      // Present row r is buffer row r + dropped. The rows past the residents
      // read a clamped row and are zeroed.
      Value dropped = Torch::AtenSubTensorOp::create(
          b, perBatchType, attentionTotal, residentAfter, one);
      Value presentRows =
          createIndexRange(rewriter, loc, pastLenV, pastLen, 2, 4);
      auto rowsType = b.getType<Torch::ValueTensorType>(
          ArrayRef<int64_t>{batch, 1, pastLen, 1}, si64);
      Value sourceRows = Torch::AtenAddTensorOp::create(
          b, rowsType, presentRows, viewAlongDim(rewriter, loc, dropped, 0, 4),
          one);
      Value lastRow =
          rewriter.createOrFold<Torch::AtenSubIntOp>(loc, bufferLenV, one);
      sourceRows =
          Torch::AtenClampOp::create(b, rowsType, sourceRows, none, lastRow);
      compactIndex = Torch::AtenExpandOp::create(
          b, withSizes(sourceRows, {batch, kvNumHeads, pastLen, headSize}),
          sourceRows,
          createIntList(b, {batchV, createConstant(kvNumHeads), pastLenV,
                            createConstant(headSize)}),
          cstFalse);
      isResident = Torch::AtenLtTensorOp::create(
          b,
          b.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>{batch, 1, pastLen, 1}, i1),
          presentRows, viewAlongDim(rewriter, loc, residentAfter, 0, 4));
    } else {
      // The present cache holds max(total_sequence_length, past length)
      // rows: a growing cache appends, a fixed-capacity one keeps its size.
      int64_t presentLen = info.presentKeyType
                               ? info.presentKeyType.getSizes()[2]
                               : Torch::kUnknownSize;
      bufferLen = presentLen;
      if (presentLen != Torch::kUnknownSize) {
        bufferLenV = createConstant(presentLen);
      } else {
        Value total = Torch::AtenItemOp::create(b, b.getType<Torch::IntType>(),
                                                info.totalSequenceLength);
        bufferLenV = Torch::PrimMaxIntOp::create(b, total, pastLenV);
      }
      if (pastLen != presentLen || presentLen == Torch::kUnknownSize)
        padRows = rewriter.createOrFold<Torch::AtenSubIntOp>(loc, bufferLenV,
                                                             pastLenV);
    }
    // onnxruntime zeroes the rows that it neither copies from the past nor
    // writes.
    Value rows = createIndexRange(rewriter, loc, bufferLenV, bufferLen, 2, 4);
    Value isKept = Torch::AtenLtTensorOp::create(
        b,
        b.getType<Torch::ValueTensorType>(
            ArrayRef<int64_t>{batch, 1, bufferLen, 1}, i1),
        rows, viewAlongDim(rewriter, loc, keptRows, 0, 4));
    Value writeIndex;
    if (k) {
      Value writeRows = Torch::AtenAddTensorOp::create(
          b,
          b.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>{batch, 1, newRows, 1}, si64),
          viewAlongDim(rewriter, loc, writeStart, 0, 4),
          createIndexRange(rewriter, loc, newRowsV, newRows, 2, 4), one);
      writeIndex = Torch::AtenExpandOp::create(
          b, withSizes(writeRows, {batch, kvNumHeads, newRows, headSize}),
          writeRows,
          createIntList(b, {batchV, createConstant(kvNumHeads), newRowsV,
                            createConstant(headSize)}),
          cstFalse);
    }
    auto bufferType = b.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>{batch, kvNumHeads, bufferLen, headSize}, bufferDtype);
    auto buildBuffer = [&](Value past, Value current) -> Value {
      Value buffer =
          bitWidth == 4 ? unpackLastDimension(rewriter, loc, past, 4) : past;
      if (padRows)
        buffer = Torch::AtenConstantPadNdOp::create(
            b, bufferType, buffer,
            createIntList(b, {zero, zero, zero, padRows}), fill);
      buffer = Torch::AtenWhereScalarOtherOp::create(b, bufferType, isKept,
                                                     buffer, fill);
      if (current)
        buffer = Torch::AtenScatterSrcOp::create(
            b, bufferType, buffer, createConstant(2), writeIndex, current);
      return buffer;
    };
    keyBuffer = buildBuffer(info.pastKey, k);
    valueBuffer = buildBuffer(info.pastValue, v);
  }

  // The present caches, in the declared types.
  auto getPresent = [&](Value buffer, Torch::ValueTensorType type) -> Value {
    Value present = buffer;
    if (compactIndex) {
      Torch::ValueTensorType compactType = withSizes(
          buffer,
          cast<Torch::ValueTensorType>(compactIndex.getType()).getSizes());
      present = Torch::AtenGatherOp::create(
          b, compactType, buffer, createConstant(2), compactIndex, cstFalse);
      present = Torch::AtenWhereScalarOtherOp::create(
          b, compactType, isResident, present, fill);
    }
    if (bitWidth == 4)
      present = packInt4LastDimension(rewriter, loc, present);
    if (present.getType() != type)
      present = Torch::TensorStaticInfoCastOp::create(b, type, present);
    return present;
  };
  Value presentKey, presentValue;
  if (info.presentKeyType) {
    presentKey = getPresent(keyBuffer, info.presentKeyType);
    presentValue = getPresent(valueBuffer, info.presentValueType);
  }
  Value attentionKey = keyBuffer, attentionValue = valueBuffer;
  if (bitWidth) {
    attentionKey = dequantizeKvCache(rewriter, loc, keyBuffer, kScale, bitWidth,
                                     info.dtype);
    attentionValue = dequantizeKvCache(rewriter, loc, valueBuffer, vScale,
                                       bitWidth, info.dtype);
  }

  // valid[b, 0, i, j]: whether query i of batch entry b reads buffer row j.
  Value columns = createIndexRange(rewriter, loc, bufferLenV, bufferLen, 3, 4);
  Value visibleEnd = viewAlongDim(rewriter, loc, attentionTotal, 0, 4);
  int64_t maskQueries = 1;
  if (info.causal) {
    maskQueries = seqLen;
    Value queryRows = createIndexRange(rewriter, loc, seqLenV, seqLen, 2, 4);
    auto endType = b.getType<Torch::ValueTensorType>(
        ArrayRef<int64_t>{batch, 1, seqLen, 1}, si64);
    Value causalEnd = Torch::AtenAddTensorOp::create(
        b, endType, viewAlongDim(rewriter, loc, attentionPast, 0, 4),
        Torch::AtenAddScalarOp::create(b, queryRows.getType(), queryRows, one,
                                       one),
        one);
    visibleEnd =
        Torch::AtenMinimumOp::create(b, endType, causalEnd, visibleEnd);
  }
  auto maskType = b.getType<Torch::ValueTensorType>(
      ArrayRef<int64_t>{batch, 1, maskQueries, bufferLen}, i1);
  Value valid = Torch::AtenLtTensorOp::create(b, maskType, columns, visibleEnd);
  if (info.localWindowSize) {
    Value windowStart = Torch::AtenSubScalarOp::create(
        b, visibleEnd.getType(), visibleEnd,
        createConstant(info.localWindowSize), one);
    valid = Torch::AtenLogicalAndOp::create(
        b, maskType, valid,
        Torch::AtenGeTensorOp::create(b, maskType, columns, windowStart));
  }

  // attention_bias over the buffer rows. Its columns are absolute positions,
  // so a windowed buffer reads them from the position of its row 0.
  Value bias = info.attentionBias;
  if (bias) {
    auto biasType = cast<Torch::ValueTensorType>(bias.getType());
    int64_t biasHeads = biasType.getSizes()[1];
    int64_t biasLen = biasType.getSizes()[3];
    Value biasLenV = Torch::getTensorDimSize(rewriter, bias, 3);
    if (bufferStart) {
      Value biasColumns = Torch::AtenAddTensorOp::create(
          b,
          b.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>{batch, 1, 1, bufferLen}, si64),
          viewAlongDim(rewriter, loc, bufferStart, 0, 4), columns, one);
      biasColumns = Torch::AtenClampOp::create(
          b, biasColumns.getType(), biasColumns, zero,
          rewriter.createOrFold<Torch::AtenSubIntOp>(loc, biasLenV, one));
      Value heads = createConstant(biasHeads);
      Value biasRows = Torch::AtenExpandOp::create(
          b, withSizes(bias, {batch, biasHeads, seqLen, biasLen}), bias,
          createIntList(b, {batchV, heads, seqLenV, biasLenV}), cstFalse);
      Value gatherIndex = Torch::AtenExpandOp::create(
          b, withSizes(biasColumns, {batch, biasHeads, seqLen, bufferLen}),
          biasColumns, createIntList(b, {batchV, heads, seqLenV, bufferLenV}),
          cstFalse);
      bias = Torch::AtenGatherOp::create(
          b, withSizes(bias, {batch, biasHeads, seqLen, bufferLen}), biasRows,
          createConstant(3), gatherIndex, cstFalse);
    } else if (biasLen != bufferLen || bufferLen == Torch::kUnknownSize) {
      // The buffer rows past total_sequence_length hold no token and are
      // masked.
      Value padColumns =
          rewriter.createOrFold<Torch::AtenSubIntOp>(loc, bufferLenV, biasLenV);
      bias = Torch::AtenConstantPadNdOp::create(
          b,
          withSizes(bias,
                    {biasType.getSizes()[0], biasHeads, seqLen, bufferLen}),
          bias, createIntList(b, {zero, padColumns}),
          Torch::ConstantFloatOp::create(b, APFloat(0.0)));
    }
  }

  Value attention, qk;
  if (info.softcap > 0.0f || info.headSink || info.smoothSoftmax ||
      info.qkOutput) {
    std::tie(attention, qk) = createScoreAttention(
        rewriter, loc, info, q, attentionKey, attentionValue, valid, bias);
  } else {
    Value mask = valid;
    if (bias) {
      auto biasType = cast<Torch::ValueTensorType>(bias.getType());
      mask = Torch::AtenWhereScalarOtherOp::create(
          b,
          b.getType<Torch::ValueTensorType>(
              ArrayRef<int64_t>{batch, biasType.getSizes()[1], seqLen,
                                bufferLen},
              info.dtype),
          valid, bias,
          Torch::ConstantFloatOp::create(
              b, APFloat::getInf(APFloat::IEEEdouble(), /*Negative=*/true)));
    }
    Value scale = info.scale != 0.0f ? Value(Torch::ConstantFloatOp::create(
                                           b, APFloat(double(info.scale))))
                                     : none;
    attention = Torch::AtenScaledDotProductAttentionOp::create(
        b, q.getType(), q, attentionKey, attentionValue, mask,
        /*dropout_p=*/Torch::ConstantFloatOp::create(b, APFloat(0.0)),
        /*is_causal=*/cstFalse, scale,
        /*enable_gqa=*/Torch::ConstantBoolOp::create(b, true));
  }

  Value output = mergeHeads(rewriter, loc, attention);
  if (output.getType() != info.outputType)
    output = Torch::TensorStaticInfoCastOp::create(b, info.outputType, output);
  SmallVector<Value> results{output};
  if (op->getNumResults() > 1) {
    results.push_back(presentKey ? presentKey : none);
    results.push_back(presentValue ? presentValue : none);
  }
  if (qk) {
    // output_qk covers the first total_sequence_length rows of the buffer.
    int64_t qkLen = info.qkType.getSizes()[3];
    if (qkLen != bufferLen || qkLen == Torch::kUnknownSize) {
      Value qkLenV =
          qkLen != Torch::kUnknownSize
              ? createConstant(qkLen)
              : Torch::AtenItemOp::create(b, b.getType<Torch::IntType>(),
                                          info.totalSequenceLength)
                    .getResult();
      qk = createSlice(rewriter, loc, qk, 3, zero, qkLenV, qkLen);
    }
    if (qk.getType() != info.qkType)
      qk = Torch::TensorStaticInfoCastOp::create(b, info.qkType, qk);
    results.push_back(qk);
  } else if (op->getNumResults() > 3) {
    results.push_back(none);
  }
  rewriter.replaceOp(op, results);
  return success();
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
      "QuickGelu", 1, [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Value input;
        float alpha;
        Torch::ValueTensorType resultType;
        if (binder.tensorOperand(input) ||
            binder.f32FloatAttr(alpha, "alpha", 1.702f) ||
            binder.tensorResultType(resultType))
          return rewriter.notifyMatchFailure(
              binder.op, "expected one tensor input and a float alpha");
        if (!isa_and_present<Float16Type, BFloat16Type, Float32Type,
                             Float64Type>(resultType.getOptionalDtype()))
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected floating-point result");

        // Computes x * sigmoid(alpha * x), in f32 for f16 and bf16.
        Location loc = binder.getLoc();
        Type computeType = resultType;
        if (isa<Float16Type, BFloat16Type>(resultType.getDtype())) {
          Type f32Type = rewriter.getF32Type();
          input = Torch::convertTensorToDtype(rewriter, loc, input, f32Type);
          computeType = resultType.getWithSizesAndDtype(
              resultType.getOptionalSizes(), f32Type);
        }
        Value cstAlpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(alpha));
        Value scaled = Torch::AtenMulScalarOp::create(
            rewriter, loc, computeType, input, cstAlpha);
        Value sigmoid =
            Torch::AtenSigmoidOp::create(rewriter, loc, computeType, scaled);
        Value result = Torch::AtenMulTensorOp::create(
            rewriter, loc, computeType, input, sigmoid);
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

  patterns.onOp("GroupQueryAttention", 1, convertGroupQueryAttention);
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
}
