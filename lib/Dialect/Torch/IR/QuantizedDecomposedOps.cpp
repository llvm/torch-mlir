//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// Verifiers for the per-tensor, per-channel and per-channel-group PT2E
// quantized_decomposed ops.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

using torch_upstream::ScalarType;

/// The dtypes PT2E quantizes to.
static constexpr ScalarType kStorageDtypes[] = {
    ScalarType::Byte,         ScalarType::Char,   ScalarType::Short,
    ScalarType::Int,          ScalarType::UInt16, ScalarType::Float8_e5m2,
    ScalarType::Float8_e4m3fn};

/// Returns the element type of the storage dtype `dtype`, or a null type if
/// `dtype` is not a storage dtype.
static Type getStorageType(MLIRContext *context, int64_t dtype) {
  switch (dtype) {
  case static_cast<int64_t>(ScalarType::Byte):
    return IntegerType::get(context, 8, IntegerType::Unsigned);
  case static_cast<int64_t>(ScalarType::Char):
    return IntegerType::get(context, 8, IntegerType::Signed);
  case static_cast<int64_t>(ScalarType::Short):
    return IntegerType::get(context, 16, IntegerType::Signed);
  case static_cast<int64_t>(ScalarType::Int):
    return IntegerType::get(context, 32, IntegerType::Signed);
  case static_cast<int64_t>(ScalarType::UInt16):
    return IntegerType::get(context, 16, IntegerType::Unsigned);
  case static_cast<int64_t>(ScalarType::Float8_e5m2):
    return Float8E5M2Type::get(context);
  case static_cast<int64_t>(ScalarType::Float8_e4m3fn):
    return Float8E4M3FNType::get(context);
  default:
    return {};
  }
}

/// Returns true if `type` is a storage element type.
static bool isStorageType(Type type) {
  return llvm::any_of(kStorageDtypes, [&](ScalarType dtype) {
    return getStorageType(type.getContext(), static_cast<int64_t>(dtype)) ==
           type;
  });
}

/// Returns the inclusive integer range of the storage type `storageType`.
static std::pair<int64_t, int64_t> getStorageRange(Type storageType) {
  assert(isStorageType(storageType) && "expected a PT2E storage type");
  if (auto integerType = dyn_cast<IntegerType>(storageType)) {
    unsigned width = integerType.getWidth();
    if (integerType.isUnsigned())
      return {0, static_cast<int64_t>(llvm::maxUIntN(width))};
    return {llvm::minIntN(width), llvm::maxIntN(width)};
  }
  const llvm::fltSemantics &semantics =
      cast<mlir::FloatType>(storageType).getFloatSemantics();
  auto largest =
      static_cast<int64_t>(APFloat::getLargest(semantics).convertToDouble());
  return {-largest, largest};
}

/// Returns the value of `value` if it is a constant int.
static std::optional<int64_t> matchConstantInt(Value value) {
  int64_t intValue;
  if (matchPattern(value, m_TorchConstantInt(&intValue)))
    return intValue;
  return std::nullopt;
}

/// Verifies that `bound`, if set, is representable in `storageType`.
static LogicalResult verifyBound(Operation *op, StringRef name,
                                 std::optional<int64_t> bound,
                                 Type storageType) {
  if (!bound)
    return success();
  auto [storageMin, storageMax] = getStorageRange(storageType);
  if (*bound >= storageMin && *bound <= storageMax)
    return success();
  return op->emitOpError("expected ") << name << " to be representable in "
                                      << storageType << ", but got " << *bound;
}

/// Verifies that `type`, named `name` in diagnostics, has a floating-point
/// dtype if its dtype is known.
static LogicalResult verifyFloatDtype(Operation *op, StringRef name,
                                      BaseTensorType type) {
  if (type.hasDtype() && !isa<mlir::FloatType>(type.getDtype()))
    return op->emitOpError("expected ")
           << name << " to have a floating-point dtype, but got "
           << type.getDtype();
  return success();
}

/// Verifies the dtypes and the quant_min/quant_max bounds of `op`. `realType`
/// and `storageType` are the types of its floating-point and quantized tensors,
/// which diagnostics call `realName` and `storageName`.
static LogicalResult verifyQuantization(Operation *op, BaseTensorType realType,
                                        StringRef realName,
                                        BaseTensorType storageType,
                                        StringRef storageName, Value dtype,
                                        Value quantMin, Value quantMax) {
  if (failed(verifyFloatDtype(op, realName, realType)))
    return failure();

  Type storageElementType = storageType.getOptionalDtype();
  if (storageElementType && !isStorageType(storageElementType))
    return op->emitOpError("expected ")
           << storageName << " to have a PT2E storage dtype, but got "
           << storageElementType;

  int64_t dtypeCode;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeCode))) {
    Type expectedStorageType = getStorageType(op->getContext(), dtypeCode);
    if (!expectedStorageType)
      return op->emitOpError("expected dtype to be a PT2E storage dtype, but "
                             "got dtype code ")
             << dtypeCode;
    if (storageElementType && storageElementType != expectedStorageType)
      return op->emitOpError("expected dtype code ")
             << dtypeCode << " (" << expectedStorageType << ") to match the "
             << storageName << " dtype " << storageElementType;
    storageElementType = expectedStorageType;
  }

  std::optional<int64_t> quantMinValue = matchConstantInt(quantMin);
  std::optional<int64_t> quantMaxValue = matchConstantInt(quantMax);
  if (quantMinValue && quantMaxValue && *quantMinValue > *quantMaxValue)
    return op->emitOpError("expected quant_min to not exceed quant_max, but "
                           "got [")
           << *quantMinValue << ", " << *quantMaxValue << "]";
  if (!storageElementType)
    return success();
  if (failed(verifyBound(op, "quant_min", quantMinValue, storageElementType)))
    return failure();
  return verifyBound(op, "quant_max", quantMaxValue, storageElementType);
}

/// Verifies that `outDtype` is None or a floating-point dtype code and that
/// `resultType` has the dtype it selects (f32 for None).
static LogicalResult verifyOutDtype(Operation *op, BaseTensorType resultType,
                                    Value outDtype) {
  Type expected;
  int64_t outDtypeCode;
  if (isa<Torch::NoneType>(outDtype.getType())) {
    expected = Float32Type::get(op->getContext());
  } else if (matchPattern(outDtype, m_TorchConstantInt(&outDtypeCode))) {
    switch (outDtypeCode) {
    case static_cast<int64_t>(ScalarType::Half):
    case static_cast<int64_t>(ScalarType::Float):
    case static_cast<int64_t>(ScalarType::Double):
    case static_cast<int64_t>(ScalarType::BFloat16):
      expected = *getTypeForScalarType(op->getContext(),
                                       static_cast<ScalarType>(outDtypeCode));
      break;
    default:
      return op->emitOpError("expected out_dtype to be a floating-point "
                             "dtype, but got dtype code ")
             << outDtypeCode;
    }
  }
  if (expected && resultType.hasDtype() && resultType.getDtype() != expected)
    return op->emitOpError("expected the result dtype to be ")
           << expected << ", but got " << resultType.getDtype();
  return success();
}

/// Verifies the dtypes and bounds of the quantize op `op`.
template <typename OpTy> static LogicalResult verifyQuantizeOp(OpTy op) {
  return verifyQuantization(op, cast<BaseTensorType>(op.getInput().getType()),
                            "input", cast<BaseTensorType>(op.getType()),
                            "result", op.getDtype(), op.getQuantMin(),
                            op.getQuantMax());
}

/// Verifies the dtypes and bounds of the dequantize op `op`.
template <typename OpTy> static LogicalResult verifyDequantizeOp(OpTy op) {
  auto resultType = cast<BaseTensorType>(op.getType());
  if (failed(verifyQuantization(op, resultType, "result",
                                cast<BaseTensorType>(op.getInput().getType()),
                                "input", op.getDtype(), op.getQuantMin(),
                                op.getQuantMax())))
    return failure();
  return verifyOutDtype(op, resultType, op.getOutDtype());
}

/// Verifies that the scales and zero points of `op` have one element per index
/// of the input dimension `axis`.
template <typename OpTy>
static LogicalResult verifyPerChannelParameters(OpTy op) {
  auto scalesType = cast<BaseTensorType>(op.getScales().getType());
  if (failed(verifyFloatDtype(op, "scales", scalesType)))
    return failure();
  int64_t scalesLength = kUnknownSize;
  if (scalesType.hasSizes()) {
    if (scalesType.getSizes().size() != 1)
      return op.emitOpError("expected scales to have rank 1, but got rank ")
             << scalesType.getSizes().size();
    scalesLength = scalesType.getSizes()[0];
  }

  auto zeroPointsType = dyn_cast<BaseTensorType>(op.getZeroPoints().getType());
  if (zeroPointsType && zeroPointsType.hasSizes()) {
    ArrayRef<int64_t> sizes = zeroPointsType.getSizes();
    if (sizes.size() != 1)
      return op.emitOpError(
                 "expected zero_points to have rank 1, but got rank ")
             << sizes.size();
    if (sizes[0] != kUnknownSize && scalesLength != kUnknownSize &&
        sizes[0] != scalesLength)
      return op.emitOpError("expected zero_points to have the length of "
                            "scales, but got ")
             << sizes[0] << " and " << scalesLength;
  }

  auto inputType = cast<BaseTensorType>(op.getInput().getType());
  int64_t axisValue;
  if (!inputType.hasSizes() ||
      !matchPattern(op.getAxis(), m_TorchConstantInt(&axisValue)))
    return success();
  ArrayRef<int64_t> inputSizes = inputType.getSizes();
  int64_t rank = inputSizes.size();
  int64_t channelDim = toPositiveDim(axisValue, rank);
  if (!isValidDim(channelDim, rank))
    return op.emitOpError("expected axis to be in [")
           << -rank << ", " << rank << "), but got " << axisValue;
  int64_t channels = inputSizes[channelDim];
  if (channels != kUnknownSize && scalesLength != kUnknownSize &&
      channels != scalesLength)
    return op.emitOpError("expected scales to have one element per index of "
                          "input dimension ")
           << channelDim << " (" << channels << "), but got " << scalesLength;
  return success();
}

LogicalResult QuantizedDecomposedQuantizePerTensorOp::verify() {
  return verifyQuantizeOp(*this);
}

LogicalResult QuantizedDecomposedDequantizePerTensorOp::verify() {
  return verifyDequantizeOp(*this);
}

LogicalResult QuantizedDecomposedQuantizePerChannelOp::verify() {
  if (failed(verifyQuantizeOp(*this)))
    return failure();
  return verifyPerChannelParameters(*this);
}

LogicalResult QuantizedDecomposedDequantizePerChannelOp::verify() {
  if (failed(verifyDequantizeOp(*this)))
    return failure();
  return verifyPerChannelParameters(*this);
}

/// Implements `getEffectiveGroupSize` of the per-channel-group ops.
static int64_t computeEffectiveGroupSize(Value input, Value scales,
                                         int64_t groupSize) {
  auto inputType = cast<BaseTensorType>(input.getType());
  auto scalesType = cast<BaseTensorType>(scales.getType());
  if (!inputType.hasSizes() || !scalesType.hasSizes() ||
      inputType.getSizes().empty() || scalesType.getSizes().empty())
    return groupSize;
  int64_t rowLength = inputType.getSizes().back();
  if (rowLength != kUnknownSize && groupSize > rowLength &&
      scalesType.getSizes().back() == 1)
    return rowLength;
  return groupSize;
}

int64_t QuantizedDecomposedQuantizePerChannelGroupOp::getEffectiveGroupSize(
    int64_t groupSize) {
  return computeEffectiveGroupSize(getInput(), getScales(), groupSize);
}

int64_t QuantizedDecomposedDequantizePerChannelGroupOp::getEffectiveGroupSize(
    int64_t groupSize) {
  return computeEffectiveGroupSize(getInput(), getScales(), groupSize);
}

/// Verifies that the scales and zero points of `op` have one element per group
/// of `group_size` elements in each input row.
template <typename OpTy>
static LogicalResult verifyPerChannelGroupParameters(OpTy op) {
  auto inputType = cast<BaseTensorType>(op.getInput().getType());
  auto scalesType = cast<BaseTensorType>(op.getScales().getType());
  if (inputType.hasSizes() && inputType.getSizes().size() != 2)
    return op.emitOpError("expected input to have rank 2, but got rank ")
           << inputType.getSizes().size();
  if (failed(verifyFloatDtype(op, "scales", scalesType)))
    return failure();
  if (scalesType.hasSizes() && scalesType.getSizes().size() != 2)
    return op.emitOpError("expected scales to have rank 2, but got rank ")
           << scalesType.getSizes().size();

  if (auto zeroPointsType =
          dyn_cast<BaseTensorType>(op.getZeroPoints().getType())) {
    if (zeroPointsType.hasSizes() && scalesType.hasSizes()) {
      ArrayRef<int64_t> zeroPointsSizes = zeroPointsType.getSizes();
      ArrayRef<int64_t> scalesSizes = scalesType.getSizes();
      bool compatible =
          zeroPointsSizes.size() == scalesSizes.size() &&
          llvm::all_of(llvm::zip_equal(zeroPointsSizes, scalesSizes),
                       [](auto sizes) {
                         auto [zeroPointsSize, scalesSize] = sizes;
                         return zeroPointsSize == kUnknownSize ||
                                scalesSize == kUnknownSize ||
                                zeroPointsSize == scalesSize;
                       });
      if (!compatible)
        return op.emitOpError("expected zero_points to have the shape of "
                              "scales, but got [")
               << zeroPointsSizes << "] and [" << scalesSizes << "]";
    }
  }

  int64_t groupSize;
  if (!matchPattern(op.getGroupSize(), m_TorchConstantInt(&groupSize)))
    return success();
  if (groupSize <= 1)
    return op.emitOpError("expected group_size to be greater than 1, but got ")
           << groupSize;
  if (!inputType.hasSizes() || !scalesType.hasSizes())
    return success();

  int64_t rows = inputType.getSizes()[0];
  int64_t rowLength = inputType.getSizes()[1];
  int64_t scaleRows = scalesType.getSizes()[0];
  int64_t groups = scalesType.getSizes()[1];
  if (rows != kUnknownSize && scaleRows != kUnknownSize && rows != scaleRows)
    return op.emitOpError("expected scales to have one row per input row ")
           << "(" << rows << "), but got " << scaleRows;
  if (rowLength == kUnknownSize)
    return success();
  int64_t effectiveGroupSize = op.getEffectiveGroupSize(groupSize);
  // Accept a group size beyond the row length when the scale columns are
  // dynamic.
  if (effectiveGroupSize > rowLength && groups == kUnknownSize)
    return success();
  if (rowLength % effectiveGroupSize != 0)
    return op.emitOpError("expected input dimension 1 (")
           << rowLength << ") to be divisible by group_size "
           << effectiveGroupSize;
  if (groups != kUnknownSize && groups != rowLength / effectiveGroupSize)
    return op.emitOpError("expected scales to have ")
           << rowLength / effectiveGroupSize
           << " columns, one per group, but got " << groups;
  return success();
}

LogicalResult QuantizedDecomposedQuantizePerChannelGroupOp::verify() {
  if (failed(verifyQuantizeOp(*this)))
    return failure();
  return verifyPerChannelGroupParameters(*this);
}

LogicalResult QuantizedDecomposedDequantizePerChannelGroupOp::verify() {
  if (failed(verifyDequantizeOp(*this)))
    return failure();
  return verifyPerChannelGroupParameters(*this);
}
