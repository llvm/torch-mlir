//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// Verifiers for the PT2E quantized_decomposed ops. Each check mirrors an
// assertion of the reference implementation in
// torch/ao/quantization/fx/_decomposed.py and applies only when the operands
// carry the information statically.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

using torch_upstream::ScalarType;

// The dtype codes in _DTYPE_TO_QVALUE_BOUNDS, i.e. the dtypes PT2E quantizes
// to.
static constexpr ScalarType kStorageDtypes[] = {
    ScalarType::Byte,         ScalarType::Char,   ScalarType::Short,
    ScalarType::Int,          ScalarType::UInt16, ScalarType::Float8_e5m2,
    ScalarType::Float8_e4m3fn};

// Returns the tensor element type of the storage dtype `dtype`, or a null type
// if PT2E does not quantize to `dtype`.
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

static bool isStorageType(Type type) {
  return llvm::any_of(kStorageDtypes, [&](ScalarType dtype) {
    return getStorageType(type.getContext(), static_cast<int64_t>(dtype)) ==
           type;
  });
}

// Returns the inclusive range of `storageType`, a PT2E storage type. Float8
// bounds are truncated to integers, as _DTYPE_TO_QVALUE_BOUNDS does.
static std::pair<int64_t, int64_t> getStorageRange(Type storageType) {
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

// Verifies the contract every quantized_decomposed op shares. `realType` and
// `storageType` are the types of the floating-point and the quantized tensor
// operand or result of `op`.
static LogicalResult verifyQuantization(Operation *op, BaseTensorType realType,
                                        BaseTensorType storageType, Value dtype,
                                        Value quantMin, Value quantMax) {
  if (realType.hasDtype() && !isa<mlir::FloatType>(realType.getDtype()))
    return op->emitOpError(
               "expected the real tensor to have a floating-point dtype, "
               "but got ")
           << realType.getDtype();

  Type storageElementType =
      storageType.hasDtype() ? storageType.getDtype() : Type();
  if (storageElementType && !isStorageType(storageElementType))
    return op->emitOpError("expected the storage tensor to have a PT2E "
                           "storage dtype, but got ")
           << storageElementType;

  int64_t dtypeCode;
  if (matchPattern(dtype, m_TorchConstantInt(&dtypeCode))) {
    Type dtypeType = getStorageType(op->getContext(), dtypeCode);
    if (!dtypeType)
      return op->emitOpError("expected dtype to be a PT2E storage dtype, but "
                             "got dtype code ")
             << dtypeCode;
    if (storageElementType && storageElementType != dtypeType)
      return op->emitOpError("expected dtype code ")
             << dtypeCode << " (" << dtypeType
             << ") to match the storage tensor dtype " << storageElementType;
    storageElementType = dtypeType;
  }

  int64_t quantMinValue, quantMaxValue;
  bool hasQuantMin = matchPattern(quantMin, m_TorchConstantInt(&quantMinValue));
  bool hasQuantMax = matchPattern(quantMax, m_TorchConstantInt(&quantMaxValue));
  if (hasQuantMin && hasQuantMax && quantMinValue > quantMaxValue)
    return op->emitOpError("expected quant_min to not exceed quant_max, but "
                           "got [")
           << quantMinValue << ", " << quantMaxValue << "]";
  if (!storageElementType)
    return success();
  auto [storageMin, storageMax] = getStorageRange(storageElementType);
  for (auto [name, hasValue, value] :
       {std::make_tuple("quant_min", hasQuantMin, quantMinValue),
        std::make_tuple("quant_max", hasQuantMax, quantMaxValue)}) {
    if (hasValue && (value < storageMin || value > storageMax))
      return op->emitOpError("expected ")
             << name << " to be representable in " << storageElementType
             << ", but got " << value;
  }
  return success();
}

// Verifies that the dequantized `resultType` has the dtype `outDtype` selects.
// A None `outDtype` selects float32.
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
      return op->emitOpError("expected the output dtype to be a "
                             "floating-point dtype, but got dtype code ")
             << outDtypeCode;
    }
  }
  if (expected && resultType.hasDtype() && resultType.getDtype() != expected)
    return op->emitOpError("expected the result dtype to be ")
           << expected << ", but got " << resultType.getDtype();
  return success();
}

// Verifies that `scales` and `zeroPoints` hold one value per index of dimension
// `axis` of `inputType`. `zeroPoints` is None for symmetric quantization. Its
// dtype is not constrained: per_channel_affine_float_qparams observers produce
// floating-point zero points.
// https://github.com/pytorch/pytorch/blob/f3c0cf56a7577d677c37718c4edc41b8fc7fb51a/torch/ao/quantization/observer.py#L398-L405
static LogicalResult verifyPerChannelParameters(Operation *op,
                                                BaseTensorType inputType,
                                                Value scales, Value zeroPoints,
                                                Value axis) {
  auto scalesType = cast<BaseTensorType>(scales.getType());
  if (scalesType.hasDtype() && !isa<mlir::FloatType>(scalesType.getDtype()))
    return op->emitOpError("expected scales to have a floating-point dtype, "
                           "but got ")
           << scalesType.getDtype();
  int64_t scalesLength = kUnknownSize;
  if (scalesType.hasSizes()) {
    if (scalesType.getSizes().size() != 1)
      return op->emitOpError("expected scales to have rank 1, but got rank ")
             << scalesType.getSizes().size();
    scalesLength = scalesType.getSizes()[0];
  }

  if (auto zeroPointsType = dyn_cast<BaseTensorType>(zeroPoints.getType())) {
    if (zeroPointsType.hasSizes()) {
      ArrayRef<int64_t> sizes = zeroPointsType.getSizes();
      if (sizes.size() != 1)
        return op->emitOpError(
                   "expected zero_points to have rank 1, but got rank ")
               << sizes.size();
      if (sizes[0] != kUnknownSize && scalesLength != kUnknownSize &&
          sizes[0] != scalesLength)
        return op->emitOpError("expected zero_points to have the length of "
                               "scales, but got ")
               << sizes[0] << " and " << scalesLength;
    }
  }

  int64_t axisValue;
  if (!inputType.hasSizes() ||
      !matchPattern(axis, m_TorchConstantInt(&axisValue)))
    return success();
  ArrayRef<int64_t> inputSizes = inputType.getSizes();
  int64_t rank = inputSizes.size();
  int64_t channelDim = toPositiveDim(axisValue, rank);
  if (!isValidDim(channelDim, rank))
    return op->emitOpError("expected axis to be in [")
           << -rank << ", " << rank << "), but got " << axisValue;
  int64_t channels = inputSizes[channelDim];
  if (channels != kUnknownSize && scalesLength != kUnknownSize &&
      channels != scalesLength)
    return op->emitOpError("expected scales to have one element per index of "
                           "input dimension ")
           << channelDim << " (" << channels << "), but got " << scalesLength;
  return success();
}

LogicalResult QuantizedDecomposedQuantizePerTensorOp::verify() {
  return verifyQuantization(*this, cast<BaseTensorType>(getInput().getType()),
                            cast<BaseTensorType>(getType()), getDtype(),
                            getQuantMin(), getQuantMax());
}

LogicalResult QuantizedDecomposedDequantizePerTensorOp::verify() {
  auto resultType = cast<BaseTensorType>(getType());
  if (failed(verifyQuantization(*this, resultType,
                                cast<BaseTensorType>(getInput().getType()),
                                getDtype(), getQuantMin(), getQuantMax())))
    return failure();
  return verifyOutDtype(*this, resultType, getOutDtype());
}

LogicalResult QuantizedDecomposedQuantizePerChannelOp::verify() {
  auto inputType = cast<BaseTensorType>(getInput().getType());
  if (failed(verifyQuantization(*this, inputType,
                                cast<BaseTensorType>(getType()), getDtype(),
                                getQuantMin(), getQuantMax())))
    return failure();
  return verifyPerChannelParameters(*this, inputType, getScales(),
                                    getZeroPoints(), getAxis());
}

LogicalResult QuantizedDecomposedDequantizePerChannelOp::verify() {
  auto inputType = cast<BaseTensorType>(getInput().getType());
  auto resultType = cast<BaseTensorType>(getType());
  if (failed(verifyQuantization(*this, resultType, inputType, getDtype(),
                                getQuantMin(), getQuantMax())) ||
      failed(verifyOutDtype(*this, resultType, getOutDtype())))
    return failure();
  return verifyPerChannelParameters(*this, inputType, getScales(),
                                    getZeroPoints(), getAxis());
}

static int64_t getEffectiveGroupSize(Value input, Value scales,
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
  return ::getEffectiveGroupSize(getInput(), getScales(), groupSize);
}

int64_t QuantizedDecomposedDequantizePerChannelGroupOp::getEffectiveGroupSize(
    int64_t groupSize) {
  return ::getEffectiveGroupSize(getInput(), getScales(), groupSize);
}

// Verifies that the rank-2 `scales` and `zeroPoints` of `OpTy` hold one value
// per group of `group_size` consecutive elements in each row of the rank-2
// input. `zeroPoints` is None for symmetric dequantization. Its dtype is not
// constrained: torchao's group_quantize_tensor_symmetric
// (torchao/quantization/utils.py) passes zero points in the float precision
// of the scales.
template <typename OpTy>
static LogicalResult verifyPerChannelGroupParameters(OpTy op) {
  auto inputType = cast<BaseTensorType>(op.getInput().getType());
  auto scalesType = cast<BaseTensorType>(op.getScales().getType());
  if (inputType.hasSizes() && inputType.getSizes().size() != 2)
    return op.emitOpError("expected input to have rank 2, but got rank ")
           << inputType.getSizes().size();
  if (scalesType.hasDtype() && !isa<mlir::FloatType>(scalesType.getDtype()))
    return op.emitOpError("expected scales to have a floating-point dtype, "
                          "but got ")
           << scalesType.getDtype();
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
  // Whether a group size beyond the row length means one group per row
  // depends on the dynamic number of group columns.
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
  if (failed(verifyQuantization(*this,
                                cast<BaseTensorType>(getInput().getType()),
                                cast<BaseTensorType>(getType()), getDtype(),
                                getQuantMin(), getQuantMax())))
    return failure();
  return verifyPerChannelGroupParameters(*this);
}

LogicalResult QuantizedDecomposedDequantizePerChannelGroupOp::verify() {
  auto resultType = cast<BaseTensorType>(getType());
  if (failed(verifyQuantization(*this, resultType,
                                cast<BaseTensorType>(getInput().getType()),
                                getDtype(), getQuantMin(), getQuantMax())) ||
      failed(verifyOutDtype(*this, resultType, getOutDtype())))
    return failure();
  return verifyPerChannelGroupParameters(*this);
}
