//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "PopulatePatterns.h"
#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

// AtenQuantizePerTensorOp
// torch-mlir uses AtenQuantizePerTensorOp and AtenIntReprOp for per tensor
// quantization. These two ops are processed and converted together to
// stablehlo.uniform_quantize op.
namespace {
class ConvertAtenQuantizePerTensorOp
    : public OpConversionPattern<AtenQuantizePerTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenQuantizePerTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *zeroPoint = op.getZeroPoint().getDefiningOp();
    if (!zeroPoint || !isa<ConstantIntOp>(zeroPoint)) {
      return failure();
    }
    auto zeroPointConstantOp = mlir::cast<ConstantIntOp>(zeroPoint);
    auto zeroPointValue = zeroPointConstantOp.getValueAttr().getInt();

    auto scale = op.getScale().getDefiningOp();
    if (!scale || !isa<ConstantFloatOp>(scale)) {
      return failure();
    }

    auto scaleConstantOp = mlir::cast<ConstantFloatOp>(scale);
    auto scaleValue =
        scaleConstantOp.getValueAttr().getValue().convertToDouble();

    auto users = op.getResult().getUsers();
    auto opUser = *op.getResult().user_begin();
    if (!(std::distance(users.begin(), users.end()) == 1) ||
        !isa<AtenIntReprOp>(opUser)) {
      return failure();
    }

    auto inputElemType =
        mlir::cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getOperands().front().getType()))
            .getElementType();

    mlir::Type dtype =
        cast<ValueTensorType>(op->getResult(0).getType()).getDtype();
    int32_t bitWidth = 0;
    int32_t flags = quant::QuantizationFlags::FlagValue::Signed;
    if (isa<QUInt8Type>(dtype)) {
      bitWidth = 8;
      flags = 0;
    } else if (isa<QInt8Type>(dtype)) {
      bitWidth = 8;
    } else if (isa<QInt16Type>(dtype)) {
      bitWidth = 16;
    } else if (isa<QInt32Type>(dtype)) {
      bitWidth = 32;
    } else {
      return failure();
    }
    auto storageType = IntegerType::get(getContext(), bitWidth);

    // Minimum and maximum values for unsigned integer.
    int64_t minValue = 0;
    int64_t maxValue = (1LL << bitWidth) - 1;
    // Update the minimum and maximum for signed integer.
    if (flags) {
      // For signed integers (2's complement representation)
      minValue = -(1LL << (bitWidth - 1));
      maxValue = (1LL << (bitWidth - 1)) - 1;
    }

    auto qty = quant::UniformQuantizedType::get(
        flags, storageType, inputElemType, scaleValue, zeroPointValue, minValue,
        maxValue);

    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    mlir::TensorType new_type = outputType.clone(qty);

    stablehlo::UniformQuantizeOp qunatize =
        rewriter.replaceOpWithNewOp<stablehlo::UniformQuantizeOp>(
            opUser, new_type, adaptor.getOperands().front());

    opUser->getResults().front().replaceAllUsesWith(
        qunatize->getResults().front());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// Aten_MakePerTensorQuantizedTensorOp
// torch-mlir uses Aten_MakePerTensorQuantizedTensorOp and AtenDequantizeSelfOp
// in pair to represent per channel dequantization. These two ops are converted
// together to stablehlo.uniform_dequantize op
namespace {
class ConvertAten_MakePerTensorQuantizedTensorOp
    : public OpConversionPattern<Aten_MakePerTensorQuantizedTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_MakePerTensorQuantizedTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opUser = *op.getResult().user_begin();
    auto users = op.getResult().getUsers();
    if (!(std::distance(users.begin(), users.end()) == 1) ||
        !isa<AtenDequantizeSelfOp>(opUser)) {
      return failure();
    }
    // [TODO] verify that zeroPoint and Scale matches with the input operand
    // type.
    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(opUser->getResult(0).getType()));

    rewriter.replaceOpWithNewOp<stablehlo::UniformDequantizeOp>(
        opUser, outputType, adaptor.getOperands().front());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenQuantizePerChannelOp
    : public OpConversionPattern<AtenQuantizePerChannelOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenQuantizePerChannelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *zeroPoints = op.getZeroPoints().getDefiningOp();
    if (!zeroPoints || !isa<ValueTensorLiteralOp>(zeroPoints)) {
      return failure();
    }
    auto zeroPointsOp = mlir::cast<ValueTensorLiteralOp>(zeroPoints);

    llvm::SmallVector<int64_t, 4> zeroPointsVec;
    for (auto zp : zeroPointsOp.getValue().getValues<llvm::APInt>()) {
      zeroPointsVec.emplace_back(zp.getSExtValue());
    }

    auto scales = op.getScales().getDefiningOp();
    if (!scales || !isa<ValueTensorLiteralOp>(scales)) {
      return failure();
    }

    llvm::SmallVector<double, 4> scalesVec;
    auto scalesOp = mlir::cast<ValueTensorLiteralOp>(scales);
    for (auto scale : scalesOp.getValue().getValues<llvm::APFloat>()) {
      scalesVec.emplace_back(scale.convertToDouble());
    }

    auto axis = op.getAxis().getDefiningOp();
    if (!axis || !isa<ConstantIntOp>(axis)) {
      return failure();
    }
    auto axisOp = mlir::cast<ConstantIntOp>(axis);
    auto axisValue = axisOp.getValueAttr().getInt();

    auto users = op.getResult().getUsers();
    auto opUser = *op.getResult().user_begin();
    if (!(std::distance(users.begin(), users.end()) == 1) ||
        !isa<AtenIntReprOp>(opUser)) {
      return failure();
    }

    auto inputElemType =
        mlir::cast<RankedTensorType>(
            getTypeConverter()->convertType(op.getOperands().front().getType()))
            .getElementType();

    mlir::Type dtype =
        cast<ValueTensorType>(op->getResult(0).getType()).getDtype();
    int32_t bitWidth = 0;
    int32_t flags = quant::QuantizationFlags::FlagValue::Signed;
    if (isa<QUInt8Type>(dtype)) {
      bitWidth = 8;
      flags = 0;
    } else if (isa<QInt8Type>(dtype)) {
      bitWidth = 8;
    } else if (isa<QInt16Type>(dtype)) {
      bitWidth = 16;
    } else if (isa<QInt32Type>(dtype)) {
      bitWidth = 32;
    } else {
      return failure();
    }
    auto storageType = IntegerType::get(getContext(), bitWidth);

    // Minimum and maximum values for unsigned integer.
    int64_t minValue = 0;
    int64_t maxValue = (1LL << bitWidth) - 1;
    // Update the minimum and maximum for signed integer.
    if (flags) {
      // For signed integers (2's complement representation)
      minValue = -(1LL << (bitWidth - 1));
      maxValue = (1LL << (bitWidth - 1)) - 1;
    }

    auto qty = quant::UniformQuantizedPerAxisType::get(
        flags, storageType, inputElemType, scalesVec, zeroPointsVec, axisValue,
        minValue, maxValue);

    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    mlir::TensorType new_type = outputType.clone(qty);

    stablehlo::UniformQuantizeOp quantize =
        rewriter.replaceOpWithNewOp<stablehlo::UniformQuantizeOp>(
            opUser, new_type, adaptor.getOperands().front());

    opUser->getResults().front().replaceAllUsesWith(
        quantize->getResults().front());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class ConvertAten_MakePerChannelQuantizedTensorOp
    : public OpConversionPattern<Aten_MakePerChannelQuantizedTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_MakePerChannelQuantizedTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opUser = *op.getResult().user_begin();
    auto users = op.getResult().getUsers();
    if (!(std::distance(users.begin(), users.end()) == 1) ||
        !isa<AtenDequantizeSelfOp>(opUser)) {
      return failure();
    }
    // [TODO] verify that zeroPoint and Scale matches with the input operand
    // type.
    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(opUser->getResult(0).getType()));

    rewriter.replaceOpWithNewOp<stablehlo::UniformDequantizeOp>(
        opUser, outputType, adaptor.getOperands().front());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

// Compute dim sizes and validate them, and return
LogicalResult validateDistanceInputs(RankedTensorType x1Type,
                                     RankedTensorType x2Type, int64_t &rank,
                                     int64_t &n, int64_t &m, int64_t &d,
                                     SmallVector<int64_t> &batchDims,
                                     SmallVector<int64_t> &batchDimsIndices,
                                     std::string &errorMsg) {
  if (!x1Type.hasStaticShape() || !x2Type.hasStaticShape()) {
    errorMsg = "only static shapes supported";
    return failure();
  }

  auto x1Shape = x1Type.getShape();
  auto x2Shape = x2Type.getShape();
  rank = x1Shape.size();

  if (x1Shape.size() != x2Shape.size()) {
    errorMsg = "x1 and x2 must have same rank";
    return failure();
  }

  n = x1Shape[rank - 2];
  m = x2Shape[rank - 2];
  d = x1Shape[rank - 1];

  if (d != x2Shape[rank - 1]) {
    errorMsg = "feature dimensions must match";
    return failure();
  }

  batchDims.assign(x1Shape.begin(), x1Shape.end() - 2);
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (x1Shape[i] != x2Shape[i]) {
      errorMsg = "batch dimensions must match";
      return failure();
    }
    batchDimsIndices.push_back(i);
  }

  return success();
}

namespace {
class ConvertAten_CdistForwardOp
    : public OpConversionPattern<Aten_CdistForwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_CdistForwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = adaptor.getX1();
    Value x2 = adaptor.getX2();
    auto x1Type = cast<RankedTensorType>(x1.getType());
    auto x2Type = cast<RankedTensorType>(x2.getType());

    double p;
    if (!matchPattern(op.getP(), m_TorchConstantFloat(&p))) {
      return rewriter.notifyMatchFailure(op, "p must be constant");
    }

    // Validate compute_mode parameter
    if (!isa<Torch::NoneType>(op.getComputeMode().getType())) {
      int64_t computeMode;
      if (!matchPattern(op.getComputeMode(),
                        m_TorchConstantInt(&computeMode))) {
        return rewriter.notifyMatchFailure(op, "compute_mode must be constant");
      }
      if (computeMode != 1 && computeMode != 2) {
        return rewriter.notifyMatchFailure(
            op, "compute_mode must be None, 1, or 2");
      }
    }

    int64_t rank, n, m, d;
    SmallVector<int64_t> batchDims, batchDimsIndices;
    std::string errorMsg;
    if (failed(validateDistanceInputs(x1Type, x2Type, rank, n, m, d, batchDims,
                                      batchDimsIndices, errorMsg))) {
      return rewriter.notifyMatchFailure(op, errorMsg);
    }

    // Reshape x1 to [...batch, N, 1, D] and x2 to [...batch, 1, M, D]
    SmallVector<int64_t> reshapeShape1(batchDims);
    reshapeShape1.append({n, 1, d});
    SmallVector<int64_t> reshapeShape2(batchDims);
    reshapeShape2.append({1, m, d});
    auto reshapeType1 =
        RankedTensorType::get(reshapeShape1, x1Type.getElementType());
    auto reshapeType2 =
        RankedTensorType::get(reshapeShape2, x2Type.getElementType());

    Value x1Reshaped =
        stablehlo::ReshapeOp::create(rewriter, loc, reshapeType1, x1);
    Value x2Reshaped =
        stablehlo::ReshapeOp::create(rewriter, loc, reshapeType2, x2);

    // Broadcast subtract: [...batch, N, M, D]
    SmallVector<int64_t> diffShape(batchDims);
    diffShape.append({n, m, d});
    auto diffType = RankedTensorType::get(diffShape, x1Type.getElementType());
    Value diff = chlo::BroadcastSubOp::create(rewriter, loc, x1Reshaped,
                                              x2Reshaped, nullptr);

    // Compute norm
    Value result;
    SmallVector<int64_t> resultShape(batchDims);
    resultShape.append({n, m});
    auto resultType =
        RankedTensorType::get(resultShape, x1Type.getElementType());
    auto scalarType = RankedTensorType::get({}, x1Type.getElementType());
    if (p == 2.0) {
      Value squared =
          stablehlo::MulOp::create(rewriter, loc, diffType, diff, diff);
      Value initValue = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(x1Type.getElementType()));
      stablehlo::ReduceOp sum = stablehlo::ReduceOp::create(
          rewriter, loc, resultType, ValueRange{squared}, ValueRange{initValue},
          rewriter.getDenseI64ArrayAttr({rank}));
      Block &block = sum.getBody().emplaceBlock();
      block.addArgument(scalarType, loc);
      block.addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(&block);
      Value add = stablehlo::AddOp::create(rewriter, loc, block.getArgument(0),
                                           block.getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, add);
      rewriter.setInsertionPointAfter(sum);
      result = stablehlo::SqrtOp::create(rewriter, loc, sum.getResults()[0]);
    } else {
      Value abs = stablehlo::AbsOp::create(rewriter, loc, diffType, diff);
      Value pConst = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(x1Type.getElementType(), p));
      Value powered =
          chlo::BroadcastPowOp::create(rewriter, loc, abs, pConst, nullptr);
      Value initValue = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(x1Type.getElementType()));
      stablehlo::ReduceOp sum = stablehlo::ReduceOp::create(
          rewriter, loc, resultType, ValueRange{powered}, ValueRange{initValue},
          rewriter.getDenseI64ArrayAttr({rank}));
      Block &block = sum.getBody().emplaceBlock();
      block.addArgument(scalarType, loc);
      block.addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(&block);
      Value add = stablehlo::AddOp::create(rewriter, loc, block.getArgument(0),
                                           block.getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, add);
      rewriter.setInsertionPointAfter(sum);
      Value invP = stablehlo::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(x1Type.getElementType(), 1.0 / p));
      result = chlo::BroadcastPowOp::create(rewriter, loc, sum.getResults()[0],
                                            invP, nullptr);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAten_PdistForwardOp
    : public OpConversionPattern<Aten_PdistForwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_PdistForwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x = adaptor.getSelf();
    auto xType = cast<RankedTensorType>(x.getType());

    if (!xType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only static shapes supported");
    }

    double p;
    if (!matchPattern(op.getP(), m_TorchConstantFloat(&p))) {
      return rewriter.notifyMatchFailure(op, "p must be constant");
    }

    auto xShape = xType.getShape();
    int64_t rank = xShape.size();
    int64_t n = xShape[rank - 2];
    int64_t d = xShape[rank - 1];
    SmallVector<int64_t> batchDims(xShape.begin(), xShape.end() - 2);

    // Reshape x to [...batch, N, 1, D] and [...batch, 1, N, D]
    SmallVector<int64_t> reshapeShape1(batchDims);
    reshapeShape1.append({n, 1, d});
    SmallVector<int64_t> reshapeShape2(batchDims);
    reshapeShape2.append({1, n, d});
    auto reshapeType1 =
        RankedTensorType::get(reshapeShape1, xType.getElementType());
    auto reshapeType2 =
        RankedTensorType::get(reshapeShape2, xType.getElementType());

    Value x1Reshaped =
        stablehlo::ReshapeOp::create(rewriter, loc, reshapeType1, x);
    Value x2Reshaped =
        stablehlo::ReshapeOp::create(rewriter, loc, reshapeType2, x);

    // Broadcast subtract: [...batch, N, N, D]
    SmallVector<int64_t> diffShape(batchDims);
    diffShape.append({n, n, d});
    auto diffType = RankedTensorType::get(diffShape, xType.getElementType());
    Value diff = chlo::BroadcastSubOp::create(rewriter, loc, x1Reshaped,
                                              x2Reshaped, nullptr);

    // Compute norm for full NxN matrix
    SmallVector<int64_t> fullResultShape(batchDims);
    fullResultShape.append({n, n});
    auto fullResultType =
        RankedTensorType::get(fullResultShape, xType.getElementType());
    auto scalarType = RankedTensorType::get({}, xType.getElementType());

    Value fullResult;
    if (p == 2.0) {
      Value squared =
          stablehlo::MulOp::create(rewriter, loc, diffType, diff, diff);
      Value initValue = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(xType.getElementType()));
      stablehlo::ReduceOp sum = stablehlo::ReduceOp::create(
          rewriter, loc, fullResultType, ValueRange{squared},
          ValueRange{initValue}, rewriter.getDenseI64ArrayAttr({rank}));
      Block &block = sum.getBody().emplaceBlock();
      block.addArgument(scalarType, loc);
      block.addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(&block);
      Value add = stablehlo::AddOp::create(rewriter, loc, block.getArgument(0),
                                           block.getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, add);
      rewriter.setInsertionPointAfter(sum);
      fullResult =
          stablehlo::SqrtOp::create(rewriter, loc, sum.getResults()[0]);
    } else {
      Value abs = stablehlo::AbsOp::create(rewriter, loc, diffType, diff);
      Value pConst = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(xType.getElementType(), p));
      Value powered =
          chlo::BroadcastPowOp::create(rewriter, loc, abs, pConst, nullptr);
      Value initValue = stablehlo::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(xType.getElementType()));
      stablehlo::ReduceOp sum = stablehlo::ReduceOp::create(
          rewriter, loc, fullResultType, ValueRange{powered},
          ValueRange{initValue}, rewriter.getDenseI64ArrayAttr({rank}));
      Block &block = sum.getBody().emplaceBlock();
      block.addArgument(scalarType, loc);
      block.addArgument(scalarType, loc);
      rewriter.setInsertionPointToStart(&block);
      Value add = stablehlo::AddOp::create(rewriter, loc, block.getArgument(0),
                                           block.getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, add);
      rewriter.setInsertionPointAfter(sum);
      Value invP = stablehlo::ConstantOp::create(
          rewriter, loc,
          rewriter.getFloatAttr(xType.getElementType(), 1.0 / p));
      fullResult = chlo::BroadcastPowOp::create(
          rewriter, loc, sum.getResults()[0], invP, nullptr);
    }

    // Create constant indices for gathering upper triangular elements
    int64_t condensedSize = n * (n - 1) / 2;
    SmallVector<int64_t> gatherIndices;
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = i + 1; j < n; ++j) {
        gatherIndices.push_back(i * n + j);
      }
    }

    SmallVector<int64_t> condensedShape(batchDims);
    condensedShape.push_back(condensedSize);
    auto condensedType =
        RankedTensorType::get(condensedShape, xType.getElementType());

    Value flatResult = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get({n * n}, xType.getElementType()),
        fullResult);

    Value indices = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseIntElementsAttr::get(
            RankedTensorType::get({condensedSize, 1}, rewriter.getI64Type()),
            ArrayRef<int64_t>(gatherIndices)));

    auto dimsAttr = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/{},
        /*collapsedSliceDims=*/{0},
        /*operandBatchingDims=*/{},
        /*startIndicesBatchingDims=*/{},
        /*startIndexMap=*/{0},
        /*indexVecDim=*/1);

    Value result = stablehlo::GatherOp::create(
        rewriter, loc, condensedType, flatResult, indices, dimsAttr,
        rewriter.getDenseI64ArrayAttr({1}), rewriter.getBoolAttr(false));

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class ConvertAten_EuclideanDistOp
    : public OpConversionPattern<Aten_EuclideanDistOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_EuclideanDistOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value x1 = adaptor.getX1();
    Value x2 = adaptor.getX2();
    auto x1Type = cast<RankedTensorType>(x1.getType());
    auto x2Type = cast<RankedTensorType>(x2.getType());

    int64_t rank, n, m, d;
    SmallVector<int64_t> batchDims, batchDimsIndices;
    std::string errorMsg;
    if (failed(validateDistanceInputs(x1Type, x2Type, rank, n, m, d, batchDims,
                                      batchDimsIndices, errorMsg))) {
      return rewriter.notifyMatchFailure(op, errorMsg);
    }

    SmallVector<int64_t> resultShape(batchDims);
    resultShape.append({n, m});
    auto resultType =
        RankedTensorType::get(resultShape, x1Type.getElementType());
    auto scalarType = RankedTensorType::get({}, x1Type.getElementType());

    // Use the identity ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1*x2^T
    Value x1Norm = stablehlo::MulOp::create(rewriter, loc, x1Type, x1, x1);
    SmallVector<int64_t> x1NormShape(batchDims);
    x1NormShape.push_back(n);
    stablehlo::ReduceOp x1NormSum = stablehlo::ReduceOp::create(
        rewriter, loc,
        RankedTensorType::get(x1NormShape, x1Type.getElementType()),
        ValueRange{x1Norm},
        ValueRange{stablehlo::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(x1Type.getElementType()))},
        rewriter.getDenseI64ArrayAttr({rank - 1}));
    Block &block1 = x1NormSum.getBody().emplaceBlock();
    block1.addArgument(scalarType, loc);
    block1.addArgument(scalarType, loc);
    rewriter.setInsertionPointToStart(&block1);
    Value x1NormAdd = stablehlo::AddOp::create(
        rewriter, loc, block1.getArgument(0), block1.getArgument(1));
    stablehlo::ReturnOp::create(rewriter, loc, x1NormAdd);
    rewriter.setInsertionPointAfter(x1NormSum);

    Value x2Norm = stablehlo::MulOp::create(rewriter, loc, x2Type, x2, x2);
    SmallVector<int64_t> x2NormShape(batchDims);
    x2NormShape.push_back(m);
    stablehlo::ReduceOp x2NormSum = stablehlo::ReduceOp::create(
        rewriter, loc,
        RankedTensorType::get(x2NormShape, x2Type.getElementType()),
        ValueRange{x2Norm},
        ValueRange{stablehlo::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(x2Type.getElementType()))},
        rewriter.getDenseI64ArrayAttr({rank - 1}));
    Block &block2 = x2NormSum.getBody().emplaceBlock();
    block2.addArgument(scalarType, loc);
    block2.addArgument(scalarType, loc);
    rewriter.setInsertionPointToStart(&block2);
    Value x2NormAdd = stablehlo::AddOp::create(
        rewriter, loc, block2.getArgument(0), block2.getArgument(1));
    stablehlo::ReturnOp::create(rewriter, loc, x2NormAdd);
    rewriter.setInsertionPointAfter(x2NormSum);

    SmallVector<int64_t> transposeIndices(batchDimsIndices);
    transposeIndices.append({rank - 1, rank - 2});
    SmallVector<int64_t> x2TShape(batchDims);
    x2TShape.append({d, m});
    Value x2T = stablehlo::TransposeOp::create(
        rewriter, loc, RankedTensorType::get(x2TShape, x2Type.getElementType()),
        x2, rewriter.getDenseI64ArrayAttr(transposeIndices));
    Value matmul = stablehlo::DotGeneralOp::create(
        rewriter, loc, resultType, x1, x2T,
        stablehlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(), batchDimsIndices, batchDimsIndices,
            {rank - 1}, {rank - 2}),
        nullptr, nullptr);
    Value twoMatmul = chlo::BroadcastMulOp::create(
        rewriter, loc, resultType, matmul,
        stablehlo::ConstantOp::create(
            rewriter, loc, rewriter.getFloatAttr(x1Type.getElementType(), 2.0)),
        nullptr);

    SmallVector<int64_t> x1UnsqueezedShape(batchDims);
    x1UnsqueezedShape.append({n, 1});
    SmallVector<int64_t> x2UnsqueezedShape(batchDims);
    x2UnsqueezedShape.append({1, m});
    Value x1Unsqueezed = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(x1UnsqueezedShape, x1Type.getElementType()),
        x1NormSum.getResult(0));
    Value x2Unsqueezed = stablehlo::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(x2UnsqueezedShape, x2Type.getElementType()),
        x2NormSum.getResult(0));

    Value distSquared = chlo::BroadcastAddOp::create(
        rewriter, loc, x1Unsqueezed, x2Unsqueezed, nullptr);
    distSquared =
        stablehlo::SubtractOp::create(rewriter, loc, distSquared, twoMatmul);
    Value result = stablehlo::SqrtOp::create(rewriter, loc, distSquared);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_stablehlo::populateUncategorizedPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenQuantizePerTensorOp>();
  target.addIllegalOp<AtenIntReprOp>();
  patterns.add<ConvertAtenQuantizePerTensorOp>(typeConverter, context);
  target.addIllegalOp<Aten_MakePerTensorQuantizedTensorOp>();
  target.addIllegalOp<AtenDequantizeSelfOp>();
  patterns.add<ConvertAten_MakePerTensorQuantizedTensorOp>(typeConverter,
                                                           context);
  target.addIllegalOp<AtenQuantizePerChannelOp>();
  patterns.add<ConvertAtenQuantizePerChannelOp>(typeConverter, context);
  patterns.add<ConvertAten_MakePerChannelQuantizedTensorOp>(typeConverter,
                                                            context);
}
