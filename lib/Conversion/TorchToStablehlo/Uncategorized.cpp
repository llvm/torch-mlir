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

// Returns the storage metadata (bitwidth, quantization flags, min/max storage
// values) for a given Torch quantized dtype.  Returns failure() for unsupported
// dtypes so callers can propagate a clean matchAndRewrite failure.
static LogicalResult getQuantStorageMeta(Type dtype, int32_t &bitWidth,
                                         int32_t &flags, int64_t &minValue,
                                         int64_t &maxValue) {
  flags = quant::QuantizationFlags::FlagValue::Signed;
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
  // Minimum and maximum values for unsigned integer.
  minValue = 0;
  maxValue = (1LL << bitWidth) - 1;
  // Update the minimum and maximum for signed integer (2's complement).
  if (flags) {
    minValue = -(1LL << (bitWidth - 1));
    maxValue = (1LL << (bitWidth - 1)) - 1;
  }
  return success();
}

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
    int32_t flags = 0;
    int64_t minValue = 0;
    int64_t maxValue = 0;
    if (failed(getQuantStorageMeta(dtype, bitWidth, flags, minValue, maxValue)))
      return failure();
    auto storageType = IntegerType::get(getContext(), bitWidth);

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
    int32_t flags = 0;
    int64_t minValue = 0;
    int64_t maxValue = 0;
    if (failed(getQuantStorageMeta(dtype, bitWidth, flags, minValue, maxValue)))
      return failure();
    auto storageType = IntegerType::get(getContext(), bitWidth);

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

// QuantizedDecomposedQuantizePerTensorOp
// Legalizes the PT2E quantize_per_tensor op to plain arithmetic:
//   scaled = input * (1/scale)
//   rounded = round_nearest_even(scaled)
//   shifted = rounded + float(zp)
//   clamped = clamp(shifted, quant_min, quant_max)
//   result  = convert(clamped) : -> int storage type
// This avoids the !quant.uniform intermediate that trips StablehloRefineShapes.
namespace {
class ConvertQuantizedDecomposedQuantizePerTensorOp
    : public OpConversionPattern<QuantizedDecomposedQuantizePerTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizedDecomposedQuantizePerTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PerTensorQParams qparams;
    if (failed(getConstantPerTensorQParams(
            rewriter, op, op.getScale(), op.getZeroPoint(), op.getQuantMin(),
            op.getQuantMax(), /*requireNonZeroScale=*/true,
            /*requireClampRange=*/true, qparams)))
      return failure();
    auto zpVal = qparams.zeroPoint;
    auto scaleVal = qparams.scale;
    auto quantMinVal = qparams.quantMin;
    auto quantMaxVal = qparams.quantMax;

    auto resultTensorType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    if (!resultTensorType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    auto storageIntType =
        dyn_cast<IntegerType>(resultTensorType.getElementType());
    if (!storageIntType)
      return rewriter.notifyMatchFailure(
          op, "result element type must be an integer type");

    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputTensorType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTensorType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
    if (!inputTensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");
    auto floatElemType =
        dyn_cast<mlir::FloatType>(inputTensorType.getElementType());
    if (!floatElemType)
      return rewriter.notifyMatchFailure(
          op, "input element type must be a float type");
    // Scalar tensor type for broadcasting constants.
    auto scalarFloatType = RankedTensorType::get({}, floatElemType);

    // Build APFloat values for the float element type.
    double invScale = 1.0 / scaleVal;
    auto makeFloatAttr = [&](double v) -> DenseElementsAttr {
      APFloat apv(v);
      bool lossy = false;
      apv.convert(floatElemType.getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &lossy);
      return DenseElementsAttr::get(scalarFloatType, apv);
    };

    // stablehlo binary ops require identical operand types; broadcast scalar
    // constants to the full input shape before use.
    auto broadcastToInputShape = [&](Value scalarTensor) -> Value {
      // scalarTensor has type tensor<floatElemType>; broadcast to
      // inputTensorType.
      SmallVector<int64_t> broadcastDims;
      return stablehlo::BroadcastInDimOp::create(rewriter, loc, inputTensorType,
                                                 scalarTensor, broadcastDims);
    };

    // multiply input by 1/scale
    Value invScaleScalar =
        stablehlo::ConstantOp::create(rewriter, loc, makeFloatAttr(invScale));
    Value invScaleBcast = broadcastToInputShape(invScaleScalar);
    Value scaled =
        stablehlo::MulOp::create(rewriter, loc, input, invScaleBcast);

    // round to nearest even
    Value rounded =
        stablehlo::RoundNearestEvenOp::create(rewriter, loc, scaled);

    // add zero_point (as float)
    Value zpFloatScalar = stablehlo::ConstantOp::create(
        rewriter, loc, makeFloatAttr(static_cast<double>(zpVal)));
    Value zpFloatBcast = broadcastToInputShape(zpFloatScalar);
    Value shifted =
        stablehlo::AddOp::create(rewriter, loc, rounded, zpFloatBcast);

    // clamp to [quant_min, quant_max]
    // stablehlo.clamp accepts rank-0 min/max or same-shape; use rank-0 here.
    Value qminScalar = stablehlo::ConstantOp::create(
        rewriter, loc, makeFloatAttr(static_cast<double>(quantMinVal)));
    Value qmaxScalar = stablehlo::ConstantOp::create(
        rewriter, loc, makeFloatAttr(static_cast<double>(quantMaxVal)));
    Value clamped = stablehlo::ClampOp::create(rewriter, loc, inputTensorType,
                                               qminScalar, shifted, qmaxScalar);

    // convert float -> integer storage type
    rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, resultTensorType,
                                                      clamped);
    return success();
  }
};
} // namespace

// QuantizedDecomposedDequantizePerTensorOp
// Legalizes the PT2E dequantize_per_tensor op to plain arithmetic:
//   wide  = convert(input) : int -> i32 (for sub without overflow)
//   subtr = wide - zp
//   fp    = convert(subtr) : i32 -> float
//   result = fp * scale
// This avoids the !quant.uniform intermediate that trips StablehloRefineShapes.
namespace {
class ConvertQuantizedDecomposedDequantizePerTensorOp
    : public OpConversionPattern<QuantizedDecomposedDequantizePerTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizedDecomposedDequantizePerTensorOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Dequantize math is (input - zero_point) * scale; quant_min / quant_max
    // are unused, so they need not be constant, and a zero scale is harmless.
    PerTensorQParams qparams;
    if (failed(getConstantPerTensorQParams(
            rewriter, op, op.getScale(), op.getZeroPoint(), op.getQuantMin(),
            op.getQuantMax(), /*requireNonZeroScale=*/false,
            /*requireClampRange=*/false, qparams)))
      return failure();
    auto scaleVal = qparams.scale;
    auto zpVal = qparams.zeroPoint;

    auto inputTensorType =
        dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!inputTensorType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
    if (!inputTensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    auto storageIntType =
        dyn_cast<IntegerType>(inputTensorType.getElementType());
    if (!storageIntType)
      return rewriter.notifyMatchFailure(
          op, "input element type must be an integer type");

    auto resultTensorType = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    if (!resultTensorType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");

    auto outputFloatType =
        dyn_cast<mlir::FloatType>(resultTensorType.getElementType());
    if (!outputFloatType)
      return rewriter.notifyMatchFailure(
          op, "result element type must be a float type");

    auto loc = op.getLoc();
    Value input = adaptor.getInput();

    // Use i32 as the wide integer type for the subtraction (avoids i8
    // overflow).
    auto i32Type = rewriter.getIntegerType(32);
    auto wideIntTensorType = inputTensorType.clone(i32Type);

    // convert int storage -> i32
    Value wideInt =
        stablehlo::ConvertOp::create(rewriter, loc, wideIntTensorType, input);

    // subtract zero_point (broadcast scalar to wide int shape first)
    auto scalarI32Type = RankedTensorType::get({}, i32Type);
    Value zpScalar = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            scalarI32Type,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(zpVal))));
    Value zpBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, wideIntTensorType, zpScalar, SmallVector<int64_t>{});
    Value subtracted =
        stablehlo::SubtractOp::create(rewriter, loc, wideInt, zpBcast);

    // convert i32 -> output float type
    auto floatTensorType = inputTensorType.clone(outputFloatType);
    Value asFloat = stablehlo::ConvertOp::create(rewriter, loc, floatTensorType,
                                                 subtracted);

    // multiply by scale (broadcast scalar to float tensor shape)
    auto scalarFloatType = RankedTensorType::get({}, outputFloatType);
    APFloat scaleApf(scaleVal);
    bool lossy = false;
    scaleApf.convert(outputFloatType.getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &lossy);
    Value scaleScalar = stablehlo::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(scalarFloatType, scaleApf));
    Value scaleBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, floatTensorType, scaleScalar, SmallVector<int64_t>{});
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, asFloat, scaleBcast);
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
  target.addIllegalOp<QuantizedDecomposedQuantizePerTensorOp>();
  patterns.add<ConvertQuantizedDecomposedQuantizePerTensorOp>(typeConverter,
                                                              context);
  target.addIllegalOp<QuantizedDecomposedDequantizePerTensorOp>();
  patterns.add<ConvertQuantizedDecomposedDequantizePerTensorOp>(typeConverter,
                                                                context);
}
