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

#include "../PassDetail.h"
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
