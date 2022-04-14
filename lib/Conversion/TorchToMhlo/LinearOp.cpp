//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include "../PassDetail.h"
#include "./MhloLegalizeUtils.h"
#include "./PopulatePattern.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <iostream>
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// AtenMmOp
class ConvertAtenMmOp : public OpConversionPattern<AtenMmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    Value rhs = adaptor.mat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported now.");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    // Mm takes two 2D tensors.
    if (lhsRank != 2 || rhsRank != 2)
      return op.emitError("aten.mm called but matrix rank != 2");

    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

    ArrayAttr precision_config;
    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, outType, lhs, rhs,
                                             precision_config);
    return success();
  }
};

// AtenBmmOp
class ConvertAtenBmmOp : public OpConversionPattern<AtenBmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    Value rhs = adaptor.mat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported now.");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    // Bmm takes two 3D tensors.
    if (lhsRank != 3 || rhsRank != 3)
      return op.emitError("aten.bmm called but matrix rank != 3");

    auto outType = getTypeConverter()
                       ->convertType(op->getResult(0).getType())
                       .cast<RankedTensorType>();

    MLIRContext *context = op->getContext();
    ArrayRef<int64_t> lhsBatchingDimensions{0};
    ArrayRef<int64_t> rhsBatchingDimensions{0};
    ArrayRef<int64_t> lhsContractingDimensions{2};
    ArrayRef<int64_t> rhsContractingDimensions{1};
    auto dot_dimension_numbers = mhlo::DotDimensionNumbersAttr::get(
        context, lhsBatchingDimensions, rhsBatchingDimensions,
        lhsContractingDimensions, rhsContractingDimensions);

    ArrayAttr precision_config;
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, outType, lhs, rhs, dot_dimension_numbers, precision_config);
    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenLinearOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
// AtenLinearOp
template <>
LogicalResult ConvertAtenLinearOp<AtenLinearOp>::matchAndRewrite(
    AtenLinearOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.input();
  Value weight = adaptor.weight();
  Value bias = adaptor.bias();

  auto inputTy = input.getType().cast<RankedTensorType>();
  auto weightTy = weight.getType().cast<RankedTensorType>();
  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .template cast<RankedTensorType>();

  if (weightTy.getRank() != 2) {
    return op.emitError("Only weight tensor with rank 2 supported");
  }
  if (succeeded(checkNotNone(rewriter, op, bias))) {
    auto biasRank = bias.getType().cast<RankedTensorType>().getRank();
    if (biasRank != 1) {
      return op.emitError("Only bias with rank 1 supported");
    }
    if (bias.getType().cast<RankedTensorType>().getElementType() !=
        inputTy.getElementType()) {
      return op.emitError("Bias element type and input element type mismatch");
    }
  }
  // By default, torch.linear doesn't support automatic type promotion
  if (weightTy.getElementType() != inputTy.getElementType()) {
    return op.emitError("Weight element type and input element type mismatch");
  }

  // Input x transposed weight
  // By default, input features lie at the last dimension.
  mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
      mhlo::DotDimensionNumbersAttr::get(
          rewriter.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{inputTy.getRank() - 1},
          /*rhsContractingDimensions=*/{1});
  mlir::ArrayAttr precisionConfig;
  auto mulResult =
      rewriter
          .create<mhlo::DotGeneralOp>(op.getLoc(), outTy, input, weight,
                                      dotDimensionNumbers, precisionConfig)
          .getResult();

  // Add bias if given
  if (succeeded(checkNotNone(rewriter, op, bias))) {
    auto bcastBias = mhlo::promoteAndBroadcast(rewriter, bias, outTy);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(op, outTy, mulResult, bcastBias);
    return success();
  }
  rewriter.replaceOp(op, mulResult);
  return success();
}

// AtenMatmulOp
template <>
LogicalResult ConvertAtenLinearOp<AtenMatmulOp>::matchAndRewrite(
    AtenMatmulOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value lhs = adaptor.self();
  Value rhs = adaptor.other();

  auto lhsTy = lhs.getType().cast<RankedTensorType>();
  auto rhsTy = rhs.getType().cast<RankedTensorType>();
  auto resultTy =
      getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();

  auto lhsRank = lhsTy.getRank();
  auto rhsRank = rhsTy.getRank();
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  if (lhsTy.getElementType() != rhsTy.getElementType()) {
    return op.emitError("mismatching operands element types");
  }

  // First case: Vec x Vec, Mat x Mat or Mat x Vec
  if ((lhsRank == 1 && rhsRank == 1) || (lhsRank == 2 && rhsRank == 1) ||
      (lhsRank == 2 && rhsRank == 2)) {
    if (lhsShape[lhsShape.size() - 1] != rhsShape[0]) {
      return op.emitError("mismatching contracting dimension");
    }
    ArrayAttr precisionConfig;
    rewriter.replaceOpWithNewOp<mhlo::DotOp>(op, resultTy, lhs, rhs,
                                             precisionConfig);
    return success();
  }

  // Second case: Vec x Mat
  if (lhsRank == 1 && rhsRank == 2) {
    if (lhsShape[0] != rhsShape[0]) {
      return op.emitError("mismatching contracting dimension");
    }
    mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
        mhlo::DotDimensionNumbersAttr::get(rewriter.getContext(),
                                           /*lhsBatchingDimensions=*/{},
                                           /*rhsBatchingDimensions=*/{},
                                           /*lhsContractingDimensions=*/{0},
                                           /*rhsContractingDimensions=*/{0});

    mlir::ArrayAttr precisionConfig;
    rewriter
        .replaceOpWithNewOp<mhlo::DotGeneralOp>(
            op, resultTy, lhs, rhs, dotDimensionNumbers, precisionConfig)
        .getResult();
    return success();
  }

  // Third case: Batch Mat x [Mat,Vec]
  if (lhsRank >= 1 && rhsRank >= 1 && (lhsRank >= 3 || rhsRank >= 3)) {
    if (lhsRank > rhsRank) {
      SmallVector<int64_t> rhsReshape(lhsRank, 1);
      int64_t rankDelta = lhsRank - rhsRank;

      // append 1 dim to rhs tensor, for the purpose of matrix multiply
      if (rhsRank == 1) {
        rankDelta--;
      }
      std::copy(rhsShape.begin(), rhsShape.end(),
                rhsReshape.begin() + rankDelta);
      auto rhsReshapeConst =
          mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(rhsReshape),
                               {static_cast<int64_t>(rhsReshape.size())})
              .getValue();
      rhs = rewriter
                .create<mhlo::DynamicReshapeOp>(
                    op->getLoc(),
                    RankedTensorType::get(rhsReshape, rhsTy.getElementType()),
                    rhs, rhsReshapeConst)
                .getResult();
      rhsTy = rhs.getType().cast<RankedTensorType>();
      rhsShape = rhsTy.getShape();
      rhsRank = rhsTy.getRank();
    } else if (rhsRank > lhsRank) {
      SmallVector<int64_t> lhsReshape(rhsRank, 1);
      int64_t rankDelta = rhsRank - lhsRank;
      std::copy(lhsShape.begin(), lhsShape.end(),
                lhsReshape.begin() + rankDelta);
      auto lhsReshapeConst =
          mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(lhsReshape),
                               {static_cast<int64_t>(lhsReshape.size())})
              .getValue();
      lhs = rewriter
                .create<mhlo::DynamicReshapeOp>(
                    op->getLoc(),
                    RankedTensorType::get(lhsReshape, lhsTy.getElementType()),
                    lhs, lhsReshapeConst)
                .getResult();
      lhsTy = lhs.getType().cast<RankedTensorType>();
      lhsShape = lhsTy.getShape();
      lhsRank = lhsTy.getRank();
    }

    SmallVector<int64_t> lhsBroadcastShape(lhsShape.begin(), lhsShape.end());
    SmallVector<int64_t> rhsBroadcastShape(rhsShape.begin(), rhsShape.end());
    SmallVector<int64_t> batchDims;

    // loop along batch dims
    for (int64_t i = 0; i < lhsRank - 2; i++) {
      batchDims.push_back(i);
      if (lhsBroadcastShape[i] == rhsBroadcastShape[i]) {
        continue;
      } else if (lhsBroadcastShape[i] == 1) {
        lhsBroadcastShape[i] = rhsBroadcastShape[i];
      } else if (rhsBroadcastShape[i] == 1) {
        rhsBroadcastShape[i] = lhsBroadcastShape[i];
      } else {
        return op.emitError("unbroadcastable operands");
      }
    }

    SmallVector<int64_t> outShape(lhsBroadcastShape.begin(),
                                  lhsBroadcastShape.end());
    outShape[outShape.size() - 1] =
        rhsBroadcastShape[rhsBroadcastShape.size() - 1];
    mhlo::DotDimensionNumbersAttr dotDimensionNumbers =
        mhlo::DotDimensionNumbersAttr::get(
            rewriter.getContext(), /*lhsBatchingDimensions=*/batchDims,
            /*rhsBatchingDimensions=*/batchDims,
            /*lhsContractingDimensions=*/{lhsRank - 1},
            /*rhsContractingDimensions=*/{rhsRank - 2});
    mlir::ArrayAttr precisionConfig;
    lhs = mhlo::promoteAndBroadcast(
        rewriter, lhs,
        RankedTensorType::get(lhsBroadcastShape, lhsTy.getElementType()));
    rhs = mhlo::promoteAndBroadcast(
        rewriter, rhs,
        RankedTensorType::get(rhsBroadcastShape, rhsTy.getElementType()));
    auto mulResult = rewriter.create<mhlo::DotGeneralOp>(
        op.getLoc(), RankedTensorType::get(outShape, resultTy.getElementType()),
        lhs, rhs, dotDimensionNumbers, precisionConfig);
    auto outShapeConst =
        mhlo::getConstTensor(rewriter, op, resultTy.getShape(),
                             {static_cast<int64_t>(resultTy.getShape().size())})
            .getValue();
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(op, resultTy, mulResult,
                                                        outShapeConst);
    return success();
  }
  return failure();
}

// AtenConvolutionOp
template <>
LogicalResult ConvertAtenLinearOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value input = adaptor.input();
  Value weight = adaptor.weight();

  // The input shape is [N x C x H x W]
  auto inputTy = input.getType().template cast<RankedTensorType>();
  // The weight shape is [OC x (IC // groups) x KH x KW]
  // If tranposed is set to true, the weight shape changes to [IC x (OC //
  // groups) x KH x KW]
  auto weightTy = weight.getType().template cast<RankedTensorType>();
  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .template cast<RankedTensorType>();

  if (!inputTy || !weightTy || !outTy) {
    return op.emitError(
        "Input, weight and output to Convolution must be ranked tensors");
  }

  if (inputTy.getRank() < 3)
    return op.emitError(
        "Convolution op only operates on input with at least 3 dims");

  SmallVector<int64_t> stride;
  if (!matchPattern(op.stride(), m_TorchConstantIntList(stride))) {
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");
  }

  SmallVector<int64_t> padding;
  if (!matchPattern(op.padding(), m_TorchConstantIntList(padding))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");
  }

  SmallVector<int64_t> dilation;
  if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilation))) {
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");
  }
  SmallVector<int64_t> outputPadding;
  if (!matchPattern(op.output_padding(),
                    m_TorchConstantIntList(outputPadding))) {
    return rewriter.notifyMatchFailure(
        op, "non-const output_padding list unsupported");
  }
  // Just ignore the outputPadding attribute
  for (int64_t item : outputPadding) {
    if (item != 0)
      return op.emitError(
          "Unimplemented: only zero output_padding list supported");
  }

  int64_t groups;
  if (!matchPattern(op.groups(), m_TorchConstantInt(&groups))) {
    return rewriter.notifyMatchFailure(op, "non-int groups unsupported");
  }

  bool transposed;
  if (!matchPattern(op.transposed(), m_TorchConstantBool(&transposed))) {
    return rewriter.notifyMatchFailure(op, "non-bool transposed unsupported");
  }
  if (transposed) {
    return op.emitError(
        "Unimplemented: only tranposed param with value 'false' supported!");
  }

  // Get mhlo::ConvOp attributes
  DenseIntElementsAttr mhloWindowStride = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<long int>(stride.size())},
                            rewriter.getI64Type()),
      stride);
  std::vector<int64_t> mhloPaddingVec;
  for (size_t i = 0; i < padding.size(); i++) {
    mhloPaddingVec.emplace_back(padding[i]);
    mhloPaddingVec.emplace_back(padding[i]);
  }

  DenseIntElementsAttr mhloPadding = DenseIntElementsAttr::get(
      RankedTensorType::get(
          {static_cast<long int>(stride.size()), static_cast<long int>(2)},
          rewriter.getI64Type()),
      mhloPaddingVec);
  
  DenseIntElementsAttr mhloRhsDilation = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<long int>(dilation.size())},
                            rewriter.getI64Type()),
      dilation);

  SmallVector<int64_t> spatialDimensions;
  for (int64_t i = 2; i < inputTy.getRank(); i++) {
    spatialDimensions.emplace_back(i);
  }
  mhlo::ConvDimensionNumbersAttr dimensionNumbers =
      mhlo::ConvDimensionNumbersAttr::get(
          rewriter.getContext(), 0, 1, spatialDimensions, 1, 0,
          spatialDimensions, 0, 1, spatialDimensions);
  
  IntegerAttr featureGroupCount =
      IntegerAttr::get(rewriter.getI64Type(), groups);
  IntegerAttr batchGroupCount = IntegerAttr::get(rewriter.getI64Type(), 1);

  // mhlo::ConvOp optional attributes, leave them as default
  DenseIntElementsAttr mhloLhsDilation;
  DenseElementsAttr windowReversal;
  ArrayAttr precisionConfig;

  auto mhloConvOp = rewriter.create<mhlo::ConvOp>(
      op->getLoc(), outTy, input, weight, mhloWindowStride, mhloPadding,
      mhloLhsDilation, mhloRhsDilation, windowReversal, dimensionNumbers,
      featureGroupCount, batchGroupCount, precisionConfig);

  auto bias = adaptor.bias();
  
  if (failed(checkNotNone(rewriter, op, op.bias()))) { // No bias given
    rewriter.replaceOp(op, mhloConvOp.getResult());
    return success();
  } else {
    if (!bias.getType().cast<RankedTensorType>()) {
      return op.emitError("Bias provided but not a ranked tensor");
    }

    auto biasTy = bias.getType().template cast<RankedTensorType>();
    if (!biasTy.getElementType().isIntOrFloat()) {
      return op.emitError("Only floating-point or integer datatype "
                          "legalization for bias supported");
    }

    // Reshape and promote bias
    SmallVector<int64_t> outBiasShape(outTy.getRank(), 1);
    // The output feature dimension are in 1-th dim.
    outBiasShape[1] = biasTy.getShape()[0];

    llvm::Optional<Value> outBiasShapeConst =
        mhlo::getConstTensor(rewriter, op, llvm::makeArrayRef(outBiasShape),
                             {static_cast<int64_t>(outBiasShape.size())});

    auto reshapeOutTy =
        RankedTensorType::get(outBiasShape, biasTy.getElementType());
    Value reshapeBias =
        rewriter
            .create<mhlo::DynamicReshapeOp>(op.getLoc(), reshapeOutTy, bias,
                                            outBiasShapeConst.getValue())
            .getResult();
    auto biasTensor = mhlo::promoteAndBroadcast(rewriter, reshapeBias, outTy);

    // Add bias
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(op, outTy, mhloConvOp.getResult(),
                                             biasTensor);
    return success();
  }
}

} // namespace

void mlir::torch::torch_to_mhlo::populateLinearOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenMmOp>();
  patterns.add<ConvertAtenMmOp>(typeConverter, context);

  target.addIllegalOp<AtenBmmOp>();
  patterns.add<ConvertAtenBmmOp>(typeConverter, context);

#define INSERT_ATEN_LINEAR_OP_PATTERN(AtenOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);

  INSERT_ATEN_LINEAR_OP_PATTERN(AtenConvolutionOp);
  INSERT_ATEN_LINEAR_OP_PATTERN(AtenMatmulOp);
  INSERT_ATEN_LINEAR_OP_PATTERN(AtenLinearOp);
#undef INSERT_ATEN_LINEAR_OP_PATTERN
}