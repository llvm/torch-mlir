//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenSizeIntOp : public OpConversionPattern<AtenSizeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSizeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.getSelf();
    Value dim = adaptor.getDim();
    auto type = self.getType().cast<RankedTensorType>();
    Value inputRank = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(type.getRank()));
    Value dimPositive = toPositiveDimDynamic(rewriter, loc, dim, inputRank);
    if (!isAssumingStrictSymbolicShapes(rewriter)) {
      assertIsValidDim(rewriter, loc, dimPositive, inputRank);
    }
    Value size = rewriter.create<tensor::DimOp>(
        loc, adaptor.getSelf(), castIntToIndex(rewriter, loc, dimPositive));
    rewriter.replaceOp(op, castIndexToInt64(rewriter, loc, size));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenNumelOp : public OpConversionPattern<AtenNumelOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNumelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value tensorSize = getTensorSize(rewriter, loc, adaptor.getSelf());
    rewriter.replaceOp(op, tensorSize);
    return success();
  }
};
} // namespace

namespace {
// Casts a tensor of exactly one element to an elemental type.
template <typename OpTy>
class ConvertAtenTensorToScalarLikeOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value input = adaptor.getA();
    SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
    int64_t inputRank = inputSizes.size();
    Type inputDtype =
        op.getA().getType().template cast<BaseTensorType>().getDtype();

    // The `input` tensor must contain exactly one element, i.e., either the
    // `input` is a zero rank tensor or all the dimensions of the `input` tensor
    // are unit.
    Value constantOne =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(1));
    for (int64_t i = 0; i < inputRank; i++)
      checkDimEqualHelper(rewriter, loc, inputSizes[i], constantOne);

    // Extract the only element from the `input` tensor.
    Value constantZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    SmallVector<Value> indices(inputRank, constantZero);
    Value result = rewriter.create<tensor::ExtractOp>(loc, input, indices);
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.replaceOp(op, convertScalarToDtype(rewriter, loc, result,
                                                resultType, inputDtype));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenScalarToTensorLike : public ConversionPattern {
public:
  ConvertAtenScalarToTensorLike(TypeConverter &typeConverter,
                                MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenTensorIntOp, AtenTensorFloatOp>(op))
      return rewriter.notifyMatchFailure(
          op, "not a supported Scalar to Tensor like op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value elemVal, dtype, device, requires_grad;
    if (AtenTensorIntOp tensorIntOp = dyn_cast<AtenTensorIntOp>(op)) {
      AtenTensorIntOp::Adaptor adaptor(operands);
      elemVal = adaptor.getT();
      dtype = tensorIntOp.getDtype();
      device = tensorIntOp.getDevice();
      requires_grad = tensorIntOp.getRequiresGrad();
    }
    if (AtenTensorFloatOp tensorFloatOp = dyn_cast<AtenTensorFloatOp>(op)) {
      AtenTensorFloatOp::Adaptor adaptor(operands);
      elemVal = adaptor.getT();
      dtype = tensorFloatOp.getDtype();
      device = tensorFloatOp.getDevice();
      requires_grad = tensorFloatOp.getRequiresGrad();
    }
    // TODO: Dtype conversion.
    if (!dtype.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "Unimplemented non-None dtype");

    // TODO: Device information.
    if (!device.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None device information");

    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();
    Type outElementType = resultType.getElementType();
    Value elemValProm =
        convertScalarToDtype(rewriter, loc, elemVal, outElementType);
    Value zeroDTensor =
        createInitTensor(rewriter, loc, {}, outElementType, elemValProm);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, zeroDTensor);
    return success();
  }
};
} // namespace

namespace {
class ConvertPrimNumToTensorScalarOp
    : public OpConversionPattern<PrimNumToTensorScalarOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PrimNumToTensorScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value a = adaptor.getA();
    Value outTensor =
        rewriter
            .create<tensor::EmptyOp>(loc, ArrayRef<OpFoldResult>{}, a.getType())
            ->getResult(0);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, a, outTensor);

    return success();
  }
};
} // namespace

namespace {
// Converts a tensor with one element to a scalar value.
template <typename OpTy>
class ConvertAtenImplicitLikeOp : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OpConversionPattern<OpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, adaptor.getA());
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::
    populateTensorScalarInteropPatternsAndLegality(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenSizeIntOp>();
  patterns.add<ConvertAtenSizeIntOp>(typeConverter, context);
  target.addIllegalOp<AtenNumelOp>();
  patterns.add<ConvertAtenNumelOp>(typeConverter, context);
  target.addIllegalOp<AtenIntTensorOp, AtenFloatTensorOp, AtenBoolTensorOp>();
  patterns.add<ConvertAtenTensorToScalarLikeOp<AtenIntTensorOp>>(typeConverter,
                                                                 context);
  patterns.add<ConvertAtenTensorToScalarLikeOp<AtenFloatTensorOp>>(
      typeConverter, context);
  patterns.add<ConvertAtenTensorToScalarLikeOp<AtenBoolTensorOp>>(typeConverter,
                                                                  context);
  target.addIllegalOp<AtenTensorIntOp, AtenTensorFloatOp>();
  patterns.add<ConvertAtenScalarToTensorLike>(typeConverter, context);
  target.addIllegalOp<PrimNumToTensorScalarOp>();
  patterns.add<ConvertPrimNumToTensorScalarOp>(typeConverter, context);
  patterns.add<ConvertAtenImplicitLikeOp<AtenScalarImplicitOp>>(typeConverter,
                                                                context);
  patterns.add<ConvertAtenImplicitLikeOp<AtenFloatImplicitOp>>(typeConverter,
                                                               context);
  patterns.add<ConvertAtenImplicitLikeOp<AtenIntImplicitOp>>(typeConverter,
                                                             context);
  target.addIllegalOp<AtenScalarImplicitOp, AtenFloatImplicitOp,
                      AtenIntImplicitOp>();
}
