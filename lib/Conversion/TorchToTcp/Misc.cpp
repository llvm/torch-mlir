//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTcp/TorchToTcp.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertAtenBroadcastToOp : public OpConversionPattern<AtenBroadcastToOp> {
public:
  using OpConversionPattern<AtenBroadcastToOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenBroadcastToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();

    SmallVector<Value> newDimSizes;
    if (!getListConstructElements(op.getSize(), newDimSizes))
      return rewriter.notifyMatchFailure(
          op, "Broadcasted shape must be a list of scalars");

    int64_t newLeadingDims = newDimSizes.size() - inputType.getRank();
    if (newLeadingDims > 0) {
      input = torch_to_tcp::broadcastRankInLeadingDims(rewriter, input,
                                                       newLeadingDims);
    }

    SmallVector<int64_t> axes;
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < static_cast<int64_t>(newDimSizes.size()); ++i) {
      Value newDimSize = newDimSizes[i];
      int64_t staticDimSize;
      if (i < newLeadingDims ||
          !matchPattern(newDimSize, m_TorchConstantInt(&staticDimSize)) ||
          staticDimSize != -1) {
        axes.push_back(i);
        newDimSize = rewriter.create<torch::TorchConversion::ToI64Op>(
            op->getLoc(), newDimSize);
        resultShape.push_back(rewriter.create<arith::IndexCastOp>(
            op->getLoc(), rewriter.getIndexType(), newDimSize));
      }
    }

    RankedTensorType resultType =
        OpConversionPattern<AtenBroadcastToOp>::getTypeConverter()
            ->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();

    auto axesAttr = rewriter.getI64ArrayAttr(axes);
    rewriter.replaceOpWithNewOp<tcp::BroadcastOp>(op, resultType, input,
                                                  resultShape, axesAttr);
    return success();
  }
};

class ConvertValueTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  using OpConversionPattern<ValueTensorLiteralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType =
        OpConversionPattern<ValueTensorLiteralOp>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    if (auto elements = op.getValueAttr().dyn_cast<DenseIntElementsAttr>()) {
      Type elementType = resultType.getElementType();
      auto denseIntAttr = elements.mapValues(elementType, [&](const APInt &v) {
        return APInt(elementType.getIntOrFloatBitWidth(), v.getSExtValue());
      });
      rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType, denseIntAttr);
      return success();
    }

    rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType,
                                              adaptor.getValue());
    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesPatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<RankedTensorType>();

    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Output tensors must have integer or floating-point datatype");

    if (!op.getLayout().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op,
                                         "Only default layout is supported");

    bool pinMemory;
    if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported pin_memory, should be either None or false");
    }

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output type");
    }

    Operation *primListOp = op.getSize().getDefiningOp();
    auto listConstruct = dyn_cast<Torch::PrimListConstructOp>(primListOp);
    if (!listConstruct) {
      return rewriter.notifyMatchFailure(
          op, "Size must come from PrimListConstructOp");
    }
    SmallVector<Value> primListVal;
    for (Value value : listConstruct.getElements()) {
      primListVal.push_back(value);
    }

    Value resultOp = torch_to_tcp::broadcast0DOr1DFromPrimList(
        rewriter, constOp, primListVal);

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesLikePatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<RankedTensorType>();

    if (!outType) {
      return rewriter.notifyMatchFailure(
          op, "Only Ranked Tensor types are supported in TCP");
    }

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Output tensors must have integer or floating-point datatype");
    }

    // TODO: Check the attribute for input vtensor
    int64_t memoryLayout;
    if (!op.getLayout().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getLayout(), m_TorchConstantInt(&memoryLayout)) ||
         memoryLayout != 0)) {
      return rewriter.notifyMatchFailure(op,
                                         "Only default layout is supported");
    }

    bool pinMemory;
    if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported pin_memory, should be either None or false");
    }

    if (!op.getMemoryFormat().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Only default memory format is supported");

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output type");
    }

    Value resultOp = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, constOp, input, /*axisInOutput=*/0,
        /*useInputAsResultType=*/true);

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

} // namespace

void torch_to_tcp::populateMiscPatternsAndLegality(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<AtenBroadcastToOp>();
  patterns.add<ConvertAtenBroadcastToOp>(typeConverter, context);

  target.addIllegalOp<ValueTensorLiteralOp>();
  patterns.add<ConvertValueTensorLiteralOp>(typeConverter, context);

  target.addIllegalOp<AtenZerosOp>();
  patterns.add<ConvertAtenZerosOnesPatternOp<AtenZerosOp, 0>>(typeConverter,
                                                              context);
  target.addIllegalOp<AtenOnesOp>();
  patterns.add<ConvertAtenZerosOnesPatternOp<AtenOnesOp, 1>>(typeConverter,
                                                             context);

  target.addIllegalOp<AtenZerosLikeOp>();
  patterns.add<ConvertAtenZerosOnesLikePatternOp<AtenZerosLikeOp, 0>>(
      typeConverter, context);
  target.addIllegalOp<AtenOnesLikeOp>();
  patterns.add<ConvertAtenZerosOnesLikePatternOp<AtenOnesLikeOp, 1>>(
      typeConverter, context);
}
