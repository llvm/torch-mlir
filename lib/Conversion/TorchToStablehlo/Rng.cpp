//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "./PopulatePatterns.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

template <>
LogicalResult ConvertAtenOp<AtenUniformOp>::matchAndRewrite(
    AtenUniformOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  Value generator = adaptor.getGenerator();
  Location loc = op.getLoc();

  if (!isa<Torch::NoneType>(generator.getType()))
    return rewriter.notifyMatchFailure(
        op, "The generator has to be None because only global default "
            "generator is supported");

  auto elements = cast<RankedTensorType>(self.getType()).getShape();
  if (llvm::any_of(elements,
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return rewriter.notifyMatchFailure(op, "Dynamic shape support TBD");
  auto shape_tensor = stablehlo::ConstantOp::create(
      rewriter, loc, rewriter.getI64TensorAttr(elements));
  auto outTy = getTypeConverter()->convertType(op.getType());
  auto outElemTy = cast<RankedTensorType>(outTy).getElementType();
  Value from =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getFrom(), outElemTy);
  Value to =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getTo(), outElemTy);
  rewriter.replaceOpWithNewOp<stablehlo::RngOp>(
      op, outTy, from, to, shape_tensor, stablehlo::RngDistribution::UNIFORM);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenRandnGeneratorOp>::matchAndRewrite(
    AtenRandnGeneratorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value generator = adaptor.getGenerator();
  Location loc = op.getLoc();

  if (!isa<Torch::NoneType>(generator.getType())) {
    return rewriter.notifyMatchFailure(
        op, "The generator has to be None because only global default "
            "generator is supported");
  }
  llvm::SmallVector<int64_t> shape;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(shape))) {
    return rewriter.notifyMatchFailure(op, "size must be constant");
  }

  auto outTy = getTypeConverter()->convertType(op.getType());
  auto outElemTy = cast<RankedTensorType>(outTy).getElementType();
  if (!isa<mlir::FloatType>(outElemTy)) {
    return rewriter.notifyMatchFailure(op,
                                       "only support output with float type");
  }
  auto scalarTy = RankedTensorType::get({}, outElemTy);

  Value shapeTensor = stablehlo::ConstantOp::create(
      rewriter, loc, rewriter.getI64TensorAttr(shape));
  Value mean = stablehlo::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(scalarTy, rewriter.getFloatAttr(outElemTy, 0.0)));
  Value var = stablehlo::ConstantOp::create(
      rewriter, loc,
      DenseElementsAttr::get(scalarTy, rewriter.getFloatAttr(outElemTy, 1.0)));

  rewriter.replaceOpWithNewOp<stablehlo::RngOp>(
      op, outTy, mean, var, shapeTensor, stablehlo::RngDistribution::NORMAL);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenNormalFunctionalOp>::matchAndRewrite(
    AtenNormalFunctionalOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  Value generator = adaptor.getGenerator();
  Location loc = op.getLoc();

  if (!isa<Torch::NoneType>(generator.getType()))
    return rewriter.notifyMatchFailure(
        op, "The generator has to be None because only global default "
            "generator is supported");

  auto elements = cast<RankedTensorType>(self.getType()).getShape();
  if (llvm::any_of(elements,
                   [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return rewriter.notifyMatchFailure(op, "Dynamic shape support TBD");
  auto shapeTensor = stablehlo::ConstantOp::create(
      rewriter, loc, rewriter.getI64TensorAttr(elements));
  auto outTy = getTypeConverter()->convertType(op.getType());
  auto outElemTy = cast<RankedTensorType>(outTy).getElementType();
  Value mean =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getMean(), outElemTy);
  Value std =
      hlo::scalarToStablehloTensor(rewriter, op, adaptor.getStd(), outElemTy);
  rewriter.replaceOpWithNewOp<stablehlo::RngOp>(
      op, outTy, mean, std, shapeTensor, stablehlo::RngDistribution::NORMAL);
  return success();
}

void mlir::torch::torch_to_stablehlo::populateRngOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context, options)

  INSERT_ATENOP_PATTERN(AtenUniformOp);
  INSERT_ATENOP_PATTERN(AtenRandnGeneratorOp);
  INSERT_ATENOP_PATTERN(AtenNormalFunctionalOp);
#undef INSERT_ATENOP_PATTERN
}
