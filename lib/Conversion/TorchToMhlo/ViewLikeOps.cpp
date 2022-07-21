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
#include "./PopulatePatterns.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;


namespace {

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenViewOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto rankType =
        adaptor.self().getType().template dyn_cast<RankedTensorType>();
    if (!rankType)
      return op.emitError("Only ranked tensor types are currently supported");

    SmallVector<Value, 4> dimSizes;
    if (!getAtenViewOpSizes(op, adaptor, rewriter, dimSizes)) {
      return op.emitError("Dims size must be a list of Scalar");
    }

    auto loc = op.getLoc();
    auto newRank = dimSizes.size();
    if (newRank == 0 || rankType.getRank() == 0) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          adaptor.self());
      return success();
    }

    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
      dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
      return dSize;
    });

#ifdef TORCH_MLIR_ENABLE_MHLO_TRUNC_DIMSIZE_TO_I32
    // The i64 calculation is much slower than i32 on some devices, such as Nvidia GPU.
    // One can truncate from i64 to i32 since dimension sizes are unlikely to exceed
    // the range of i32(4GiB)
    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
      // dimSize: cast i64 -> i32
      dSize = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), dSize);
      return dSize;
    });
#endif

    Value mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
    rewriter.replaceOpWithNewOp<chlo::DynamicReshapeOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self(),
        mhloShape);
    return success();
  }

  bool getAtenViewOpSizes(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<Value, 4>& dimSizes) const;
};

template <>
bool ConvertAtenViewOp<AtenViewOp>::getAtenViewOpSizes(
    AtenViewOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter,
    SmallVector<Value, 4>& dimSizes) const {
  return getListConstructElements(adaptor.size(), dimSizes);
}

template <>
bool ConvertAtenViewOp<AtenReshapeOp>::getAtenViewOpSizes(
    AtenReshapeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter,
    SmallVector<Value, 4>& dimSizes) const {
  return getListConstructElements(adaptor.shape(), dimSizes);
}

} // namespace

void mlir::torch::torch_to_mhlo::populateViewLikeOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

#define INSERT_VIEW_OP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();       \
  patterns.add<ConvertAtenViewOp<AtenOp>>(typeConverter, context);
    INSERT_VIEW_OP_PATTERN(AtenViewOp);
    INSERT_VIEW_OP_PATTERN(AtenReshapeOp);
#undef INSERT_VIEW_OP_PATTERN
}
