//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_LIB_CONVERSION_TORCHTOSTABLEHLO_POPULATEPATTERNS_H
#define TORCHMLIR_LIB_CONVERSION_TORCHTOSTABLEHLO_POPULATEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace torch {
namespace torch_to_stablehlo {

struct TorchToStablehloOptions {
  bool enableStaticShape = false;
  size_t dimSizeIndexBits = 64;
  bool defaultComputeOnFp32 = false;
};

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpAdaptor = typename AtenOpT::Adaptor;
  ConvertAtenOp(TypeConverter &typeConverter, MLIRContext *context,
                const TorchToStablehloOptions &options)
      : OpConversionPattern<AtenOpT>(typeConverter, context) {
    this->options = options;
  }
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(op, "haven't been implemented");
  }
  const TorchToStablehloOptions &getOptions() const { return options; }

private:
  TorchToStablehloOptions options;
};

void populateBasicOpPatternsAndLegality(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target,
                                        const TorchToStablehloOptions &options);
void populateViewLikeOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options);
void populateGatherScatterOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options);
void populateReductionOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options);
void populateLinearOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options);

void populatePoolingOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const TorchToStablehloOptions &options);

void populateRngOpPatternsAndLegality(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      ConversionTarget &target,
                                      const TorchToStablehloOptions &options);

} // namespace torch_to_stablehlo
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_LIB_CONVERSION_TORCHTOSTABLEHLO_POPULATEPATTERNS_H
