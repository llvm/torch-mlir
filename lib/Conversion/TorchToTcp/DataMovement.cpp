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
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern<AtenCatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenCatOp catOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> inputs;
    if (!getListConstructElements(adaptor.getTensors(), inputs))
      return rewriter.notifyMatchFailure(
          catOp, "aten.cat operands must be a list of tensors");

    auto tensorInputs = getTypeConvertedValues(rewriter, catOp->getLoc(),
                                               getTypeConverter(), inputs);

    int64_t dim;
    if (!matchPattern(catOp.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          catOp, "aten.cat dim must be constant integer");

    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(catOp.getType())
                                      .cast<RankedTensorType>();

    dim = toPositiveDim(dim, resultType.getRank());

    rewriter.replaceOpWithNewOp<tcp::ConcatOp>(
        catOp, resultType, tensorInputs,
        rewriter.getIntegerAttr(rewriter.getI64Type(), dim));
    return success();
  }
};
} // namespace

void torch_to_tcp::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenCatOp>();
  patterns.add<ConvertAtenCatOp>(typeConverter, context);
}
