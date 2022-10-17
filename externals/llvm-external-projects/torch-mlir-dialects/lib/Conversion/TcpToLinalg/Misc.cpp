//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Conversion/TcpToLinalg/TcpToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpDialect.h"
#include "torch-mlir-dialects/Dialect/Tcp/IR/TcpOps.h"

using namespace mlir;
using namespace mlir::tcp;

namespace {

void getValuesFromIndexArrayAttribute(ArrayAttr attr,
                                      SmallVector<int64_t> &arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

class ConvertBroadcastOp : public OpConversionPattern<BroadcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
              BroadcastOp op,
              OpAdaptor adaptor,
              ConversionPatternRewriter &b) const override {
    Location loc = op->getLoc();
    auto resultTensorType = OpConversionPattern::getTypeConverter()
                  ->convertType(op->getResult(0).getType()).template cast<RankedTensorType>();
    auto inputTensor = op->getOperands()[0];

    SmallVector<int64_t> axes;
    getValuesFromIndexArrayAttribute(op.getAxes(), axes);

    auto resultRank = resultTensorType.getRank();
    SmallVector<Value> resultDimSizes;
    SmallVector<AffineExpr> exprs;
    int64_t pos = 0;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (pos < static_cast<int64_t>(axes.size()) && axes[pos] == i) {
        resultDimSizes.push_back(op->getOperands()[pos+1]);
        exprs.push_back(b.getAffineConstantExpr(0));
        ++pos;
      } else {
        resultDimSizes.push_back(b.createOrFold<tensor::DimOp>(loc, inputTensor, i));
        exprs.push_back(b.getAffineDimExpr(i));
      }
    }

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(AffineMap::get(resultRank, 0, exprs, b.getContext()));
    indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

    SmallVector<StringRef> iteratorTypes(resultRank,
                                         getParallelIteratorTypeName());

    Value emptyTensor = b.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(resultDimSizes), resultTensorType.getElementType());

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      b.create<linalg::YieldOp>(loc, payloadArgs[0]);
    };
    Value generic = b.create<linalg::GenericOp>(loc,
                                       emptyTensor.getType(),
                                       inputTensor,
                                       emptyTensor,
                                       indexingMaps,
                                       iteratorTypes,
                                       bodyBuilder).getResult(0);
    b.replaceOp(op, generic);
    return success();
  }
};

} // namespace

void mlir::tcp_to_linalg::populateMiscPatternsAndLegality(TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<BroadcastOp>();
  patterns.add<ConvertBroadcastOp>(typeConverter, context);
}