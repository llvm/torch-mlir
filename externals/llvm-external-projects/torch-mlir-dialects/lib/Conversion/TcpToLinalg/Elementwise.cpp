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

Value createElementwiseLinalgGeneric(
    OpBuilder &b, Location loc, ValueRange tensorOperands,
    RankedTensorType resultTensorType,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilder) {
  auto resultRank = resultTensorType.getRank();
  SmallVector<Value> resultDimSizes;
  SmallVector<AffineMap> indexingMaps;

  // In order to populate the resultDimSizes, we only need to look at one of
  // the tensorOperands, since all the operands are expected to have the same
  // shape.
  auto tensorOperand = tensorOperands[0];
  for (int64_t i = 0; i < resultRank; ++i) {
    resultDimSizes.push_back(b.createOrFold<tensor::DimOp>(loc, tensorOperand, i));
  }

  // Add indexing maps for all the tensor operands and for the result.
  for (size_t i = 0; i < tensorOperands.size(); ++i) {
    indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));
  }
  indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

  SmallVector<StringRef> iteratorTypes(resultRank,
                                       getParallelIteratorTypeName());

  Value emptyTensor = b.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(resultDimSizes), resultTensorType.getElementType());
  return b.create<linalg::GenericOp>(
      loc, emptyTensor.getType(), tensorOperands, emptyTensor, indexingMaps,
      iteratorTypes, bodyBuilder).getResult(0);
}

template<typename TcpOpT>
Value createLinalgPayloadForElementwiseOp(
    TcpOpT op, RankedTensorType resultTensorType, OpBuilder &b, ValueRange payloadArgs) {
  Location loc = op->getLoc();
  if (isa<TanhOp>(op)) {
    return b.create<math::TanhOp>(loc, payloadArgs[0]);
  }
  if (isa<AddOp>(op)) {
    auto elemType = resultTensorType.getElementType();
    if (elemType.isa<mlir::FloatType>()) {
      return b.create<arith::AddFOp>(loc, payloadArgs[0], payloadArgs[1]);
    } else {
      return b.create<arith::AddIOp>(loc, payloadArgs[0], payloadArgs[1]);
    }
  }
  op->emitError("unimplemented lowering in createLinalgPayloadForElementwiseOp");
  return nullptr;
}

template<typename TcpOpT>
class ConvertElementwiseOp : public OpConversionPattern<TcpOpT> {
public:
  using OpConversionPattern<TcpOpT>::OpConversionPattern;
  using OpAdaptor = typename TcpOpT::Adaptor;

  LogicalResult matchAndRewrite(
            TcpOpT op,
            OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTensorType = OpConversionPattern<TcpOpT>::getTypeConverter()
                  ->convertType(op->getResult(0).getType()).template cast<RankedTensorType>();
    auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
        adaptor.getOperands(), [](Value v) { return v.getType().isa<RankedTensorType>(); }));

    // Create Linalg payload
    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      Value result = createLinalgPayloadForElementwiseOp<TcpOpT>(op, resultTensorType, b, payloadArgs);
      b.create<linalg::YieldOp>(loc, result);
    };

    Value generic = createElementwiseLinalgGeneric(
        rewriter, loc, tensorOperands, resultTensorType, bodyBuilder);
    rewriter.replaceOp(op, generic);
    return success();
  }
};

} // namespace

void mlir::tcp_to_linalg::populateElementwisePatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<TanhOp>();
  patterns.add<ConvertElementwiseOp<TanhOp>>(typeConverter, context);

  target.addIllegalOp<AddOp>();
  patterns.add<ConvertElementwiseOp<AddOp>>(typeConverter, context);
}
