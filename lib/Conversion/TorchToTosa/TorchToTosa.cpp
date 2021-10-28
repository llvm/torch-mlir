//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
#define DECL_CONVERT_ATENOP(aten_op)                                              \
class ConvertAten##aten_op##Op : public OpConversionPattern<Aten##aten_op##Op> {  \
public:                                                                           \
  using OpConversionPattern::OpConversionPattern;                                 \
  LogicalResult                                                                   \
  matchAndRewrite(Aten##aten_op##Op op, ArrayRef<Value> operands,                 \
                  ConversionPatternRewriter &rewriter) const override;            \
}; 
DECL_CONVERT_ATENOP(Tanh)
DECL_CONVERT_ATENOP(Sigmoid)
#undef DECL_CONVERT_ATENOP

LogicalResult
ConvertAtenTanhOp::matchAndRewrite(AtenTanhOp op, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const {
  AtenTanhOp::Adaptor adaptor(operands);
  rewriter.replaceOpWithNewOp<tosa::TanhOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self());
  return success();
}

LogicalResult
ConvertAtenSigmoidOp::matchAndRewrite(AtenSigmoidOp op, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const {
  AtenSigmoidOp::Adaptor adaptor(operands);
  rewriter.replaceOpWithNewOp<tosa::SigmoidOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.self());
  return success();
}

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosa Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosa
    : public ConvertTorchToTosaBase<ConvertTorchToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

#define INSERT_NEW_PATTERN(aten_op)                               \
  target.addIllegalOp<Aten##aten_op##Op>();                       \
  patterns.add<ConvertAten##aten_op##Op>(typeConverter, context); 
INSERT_NEW_PATTERN(Tanh);
INSERT_NEW_PATTERN(Sigmoid);
#undef INSERT_NEW_PATTERN

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::createConvertTorchToTosaPass() {
  return std::make_unique<ConvertTorchToTosa>();
}
