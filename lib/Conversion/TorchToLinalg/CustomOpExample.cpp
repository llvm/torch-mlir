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
#include "Utils.h"
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
class ConvertCustomOpExample
  : public OpConversionPattern<TorchMlirCustomOpExampleIdentityOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(TorchMlirCustomOpExampleIdentityOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter
                                ) const override {
    // Type checks.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    // Since the example op does nothing, we simply replace the uses of the
    // return value with its argument, then remove the op.
    rewriter.replaceOp(op, op->getOperands());

    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateCustomOpExamplePatternsAndLegality(
    TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<TorchMlirCustomOpExampleIdentityOp>();
  patterns.add<ConvertCustomOpExample>(typeConverter, context);
}
