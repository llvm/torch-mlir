//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class EvaluateRandnOp : public OpRewritePattern<AtenRandnOp> {
public:

EvaluateRandnOp(MLIRContext *context, ArrayRef<float> randomValues, int& randomValuesItr)
      : OpRewritePattern(context), randomValues(randomValues), randomValuesItr(randomValuesItr) {}

  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnOp op,
                                PatternRewriter &rewriter) const override {

    // Get the result type of the op
    auto resultType = dyn_cast<ValueTensorType>(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "Result type is not a ValueTensorType");
    }
    
    auto resultShape = resultType.getSizes();
    auto resultDtype = resultType.getDtype();
    
    // Calculate the total number of elements
    int64_t numElements = 1;
    for (auto dim : resultShape) {
      numElements *= dim;
    }

    // Check if we have enough random values
    if(randomValuesItr + numElements > (int64_t)randomValues.size()) {
      return rewriter.notifyMatchFailure(op, "Not enough random values provided");
    }
    
    DenseElementsAttr attr;
    auto attrType = RankedTensorType::get(resultShape, resultDtype);

    if (isa<mlir::FloatType>(resultDtype)) {

      SmallVector<APFloat> values;
      for (int64_t i = 0; i < numElements; ++i) {
       values.push_back(APFloat(randomValues[randomValuesItr++]));
      }
      attr = DenseElementsAttr::get(attrType, values);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported dtype for AtenRandnOp");
    }
    
    // Replace with torch.vtensor.literal
    rewriter.replaceOpWithNewOp<Torch::ValueTensorLiteralOp>(op, resultType, attr);

    return success();
  }

  private:
    ArrayRef<float> randomValues;
    int& randomValuesItr;
};

class EvaluateRandnOpsPass : public EvaluateRandnOpsBase<EvaluateRandnOpsPass> {
public:
   
  void runOnOperation() override {
   
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    int randomValuesItr = 0;
    patterns.insert<EvaluateRandnOp>(context, randomValues, randomValuesItr);
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
    
    // Warn if not all random values were used
    if(randomValuesItr < (int)randomValues.size()) {
      getOperation().emitWarning() << "Not all random values were used";
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createEvaluateRandnOpsPass(ArrayRef<float> randomValues) {
  return std::make_unique<EvaluateRandnOpsPass>();
}
