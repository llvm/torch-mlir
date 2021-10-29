//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Helper funtion to get rank of `Base tensor type`.
// -1 is returned if the tensorRank can't be determined.
static int getTensorRank(Value tensor) {
  int tensorRank = -1;
  BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();

  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> tensorShape = tensorType.getSizes();
    tensorRank = tensorShape.size();
  }
  return tensorRank;
}

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.self();
    Value dim = op.dim();
    if (!op.dtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    // exp(x)
    Value exp = rewriter.create<AtenExpOp>(loc, tensorType, self);

    // sum(exp(x))
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(dim.getType()), dim);
    Value keepDim = rewriter.create<ConstantBoolOp>(loc, true);
    Value dtype = rewriter.create<ConstantNoneOp>(loc);
    SmallVector<int64_t> sizes;
    int64_t dimInt;
    if (tensorType.hasSizes()) {
      ArrayRef<int64_t> inputShape = tensorType.getSizes();
      int64_t inputRank = inputShape.size();
      if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
        dimInt = toPositiveDim(dimInt, inputRank);
        if (!isValidDim(dimInt, inputRank))
          return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
        sizes.append(inputShape.begin(), inputShape.end());
        sizes[dimInt] = 1;
      } else {
        sizes.resize(inputRank, kUnknownSize);
      }
    }
    Type resultType = tensorType.getWithSizesAndDtype(
        sizes.size() == 0 ? Optional<ArrayRef<int64_t>>()
                          : llvm::makeArrayRef(sizes),
        tensorType.getDtype());
    Value sum = rewriter.create<AtenSumDimIntListOp>(loc, resultType, exp,
                                                     dimList, keepDim, dtype);
    // exp(x) / sum(exp(x))
    Value result = rewriter.create<AtenDivTensorOp>(loc, tensorType, exp, sum);
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(),
                                                        result);
    return success();
  }
};
} // namespace

// Decompose torch.matmul into: torch.mm and torch.bmm according to ranks.
namespace {
class DecomposeAtenMatmulOp : public OpRewritePattern<AtenMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.self();
    Value rhs = op.other();

    int lhsRank = getTensorRank(lhs);
    int rhsRank = getTensorRank(rhs);

    // If both lhs and rhs ranks are 2 then map it to `aten.mm` op.
    if (lhsRank == 2 && rhsRank == 2)
      rewriter.replaceOpWithNewOp<AtenMmOp>(op, op.getType(), lhs, rhs);

    // If both lhs and rhs ranks are 3 then map it to `aten.bmm` op.
    if (lhsRank == 3 && rhsRank == 3)
      rewriter.replaceOpWithNewOp<AtenBmmOp>(op, op.getType(), lhs, rhs);

    return success();
  }
};
} // namespace

namespace {
class DecomposeComplexOpsPass
    : public DecomposeComplexOpsBase<DecomposeComplexOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    patterns.add<DecomposeAtenSoftmaxIntOp>(context);
    target.addIllegalOp<AtenSoftmaxIntOp>();
    patterns.add<DecomposeAtenMatmulOp>(context);
    target.addDynamicallyLegalOp<AtenMatmulOp>([](AtenMatmulOp op) {
      int lhsRank = getTensorRank(op.self());
      int rhsRank = getTensorRank(op.other());

      // Make aten.matmul legal if the following condition is satisfied.
      return (lhsRank != 2 || rhsRank != 2) && (lhsRank != 3 || rhsRank != 3);
    });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createDecomposeComplexOpsPass() {
  return std::make_unique<DecomposeComplexOpsPass>();
}
