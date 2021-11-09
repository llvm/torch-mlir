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

static Value createAtenSum(PatternRewriter &rewriter, Location loc,
                           Operation *op, Value input, Value dim,
                           bool keepDim) {
  BaseTensorType tensorType = input.getType().cast<BaseTensorType>();
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc, Torch::ListType::get(dim.getType()), dim);
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  Value dtype = rewriter.create<ConstantNoneOp>(loc);
  SmallVector<int64_t> sizes;
  int64_t dimInt;
  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> inputShape = tensorType.getSizes();
    int64_t inputRank = inputShape.size();
    if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
      dimInt = toPositiveDim(dimInt, inputRank);
      if (!isValidDim(dimInt, inputRank)) {
        (void)rewriter.notifyMatchFailure(op, "dim is not a valid dim");
        return nullptr;
      }
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
  Value sum = rewriter.create<AtenSumDimIntListOp>(loc, resultType, input,
                                                   dimList, keepDimCst, dtype);
  return sum;
}

namespace {
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.self();
    MLIRContext *context = op.getContext();
    int64_t rank = getTensorRank(self);
    if (rank < 0)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    SmallVector<Value> sizes;
    for (int i = 0; i < rank; i++) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      sizes.push_back(rewriter.create<AtenSizeIntOp>(loc, self, dim));
    }

    Value sizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), sizes);
    rewriter.replaceOp(op, sizeList);
    return success();
  }
};
} // namespace

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
    Value sum = createAtenSum(rewriter, loc, op, exp, dim, /*keepDim=*/true);
    if (!sum)
      return failure();
    // exp(x) / sum(exp(x))
    Value result = rewriter.create<AtenDivTensorOp>(loc, tensorType, exp, sum);
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(),
                                                        result);
    return success();
  }
};
} // namespace

// Aten_SoftmaxBackwardDataOp(gradOutput, output, dim) =>
//    newGrad = gradOutput * output
//    result = newGrad - output * sum(newGrad, dim))
//
// Refer to
// https://github.com/pytorch/pytorch/blob/15fecc4c830a3907fde4b44c9962dc4144da50a4/torch/csrc/jit/codegen/cuda/ops/normalization.cpp#L31
namespace {
class DecomposeAten_SoftmaxBackwardDataOp
    : public OpRewritePattern<Aten_SoftmaxBackwardDataOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SoftmaxBackwardDataOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value gradOutput = op.grad_output();
    Value output = op.output();
    Value dim = op.dim();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value newGrad =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, gradOutput, output);
    // temp = output * sum(newGrad, dim)
    Value sum =
        createAtenSum(rewriter, loc, op, newGrad, dim, /*keepDim=*/true);
    if (!sum)
      return failure();
    auto broadcastSizeType = Torch::ListType::get(Torch::IntType::get(context));
    Value broadcastSize =
        rewriter.create<AtenSizeOp>(loc, broadcastSizeType, output);
    Value sumBroadcast =
        rewriter.create<AtenBroadcastToOp>(loc, tensorType, sum, broadcastSize);
    Value temp =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, output, sumBroadcast);

    // newGrad - temp
    Value alpha =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
    Value sub =
        rewriter.create<AtenSubTensorOp>(loc, tensorType, newGrad, temp, alpha);

    rewriter.replaceOp(op, sub);
    return success();
  }
};
} // namespace

// AtenTanhBackwardOp(gradOutput, output) =>
//    result = gradOutput * (1 - output^2)
// To get away from broadcasts the above formula is expanded i.e.,
// result = gradOutput - (gradOutput * output^2)
namespace {
class DecomposeAtenTanhBackwardOp
    : public OpRewritePattern<AtenTanhBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTanhBackwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.grad_output();

    // `output` is the value flowing out from tanh. Hence, tanh(x) = output.
    //  Since, dTanh(x) = (1 - tanh(x)^2) hence, dOutput = (1 - output^2). 
    Value output = op.output();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value tanhSquare =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, output, output);
    Value gradMulTanhSquare = rewriter.create<AtenMulTensorOp>(
        loc, tensorType, tanhSquare, gradOutput);

    Value alpha =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
    Value newGrad = rewriter.create<AtenSubTensorOp>(
        loc, tensorType, gradOutput, gradMulTanhSquare, alpha);
    rewriter.replaceOp(op, newGrad);
    return success();
  }
};
} // namespace

// Decompose aten.log_softmax op into: log(softmax(x))
namespace {
class DecomposeAtenLogSoftmaxIntOp
    : public OpRewritePattern<AtenLogSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLogSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.self();
    Value dim = op.dim();
    if (!op.dtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for log_softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    // softmax(x, dim)
    Value softmax = rewriter.create<AtenSoftmaxIntOp>(loc, tensorType, self,
                                                      dim, op.dtype());
    rewriter.replaceOpWithNewOp<AtenLogOp>(op, op.getType(), softmax);
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

// Decompose torch.expand into torch.broadcast_to op.
namespace {
class DecomposeAtenExpandOp : public OpRewritePattern<AtenExpandOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExpandOp op,
                                PatternRewriter &rewriter) const override {
    bool implicit = false;
    if (!matchPattern(op.implicit(), m_TorchConstantBool(&implicit)) ||
        implicit) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires implicit to be false");
    }
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(), op.self(),
                                                   op.size());
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
    patterns.add<DecomposeAtenLogSoftmaxIntOp>(context);
    target.addIllegalOp<AtenLogSoftmaxIntOp>();
    patterns.add<DecomposeAtenExpandOp>(context);
    target.addIllegalOp<AtenExpandOp>();
    patterns.add<DecomposeAtenSizeOp>(context);
    target.addIllegalOp<AtenSizeOp>();
    patterns.add<DecomposeAten_SoftmaxBackwardDataOp>(context);
    target.addIllegalOp<Aten_SoftmaxBackwardDataOp>();
    patterns.add<DecomposeAtenTanhBackwardOp>(context);
    target.addIllegalOp<AtenTanhBackwardOp>();
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
