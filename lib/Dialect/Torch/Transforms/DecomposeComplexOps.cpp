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

static Value createSumAlongDimension(PatternRewriter &rewriter, Location loc,
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

// Helper for creating `aten::sub_tensor_op`.
static Value createTensorSub(PatternRewriter &rewriter, Location loc,
                                   Type tensorType, Value lhs, Value rhs) {
  Value alpha =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
  Value sub =
      rewriter.create<AtenSubTensorOp>(loc, tensorType, lhs, rhs, alpha);
  return sub;
}

// Share code between `softmax_backward` and `log_softmax_backward` ops.
// Returns x - y * sum(z, dim).
static Value createSoftmaxBackwardCommonKernel(PatternRewriter &rewriter,
                                               Location loc, Operation *op,
                                               Type tensorType, Value x,
                                               Value y, Value z, Value dim) {
  Value sum = createSumAlongDimension(rewriter, loc, op, z, dim, /*keepDim=*/true);
  if (!sum)
    return nullptr;
  auto broadcastSizeType =
      Torch::ListType::get(Torch::IntType::get(op->getContext()));
  Value broadcastSize = rewriter.create<AtenSizeOp>(loc, broadcastSizeType, z);
  Value sumBroadcast =
      rewriter.create<AtenBroadcastToOp>(loc, tensorType, sum, broadcastSize);
  Value temp =
      rewriter.create<AtenMulTensorOp>(loc, tensorType, y, sumBroadcast);

  Value sub = createTensorSub(rewriter, loc, tensorType, x, temp);
  return sub;
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

namespace {
class DecomposeAtenSelectIntOp : public OpRewritePattern<AtenSelectIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSelectIntOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value end =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), op.index(), one);
    rewriter.replaceOpWithNewOp<AtenSliceTensorOp>(op, op.getResult().getType(),
                                                   op.self(), op.dim(),
                                                   op.index(), end, one);

    return success();
  }
};
} // namespace

// Calculates the softmax function on the given `input` tensor. Softmax(x) =
// exp(x)/sum(exp(x)).
template <typename OpTy>
static Value getSoftmaxResult(OpTy op, Type resultType,
                              PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.dim();
  Value self = op.self();

  // exp(x)
  Value exp = rewriter.create<AtenExpOp>(loc, resultType, self);
  // sum(exp(x))
  Value sum =
      createSumAlongDimension(rewriter, loc, op, exp, dim, /*keepDim=*/true);
  if (!sum)
    return nullptr;
  // exp(x) / sum(exp(x))
  return rewriter.create<AtenDivTensorOp>(loc, resultType, exp, sum);
}

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.self();
    if (!op.dtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value result = getSoftmaxResult(op, tensorType, rewriter);
    if (!result)
      return failure();
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(),
                                                        result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_SoftmaxOp : public OpRewritePattern<Aten_SoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.self();
    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    bool halfToFloat;
    if (!matchPattern(op.half_to_float(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    // Currently, setting `halfToFloat` is not supported as the E2E testing for
    // the same is not present on CPU.
    if (halfToFloat)
      return rewriter.notifyMatchFailure(
          op, "halfToFloat is currently not supported.");

    Value result = getSoftmaxResult(op, tensorType, rewriter);
    if (!result)
      return op.emitError("failed to get softmax result");
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
    Value gradOutput = op.grad_output();
    Value output = op.output();
    Value dim = op.dim();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value newGrad =
        rewriter.create<AtenMulTensorOp>(loc, tensorType, gradOutput, output);
    Value result = createSoftmaxBackwardCommonKernel(
        rewriter, loc, op, tensorType, newGrad, output, newGrad, dim);
    if (!result)
      return rewriter.notifyMatchFailure(
          op,
          "nullptr returned by createSoftmaxBackwardCommonKernel function.");
    rewriter.replaceOp(op, result);
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

    Value newGrad = createTensorSub(rewriter, loc, tensorType, gradOutput,
                                          gradMulTanhSquare);
    rewriter.replaceOp(op, newGrad);
    return success();
  }
};
} // namespace

// Aten_LogSoftmaxBackwardDataOp(gradOutput, output, dim) =>
//    result = gradOutput - (exp(output) * sum(gradOutput, dim))
namespace {
class DecomposeAten_LogSoftmaxBackwardDataOp
    : public OpRewritePattern<Aten_LogSoftmaxBackwardDataOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_LogSoftmaxBackwardDataOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.grad_output();
    Value output = op.output();
    Value dim = op.dim();

    BaseTensorType tensorType = gradOutput.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value expOut = rewriter.create<AtenExpOp>(loc, tensorType, output);
    Value result = createSoftmaxBackwardCommonKernel(
        rewriter, loc, op, tensorType, gradOutput, expOut, gradOutput, dim);
    if (!result)
      return rewriter.notifyMatchFailure(
          op,
          "nullptr returned by createSoftmaxBackwardCommonKernel function.");
    rewriter.replaceOp(op, result);
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

// Decompose torch.addmm into torch.mm and torch.add.Tensor op.
namespace {
class DecomposeAtenAddmmOp : public OpRewritePattern<AtenAddmmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAddmmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.self();
    Value mat1 = op.mat1();
    Value mat2 = op.mat2();

    // The operands `mat1`, `mat2` to aten.addmm must be of rank 2.
    if (getTensorRank(mat1) != 2 || getTensorRank(mat2) != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected mat1, mat2 operands to aten.addmm to be rank 2");
    }

    // TODO: Handle integer type operands.
    if (!input.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non-floating point dtype");
    }

    // matrix multiplication: matmul = mat1 @ mat2
    Value matmul = rewriter.create<AtenMmOp>(loc, op.getType(), mat1, mat2);
    // scaledInput = self * beta
    Value scaledInput = rewriter.create<AtenMulScalarOp>(loc, input.getType(),
                                                         input, op.beta());
    // result = scaledInput + alpha * matmul
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), scaledInput,
                                                 matmul, op.alpha());
    return success();
  }
};
} // namespace

// Decompose torch.mean into: sum(x)/div(numTensorElements).
namespace {
class DecomposeAtenMeanOp : public OpRewritePattern<AtenMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.self();
    Value output = op.result();
    BaseTensorType outputTensorType = output.getType().cast<BaseTensorType>();
    Value sum = rewriter.create<AtenSumOp>(loc, outputTensorType, input, op.dtype());
    Value numTensorElements = rewriter.create<AtenNumelOp>(loc, input);
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, outputTensorType, sum,
                                                 numTensorElements);
    return success();
  }
};
} // namespace

namespace {
template<typename OpTy, typename T1T2Op>
class DecomposeAtenAddCLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.self();
    Value tensor1 = op.tensor1();
    Value tensor2 = op.tensor2();
    Value value = op.value();

    Value product = rewriter.create<T1T2Op>(loc, op.getType(), tensor1, tensor2);
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), input, product,
                                                  value);
    return success();
  }
};

class DecomposeAtenLayerNormOp : public OpRewritePattern<AtenLayerNormOp> {
  using OpRewritePattern<AtenLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto input = op.input().getType().cast<BaseTensorType>();
    if (!input.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = input.getSizes().size();
    Value normalizedShape = op.normalized_shape();
    SmallVector<Value> normalizedShapeSizesTorchInt;
    getListConstructElements(normalizedShape, normalizedShapeSizesTorchInt);
    std::vector<int64_t> meanVarSizes;
    for (int i = normalizedShapeSizesTorchInt.size(); i < inputRank; i++)
      meanVarSizes.push_back(input.getSizes()[i]);
    auto meanVarType = input.getWithSizesAndDtype(
        llvm::makeArrayRef(meanVarSizes), input.getDtype());
    auto nativeLayerNorm = rewriter.create<AtenNativeLayerNormOp>(
        loc, op.getType(), meanVarType, meanVarType, op.input(),
        op.normalized_shape(), op.weight(), op.bias(), op.eps());
    rewriter.replaceOp(op, nativeLayerNorm.getResult(0));
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
    patterns.add<DecomposeAten_SoftmaxOp>(context);
    target.addIllegalOp<Aten_SoftmaxOp>();
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
    patterns.add<DecomposeAtenAddmmOp>(context);
    target.addIllegalOp<AtenAddmmOp>();
    patterns.add<DecomposeAtenMeanOp>(context);
    target.addIllegalOp<AtenMeanOp>();
    patterns.add<DecomposeAtenSelectIntOp>(context);
    target.addIllegalOp<AtenSelectIntOp>();
    patterns.add<DecomposeAtenMatmulOp>(context);
    patterns.add<DecomposeAten_LogSoftmaxBackwardDataOp>(context);
    target.addIllegalOp<Aten_LogSoftmaxBackwardDataOp>();
    target.addDynamicallyLegalOp<AtenMatmulOp>([](AtenMatmulOp op) {
      int lhsRank = getTensorRank(op.self());
      int rhsRank = getTensorRank(op.other());

      // Make aten.matmul legal if the following condition is satisfied.
      return (lhsRank != 2 || rhsRank != 2) && (lhsRank != 3 || rhsRank != 3);
    });
    patterns.add<DecomposeAtenAddCLikeOp<AtenAddcmulOp, AtenMulTensorOp>>(context);
    target.addIllegalOp<AtenAddcmulOp>();
    patterns.add<DecomposeAtenAddCLikeOp<AtenAddcdivOp, AtenDivTensorOp>>(context);
    target.addIllegalOp<AtenAddcdivOp>();
    target.addIllegalOp<AtenLayerNormOp>();
    patterns.add<DecomposeAtenLayerNormOp>(context);

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
