//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Helper function to check whether the `dtype` is None or Float type.
static bool isNoneOrFloatDtype(MLIRContext *context, Value dtype) {
  if (dtype.getType().isa<Torch::NoneType>())
    return true;
  int64_t dtypeInt;
  if (!matchPattern(dtype, m_TorchConstantInt(&dtypeInt)))
    return false;
  Type resDtype =
      getTypeForScalarType(context, (torch_upstream::ScalarType)dtypeInt);
  return resDtype.isa<mlir::FloatType>();
}

// Helper function to compute the return type of the reduction function.
// `dim` specifies the dimension to reduce and `keepDim` preserves the rank of
// the input tensor.
static Type computeReductionType(PatternRewriter &rewriter, Operation *op,
                                 BaseTensorType tensorType, Value dim,
                                 bool keepDim) {
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
      // The dimension to be reduced is set to 1 when `keepDim` is true else it
      // is removed.
      if (keepDim)
        sizes[dimInt] = 1;
      else
        sizes.erase(sizes.begin() + dimInt);
    } else {
      unsigned reducedRank = keepDim ? inputRank : inputRank - 1;
      sizes.resize(reducedRank, kUnknownSize);
    }
  }

  Type resultType = tensorType.getWithSizesAndDtype(
      sizes.size() == 0 ? std::optional<ArrayRef<int64_t>>()
                        : llvm::makeArrayRef(sizes),
      tensorType.getOptionalDtype());
  return resultType;
}

// Reduction function to calculate sum along given `dim`.
static Value createSumAlongDimension(PatternRewriter &rewriter, Location loc,
                                     Operation *op, Value input, Value dim,
                                     bool keepDim) {
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc, Torch::ListType::get(dim.getType()), dim);
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  Value dtype = rewriter.create<ConstantNoneOp>(loc);
  Type resultType = computeReductionType(
      rewriter, op, input.getType().cast<BaseTensorType>(), dim, keepDim);
  if (!resultType)
    return nullptr;
  return rewriter.create<AtenSumDimIntListOp>(loc, resultType, input, dimList,
                                              keepDimCst, dtype);
}

// Redunction function to calculate max along given `dim`.
static Value createMaxAlongDimension(PatternRewriter &rewriter, Location loc,
                                     Operation *op, Value input, Value dim,
                                     bool keepDim) {
  Value keepDimCst = rewriter.create<ConstantBoolOp>(loc, keepDim);
  BaseTensorType valueType =
      computeReductionType(rewriter, op, input.getType().cast<BaseTensorType>(),
                           dim, keepDim)
          .cast<BaseTensorType>();
  if (!valueType)
    return nullptr;
  BaseTensorType indexType =
      valueType
          .getWithSizesAndDtype(
              !valueType.hasSizes() ? std::optional<ArrayRef<int64_t>>()
                                    : llvm::makeArrayRef(valueType.getSizes()),
              IntegerType::get(op->getContext(), 64, IntegerType::Signed))
          .cast<BaseTensorType>();
  return rewriter
      .create<AtenMaxDimOp>(loc, valueType, indexType, input, dim, keepDimCst)
      .getValues();
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

// Helper to create a tensor filled with the given scalar. Scalar would be
// converted the to the element type of the given tensor type.
static Value createInitTensor(PatternRewriter &rewriter, Location loc,
                              Type resultType, Value scalar, Value sizeList) {
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  return rewriter.create<AtenFullOp>(
      loc, resultType, sizeList, scalar, /*dtype=*/noneVal, /*layout=*/noneVal,
      /*device=*/noneVal, /*memory_format=*/noneVal);
}

// Helper to create a rank 0 tensor filled with the given `scalar`. `scalar`
// would be converted to the element type of the given `inputType`.
static Value createRank0Tensor(PatternRewriter &rewriter, Location loc,
                               BaseTensorType inputType, Value scalar) {
  SmallVector<int64_t> sizes;
  Type rank0TensorTy = inputType.getWithSizesAndDtype(
      makeArrayRef(sizes), inputType.getOptionalDtype());
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc, Torch::ListType::get(Torch::IntType::get(inputType.getContext())),
      ValueRange{});
  return createInitTensor(rewriter, loc, rank0TensorTy, scalar, dimList);
}

// Share code between `softmax_backward` and `log_softmax_backward` ops.
// Returns x - y * sum(z, dim).
static Value createSoftmaxBackwardCommonKernel(PatternRewriter &rewriter,
                                               Location loc, Operation *op,
                                               Type tensorType, Value x,
                                               Value y, Value z, Value dim) {
  Value sum =
      createSumAlongDimension(rewriter, loc, op, z, dim, /*keepDim=*/true);
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
/// We decompose aten.amax into a set of aten.max.dim op(s) depending on the
/// number of dimensions across which the max needs to be computed.
/// Eg:
///   INPUT:
///      final_output = aten.amax(initial_input, dim=(0, 2, 1), keepdim=False)
///
///   OUTPUT:
///      input_1 = aten.max.dim(initial_input, 2, keepdim)  #1
///      input_2 = aten.max.dim(input_1, 1, keepdim)   #2
///      final_output = aten.max.dim(input_2, 0, keepdim) #3
///
/// NOTE: We iterate over, in reverse order, every dimension included in `dim`
///       of the `aten.amax` op and create an `aten.amax.dim` op.
///       Input tensor to the next `aten.amax.dim` op is thus the output of the
///       previous `aten.amax.dim` op.
class DecomposeAtenAmaxOp : public OpRewritePattern<AtenAmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<int64_t, 4> dims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");

    bool keepDim;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant boolean value for keepDim");

    Value input = op.getSelf();
    // For every dimension included in `dim` of the op, iterated over in
    // reverse order, we create a call to aten.max.dim.
    std::sort(dims.begin(), dims.end());
    std::reverse(dims.begin(), dims.end());
    for (int64_t dimInt : dims) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimInt));
      // The input to the next invocation of aten.max.dim is the output of the
      // previous aten.max.dim op.
      input = createMaxAlongDimension(rewriter, loc, op, input, dim, keepDim);
    }
    rewriter.replaceOp(op, input);
    return success();
  }
};
} // end namespace

namespace {
class DecomposeAtenSizeOp : public OpRewritePattern<AtenSizeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;
    SmallVector<Value> sizes;
    for (unsigned i = 0; i < rank; i++) {
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
    Value start = op.getIndex();
    Value dim = op.getDim();
    Value self = op.getSelf();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusOne =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, one);
    Value slice = rewriter.create<AtenSliceTensorOp>(
        loc,
        computeReductionType(rewriter, op,
                             self.getType().cast<BaseTensorType>(), dim,
                             /*keepDim=*/true),
        op.getSelf(), dim, start, startPlusOne, /*step=*/one);

    // `aten.slice.tensor` doesn't squeeze the dim even when it's size 1 after
    // slicing, while `aten.select.int` does.
    rewriter.replaceOpWithNewOp<AtenSqueezeDimOp>(op, op.getResult().getType(),
                                                  slice, op.getDim());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNarrowOp : public OpRewritePattern<AtenNarrowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNarrowOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value start = op.getStart();
    Value dim = op.getDim();
    Value length = op.getLength();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusLength =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, length);

    rewriter.replaceOpWithNewOp<AtenSliceTensorOp>(
        op, op.getResult().getType(), op.getSelf(), /*dim=*/dim, /*start=*/start,
        /*end=*/startPlusLength, /*step=*/one);

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenZeroOp
    : public OpRewritePattern<AtenZeroOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenZeroOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<ConstantIntOp>(op.getLoc(),
                                                rewriter.getI64IntegerAttr(0));
    rewriter.replaceOpWithNewOp<AtenFillScalarOp>(op, op.getType(), op.getSelf(),
                                                  zero);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenReshapeOp : public OpRewritePattern<AtenReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getSelf();
    // TODO: Handle non value tensor type operands.
    if (!input.getType().isa<ValueTensorType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only value tensor type operands are supported");
    }
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), input,
                                            op.getShape());
    return success();
  }
};
} // namespace

// Calculates the softmax function on the given `input` tensor. Softmax(x) =
// exp(x)/sum(exp(x)).
// To avoid overflow we use the following decomposition rule:
//     x_max = max(input, dim, keepdim = True)
//     unnorm = aten.exp(input - x_max)
//     softmax = unnorm / sum(unnorm, dim, keepdim = True)
template <typename OpTy>
static Value getSoftmaxResult(OpTy op, Value self, Type resultType,
                              PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.getDim();
  Value xMax =
      createMaxAlongDimension(rewriter, loc, op, self, dim, /*keepDim=*/true);
  if (!xMax)
    return nullptr;
  Value unNormalized = createTensorSub(rewriter, loc, resultType, self, xMax);
  Value unNormalizedExp =
      rewriter.create<AtenExpOp>(loc, resultType, unNormalized);
  Value sum = createSumAlongDimension(rewriter, loc, op, unNormalizedExp, dim,
                                      /*keepDim=*/true);
  if (!sum)
    return nullptr;
  return rewriter.create<AtenDivTensorOp>(loc, resultType, unNormalizedExp,
                                          sum);
}

// Decompose softmax into: exp(x) / sum(exp(x))
namespace {
class DecomposeAtenSoftmaxIntOp : public OpRewritePattern<AtenSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    if (!op.getDtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value result = getSoftmaxResult(op, self, tensorType, rewriter);
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
    Value self = op.getSelf();
    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    BaseTensorType resultTensorType = op.getType().cast<BaseTensorType>();
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultTensorDtype = resultTensorType.getDtype();
    // `torch.ops.aten._softmax`'s softmax with half to float conversion is not
    // supported on CPU, but we go ahead with the decomposing.
    // TODO: Add an e2e test once upstream support is added.
    // If `half_to_float` is set, we convert the input's elemental type to match
    // that of output's.
    if (halfToFloat) {
      Location loc = op.getLoc();
      Value none = rewriter.create<ConstantNoneOp>(loc);
      Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
      self = rewriter.create<AtenToDtypeOp>(
          loc, resultTensorType, self,
          getDtypeIntValueForType(rewriter, loc, resultTensorDtype),
          /*non_blocking=*/cstFalse, /*copy=*/cstFalse, /*memory_format=*/none);
    }
    Value result = getSoftmaxResult(op, self, resultTensorType, rewriter);
    if (!result)
      return op.emitError("failed to get softmax result");
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTensorType,
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
    Value gradOutput = op.getGradOutput();
    Value output = op.getOutput();
    Value dim = op.getDim();

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
    Value gradOutput = op.getGradOutput();

    // `output` is the value flowing out from tanh. Hence, tanh(x) = output.
    //  Since, dTanh(x) = (1 - tanh(x)^2) hence, dOutput = (1 - output^2).
    Value output = op.getOutput();

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
    Value gradOutput = op.getGradOutput();
    Value output = op.getOutput();
    Value dim = op.getDim();

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

// Decompose `AtenArgMaxOp` into `AtenMaxDimOp`.
namespace {
class DecomposeAtenArgMaxOp : public OpRewritePattern<AtenArgmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenArgmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value dim = op.getDim();
    Value keepDim = op.getKeepdim();
    Value result = op.getResult();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    BaseTensorType indicesTensorType = result.getType().cast<BaseTensorType>();
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensor to have a rank");
    }
    unsigned inputRank = *maybeInputRank;
    if (!indicesTensorType.hasSizes())
      return failure();
    BaseTensorType valueTensorType =
        inputType
            .getWithSizesAndDtype(indicesTensorType.getOptionalSizes(),
                                  inputType.getOptionalDtype())
            .cast<BaseTensorType>();

    // If the dim type is `NoneType` i.e. reduce along all the dimensions.
    // `AtenMaxDimOp` doesn't support dim as `NoneType` so first the input
    // tensor is flattened to 1d tensor and then the reduction happens on the
    // 0th dimension.
    if (dim.getType().isa<Torch::NoneType>()) {
      BaseTensorType flattenType =
          inputType
              .getWithSizesAndDtype({kUnknownSize},
                                    inputType.getOptionalDtype())
              .cast<BaseTensorType>();
      dim = rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value end = rewriter.create<ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(inputRank - 1));
      input = rewriter.create<AtenFlattenUsingIntsOp>(loc, flattenType, input,
                                                      dim, end);
    }
    Value maxResult =
        rewriter
            .create<AtenMaxDimOp>(loc, valueTensorType, indicesTensorType,
                                  input, dim, keepDim)
            .getIndices();

    rewriter.replaceOp(op, maxResult);
    return success();
  }
};
} // namespace

// To avoid overflow we use the following decomposition rule:
//  x_max = aten.max(x, dim, keepdim=True)[0]
//  shifted = x - x_max
//  shifted_logsumexp = aten.log(aten.sum(aten.exp(shifted), dim, keepdim=True))
//  log_softmax = shifted - shifted_logsumexp
template <typename OpTy>
static Value getLogSoftmaxResult(OpTy op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  Value dim = op.getDim();
  Value self = op.getSelf();
  BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
  Value xMax =
      createMaxAlongDimension(rewriter, loc, op, self, dim, /*keepDim=*/true);
  if (!xMax)
    return nullptr;

  Value shifted = createTensorSub(rewriter, loc, tensorType, self, xMax);
  Value shiftedExp = rewriter.create<AtenExpOp>(loc, tensorType, shifted);
  Value shiftedSumExp =
      createSumAlongDimension(rewriter, loc, op, shiftedExp, dim,
                              /*keepDim=*/true);
  if (!shiftedSumExp)
    return nullptr;

  Value shiftedLogSumExp =
      rewriter.create<AtenLogOp>(loc, shiftedSumExp.getType(), shiftedSumExp);
  Value result =
      createTensorSub(rewriter, loc, op.getType(), shifted, shiftedLogSumExp);
  return result;
}

namespace {
class DecomposeAtenLogSoftmaxIntOp
    : public OpRewritePattern<AtenLogSoftmaxIntOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLogSoftmaxIntOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    if (!op.getDtype().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Unimplemented non-None dtype for log_softmax");

    BaseTensorType tensorType = self.getType().cast<BaseTensorType>();
    if (!tensorType.hasDtype() || !tensorType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(op, "Only support floating type");

    Value logSoftmax = getLogSoftmaxResult(op, rewriter);
    if (!logSoftmax)
      return rewriter.notifyMatchFailure(
          op, "getLogSoftmaxResult function returned nullptr");
    rewriter.replaceOp(op, logSoftmax);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_LogSoftmaxOp : public OpRewritePattern<Aten_LogSoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_LogSoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    bool halfToFloat;
    if (!matchPattern(op.getHalfToFloat(), m_TorchConstantBool(&halfToFloat)))
      return rewriter.notifyMatchFailure(
          op, "Expected a boolean value for half_to_float");

    // Currently, setting `halfToFloat` is not supported as the E2E testing for
    // the same is not present on CPU.
    if (halfToFloat)
      return rewriter.notifyMatchFailure(
          op, "halfToFloat is currently not supported.");
    Value _logSoftmax = getLogSoftmaxResult(op, rewriter);
    if (!_logSoftmax)
      return rewriter.notifyMatchFailure(
          op, "getLogSoftmaxResult function returned nullptr");
    rewriter.replaceOp(op, _logSoftmax);
    return success();
  }
};
} // namespace

// Decompose aten.matmul into: aten.mm and aten.bmm according to ranks.
namespace {
class DecomposeAtenMatmulOp : public OpRewritePattern<AtenMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    Value rhs = op.getOther();

    std::optional<unsigned> maybeLhsRank = getTensorRank(lhs);
    std::optional<unsigned> maybeRhsRank = getTensorRank(rhs);
    if (!maybeLhsRank || !maybeRhsRank) {
      return rewriter.notifyMatchFailure(
          op, "expected input tensors to have a rank");
    }
    unsigned lhsRank = *maybeLhsRank;
    unsigned rhsRank = *maybeRhsRank;

    if (lhsRank == 2 && rhsRank == 2) {
      // If both lhs and rhs ranks are 2 then map it to `aten.mm` op.
      rewriter.replaceOpWithNewOp<AtenMmOp>(op, op.getType(), lhs, rhs);
    } else if (lhsRank == 3 && rhsRank == 3) {
      // If both lhs and rhs ranks are 3 then map it to `aten.bmm` op.
      rewriter.replaceOpWithNewOp<AtenBmmOp>(op, op.getType(), lhs, rhs);
    } else {
      return failure();
    }

    return success();
  }
};
} // namespace

// Decompose aten.mv into: aten.matmul.
namespace {
class DecomposeAtenMvOp : public OpRewritePattern<AtenMvOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMvOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    Value rhs = op.getVec();
    rewriter.replaceOpWithNewOp<AtenMatmulOp>(op, op.getType(), lhs, rhs);
    return success();
  }
};
} // namespace

// ReLU6(x) = min(max(0, x), 6) = min(Relu(x), 6)
static Value getRelu6Results(PatternRewriter &rewriter, Location loc,
                             Value input) {
  BaseTensorType inputType = input.getType().cast<BaseTensorType>();

  Value relu = rewriter.create<AtenReluOp>(loc, inputType, input);
  Value cst6 =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(6));
  Value sixTensor = createRank0Tensor(rewriter, loc, inputType, cst6);
  Value relu6Out =
      rewriter.create<AtenMinimumOp>(loc, inputType, relu, sixTensor);
  return relu6Out;
}

namespace {
class DecomposeAtenRelu6Op : public OpRewritePattern<AtenRelu6Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRelu6Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value relu6 = getRelu6Results(rewriter, loc, op.getSelf());
    rewriter.replaceOp(op, relu6);
    return success();
  }
};
} // namespace

// Hardswish(x) = x * Relu6(x+3)/6
namespace {
class DecomposeAtenHardswishOp : public OpRewritePattern<AtenHardswishOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardswishOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type inputType = input.getType();

    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantThree = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(3));
    Value constantSix = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(6));
    Value inputPlusThree = rewriter.create<AtenAddScalarOp>(
        loc, inputType, input, constantThree, /*alpha=*/constantOne);
    Value relu6 = getRelu6Results(rewriter, loc, inputPlusThree);
    Value divTensor =
        rewriter.create<AtenDivScalarOp>(loc, inputType, relu6, constantSix);
    Value mulTensor =
        rewriter.create<AtenMulTensorOp>(loc, inputType, divTensor, input);

    rewriter.replaceOp(op, mulTensor);
    return success();
  }
};
} // namespace

// LeakyRelu = max(0,x) + negative_slope * min(0,x)
namespace {
class DecomposeAtenLeakyReluOp : public OpRewritePattern<AtenLeakyReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLeakyReluOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value negativeSlope = op.getNegativeSlope();
    auto resType = op.getType().cast<BaseTensorType>();

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, input);
    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value scaledNegativeOutput = rewriter.create<AtenMulScalarOp>(
        loc, resType, negativeOutput, negativeSlope);
    Value leakyReluOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, scaledNegativeOutput, constantOne);

    rewriter.replaceOp(op, leakyReluOutput);
    return success();
  }
};
} // namespace

// LeakyReluBackward = max(0,grad) + negative_slope * min(0,x)
namespace {
class DecomposeAtenLeakyReluBackwardOp
    : public OpRewritePattern<AtenLeakyReluBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLeakyReluBackwardOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value gradOutput = op.getGradOutput();
    Value input = op.getSelf();
    Value negativeSlope = op.getNegativeSlope();
    auto resType = op.getType().cast<BaseTensorType>();

    bool selfIsResult = false;
    if (!matchPattern(op.getSelfIsResult(),
                      m_TorchConstantBool(&selfIsResult)) ||
        selfIsResult)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: self_is_result should be false");

    Value constantZero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value constantOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value zeroTensor = createRank0Tensor(rewriter, loc, resType, constantZero);
    Value positiveOutput =
        rewriter.create<AtenMaximumOp>(loc, resType, zeroTensor, gradOutput);
    Value negativeOutput =
        rewriter.create<AtenMinimumOp>(loc, resType, zeroTensor, input);
    Value scaledNegativeOutput = rewriter.create<AtenMulScalarOp>(
        loc, resType, negativeOutput, negativeSlope);
    Value leakyReluBackwardOutput = rewriter.create<AtenAddTensorOp>(
        loc, resType, positiveOutput, scaledNegativeOutput, constantOne);

    rewriter.replaceOp(op, leakyReluBackwardOutput);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenTOp : public OpRewritePattern<AtenTOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenTOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getSelf();
    std::optional<unsigned> lhsRank = getTensorRank(lhs);
    auto loc = op.getLoc();

    if (!lhsRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    } else if (*lhsRank > 2) {
      std::string errorMessage =
          "t() expects a tensor with <=2 dimensions, but self is " +
          std::to_string(*lhsRank) + "D";
      return rewriter.notifyMatchFailure(op, errorMessage.c_str());
    } else if (*lhsRank < 2)
      rewriter.replaceOp(op, lhs);
    else {
      Value zero =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
      Value one =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      rewriter.replaceOpWithNewOp<AtenTransposeIntOp>(op, op.getType(), lhs,
                                                      zero, one);
    }
    return success();
  }
};
} // namespace

// Decompose aten.roll into aten.slice and aten.cat ops.
// https://pytorch.org/docs/stable/generated/torch.roll.html
namespace {
class DecomposeAtenRollOp : public OpRewritePattern<AtenRollOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRollOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> shifts;
    if (!getListConstructElements(op.getShifts(), shifts))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: shifts not list of Scalar");
    SmallVector<Value> dims;
    if (!getListConstructElements(op.getDims(), dims))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dims not list of Scalar");

    if (shifts.size() != dims.size())
      return op.emitError("list sizes of shifts and dims are not the same");

    auto loc = op.getLoc();
    Value constNone = rewriter.create<ConstantNoneOp>(loc);
    Value constZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    auto self = op.getSelf();
    auto selfTy = self.getType().cast<BaseTensorType>();
    // roll(input, shift, dim) = cat({
    //   slice(input, dim, -shift, none),
    //   slice(input, dim, 0, -shift)}, dim)
    auto imitateRoll = [&](Value input, Value shift, Value dim,
                           int64_t cstDim) {
      Value negShift = rewriter.create<AtenNegIntOp>(loc, shift);
      ArrayRef<int64_t> inputShape = selfTy.getSizes();
      SmallVector<int64_t> sizes;
      sizes.append(inputShape.begin(), inputShape.end());
      sizes[cstDim] = kUnknownSize;
      Type sliceTy = selfTy.getWithSizesAndDtype(llvm::makeArrayRef(sizes),
                                                 selfTy.getOptionalDtype());
      Value slice0 = rewriter.create<AtenSliceTensorOp>(
          loc, sliceTy, input, dim, negShift, constNone, constOne);
      Value slice1 = rewriter.create<AtenSliceTensorOp>(
          loc, sliceTy, input, dim, constZero, negShift, constOne);

      Type listType = Torch::ListType::get(sliceTy);
      Value slices = rewriter.create<PrimListConstructOp>(
          loc, listType, llvm::ArrayRef<Value>{slice0, slice1});
      return rewriter.create<AtenCatOp>(loc, self.getType(), slices, dim);
    };
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;
    Value output = self;
    auto nShifts = shifts.size();
    for (size_t k = 0; k < nShifts; ++k) {
      auto dim = dims[k];
      int64_t cstDim = -1;
      if (!matchPattern(dim, m_TorchConstantInt(&cstDim)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: dim must be constant");

      cstDim = toPositiveDim(cstDim, rank);
      output = imitateRoll(output, shifts[k], dim, cstDim);
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

// Decompose aten.repeat into aten.expand and aten.view ops.
//
// Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
//
// For shape [S1, S2, S3] and repeats [M0, M1, M2, M3]
//     MS0 = M0; MS1 = M1 * S1; MS2 = M2 * S2; MS3 = M3 * S3
//
// def aten_repeat(self, repeats):
//     sizes = self.size()
//     unsqueezed_sizes = []
//     expanded_sizes = []
//     reshape_sizes = []
//     leading_rank = repeats.size() - sizes.size()
//     for r in range(leading_rank):
//         unsqueezed_sizes.append(1)
//         expanded_sizes.append(repeats[r])
//         reshaped_sizes.append(repeats[r])
//
//     for s, m in zip(sizes, repeats[leading_rank:]):
//         unsqueezed_sizes += [1, s]
//         expanded_sizes += [m, s]
//         reshaped_sizes += [m * s]
//     return
//     self.view(unsqueezed_sizes).expand(expanded_sizes).view(reshaped_sizes)
//
namespace {
class DecomposeAtenRepeatOp : public OpRewritePattern<AtenRepeatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRepeatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "Unimplemented: unranked tensor");
    unsigned rank = *maybeRank;

    SmallVector<Value> repeats;
    if (!getListConstructElements(op.getRepeats(), repeats))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: repeats not list of Scalar");

    if (rank > repeats.size()) {
      return rewriter.notifyMatchFailure(
          op, "repeats are not matched with self's rank");
    }

    auto insertDimSizes = [](SmallVector<Value> &dimSizes,
                             SmallVector<int64_t> &shape,
                             const ArrayRef<Value> &vals) {
      dimSizes.insert(dimSizes.end(), vals.begin(), vals.end());
      std::transform(vals.begin(), vals.end(), std::back_inserter(shape),
                     [&](Value val) -> int64_t {
                       int64_t cst_val;
                       if (matchPattern(val, m_TorchConstantInt(&cst_val))) {
                         return cst_val;
                       } else {
                         return kUnknownSize;
                       }
                     });
    };

    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    SmallVector<Value> unsqueezedSizes, expandedSizes, reshapedSizes;
    SmallVector<int64_t> unsqueezedIntSizes, expandedIntSizes;
    auto leadingRank = repeats.size() - rank;
    assert(leadingRank >= 0 && "leadingRank should greater than 0");
    for (size_t i = 0; i < leadingRank; ++i) {
      insertDimSizes(unsqueezedSizes, unsqueezedIntSizes, ArrayRef<Value>{one});
      insertDimSizes(expandedSizes, expandedIntSizes,
                     ArrayRef<Value>{repeats[i]});
      reshapedSizes.push_back(repeats[i]);
    }

    auto selfType = self.getType().dyn_cast<BaseTensorType>();
    auto selfShape = selfType.getSizes();
    for (unsigned i = 0; i < rank; i++) {
      auto scale = repeats[i + leadingRank];
      Value dimSize;
      if (selfShape[i] == kUnknownSize) {
        Value dim = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(i));
        dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
      } else {
        dimSize = rewriter.create<Torch::ConstantIntOp>(
            loc, rewriter.getI64IntegerAttr(selfShape[i]));
      }

      insertDimSizes(unsqueezedSizes, unsqueezedIntSizes,
                     ArrayRef<Value>{one, dimSize});
      insertDimSizes(expandedSizes, expandedIntSizes,
                     ArrayRef<Value>{scale, dimSize});

      Value scaledSize = rewriter.create<AtenMulIntOp>(loc, dimSize, scale);
      reshapedSizes.push_back(scaledSize);
    }

    Type dtype = self.getType().cast<ValueTensorType>().getOptionalDtype();
    Type unsqueezedType = ValueTensorType::get(
        context, llvm::makeArrayRef(unsqueezedIntSizes), dtype);
    Type expandedType = ValueTensorType::get(
        context, llvm::makeArrayRef(expandedIntSizes), dtype);

    auto listType = Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value unsqueezedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, unsqueezedSizes);
    Value expandedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, expandedSizes);
    Value reshapedDims =
        rewriter.create<PrimListConstructOp>(loc, listType, reshapedSizes);
    auto reshaped = rewriter.create<AtenViewOp>(loc, unsqueezedType, op.getSelf(),
                                                unsqueezedDims);
    auto expanded = rewriter.create<AtenBroadcastToOp>(loc, expandedType,
                                                       reshaped, expandedDims);

    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), expanded,
                                            reshapedDims);
    return success();
  }
};
} // namespace

// Decompose aten.flatten.using_ints into aten.view op.
namespace {
class DecomposeAtenFlattenUsingIntsOp
    : public OpRewritePattern<AtenFlattenUsingIntsOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFlattenUsingIntsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    MLIRContext *context = op.getContext();
    std::optional<unsigned> maybeRank = getTensorRank(self);
    if (!maybeRank)
      return rewriter.notifyMatchFailure(op, "unimplemented: unranked tensor");
    unsigned rank = *maybeRank;

    int64_t start, end;
    if (!matchPattern(op.getStartDim(), m_TorchConstantInt(&start)) ||
        !matchPattern(op.getEndDim(), m_TorchConstantInt(&end))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires start and end dims to be constants");
    }

    SmallVector<Value, 4> newSizes;
    if (rank == 0) {
      Value one =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
      newSizes.push_back(one);
    } else {
      start = toPositiveDim(start, rank);
      end = toPositiveDim(end, rank);

      if (start > end) {
        return rewriter.notifyMatchFailure(
            op, "expected end dim larger than start dim");
      }

      newSizes.reserve(rank - end + start);
      for (int64_t k = 0; k < start; ++k) {
        Value dim =
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(k));
        newSizes.push_back(
            rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/dim));
      }
      Value flattenDimSize =
          rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(-1));
      newSizes.push_back(flattenDimSize);
      for (int64_t k = end + 1; k < rank; ++k) {
        Value dim =
            rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(k));
        newSizes.push_back(
            rewriter.create<AtenSizeIntOp>(loc, self, /*dim=*/dim));
      }
    }
    Value newSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), newSizes);
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            newSizeList);
    return success();
  }
};
} // namespace

// Decompose aten.expand into aten.broadcast_to op.
namespace {
class DecomposeAtenExpandOp : public OpRewritePattern<AtenExpandOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExpandOp op,
                                PatternRewriter &rewriter) const override {
    bool implicit = false;
    if (!matchPattern(op.getImplicit(), m_TorchConstantBool(&implicit)) ||
        implicit) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires implicit to be false");
    }
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(), op.getSelf(),
                                                   op.getSize());
    return success();
  }
};
} // namespace

// Decompose aten.where.Scalar into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarOp : public OpRewritePattern<AtenWhereScalarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    Value selfTensor = createRank0Tensor(rewriter, loc, resType, op.getSelf());
    Value otherTensor = createRank0Tensor(rewriter, loc, resType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 selfTensor, otherTensor);
    return success();
  }
};
} // namespace

// Decompose aten.where.ScalarOther into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarOtherOp
    : public OpRewritePattern<AtenWhereScalarOtherOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarOtherOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    Value otherTensor = createRank0Tensor(rewriter, loc, resType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 op.getSelf(), otherTensor);
    return success();
  }
};
} // namespace

// Decompose aten.where.ScalarSelf into aten.where.self op.
namespace {
class DecomposeAtenWhereScalarSelfOp
    : public OpRewritePattern<AtenWhereScalarSelfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenWhereScalarSelfOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resType = op.getType().cast<BaseTensorType>();
    Value selfTensor = createRank0Tensor(rewriter, loc, resType, op.getSelf());
    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, resType, op.getCondition(),
                                                 selfTensor, op.getOther());
    return success();
  }
};
} // namespace

// Decompose aten.convolution_overrideable to aten.convolution op.
namespace {
class DecomposeAtenConvolutionOverrideableOp
    : public OpRewritePattern<AtenConvolutionOverrideableOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvolutionOverrideableOp op,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten._convolution-like to aten.convolution
namespace {
template <typename ConvolutionLikeOp>
class DecomposeAten_ConvolutionLikeOp
    : public OpRewritePattern<ConvolutionLikeOp> {
public:
  using OpRewritePattern<ConvolutionLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvolutionLikeOp op,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten.conv2d to aten.convolution
namespace {
class DecomposeAtenConv2dOp : public OpRewritePattern<AtenConv2dOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConv2dOp op,
                                PatternRewriter &rewriter) const override {

    Value emptyList = rewriter.create<PrimListConstructOp>(
        op.getLoc(), Torch::ListType::get(Torch::IntType::get(op.getContext())),
        SmallVector<Value>());
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), cstFalse, emptyList,
        op.getGroups());

    return success();
  }
};
} // namespace

// Decompose aten.conv_transpose2d to aten.convolution
namespace {
class DecomposeAtenConvTranspose2dOp
    : public OpRewritePattern<AtenConvTranspose2dInputOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvTranspose2dInputOp op,
                                PatternRewriter &rewriter) const override {

    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), true);
    rewriter.replaceOpWithNewOp<AtenConvolutionOp>(
        op, op->getResultTypes(), op.getInput(), op.getWeight(), op.getBias(),
        op.getStride(), op.getPadding(), op.getDilation(), /*transposed=*/cstTrue,
        op.getOutputPadding(), op.getGroups());
    return success();
  }
};
} // namespace

// Decompose aten.convolution_backward_overrideable to aten.convolution_backward
// op.
namespace {
class DecomposeAtenConvolutionBackwardOverrideableOp
    : public OpRewritePattern<AtenConvolutionBackwardOverrideableOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvolutionBackwardOverrideableOp op,
                                PatternRewriter &rewriter) const override {

    Value none = rewriter.create<ConstantNoneOp>(op->getLoc());
    rewriter.replaceOpWithNewOp<AtenConvolutionBackwardOp>(
        op, op.getResultTypes(), op.getGradOutput(), op.getInput(), op.getWeight(),
        none, op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups(), op.getOutputMask());

    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenConvolutionBackwardOp
    : public OpRewritePattern<AtenConvolutionBackwardOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenConvolutionBackwardOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value gradOutput = op.getGradOutput();
    std::optional<unsigned> maybeGradRank = getTensorRank(gradOutput);
    if (!maybeGradRank) {
      return rewriter.notifyMatchFailure(op,
                                         "expected grad output to have a rank");
    }
    unsigned gradRank = *maybeGradRank;
    Value cstZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value cstOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value cstTwo = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(2));
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(
        loc, rewriter.getBoolAttr(false));

    Value input = op.getInput();
    Value weight = op.getWeight();

    if (gradRank != 4)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only 2D convolutions supported.");

    SmallVector<Value> padding;
    if (!getListConstructElements(op.getPadding(), padding))
      return rewriter.notifyMatchFailure(op, "padding must be a list.");

    SmallVector<Value> strides;
    if (!getListConstructElements(op.getStride(), strides))
      return rewriter.notifyMatchFailure(op, "stride must be a list.");
    for (Value stride : strides) {
      Value cmp = rewriter.create<Torch::AtenEqIntOp>(loc, stride, cstOne);
      rewriter.create<Torch::RuntimeAssertOp>(
          loc, cmp, "unimplemented: only strides of 1 supported.");
    }

    SmallVector<Value> dilations;
    if (!getListConstructElements(op.getDilation(), dilations))
      return rewriter.notifyMatchFailure(op, "dilation must be a list.");
    for (Value dilation : dilations) {
      Value cmp = rewriter.create<Torch::AtenEqIntOp>(loc, dilation, cstOne);
      rewriter.create<Torch::RuntimeAssertOp>(
          loc, cmp, "unimplemented: only dilations of 1 supported.");
    }

    SmallVector<bool> outMask;
    if (!matchPattern(op.getOutputMask(), m_TorchListOfConstantBools(outMask)))
      return rewriter.notifyMatchFailure(
          op, "only constant bool output_mask is supported.");
    // Support for `False` values for output mask unimplemented.
    if (!llvm::all_of(outMask, [](bool mask) { return mask; }))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only true values for output_mask supported.");

    bool transposed;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return rewriter.notifyMatchFailure(
          op, "only constant transposed is supported.");
    if (transposed)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: transposed convolutions are not supported.");

    // Rotate weight.
    SmallVector<Value> axes;
    for (unsigned i = 2; i < gradRank; i++) {
      axes.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    }
    Value axesList = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())), axes);
    weight = rewriter.create<Torch::AtenFlipOp>(loc, weight.getType(), weight,
                                                axesList);
    // Calculate padding for first convolution.
    SmallVector<Value> gradInputPaddingValues;
    for (unsigned i = 2; i < gradRank; i++) {
      Value dim = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i));
      Value outDim = rewriter.create<Torch::AtenSizeIntOp>(loc, input, dim);

      // Calculate 1 + (weightDim // 2) * 2, which fixes issues with
      // even-sized weight.
      Value weightDim = rewriter.create<Torch::AtenSizeIntOp>(loc, weight, dim);
      weightDim =
          rewriter.create<Torch::AtenFloordivIntOp>(loc, weightDim, cstTwo);
      weightDim = rewriter.create<Torch::AtenMulIntOp>(loc, weightDim, cstTwo);
      weightDim = rewriter.create<Torch::AtenAddIntOp>(loc, weightDim, cstOne);
      Value gradOutDim =
          rewriter.create<Torch::AtenSizeIntOp>(loc, gradOutput, dim);

      // Calculate (((outDim - 1) * stride) + weightDim - gradOutDim) // 2,
      // the padding value for this dimension. Derived from the formula at
      // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
      Value padVal = rewriter.create<Torch::AtenSubIntOp>(loc, outDim, cstOne);
      padVal =
          rewriter.create<Torch::AtenMulIntOp>(loc, padVal, strides[i - 2]);
      padVal = rewriter.create<Torch::AtenAddIntOp>(loc, padVal, weightDim);
      padVal = rewriter.create<Torch::AtenSubIntOp>(loc, padVal, gradOutDim);
      padVal = rewriter.create<Torch::AtenFloordivIntOp>(loc, padVal, cstTwo);

      gradInputPaddingValues.push_back(padVal);
    }
    Value gradInputPadding = rewriter.create<Torch::PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), gradInputPaddingValues);
    Value weightTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
        loc, weight.getType(), weight, cstZero, cstOne);
    // Convolve grad_output with weight.
    Value gradInput = rewriter.create<Torch::AtenConvolutionOp>(
        loc, op.getResultTypes()[0], gradOutput, weightTransposed, cstNone,
        op.getStride(), gradInputPadding, op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups());

    Value gradOutputTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
        loc, gradOutput.getType(), gradOutput, cstZero, cstOne);
    Value inputTransposed = rewriter.create<Torch::AtenTransposeIntOp>(
        loc, input.getType(), input, cstZero, cstOne);
    // Convolve input with grad_output.
    Value gradWeight = rewriter.create<Torch::AtenConvolutionOp>(
        loc, op.getResultTypes()[1], inputTransposed, gradOutputTransposed,
        cstNone, op.getStride(), op.getPadding(), op.getDilation(), op.getTransposed(),
        op.getOutputPadding(), op.getGroups());
    gradWeight = rewriter.create<Torch::AtenTransposeIntOp>(
        loc, gradWeight.getType(), gradWeight, cstZero, cstOne);

    SmallVector<Value> dimIntList{cstZero};
    for (unsigned i = 2; i < gradRank; i++)
      dimIntList.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    Value gradIntList = rewriter.create<Torch::PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        dimIntList);
    // Sum grad_output along dim 1.
    Value gradBias = rewriter.create<Torch::AtenSumDimIntListOp>(
        loc, op.getResultTypes()[2], gradOutput, gradIntList, cstFalse,
        cstNone);

    rewriter.replaceOp(op, {gradInput, gradWeight, gradBias});
    return success();
  }
};
} // namespace

// Decompose aten.addmm into aten.mm and aten.add.Tensor op.
namespace {
class DecomposeAtenAddmmOp : public OpRewritePattern<AtenAddmmOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAddmmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value mat1 = op.getMat1();
    Value mat2 = op.getMat2();
    std::optional<unsigned> mat1Rank = getTensorRank(mat1);
    std::optional<unsigned> mat2Rank = getTensorRank(mat2);

    // The operands `mat1`, `mat2` to aten.addmm must be of rank 2.
    if (!mat1Rank || !mat2Rank || *mat1Rank != 2 || *mat2Rank != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected mat1, mat2 operands to aten.addmm to be rank 2");
    }

    // TODO: Handle integer type operands.
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non-floating point dtype");
    }

    // matrix multiplication: matmul = mat1 @ mat2
    Value matmul = rewriter.create<AtenMmOp>(loc, op.getType(), mat1, mat2);
    // scaledInput = self * beta
    Value scaledInput = rewriter.create<AtenMulScalarOp>(loc, input.getType(),
                                                         input, op.getBeta());
    // result = scaledInput + alpha * matmul
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), scaledInput,
                                                 matmul, op.getAlpha());
    return success();
  }
};
} // namespace

// Decompose aten.mean into: sum(x)/div(numTensorElements).
namespace {
class DecomposeAtenMeanOp : public OpRewritePattern<AtenMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value output = op.getResult();
    BaseTensorType outputTensorType = output.getType().cast<BaseTensorType>();
    Value sum =
        rewriter.create<AtenSumOp>(loc, outputTensorType, input, op.getDtype());
    Value numTensorElements = rewriter.create<AtenNumelOp>(loc, input);
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, outputTensorType, sum,
                                                 numTensorElements);
    return success();
  }
};
} // namespace

// productDimSize = product(size(dim) for dim in dims)
// aten.mean(x, dims) = aten.sum(x, dims) / productDimSize.
namespace {
class DecomposeAtenMeanDimOp : public OpRewritePattern<AtenMeanDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMeanDimOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;

    Value dimList = op.getDim();
    Value keepDim = op.getKeepdim();
    Value dtype = op.getDtype();
    Type outputType = op.getType();
    MLIRContext *context = op.getContext();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>() ||
        !isNoneOrFloatDtype(context, dtype)) {
      return rewriter.notifyMatchFailure(
          op, "only floating-point type is supported");
    }

    SmallVector<Value> dimListElements;
    if (!getListConstructElements(dimList, dimListElements) &&
        !dimList.getType().isa<Torch::NoneType>()) {
      return rewriter.notifyMatchFailure(
          op, "expected `dim` to be `None` or constructed from list construct");
    }

    // Compute sum along dimensions specified in `dimList`.
    Value sumAlongDims = rewriter.create<AtenSumDimIntListOp>(
        loc, outputType, input, dimList, keepDim, dtype);

    // `productDimSize` is product of sizes of dimensions to be reduced.
    Value productDimSize;
    // Case: Reduce along all dims.
    if (dimListElements.empty() && inputRank != 0) {
      productDimSize = rewriter.create<AtenNumelOp>(loc, input);
    } else {
      productDimSize = rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(1));
      for (Value dim : dimListElements) {
        Value dimSize = rewriter.create<AtenSizeIntOp>(loc, input, dim);
        productDimSize =
            rewriter.create<AtenMulIntOp>(loc, productDimSize, dimSize);
      }
    }
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, outputType, sumAlongDims,
                                                 productDimSize);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenSquareOp : public OpRewritePattern<AtenSquareOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSquareOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), self, self);
    return success();
  }
};
} // namespace

// Silu(x) = sigmoid(x) * x
namespace {
class DecomposeAtenSiluOp : public OpRewritePattern<AtenSiluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSiluOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    Value sigmoid =
        rewriter.create<AtenSigmoidOp>(op.getLoc(), op.getType(), self);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), sigmoid,
                                                 self);
    return success();
  }
};
} // namespace

// pDash = 1.0 - p
// boolMask = aten.rand_like(input) < pDash
// dropout(input, p, train=True) = (boolMask * input) / pDash
// dropout(input, p, train=False) = input
namespace {
class DecomposeAtenDropoutOp : public OpRewritePattern<AtenDropoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenDropoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value prob = op.getP();
    bool train = false;
    if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train)))
      return rewriter.notifyMatchFailure(op,
                                         "train must be a boolean constant");
    if (!train) {
      rewriter.replaceOp(op, input);
      return success();
    }
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>())
      return rewriter.notifyMatchFailure(
          op, "only support floating type input for training mode");
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value floatOne =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value oneMinusP = rewriter.create<AtenSubFloatOp>(loc, floatOne, prob);
    Value boolMask = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
        loc, inputType, input, oneMinusP, /*generator=*/noneVal);
    Value maskedInput =
        rewriter.create<AtenMulTensorOp>(loc, inputType, boolMask, input);
    rewriter.replaceOpWithNewOp<AtenDivScalarOp>(op, op.getType(), maskedInput,
                                                 oneMinusP);
    return success();
  }
};
} // namespace

// Decompose aten.var into: aten.var.dim op.
namespace {
class DecomposeAtenVarOp : public OpRewritePattern<AtenVarOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(self);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;
    BaseTensorType rank0FloatTensorTy = op.getType().cast<BaseTensorType>();
    if (!rank0FloatTensorTy.hasSizes() ||
        rank0FloatTensorTy.getSizes().size() != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected aten.var to have a rank 0 tensor type");
    }

    SmallVector<Value> dims;
    for (unsigned i = 0; i < inputRank; i++)
      dims.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())), dims);

    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<AtenVarDimOp>(op, rank0FloatTensorTy, self,
                                              dimList, op.getUnbiased(),
                                              /*keepdim=*/cstFalse);
    return success();
  }
};
} // namespace

// Decompose aten.std to sqrt(var(x))
namespace {
class DecomposeAtenStdOp : public OpRewritePattern<AtenStdOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorTy = self.getType().cast<BaseTensorType>();
    if (!inputTensorTy.hasDtype() ||
        !inputTensorTy.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "Only aten.std support floating type");
    }
    Value var = rewriter.create<AtenVarOp>(op->getLoc(), op.getType(),
                                           op.getSelf(), op.getUnbiased());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), var);
    return success();
  }
};
} // namespace

// Softplus(x, beta, threshold) =
//   x * beta > threshold ? x : log(1 + exp(x * beta)) / beta
namespace {
class DecomposeAtenSoftplusOp : public OpRewritePattern<AtenSoftplusOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSoftplusOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();

    Value inputTimesBeta =
        rewriter.create<AtenMulScalarOp>(loc, inputType, input, op.getBeta());

    // out = log1p(exp(input * beta)) / beta
    Value exp = rewriter.create<AtenExpOp>(loc, inputType, inputTimesBeta);
    Value log1p = rewriter.create<AtenLog1pOp>(loc, inputType, exp);
    Value out =
        rewriter.create<AtenDivScalarOp>(loc, inputType, log1p, op.getBeta());

    // Select where x * beta > threshold
    auto boolResType = inputType.getWithSizesAndDtype(inputType.getSizes(),
                                                      rewriter.getI1Type());
    Value condition = rewriter.create<AtenGtScalarOp>(
        loc, boolResType, inputTimesBeta, op.getThreshold());

    rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(op, op.getType(), condition,
                                                 input, out);
    return success();
  }
};
} // namespace

// Decompose aten.std.dim to sqrt(var.dim(x))
namespace {
class DecomposeAtenStdDimOp : public OpRewritePattern<AtenStdDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdDimOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorType = self.getType().cast<BaseTensorType>();
    if (!inputTensorType.hasDtype() ||
        !inputTensorType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "aten.std.dim expects input tensor of floating-point type");
    }

    Value varDim =
        rewriter.create<AtenVarDimOp>(op->getLoc(), op.getType(), self,
                                      op.getDim(), op.getUnbiased(), op.getKeepdim());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), varDim);
    return success();
  }
};
} // namespace

// Decompose aten.std.correction to sqrt(var.correction(x))
namespace {
class DecomposeAtenStdCorrectionOp
    : public OpRewritePattern<AtenStdCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenStdCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    Value self = op.getSelf();
    BaseTensorType inputTensorType = self.getType().cast<BaseTensorType>();
    if (!inputTensorType.hasDtype() ||
        !inputTensorType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(
          op,
          "aten.std.correction expects input tensor of floating-point type");
    }

    Value varCorrection = rewriter.create<AtenVarCorrectionOp>(
        op->getLoc(), op.getType(), self, op.getDim(), op.getCorrection(),
        op.getKeepdim());
    rewriter.replaceOpWithNewOp<AtenSqrtOp>(op, op.getType(), varCorrection);
    return success();
  }
};
} // namespace

// Hardsigmoid(x) = max(0, min(1, (x+3)/6))
namespace {
class DecomposeAtenHardsigmoidOp : public OpRewritePattern<AtenHardsigmoidOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardsigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();

    // outputTensor = (input + 3) / 6.
    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantThree = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(3));
    Value constantSix = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(6));
    Value inputPlusThree = rewriter.create<AtenAddScalarOp>(
        loc, inputType, input, constantThree, /*alpha=*/constantOne);
    Value outputTensor = rewriter.create<AtenDivScalarOp>(
        loc, inputType, inputPlusThree, constantSix);

    // result = max(0, min(1, (input+3)/6))
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value oneTensor = createRank0Tensor(rewriter, loc, inputType, constantOne);
    Value minResult =
        rewriter.create<AtenMinimumOp>(loc, inputType, oneTensor, outputTensor);
    Value zeroTensor =
        createRank0Tensor(rewriter, loc, inputType, constantZero);
    rewriter.replaceOpWithNewOp<AtenMaximumOp>(op, op.getType(), zeroTensor,
                                               minResult);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenHardtanhOp : public OpRewritePattern<AtenHardtanhOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardtanhOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();

    // result = min(maxVal, max(minVal, x))
    Value minVal = createRank0Tensor(rewriter, loc, inputType, op.getMinVal());
    Value maxResult =
        rewriter.create<AtenMaximumOp>(loc, inputType, input, minVal);
    Value maxVal = createRank0Tensor(rewriter, loc, inputType, op.getMaxVal());
    rewriter.replaceOpWithNewOp<AtenMinimumOp>(op, op.getType(), maxVal,
                                               maxResult);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandLikeOp : public OpRewritePattern<AtenRandLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandLikeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type resultType = op.getType();
    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>()) {
      return rewriter.notifyMatchFailure(op,
                                         "only support floating-point type");
    }

    // Create a uniform random op with low and high set to 0.0 and 1.0,
    // respectively.
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value zero =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(0.0));
    Value one =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
    Value emptyTensor = rewriter.create<AtenFullLikeOp>(
        loc, resultType, input, zero, op.getDtype(), op.getLayout(), op.getDevice(),
        op.getPinMemory(), op.getMemoryFormat());
    rewriter.replaceOpWithNewOp<AtenUniformOp>(op, resultType, emptyTensor,
                                               /*from=*/zero, /*to=*/one,
                                               /*generator=*/none);
    return success();
  }
};
} // namespace

namespace {
// Bernoulli(x, p) = (randLike(float(x)) < p).cast(type(x)). Here,
// 1. p must be a float tensor.
// 2. The shape of p should be broadcastable to the shape of x.
// 3. Bernoulli(x, p) returns a tensor of the same type as that of x.
static LogicalResult decomposeBernoulliLikeOp(PatternRewriter &rewriter,
                                              Operation *op, Location loc,
                                              Value input, Value prob,
                                              Value &output) {
  auto inputType = input.getType().cast<BaseTensorType>();
  auto probType = prob.getType().cast<BaseTensorType>();
  // Both the `input` and `prob` must be ranked tensors.
  if (!inputType.hasSizes() || !inputType.hasDtype() || !probType.hasSizes() ||
      !probType.hasDtype()) {
    return rewriter.notifyMatchFailure(
        op, "can't decompose bernoulli like ops without sizes or dtype");
  }
  // The `prob` is expected to be a float type tensor.
  if (!probType.getDtype().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(
        op, "probabilities must be a float type tensor");
  }

  // Since the `aten.randLike` op expects float-type operand, create a
  // float-type tensor with the same shape as that of the `input`.
  Value floatTensor =
      convertTensorToDtype(rewriter, loc, input, rewriter.getF64Type());
  Value none = rewriter.create<ConstantNoneOp>(loc);
  Value randomVal = rewriter.create<AtenRandLikeOp>(
      loc, floatTensor.getType(), floatTensor, /*dtype=*/none, /*layout=*/none,
      /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);

  // Bernoulli(x, p) = randLike(float(x)) < p.
  auto boolResType = inputType.getWithSizesAndDtype(inputType.getSizes(),
                                                    rewriter.getI1Type());
  Value lessThanP =
      rewriter.create<AtenLtTensorOp>(loc, boolResType, randomVal, prob);

  // As the `output` is expected to be of the `input` type, convert the boolean
  // tensor `lessThanP` to a `input` type tensor.
  output = convertTensorToDtype(rewriter, loc, lessThanP, inputType.getDtype());
  return success();
}

// aten.bernoulli(x) = randLike(x) < x. Here, the input x is a tensor
// containing probabilities to be used for drawing the binary random number.
class DecomposeAtenBernoulliOp : public OpRewritePattern<AtenBernoulliOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBernoulliOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to ben None because only global default "
              "generator is supported");
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, input, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};

// aten.bernoulli.float(x, p) = (randLike(float(x)) < tensor(p)).cast(type(x)).
// Since the input x can be an integer tensor, it's important to cast it to
// float type before passing it to the `aten.randLike` op.
class DecomposeValsemVariantAtenBernoulliFloatOp
    : public OpRewritePattern<ValsemVariantAtenBernoulliFloatOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ValsemVariantAtenBernoulliFloatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value p = op.getP();
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to ben None because only global default "
              "generator is supported");

    auto inputType = input.getType().cast<BaseTensorType>();
    SmallVector<int64_t> empty;
    Type tensorType = inputType.getWithSizesAndDtype(llvm::makeArrayRef(empty),
                                                     rewriter.getF64Type());
    Value prob = rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType, p);
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, prob, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};

// aten.bernoulli.Tensor(x, p) = (randLike(float(x)) < p).cast(type(x)).
// Since the input x can be an integer tensor, it's important to cast it to
// float type before passing it to the `aten.randLike` op.
class DecomposeAtenBernoulliTensorOp
    : public OpRewritePattern<AtenBernoulliTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBernoulliTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value prob = op.getP();
    if (!op.getGenerator().getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "The generator has to ben None because only global default "
              "generator is supported");
    Value output;
    if (failed(
            decomposeBernoulliLikeOp(rewriter, op, loc, input, prob, output)))
      return rewriter.notifyMatchFailure(
          op, "decomposeBernoulliLikeOp failed to decompose the op");
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy, typename T1T2Op>
class DecomposeAtenAddCLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Value tensor1 = op.getTensor1();
    Value tensor2 = op.getTensor2();
    Value value = op.getValue();

    Value product =
        rewriter.create<T1T2Op>(loc, op.getType(), tensor1, tensor2);
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), input,
                                                 product, value);
    return success();
  }
};

class DecomposeAtenLayerNormOp : public OpRewritePattern<AtenLayerNormOp> {
  using OpRewritePattern<AtenLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto input = op.getInput().getType().cast<BaseTensorType>();
    if (!input.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = input.getSizes().size();
    Value normalizedShape = op.getNormalizedShape();
    SmallVector<Value> normalizedShapeSizesTorchInt;
    getListConstructElements(normalizedShape, normalizedShapeSizesTorchInt);
    int64_t axis = inputRank - normalizedShapeSizesTorchInt.size();
    std::vector<int64_t> meanVarSizes(inputRank, 1);
    for (int i = 0; i < axis; i++)
      meanVarSizes[i] = input.getSizes()[i];
    auto meanVarType = input.getWithSizesAndDtype(
        llvm::makeArrayRef(meanVarSizes), input.getOptionalDtype());
    auto nativeLayerNorm = rewriter.create<AtenNativeLayerNormOp>(
        loc, op.getType(), meanVarType, meanVarType, op.getInput(),
        op.getNormalizedShape(), op.getWeight(), op.getBias(), op.getEps());
    rewriter.replaceOp(op, nativeLayerNorm.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNativeLayerNormOp
    : public OpRewritePattern<AtenNativeLayerNormOp> {
  using OpRewritePattern<AtenNativeLayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeLayerNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();

    auto inputTy = op.getInput().getType().cast<BaseTensorType>();
    if (!inputTy.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "input tensor should have known sizes.");
    int64_t inputRank = inputTy.getSizes().size();
    Value normalizedShape = op.getNormalizedShape();
    SmallVector<Value> normalizedShapeSizesTorchInt;
    getListConstructElements(normalizedShape, normalizedShapeSizesTorchInt);
    int64_t axis = inputRank - normalizedShapeSizesTorchInt.size();
    auto reduceDimInts = llvm::to_vector<4>(llvm::seq<int64_t>(axis, inputRank));
    auto reducedTy = op.getResult(1).getType();
    auto sizeListType = ListType::get(IntType::get(context));

    // build reduce dims
    SmallVector<Value> reduceDimVals;
    reduceDimVals.reserve(reduceDimInts.size());
    std::transform(reduceDimInts.begin(), reduceDimInts.end(),
                   std::back_inserter(reduceDimVals), [&](int64_t d) {
                     return rewriter.create<Torch::ConstantIntOp>(
                         loc, rewriter.getI64IntegerAttr(d));
                   });
    Value reduceDimList =
        rewriter.create<PrimListConstructOp>(loc, sizeListType, reduceDimVals);
    Value one = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));

    Value cstTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value none = rewriter.create<Torch::ConstantNoneOp>(loc);
    // mean(x)
    Value inputMean = rewriter.create<AtenMeanDimOp>(
        loc, reducedTy, op.getInput(), reduceDimList, cstTrue, none);

    // x - mean(x)
    Value inputMeanExpanded =
        rewriter.create<AtenExpandAsOp>(loc, inputTy, inputMean, op.getInput());
    Value inputZeroMean = rewriter.create<AtenSubTensorOp>(
        loc, inputTy, op.getInput(), inputMeanExpanded, one);
    // var(x) = mean((x - mean(x))^2)
    Value inputZeroMeanSquare = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputZeroMean, inputZeroMean);
    Value inputVar = rewriter.create<AtenMeanDimOp>(
        loc, reducedTy, inputZeroMeanSquare, reduceDimList, cstTrue, none);

    // rsqrt(var(x) + eps)
    Value inputVarPlusEps = rewriter.create<AtenAddScalarOp>(
        loc, reducedTy, inputVar, op.getEps(), one);
    Value inputRsqrtVar =
        rewriter.create<AtenRsqrtOp>(loc, reducedTy, inputVarPlusEps);

    // (x - mean(x)) * rsqrt(var(x) + eps)
    Value inputRsqrtVarExpanded = rewriter.create<AtenExpandAsOp>(
        loc, inputTy, inputRsqrtVar, op.getInput());
    Value inputNormalized = rewriter.create<AtenMulTensorOp>(
        loc, inputTy, inputZeroMean, inputRsqrtVarExpanded);
    Value out = rewriter.create<TensorStaticInfoCastOp>(
        loc, op.getResult(0).getType(), inputNormalized);

    Value weight = op.getWeight();
    Value bias = op.getBias();
    if (!weight.getType().isa<Torch::NoneType>()) {
      out = rewriter.create<AtenMulTensorOp>(loc, out.getType(), out, weight);
    }
    if (!bias.getType().isa<Torch::NoneType>()) {
      out =
          rewriter.create<AtenAddTensorOp>(loc, out.getType(), out, bias, one);
    }
    rewriter.replaceOp(op, {out, inputMean, inputRsqrtVar});

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.emptyLike` op into `aten.size` and `aten.empty` ops.
class DecomposeAtenEmptyLikeOp : public OpRewritePattern<AtenEmptyLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenEmptyLikeOp op,
                                PatternRewriter &rewriter) const override {
    auto sizeListType =
        Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value sizeList =
        rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, op.getSelf());
    rewriter.replaceOpWithNewOp<AtenEmptyMemoryFormatOp>(
        op, op.getType(), sizeList, op.getDtype(), op.getLayout(), op.getDevice(),
        op.getPinMemory(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
// The `aten.arange` op is converted to `aten.arange.startStep` op.
class DecomposeAtenArangeOp : public OpRewritePattern<AtenArangeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenArangeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The AtenArangeOp doesn't have a start and step value. Therefore we set
    // them as default values 0 and 1, respectively.
    Value start, step;
    start = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    step = rewriter.create<Torch::ConstantIntOp>(loc,
                                                 rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<AtenArangeStartStepOp>(
        op, op.getType(), start, op.getEnd(), step, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// The `aten.arange.start` op is converted to `aten.arange.startStep` op.
class DecomposeAtenArangeStartOp : public OpRewritePattern<AtenArangeStartOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenArangeStartOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The AtenArangeStartOp doesn't have a step value. Therefore we set it as
    // default value 1.
    Value step;
    step = rewriter.create<Torch::ConstantIntOp>(loc,
                                                 rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<AtenArangeStartStepOp>(
        op, op.getType(), op.getStart(), op.getEnd(), step, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// Decompose constant tensor full like ops.
template <typename OpTy, int fillVal>
class DecomposeConstantTensorAllocLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value constVal = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(fillVal));
    rewriter.replaceOpWithNewOp<AtenFullLikeOp>(
        op, op.getType(), op.getSelf(), constVal, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenNativeBatchNormOp
    : public OpRewritePattern<AtenNativeBatchNormOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNativeBatchNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Value runningMean = op.getRunningMean();
    Value runningVar = op.getRunningVar();
    Value eps = op.getEps();

    // TODO: Add support for `training` mode.
    bool training = false;
    if (!matchPattern(op.getTraining(), m_TorchConstantBool(&training)) ||
        training)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: training mode is not supported");

    // Rank of the input tensor must be greater than or equal to 2. The shape of
    // the `input` is supposed to be (N, C, D?, H?, W?).
    std::optional<unsigned> maybeInputRank = getTensorRank(input);
    if (!maybeInputRank || *maybeInputRank < 2)
      return rewriter.notifyMatchFailure(
          op, "input must have rank greater than or equal to 2");
    unsigned inputRank = *maybeInputRank;

    // In the inference mode, the `runningMean` and `runningVar` must not be
    // None.
    if (runningMean.getType().isa<Torch::NoneType>() ||
        runningVar.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "running stats must not be None in inference mode");

    // Rank of `runningMean` and `runningVar` must be exactly 1.
    std::optional<unsigned> runningMeanRank = getTensorRank(runningMean);
    std::optional<unsigned> runningVarRank = getTensorRank(runningVar);
    if (!runningMeanRank || !runningVarRank || *runningMeanRank != 1 ||
        *runningVarRank != 1)
      return rewriter.notifyMatchFailure(
          op, "expected runningMean and runningVar to be rank 1");

    Value zero =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value numFeatures = rewriter.create<AtenSizeIntOp>(loc, input, /*dim=*/one);
    // TODO: Add Runtime Asserts to check the shape of weight, bias,
    // runningMean and runningVar to be (numFeatures).

    // The `runningMean` and `runningVar` must be reshaped to (1, C, 1?, 1?, 1?)
    // to make it broadcast-compatible with (N, C, D?, H?, W?).
    // 1. runningMean = runningMean.view(1, C, 1?, 1?, 1?)
    // 2. runningVar = runningVar.view(1, C, 1?, 1?, 1?)
    SmallVector<Value> runningStatsShape(inputRank, one);
    runningStatsShape[1] = numFeatures;
    Value runningStatsSizeList = rewriter.create<PrimListConstructOp>(
        loc, ListType::get(IntType::get(context)), runningStatsShape);

    SmallVector<int64_t> runningStatsShapeInt(inputRank, 1);
    runningStatsShapeInt[1] = kUnknownSize;
    Type dtype = input.getType().cast<ValueTensorType>().getOptionalDtype();
    Type reshapeType = ValueTensorType::get(
        context, llvm::makeArrayRef(runningStatsShapeInt), dtype);

    runningMean = rewriter.create<AtenViewOp>(loc, reshapeType, runningMean,
                                              runningStatsSizeList);
    runningVar = rewriter.create<AtenViewOp>(loc, reshapeType, runningVar,
                                             runningStatsSizeList);

    // normalizedInput = (input - runningMean) / (sqrt(runningVar + eps)).
    Value inputSubMean = rewriter.create<AtenSubTensorOp>(
        loc, input.getType(), input, runningMean, /*alpha=*/one);
    Value varEps = rewriter.create<AtenAddScalarOp>(
        loc, runningVar.getType(), runningVar, eps, /*alpha=*/one);
    Value invStd = rewriter.create<AtenRsqrtOp>(loc, varEps.getType(), varEps);
    Value normalizedInput = rewriter.create<AtenMulTensorOp>(
        loc, inputSubMean.getType(), inputSubMean, invStd);

    // The `weight` and `bias` must be reshaped to (1, C, 1?, 1?, 1?) to make it
    // broadcast-compatible with (N, C, D?, H?, W?).
    // 1. weight = weight.view(1, C, 1?, 1?, 1?)
    // 2. bias = bias.view(1, C, 1?, 1?, 1?)
    // 3. output = normalizedInput * weight + bias
    Value batchNormOutput = normalizedInput;
    if (!weight.getType().isa<Torch::NoneType>()) {
      // Rank of `weight` must be exactly 1.
      std::optional<unsigned> weightRank = getTensorRank(weight);
      if (!weightRank || *weightRank != 1)
        return rewriter.notifyMatchFailure(op, "expected weight to be rank 1");
      weight = rewriter.create<AtenViewOp>(loc, reshapeType, weight,
                                           runningStatsSizeList);
      batchNormOutput = rewriter.create<AtenMulTensorOp>(
          loc, batchNormOutput.getType(), batchNormOutput, weight);
    }
    if (!bias.getType().isa<Torch::NoneType>()) {
      // Rank of `bias` must be exactly 1.
      std::optional<unsigned> biasRank = getTensorRank(bias);
      if (!biasRank || *biasRank != 1)
        return rewriter.notifyMatchFailure(op, "expected bias to be rank 1");
      bias = rewriter.create<AtenViewOp>(loc, reshapeType, bias,
                                         runningStatsSizeList);
      batchNormOutput = rewriter.create<AtenAddTensorOp>(
          loc, batchNormOutput.getType(), batchNormOutput, bias, /*alpha=*/one);
    }

    // The `mean` and `invstd` outputs are empty tensors in inference mode.
    Value zeroList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(zero.getType()), zero);
    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value emptyMeanTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, op.getType(1), zeroList, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);
    Value emptyInvStdTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, op.getType(2), zeroList, /*dtype=*/none, /*layout=*/none,
        /*device=*/none, /*pinMemory=*/none, /*memoryFormat=*/none);

    rewriter.replaceOp(op,
                       {batchNormOutput, emptyMeanTensor, emptyInvStdTensor});
    return success();
  }
};
} // namespace

// Decompse `Aten_UnsafeViewOp` into `AtenViewOp`. UnsafeView() differs from
// view() in that the returned tensor isn't treated as a view for the purposes
// of automatic differentiation.  It's only safe to use if the `self` tensor is
// temporary. For example, the viewed tensor here (a + b) is discarded
// immediately after viewing:
//
//  res = UnsafeView(a + b, size);
//
// This is a hack because in-place operations on tensors treated like views
// can be much more expensive than the same operations on non-view tensors.

// Refer to
// https://github.com/pytorch/pytorch/blob/364055b2771ecf9b54f1d67a8bf44bb5496476d4/aten/src/ATen/native/TensorShape.cpp#L2072
namespace {
class DecomposeAten_UnsafeViewOp : public OpRewritePattern<Aten_UnsafeViewOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_UnsafeViewOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            op.getSize());
    return success();
  }
};
} // namespace

// In PyTorch, ReshapeAlias just uses an already computed stride.
// See
// https://github.com/pytorch/pytorch/blob/d8c31a819d4a65e732b5901e3b994e1869851f1a/aten/src/ATen/native/TensorShape.cpp#L1153
// Note that this is the same decomposition as in AOTAutograd
// https://github.com/pytorch/functorch/blob/a3042d94e616d4143813668b1372d9d4545be14e/functorch/Src/aotAutograd.py#L104
namespace {
class DecomposeAten_ReshapeAliasOp
    : public OpRewritePattern<Aten_ReshapeAliasOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ReshapeAliasOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenViewOp>(op, op.getType(), op.getSelf(),
                                            op.getSize());
    return success();
  }
};
} // namespace

namespace {
// Decompose constant tensor like ops.
template <typename OpTy, typename NewOpTy>
class DecomposeConstantTensorNewLikeOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::NoneType>()) {
      BaseTensorType tensorType =
          op.getSelf().getType().template cast<BaseTensorType>();
      if (!tensorType.hasDtype()) {
        return rewriter.notifyMatchFailure(
            op, "expected input tensor to have a dtype");
      }
      dtype =
          getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
    }
    rewriter.replaceOpWithNewOp<NewOpTy>(op, op.getType(), op.getSize(), dtype,
                                         op.getLayout(), op.getDevice(),
                                         op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.full` op into `aten.broadcastTo`
class DecomposeAtenFullOp : public OpRewritePattern<AtenFullOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFullOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    BaseTensorType outTy = op.getType().template cast<BaseTensorType>();
    if (!outTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    SmallVector<int64_t> empty;
    auto dtype =
        getTypeForTorchType(op.getContext(), op.getFillValue().getType());
    Type tensorType =
        outTy.getWithSizesAndDtype(llvm::makeArrayRef(empty), dtype);
    Value fillVal = rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType,
                                                             op.getFillValue());
    fillVal = convertTensorToDtype(rewriter, loc, fillVal, outTy.getDtype());
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(), fillVal,
                                                   op.getSize());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.linear` op into `aten.matmul` and `aten.add` ops.
class DecomposeAtenLinearOp : public OpRewritePattern<AtenLinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();

    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasSizes() || inputType.getSizes().size() < 2)
      return rewriter.notifyMatchFailure(
          op, "expected input to be rank 2 or greater");

    BaseTensorType weightType = weight.getType().cast<BaseTensorType>();
    // `weight` must be a rank 2 matrix.
    if (!weightType.hasSizes() || weightType.getSizes().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected weight to be a rank 2");

    SmallVector<int64_t> transposeShape =
        llvm::to_vector(llvm::reverse(weightType.getSizes()));
    Type transposeType = weightType.getWithSizesAndDtype(
        llvm::makeArrayRef(transposeShape), weightType.getOptionalDtype());
    Value transposeWeight =
        rewriter.create<AtenTOp>(loc, transposeType, weight);

    Value matmul = rewriter.create<AtenMatmulOp>(loc, op.getType(), input,
                                                 transposeWeight);
    if (bias.getType().isa<Torch::NoneType>()) {
      rewriter.replaceOp(op, matmul);
      return success();
    }

    BaseTensorType biasType = bias.getType().cast<BaseTensorType>();
    if (!biasType.hasSizes() || biasType.getSizes().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected bias to be rank 1");

    Value alpha =
        rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1));
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(op, op.getType(), matmul,
                                                 op.getBias(), alpha);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.mish` op into `aten.tanh` and `aten.softplus` ops.
// Mish(x) = x * Tanh(Softplus(x))
class DecomposeAtenMishOp : public OpRewritePattern<AtenMishOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMishOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSelf();
    Type type = op.getType();

    auto inputType = input.getType().cast<BaseTensorType>();
    if (!inputType.hasDtype())
      return rewriter.notifyMatchFailure(op, "Dtype not present");

    Type dType = inputType.getDtype();
    // Form default Value tensors for `beta` and `threshold` operands
    // of `aten.softplus` op.
    Value beta = getConstantWithGivenDtypeAndValue(rewriter, loc, 1.0, dType);
    Value threshold =
        getConstantWithGivenDtypeAndValue(rewriter, loc, 20.0, dType);
    Value softplusOp =
        rewriter.create<AtenSoftplusOp>(loc, type, input, beta, threshold);
    Value tanhOp = rewriter.create<AtenTanhOp>(loc, type, softplusOp);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, type, input, tanhOp);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.fullLike` op into `aten.emptyLike` and `aten.fill` ops.
class DecomposeAtenFullLikeOp : public OpRewritePattern<AtenFullLikeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFullLikeOp op,
                                PatternRewriter &rewriter) const override {
    BaseTensorType outTy = op.getType().template cast<BaseTensorType>();
    if (!outTy.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    SmallVector<int64_t> empty;
    auto dtype =
        getTypeForTorchType(op.getContext(), op.getFillValue().getType());
    Type tensorType =
        outTy.getWithSizesAndDtype(llvm::makeArrayRef(empty), dtype);
    Value fillVal = rewriter.create<PrimNumToTensorScalarOp>(
        op.getLoc(), tensorType, op.getFillValue());
    fillVal =
        convertTensorToDtype(rewriter, op.getLoc(), fillVal, outTy.getDtype());
    rewriter.replaceOpWithNewOp<AtenExpandAsOp>(op, op.getType(), fillVal,
                                                op.getSelf());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.indexPut` op into `valsem.aten.indexPutImpl` op.
class DecomposeAtenIndexPutOp : public OpRewritePattern<AtenIndexPutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexPutOp op,
                                PatternRewriter &rewriter) const override {
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<Aten_IndexPutImplOp>(
        op, op.getType(), op.getSelf(), op.getIndices(), op.getValues(), op.getAccumulate(),
        /*unsafe=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenExpandAsOp : public OpRewritePattern<AtenExpandAsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenExpandAsOp op,
                                PatternRewriter &rewriter) const override {

    auto sizeListType =
        Torch::ListType::get(Torch::IntType::get(op.getContext()));
    Value sizeList =
        rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, op.getOther());
    rewriter.replaceOpWithNewOp<AtenBroadcastToOp>(op, op.getType(), op.getSelf(),
                                                   sizeList);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.ToCopy` op into `valsem.aten.copy` op.
class DecomposeAten_ToCopyOp : public OpRewritePattern<Aten_ToCopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_ToCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<BaseTensorType>();
    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultDtype = resultType.getDtype();
    Value zero = getConstantWithGivenDtypeAndValue(rewriter, op.getLoc(), 0.0,
                                                   resultDtype);
    Value emptyTensor = rewriter.create<AtenFullLikeOp>(
        op.getLoc(), op.getType(), op.getSelf(), zero, op.getDtype(), op.getLayout(),
        op.getDevice(), op.getPinMemory(), op.getMemoryFormat());
    rewriter.replaceOpWithNewOp<AtenCopyOp>(op, op.getType(), emptyTensor,
                                            op.getSelf(), op.getNonBlocking());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.copy` op into `aten.to.dtype` and `aten.expand_as`.
class DecomposeAtenCopyOp : public OpRewritePattern<AtenCopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getType().cast<BaseTensorType>();
    if (!resultType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }
    Type resultDtype = resultType.getDtype();
    Value srcToDtype =
        convertTensorToDtype(rewriter, op.getLoc(), op.getSrc(), resultDtype);
    rewriter.replaceOpWithNewOp<AtenExpandAsOp>(op, op.getType(), srcToDtype,
                                                op.getSelf());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.newEmpty` op into `aten.empty.memoryFormat` op.
class DecomposeAtenNewEmptyOp : public OpRewritePattern<AtenNewEmptyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNewEmptyOp op,
                                PatternRewriter &rewriter) const override {
    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value dtype = op.getDtype();
    if (dtype.getType().isa<Torch::NoneType>()) {
      BaseTensorType tensorType = op.getSelf().getType().cast<BaseTensorType>();
      if (!tensorType.hasDtype()) {
        return rewriter.notifyMatchFailure(
            op, "expected input tensor to have a dtype");
      }
      dtype =
          getDtypeIntValueForType(rewriter, op.getLoc(), tensorType.getDtype());
    }
    rewriter.replaceOpWithNewOp<AtenEmptyMemoryFormatOp>(
        op, op.getType(), op.getSize(), dtype, op.getLayout(), op.getDevice(),
        op.getPinMemory(), /*memoryFormat=*/noneVal);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.indexPut.hackedTwin` op into `valsem.aten.indexPutImpl`
// op.
class DecomposeAtenIndexPutHackedTwinOp
    : public OpRewritePattern<AtenIndexPutHackedTwinOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexPutHackedTwinOp op,
                                PatternRewriter &rewriter) const override {
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(op.getLoc(), false);
    rewriter.replaceOpWithNewOp<Aten_IndexPutImplOp>(
        op, op.getType(), op.getSelf(), op.getIndices(), op.getValues(), op.getAccumulate(),
        /*unsafe=*/cstFalse);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.pad` op into `aten.constantPadNd` op.
class DecomposeAtenPadOp : public OpRewritePattern<AtenPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenPadOp op,
                                PatternRewriter &rewriter) const override {

    Value value = op.getValue();
    if (value.getType().isa<Torch::OptionalType>())
      return rewriter.notifyMatchFailure(op, "optional type not supported");
    if (value.getType().isa<Torch::NoneType>())
      value = rewriter.create<Torch::ConstantFloatOp>(
          op.getLoc(), rewriter.getF64FloatAttr(0));

    rewriter.replaceOpWithNewOp<AtenConstantPadNdOp>(
        op, op.getType(), op.getSelf(), op.getPad(), value);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.to.dtypeLayout` op into `aten.to.dtype` op.
class DecomposeAtenToDtypeLayoutOp
    : public OpRewritePattern<AtenToDtypeLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToDtypeLayoutOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Add support for pinMemory arg equal to `True`.
    if (!op.getPinMemory().getType().isa<Torch::NoneType>()) {
      bool pinMemory;
      if (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory must be a constant");
      else if (pinMemory)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: pinMemory is expected to be false");
    }

    // TODO: Add support for non-None device arg.
    if (!op.getDevice().getType().isa<Torch::NoneType>()) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: device arg must be None");
    }

    // TODO: Add support for non-strided layout.
    // torch.layout is by default strided i.e. 0.
    if (!op.getLayout().getType().isa<Torch::NoneType>()) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout must be a constant");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "unimplemented: layout is expected to be strided");
    }

    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(op, op.getType(), op.getSelf(),
                                               op.getDtype(), op.getNonBlocking(),
                                               op.getCopy(), op.getMemoryFormat());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.to.device` op into `aten.to.dtype` op.
class DecomposeAtenToDeviceOp : public OpRewritePattern<AtenToDeviceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenToDeviceOp op,
                                PatternRewriter &rewriter) const override {

    // Device information isn't relevant to torch-mlir, so we can drop that info
    // here.
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(op, op.getType(), op.getSelf(),
                                               op.getDtype(), op.getNonBlocking(),
                                               op.getCopy(), op.getMemoryFormat());

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.adaptiveAvgPool2d` op into `aten.avgPool2d` op.
//
// For AdaptiveAvgPool2d op, when the input size is an integer multiple of
// output size the kernelSize, stride and padding is calculated as follows:
// strideH = inH // outH
// strideW = inH // outH
// kernelH = inH - [(outH - 1) * strideH]
// kernelW = inW - [(outW - 1) * strideW]
// paddingH = 0, paddingW = 0
//
// For the special case, when the output size is one for all dimensions,
// the kernel size is same as the input size.
class DecomposeAtenAdaptiveAvgPool2dOp
    : public OpRewritePattern<AtenAdaptiveAvgPool2dOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenAdaptiveAvgPool2dOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value input = op.getSelf();
    std::optional<unsigned> maybeRank = getTensorRank(input);
    if (!maybeRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned rank = *maybeRank;
    SmallVector<Value, 2> inputHW;
    Value dimH = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rank - 2));
    inputHW.push_back(
        /*inH=*/rewriter.create<AtenSizeIntOp>(loc, input, dimH));
    Value dimW = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(rank - 1));
    inputHW.push_back(
        /*inW=*/rewriter.create<AtenSizeIntOp>(loc, input, dimW));

    Value outputShape = op.getOutputSize();
    SmallVector<Value> outputShapeSizesTorchInt;
    getListConstructElements(outputShape, outputShapeSizesTorchInt);

    // TODO: Add support for cases other than:
    // 1.) inH == outH and inW == outW.
    // 2.) outH == outW == 1
    bool unitOutputSize = true;
    for (Value outShape : outputShapeSizesTorchInt) {
      int64_t outShapeInt;
      if (!matchPattern(outShape, m_TorchConstantInt(&outShapeInt))) {
        return rewriter.notifyMatchFailure(
            op, "output size is expected to be a constant");
      }
      if (outShapeInt != 1) {
        unitOutputSize = false;
        break;
      }
    }

    Value constantOne = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(1));
    Value constantZero = rewriter.create<Torch::ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(0));
    Value constantFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value constantTrue = rewriter.create<Torch::ConstantBoolOp>(loc, true);
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    SmallVector<Value, 2> kernelSize;

    for (unsigned i = 0; i < inputHW.size(); i++) {
      if (unitOutputSize) {
        BaseTensorType inputTensorType = input.getType().cast<BaseTensorType>();
        ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
        kernelSize.push_back(inputShape[rank - 2 + i] == kUnknownSize
                                 ? inputHW[i]
                                 : rewriter.create<Torch::ConstantIntOp>(
                                       loc, rewriter.getI64IntegerAttr(
                                                inputShape[rank - 2 + i])));
      } else {
        Value cond = rewriter.create<AtenEqIntOp>(loc, inputHW[i],
                                                  outputShapeSizesTorchInt[i]);
        rewriter.create<RuntimeAssertOp>(
            loc, cond,
            "unimplemented: only support cases where input and output size are "
            "equal for non-unit output size");

        Value outMinusOne = rewriter.create<AtenSubIntOp>(
            loc, outputShapeSizesTorchInt[i], constantOne);
        kernelSize.push_back(
            rewriter.create<AtenSubIntOp>(loc, inputHW[i], outMinusOne));
      }
    }

    Value kernelSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)), kernelSize);
    // Currently we only support cases where input size is equal to the output
    // size or unit output size. For the former case, stride is always equal to
    // one and for the latter the stride value doesn't matter, since the kernel
    // size is same as the input size. Therfore, keeping the stride as one for
    // the latter case as well for the ease of implementation.
    Value strideList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantOne, constantOne});
    Value paddingSizeList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(context)),
        ValueRange{constantZero, constantZero});

    rewriter.replaceOpWithNewOp<AtenAvgPool2dOp>(
        op, op.getType(), input, kernelSizeList, strideList, paddingSizeList,
        /*ceilMode=*/constantFalse, /*countIncludePad=*/constantTrue,
        /*divisorOverride=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.clampMin` op into `aten.clamp` op.
class DecomposeAtenClampMinOp : public OpRewritePattern<AtenClampMinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenClampMinOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenClampOp>(op, op.getType(), op.getSelf(),
                                             op.getMin(), /*max=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.clampMax` op into `aten.clamp` op.
class DecomposeAtenClampMaxOp : public OpRewritePattern<AtenClampMaxOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenClampMaxOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenClampOp>(op, op.getType(), op.getSelf(),
                                             /*min=*/constantNone, op.getMax());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.baddbmm` op into `aten.bmm`, `aten.mul.Scalar`, and
// `aten.add.Tensor` op.
class DecomposeAtenBaddbmmOp : public OpRewritePattern<AtenBaddbmmOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenBaddbmmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bmm =
        rewriter.create<AtenBmmOp>(loc, op.getType(), op.getBatch1(), op.getBatch2());
    Value alphaTimesBmm =
        rewriter.create<AtenMulScalarOp>(loc, op.getType(), bmm, op.getAlpha());
    Value input = op.getSelf();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();
    BaseTensorType resultType =
        op->getResult(0).getType().cast<BaseTensorType>();
    if (inputType.hasDtype() && resultType.hasDtype() &&
        inputType.getDtype() != resultType.getDtype()) {
      input = convertTensorToDtype(rewriter, loc, input, resultType.getDtype());
    }
    rewriter.replaceOpWithNewOp<AtenAddTensorOp>(
        op, op.getType(), alphaTimesBmm, op.getSelf(), op.getBeta());
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.floorDivide` op into `aten.div.TensorMode` op.
class DecomposeAtenFloorDivideOp : public OpRewritePattern<AtenFloorDivideOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFloorDivideOp op,
                                PatternRewriter &rewriter) const override {
    // https://pytorch.org/docs/stable/generated/torch.floorDivide.html
    // PyTorch aten.floorDivide is a misnomer because it actually rounds
    // the quotient towards zero instead of taking its floor.
    Value cstStrFloor =
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "trunc");
    rewriter.replaceOpWithNewOp<AtenDivTensorModeOp>(
        op, op.getType(), op.getSelf(), op.getOther(),
        /*roundingMode=*/cstStrFloor);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.numpyT` op into `aten.permute` op.
class DecomposeAtenNumpyTOp : public OpRewritePattern<AtenNumpyTOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenNumpyTOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();
    std::optional<unsigned> maybeInputRank = getTensorRank(self);
    if (!maybeInputRank) {
      return rewriter.notifyMatchFailure(op, "expected input to have a rank");
    }
    unsigned inputRank = *maybeInputRank;

    SmallVector<Value> dimListElements;
    SmallVector<int> dimListInts(llvm::reverse(
        llvm::iota_range<int>(0, inputRank, /*inclusive=*/false)));
    for (int dimListInt : dimListInts) {
      dimListElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(dimListInt)));
    }
    Value dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op->getContext())),
        dimListElements);
    rewriter.replaceOpWithNewOp<AtenPermuteOp>(op, op.getType(), self, dimList);
    return success();
  }
};
} // namespace

template <typename OpTy>
static LogicalResult calculateVariance(OpTy op, PatternRewriter &rewriter,
                                       bool unbiased, int64_t correction) {
  Location loc = op.getLoc();
  Value self = op.getSelf();
  Value dimList = op.getDim();
  Value keepDim = op.getKeepdim();
  BaseTensorType inputTensorTy = self.getType().cast<BaseTensorType>();
  Type outputType = op.getType();
  BaseTensorType outputTensorType = outputType.cast<BaseTensorType>();
  if (!outputTensorType.hasDtype()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected result type to have a dtype");
  }
  Type newOutputType = outputTensorType.getWithSizesAndDtype(
      outputTensorType.getSizes(), rewriter.getF64Type());
  if (!inputTensorTy.hasDtype() ||
      !inputTensorTy.getDtype().isa<mlir::FloatType>()) {
    return rewriter.notifyMatchFailure(
        op, "support floating-point type input only");
  }

  // Upcasting the input tensor to `F64` dtype for higher precision during the
  // computation of the result.
  if (inputTensorTy.getDtype().getIntOrFloatBitWidth() != 64) {
    self = convertTensorToDtype(rewriter, loc, self, rewriter.getF64Type());
    inputTensorTy = self.getType().cast<BaseTensorType>();
  }

  std::optional<unsigned> maybeInputRank = getTensorRank(self);
  if (!maybeInputRank) {
    return rewriter.notifyMatchFailure(op, "expected input to have a rank");
  }
  unsigned inputRank = *maybeInputRank;
  SmallVector<Value> dimListElements;
  bool isNoneOrEmpty = true;
  if (!dimList.getType().template isa<Torch::NoneType>()) {
    if (!getListConstructElements(dimList, dimListElements))
      return rewriter.notifyMatchFailure(
          op, "expect dimList to be constructed from list construct");
    if (!dimListElements.empty() || inputRank == 0)
      isNoneOrEmpty = false;
  }
  if (isNoneOrEmpty) {
    for (unsigned i = 0; i < inputRank; i++)
      dimListElements.push_back(rewriter.create<Torch::ConstantIntOp>(
          loc, rewriter.getI64IntegerAttr(i)));
    dimList = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(Torch::IntType::get(op.getContext())),
        dimListElements);
  }
  Type meanDimResultType = inputTensorTy;
  for (unsigned i = 0; i < dimListElements.size(); i++)
    meanDimResultType = computeReductionType(
        rewriter, op, meanDimResultType.cast<BaseTensorType>(),
        dimListElements[i],
        /*keepDim=*/true);

  Value constantNone = rewriter.create<ConstantNoneOp>(loc);
  Value constantTrue = rewriter.create<ConstantBoolOp>(loc, true);
  Value meanAlongDims = rewriter.create<AtenMeanDimOp>(
      loc, meanDimResultType, self, dimList, /*keepDim=*/constantTrue,
      /*dtype=*/constantNone);
  Value subMean =
      createTensorSub(rewriter, loc, inputTensorTy, self, meanAlongDims);
  Value square = rewriter.create<AtenSquareOp>(loc, inputTensorTy, subMean);

  if (!unbiased) {
    Value result = rewriter.create<AtenMeanDimOp>(
        loc, newOutputType, square, dimList, keepDim, /*dtype=*/constantNone);
    result = convertTensorToDtype(rewriter, loc, result,
                                  outputTensorType.getDtype());
    rewriter.replaceOp(op, result);
    return success();
  }
  // Divide the square sum by productDimSize - correction.
  Value squareSum = rewriter.create<AtenSumDimIntListOp>(
      loc, newOutputType, square, dimList, keepDim, /*dtype=*/constantNone);

  // `productDimSize` is product of sizes of dimensions to be reduced.
  Value constantOne =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value productDimSize = constantOne;
  for (Value dim : dimListElements) {
    Value dimSize = rewriter.create<AtenSizeIntOp>(loc, self, dim);
    productDimSize =
        rewriter.create<AtenMulIntOp>(loc, productDimSize, dimSize);
  }
  Value cstCorrection = rewriter.create<Torch::ConstantIntOp>(
      loc, rewriter.getI64IntegerAttr(correction));
  // The `correction` value should be less than or equal to `productDimSize +
  // 1`.
  Value productDimSizePlusOne =
      rewriter.create<AtenAddIntOp>(loc, productDimSize, constantOne);
  Value cond =
      rewriter.create<AtenGeIntOp>(loc, productDimSizePlusOne, cstCorrection);
  rewriter.create<RuntimeAssertOp>(
      loc, cond,
      "correction value should be less than or equal to productDimSize + 1");
  Value productDimSizeSubCorrection =
      rewriter.create<AtenSubIntOp>(loc, productDimSize, cstCorrection);
  Value result = rewriter.create<AtenDivScalarOp>(loc, newOutputType, squareSum,
                                                  productDimSizeSubCorrection);
  result =
      convertTensorToDtype(rewriter, loc, result, outputTensorType.getDtype());
  rewriter.replaceOp(op, result);
  return success();
}

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// For Unbiased case:
// out = aten.sum(square, dims) / (productDimSize-1)
// For Biased case:
// out = aten.mean(square, dims)
namespace {
class DecomposeAtenVarDimOp : public OpRewritePattern<AtenVarDimOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarDimOp op,
                                PatternRewriter &rewriter) const override {
    bool unbiased;
    if (!matchPattern(op.getUnbiased(), m_TorchConstantBool(&unbiased))) {
      return rewriter.notifyMatchFailure(
          op, "Only support constant unbiased for aten.var");
    }
    int64_t correction = unbiased ? 1 : 0;
    if (failed(calculateVariance<AtenVarDimOp>(op, rewriter, unbiased,
                                               correction)))
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    return success();
  }
};
} // namespace

// Decompose aten.var(x, dims) into:
// sub = aten.sub(x, aten.mean(x, dims))
// square = aten.square(sub)
// out = aten.sum(square, dims) / (productDimSize - correction)
namespace {
class DecomposeAtenVarCorrectionOp
    : public OpRewritePattern<AtenVarCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    int64_t correction;
    if (!op.getCorrection().getType().isa<Torch::NoneType>()) {
      if (!matchPattern(op.getCorrection(), m_TorchConstantInt(&correction)))
        return rewriter.notifyMatchFailure(
            op, "Only support constant int correction for aten.var");
    } else {
      // The default value in case of `correction` being None is 1.
      correction = 1;
    }
    bool unbiased = correction == 0 ? false : true;
    if (failed(calculateVariance<AtenVarCorrectionOp>(op, rewriter, unbiased,
                                                      correction)))
      return rewriter.notifyMatchFailure(op, "invalid variance parameters");
    return success();
  }
};
} // namespace

namespace {
// Decompose the `aten.selectScatter` operation into `aten.sliceScatter` op.
class DecomposeAtenSelectScatterOp
    : public OpRewritePattern<AtenSelectScatterOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenSelectScatterOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value start = op.getIndex();
    Value dim = op.getDim();
    Value self = op.getSelf();
    Value src = op.getSrc();

    Value one =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value startPlusOne =
        rewriter.create<AtenAddIntOp>(loc, one.getType(), start, one);
    BaseTensorType srcTensorType = src.getType().cast<BaseTensorType>();
    SmallVector<int64_t> sizes;
    if (!srcTensorType.hasSizes())
      return rewriter.notifyMatchFailure(op, "src tensor must have size");

    ArrayRef<int64_t> srcShape = srcTensorType.getSizes();
    // `src` has a reduced rank. Hence add 1.
    int64_t srcRank = srcShape.size() + 1;
    int64_t dimInt = 0;
    if (matchPattern(dim, m_TorchConstantInt(&dimInt))) {
      dimInt = toPositiveDim(dimInt, srcRank);
      if (!isValidDim(dimInt, srcRank))
        return rewriter.notifyMatchFailure(op, "dim is not a valid dim");

      sizes.append(srcShape.begin(), srcShape.end());
      sizes.insert(sizes.begin() + dimInt, 1);

    } else {
      sizes.resize(srcShape.size() + 1, kUnknownSize);
    }
    Type srcType = srcTensorType.getWithSizesAndDtype(
        llvm::makeArrayRef(sizes), srcTensorType.getOptionalDtype());
    src = rewriter.create<AtenUnsqueezeOp>(loc, srcType, src, dim);
    rewriter.replaceOpWithNewOp<AtenSliceScatterOp>(
        op, op.getSelf().getType(), self, src, dim, start, startPlusOne,
        /*step=*/one);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAten_EmbeddingBagOp
    : public OpRewritePattern<Aten_EmbeddingBagOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten_EmbeddingBagOp op,
                                PatternRewriter &rewriter) const override {
    Value weight = op.getWeight();
    Value indices = op.getIndices();
    Value offsets = op.getOffsets();
    Value scaleGradByFreq = op.getScaleGradByFreq();
    Value mode = op.getMode();
    Value sparse = op.getSparse();
    Value perSampleWeights = op.getPerSampleWeights();
    Value includeLastOffset = op.getIncludeLastOffset();
    Value paddingIdx = op.getPaddingIdx();

    auto resultType0 = op->getResult(0).getType();
    auto resultType1 = op->getResult(1).getType();
    auto resultType2 = op->getResult(2).getType();
    auto resultType3 = op->getResult(3).getType();

    mlir::TypeRange returnTypes{resultType0, resultType1, resultType2,
                                resultType3};

    rewriter.replaceOpWithNewOp<AtenEmbeddingBagPaddingIdxOp>(
        op, returnTypes, weight, indices, offsets, scaleGradByFreq, mode,
        sparse, perSampleWeights, includeLastOffset, paddingIdx);

    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.liftFreshCopy` op into `aten.clone` op.
class DecomposeAtenLiftFreshCopyOp
    : public OpRewritePattern<AtenLiftFreshCopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenLiftFreshCopyOp op,
                                PatternRewriter &rewriter) const override {
    Value constantNone = rewriter.create<ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenCloneOp>(op, op.getType(), op.getSelf(),
                                             /*memoryFormat=*/constantNone);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.index.TensorHackedTwin` op into `aten.index.Tensor` op.
class DecomposeAtenIndexTensorHackedTwinOp
    : public OpRewritePattern<AtenIndexTensorHackedTwinOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenIndexTensorHackedTwinOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AtenIndexTensorOp>(op, op.getType(), op.getSelf(),
                                                   op.getIndices());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenMseLossOp : public OpRewritePattern<AtenMseLossOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenMseLossOp op,
                                PatternRewriter &rewriter) const override {

    // The `reduction` arg would have only three valid values.
    // 0 means no reduction.
    // 1 means mean reduction.
    // 2 means sum reduction.
    int64_t reductionType;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reductionType)))
      return rewriter.notifyMatchFailure(
          op, "Expected a constant integer value for reduction");

    Location loc = op.getLoc();
    BaseTensorType resultType = op.getType().cast<BaseTensorType>();
    BaseTensorType inputType = op.getSelf().getType().cast<BaseTensorType>();
    if (!inputType.hasSizes())
      return rewriter.notifyMatchFailure(
          op, "Expected the input tensor to have sizes");
    BaseTensorType subType =
        inputType
            .getWithSizesAndDtype(llvm::makeArrayRef(inputType.getSizes()),
                                  resultType.getOptionalDtype())
            .cast<BaseTensorType>();

    Value sub = createTensorSub(rewriter, loc, subType, op.getSelf(), op.getTarget());
    Value result = rewriter.create<AtenSquareOp>(loc, subType, sub);
    if (reductionType == torch_upstream::Reduction::None) {
      rewriter.replaceOp(op, result);
      return success();
    }
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    if (reductionType == torch_upstream::Reduction::Mean)
      result = rewriter.create<AtenMeanDimOp>(loc, resultType, result,
                                              /*dim=*/cstNone,
                                              /*keepdim=*/cstFalse,
                                              /*dtype=*/cstNone);
    else
      result = rewriter.create<AtenSumDimIntListOp>(
          loc, resultType, result, /*dim=*/cstNone, /*keepdim=*/cstFalse,
          /*dtype=*/cstNone);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenRandintLowOp : public OpRewritePattern<AtenRandintLowOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandintLowOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Type resultType = op.getType();
    BaseTensorType resultTensorType = resultType.cast<BaseTensorType>();
    if (!resultTensorType.hasDtype()) {
      return rewriter.notifyMatchFailure(
          op, "expected result type to have a dtype");
    }

    int64_t cstLow, cstHigh;
    if (!matchPattern(op.getLow(), m_TorchConstantInt(&cstLow)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: low must be a constant integer");
    if (!matchPattern(op.getHigh(), m_TorchConstantInt(&cstHigh)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: high must be a constant integer");

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value cstFalse = rewriter.create<ConstantBoolOp>(loc, false);
    Value low = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)cstLow));
    Value high = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)cstHigh));

    BaseTensorType floatResultType =
        resultTensorType
            .getWithSizesAndDtype(resultTensorType.getSizes(),
                                  rewriter.getF32Type())
            .cast<BaseTensorType>();
    Value emptyTensor = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, floatResultType, op.getSize(), /*dtype=*/none, /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pinMemory=*/op.getPinMemory(),
        /*memoryFormat=*/none);

    Value result =
        rewriter.create<AtenUniformOp>(loc, floatResultType, emptyTensor,
                                       /*from=*/low,
                                       /*to=*/high,
                                       /*generator=*/none);
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, resultType, result,
        getDtypeIntValueForType(rewriter, loc, resultTensorType.getDtype()),
        /*nonBlocking=*/cstFalse, /*copy=*/cstFalse, /*memoryFormat=*/none);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.varMean.correction` op into `aten.var.correction` and
// `aten.mean.dim` op.
class DecomposeAtenVarMeanCorrectionOp
    : public OpRewritePattern<AtenVarMeanCorrectionOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarMeanCorrectionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value var = rewriter.create<AtenVarCorrectionOp>(
        loc, op.getType(0), op.getSelf(), op.getDim(), op.getCorrection(), op.getKeepdim());
    Value mean =
        rewriter.create<AtenMeanDimOp>(loc, op.getType(0), op.getSelf(), op.getDim(),
                                       op.getKeepdim(), /*dtype=*/noneVal);
    rewriter.replaceOp(op, {var, mean});
    return success();
  }
};
} // namespace

namespace {
// Decompose `prims.convertElementType` op into `aten.to.dtype` op.
class DecomposePrimsConvertElementTypeOp
    : public OpRewritePattern<PrimsConvertElementTypeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimsConvertElementTypeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value cstFalse = rewriter.create<Torch::ConstantBoolOp>(loc, false);
    Value cstNone = rewriter.create<Torch::ConstantNoneOp>(loc);
    rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
        op, op.getType(), op.getA(), op.getDtype(), /*nonBlocking=*/cstFalse,
        /*copy=*/cstFalse, /*memoryFormat=*/cstNone);
    return success();
  }
};
} // namespace

namespace {
// The op is decomposed using the Box-Muller transform.
// Refer: https://en.wikipedia.org/wiki/Box-Muller_transform
class DecomposeAtenRandnGeneratorOp
    : public OpRewritePattern<AtenRandnGeneratorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnGeneratorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resultType = op.getType();

    Value none = rewriter.create<ConstantNoneOp>(loc);
    Value low = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)0.0));
    Value high = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)1.0));
    Value cstMinusTwo = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)-2.0));
    Value cstTwoPie = rewriter.create<Torch::ConstantFloatOp>(
        loc, rewriter.getF64FloatAttr((double)(2.0 * 3.14159)));

    Value emptyTensorA = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, resultType, op.getSize(), /*dtype=*/none, /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pin_memory=*/op.getPinMemory(),
        /*memory_format=*/none);
    Value emptyTensorB = rewriter.create<AtenEmptyMemoryFormatOp>(
        loc, resultType, op.getSize(), /*dtype=*/none, /*layout=*/op.getLayout(),
        /*device=*/op.getDevice(), /*pin_memory=*/op.getPinMemory(),
        /*memory_format=*/none);

    Value uOne = rewriter.create<AtenUniformOp>(loc, resultType, emptyTensorA,
                                                /*from=*/low,
                                                /*to=*/high,
                                                /*generator=*/op.getGenerator());
    Value uTwo = rewriter.create<AtenUniformOp>(loc, resultType, emptyTensorB,
                                                /*from=*/low,
                                                /*to=*/high,
                                                /*generator=*/op.getGenerator());

    Value logUOne = rewriter.create<AtenLogOp>(loc, resultType, uOne);
    Value minusTwoLogUOne =
        rewriter.create<AtenMulScalarOp>(loc, resultType, logUOne, cstMinusTwo);
    Value r = rewriter.create<AtenSqrtOp>(loc, resultType, minusTwoLogUOne);
    Value theta =
        rewriter.create<AtenMulScalarOp>(loc, resultType, uTwo, cstTwoPie);
    Value cosTheta = rewriter.create<AtenCosOp>(loc, resultType, theta);
    rewriter.replaceOpWithNewOp<AtenMulTensorOp>(op, op.getType(), r, cosTheta);
    return success();
  }
};
} // namespace

namespace {
// Decompose `aten.randn` op into `aten.randn.generator` op.
class DecomposeAtenRandnOp : public OpRewritePattern<AtenRandnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenRandnOp op,
                                PatternRewriter &rewriter) const override {
    Value none = rewriter.create<Torch::ConstantNoneOp>(op.getLoc());
    rewriter.replaceOpWithNewOp<AtenRandnGeneratorOp>(
        op, op.getType(), op.getSize(), /*generator=*/none, op.getDtype(),
        op.getLayout(), op.getDevice(), op.getPinMemory());
    return success();
  }
};
} // namespace

namespace {
class DecomposeAtenVarMeanOp : public OpRewritePattern<AtenVarMeanOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenVarMeanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value falseVal = rewriter.create<ConstantBoolOp>(loc, false);
    Value noneVal = rewriter.create<ConstantNoneOp>(loc);
    Value var = rewriter.create<AtenVarDimOp>(loc, op.getType(0), op.getSelf(),
                                              /*dim=*/noneVal, op.getUnbiased(),
                                              /*keepdim=*/falseVal);
    Value mean = rewriter.create<AtenMeanOp>(loc, op.getType(0), op.getSelf(),
                                             /*dtype=*/noneVal);
    rewriter.replaceOp(op, {var, mean});
    return success();
  }
};
} // namespace

namespace {
class DecomposeComplexOpsPass
    : public DecomposeComplexOpsBase<DecomposeComplexOpsPass> {
private:
  llvm::StringSet<> legalOpsSet;

  template <typename DecomposePattern>
  void addPatternIfTargetOpIsIllegal(RewritePatternSet &patterns) {
    MLIRContext *context = &getContext();
    std::optional<OperationName> opName =
        DecomposePattern(context).getRootKind();
    // Because the `DecomposeComplexOpsPass` uses a greedy algorithm
    // to apply patterns, only patterns that we for sure know we want to run
    // must be added. This restricts the set of patterns allowed in this file to
    // patterns that apply to a single op. In other words, patterns that match
    // on `Operation *` are not allowed, since there is no way of telling if
    // that pattern will match on an op in the `legalOpsSet` or not.
    assert(opName && "All decomposition patterns must target a single op");
    if (!legalOpsSet.contains(opName->getStringRef()))
      patterns.add<DecomposePattern>(context);
  }

public:
  DecomposeComplexOpsPass() = default;
  DecomposeComplexOpsPass(ArrayRef<std::string> legalOps) {
    this->legalOps = legalOps;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // The strings in the `legalOps` ArrayRef don't exist during the call to the
    // constructor `DecomposeComplexOpsPass`, so the creation of the
    // `legalOpsSet` must be delayed to when `runOnOperation` gets called.
    legalOpsSet.clear();
    legalOpsSet.insert(legalOps.begin(), legalOps.end());

    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LogSoftmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLogSoftmaxIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenEmptyLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenOnesLikeOp, 1>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorAllocLikeOp<AtenZerosLikeOp, 0>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRollOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRepeatOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFlattenUsingIntsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarOtherOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenWhereScalarSelfOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenConvolutionBackwardOverrideableOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSizeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenReshapeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_SoftmaxBackwardDataOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTanhBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAddmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMeanDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectIntOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMatmulOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMvOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenTOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_LogSoftmaxBackwardDataOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAddCLikeOp<AtenAddcmulOp, AtenMulTensorOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAtenAddCLikeOp<AtenAddcdivOp, AtenDivTensorOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLayerNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNativeLayerNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNativeBatchNormOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvolutionOverrideableOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAten_ConvolutionLikeOp<Aten_ConvolutionOp>>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeAten_ConvolutionLikeOp<Aten_ConvolutionDeprecatedOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvolutionBackwardOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConv2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenConvTranspose2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArangeStartOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenArgMaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSquareOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_UnsafeViewOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_ReshapeAliasOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBernoulliOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeValsemVariantAtenBernoulliFloatOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBernoulliTensorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenZeroOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardsigmoidOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRelu6Op>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardswishOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSoftplusOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSiluOp>(patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorNewLikeOp<AtenNewZerosOp, AtenZerosOp>>(
        patterns);
    addPatternIfTargetOpIsIllegal<
        DecomposeConstantTensorNewLikeOp<AtenNewOnesOp, AtenOnesOp>>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenHardtanhOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFullOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLinearOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMishOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFullLikeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexPutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenExpandAsOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_ToCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenDropoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNewEmptyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexPutHackedTwinOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenPadOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDtypeLayoutOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenToDeviceOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAdaptiveAvgPool2dOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMinOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenClampMaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenBaddbmmOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenFloorDivideOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNumpyTOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenSelectScatterOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenAmaxOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdDimOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenStdCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenNarrowOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAten_EmbeddingBagOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLiftFreshCopyOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenIndexTensorHackedTwinOp>(
        patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenMseLossOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandintLowOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanCorrectionOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposePrimsConvertElementTypeOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandnOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenRandnGeneratorOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenVarMeanOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluOp>(patterns);
    addPatternIfTargetOpIsIllegal<DecomposeAtenLeakyReluBackwardOp>(patterns);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createDecomposeComplexOpsPass(
    ArrayRef<std::string> legalOps) {
  return std::make_unique<DecomposeComplexOpsPass>(legalOps);
}