//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "TransformerEncoderUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/Twine.h"

#include <cmath>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

static Value createIntConstant(PatternRewriter &rewriter, Location loc,
                               int64_t value) {
  return Torch::ConstantIntOp::create(rewriter, loc,
                                      rewriter.getI64IntegerAttr(value));
}

static Value createBoolConstant(PatternRewriter &rewriter, Location loc,
                                bool value) {
  return Torch::ConstantBoolOp::create(rewriter, loc,
                                       rewriter.getBoolAttr(value));
}

static Value createFloatConstant(PatternRewriter &rewriter, Location loc,
                                 double value, Type /*dtype*/) {
  auto attr = rewriter.getF64FloatAttr(value);
  return Torch::ConstantFloatOp::create(rewriter, loc, attr);
}

static Value createIntList(PatternRewriter &rewriter, Location loc,
                           ArrayRef<int64_t> values) {
  SmallVector<Value> elems;
  elems.reserve(values.size());
  for (int64_t v : values)
    elems.push_back(createIntConstant(rewriter, loc, v));
  auto listType =
      Torch::ListType::get(Torch::IntType::get(rewriter.getContext()));
  return Torch::PrimListConstructOp::create(rewriter, loc, listType, elems);
}

static FailureOr<ValueTensorType> expectRankedTensor(Value v, int64_t rank,
                                                     StringRef name,
                                                     PatternRewriter &rewriter,
                                                     Operation *op) {
  auto tensorType = dyn_cast<ValueTensorType>(v.getType());
  if (!tensorType || !tensorType.hasSizes() ||
      tensorType.getSizes().size() != static_cast<size_t>(rank))
    return rewriter.notifyMatchFailure(
        op, Twine("expected ") + name + " tensor of rank " + Twine(rank));
  return tensorType;
}

static int64_t adaptSizeForView(int64_t size) {
  return size == Torch::kUnknownSize ? -1 : size;
}

static FailureOr<ValueTensorType>
checkTensorShape(Value value, ArrayRef<int64_t> expected, StringRef name,
                 PatternRewriter &rewriter, Operation *op) {
  auto type = dyn_cast<ValueTensorType>(value.getType());
  if (!type || !type.hasSizes())
    return rewriter.notifyMatchFailure(
        op, Twine("expected tensor operands with known sizes for ") + name);
  ArrayRef<int64_t> actual = type.getSizes();
  if (actual.size() != expected.size())
    return rewriter.notifyMatchFailure(
        op, Twine("rank mismatch for ") + name + ": expected " +
                Twine(expected.size()) + " but got " + Twine(actual.size()));
  for (size_t i = 0; i < expected.size(); ++i) {
    int64_t exp = expected[i];
    if (exp == Torch::kUnknownSize)
      continue;
    if (actual[i] != exp && actual[i] != Torch::kUnknownSize)
      return rewriter.notifyMatchFailure(
          op, Twine("dimension mismatch for ") + name + " at index " +
                  Twine(i) + ": expected " + Twine(exp) + " but got " +
                  Twine(actual[i]));
  }
  return type;
}

struct QkvProjections {
  Value query;
  Value key;
  Value value;
};

static FailureOr<QkvProjections>
buildQkvProjections(PatternRewriter &rewriter, Location loc, Value input,
                    Value qkvWeight, Value qkvBias, int64_t embedDim,
                    int64_t numHeads, ValueTensorType inputType) {
  auto dtype = inputType.getOptionalDtype();
  ArrayRef<int64_t> sizes = inputType.getSizes();
  int64_t batch = sizes[0];
  int64_t seqLen = sizes[1];
  int64_t headDim = embedDim / numHeads;

  SmallVector<int64_t, 4> linearSizes = {batch, seqLen, 3 * embedDim};
  ValueTensorType linearType =
      cast<ValueTensorType>(inputType.getWithSizesAndDtype(linearSizes, dtype));
  Value qkvLinear = AtenLinearOp::create(rewriter, loc, linearType, input,
                                         qkvWeight, qkvBias);

  SmallVector<int64_t, 5> reshapeSizes = {batch, seqLen, 3, numHeads, headDim};
  ValueTensorType reshapeType = cast<ValueTensorType>(
      inputType.getWithSizesAndDtype(reshapeSizes, dtype));
  SmallVector<int64_t, 5> reshapeShape = {
      adaptSizeForView(batch), adaptSizeForView(seqLen), 3, numHeads, headDim};
  Value reshapeList = createIntList(rewriter, loc, reshapeShape);
  Value reshaped =
      AtenViewOp::create(rewriter, loc, reshapeType, qkvLinear, reshapeList);

  Value dimTwo = createIntConstant(rewriter, loc, 2);
  SmallVector<int64_t, 4> selectSizes = {batch, seqLen, numHeads, headDim};
  ValueTensorType selectType =
      cast<ValueTensorType>(inputType.getWithSizesAndDtype(selectSizes, dtype));

  Value qIndex = createIntConstant(rewriter, loc, 0);
  Value kIndex = createIntConstant(rewriter, loc, 1);
  Value vIndex = createIntConstant(rewriter, loc, 2);
  Value q = AtenSelectIntOp::create(rewriter, loc, selectType, reshaped, dimTwo,
                                    qIndex);
  Value k = AtenSelectIntOp::create(rewriter, loc, selectType, reshaped, dimTwo,
                                    kIndex);
  Value v = AtenSelectIntOp::create(rewriter, loc, selectType, reshaped, dimTwo,
                                    vIndex);

  SmallVector<int64_t, 4> permSizes = {batch, numHeads, seqLen, headDim};
  ValueTensorType permType =
      cast<ValueTensorType>(inputType.getWithSizesAndDtype(permSizes, dtype));
  Value permList = createIntList(rewriter, loc, {0, 2, 1, 3});
  q = AtenPermuteOp::create(rewriter, loc, permType, q, permList);
  k = AtenPermuteOp::create(rewriter, loc, permType, k, permList);
  v = AtenPermuteOp::create(rewriter, loc, permType, v, permList);

  return QkvProjections{q, k, v};
}

class DecomposeTransformerEncoderLayerFwdOperatorOp
    : public OpRewritePattern<Torch::OperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Torch::OperatorOp op,
                                PatternRewriter &rewriter) const override {
    if (!isTransformerEncoderOperator(op))
      return failure();

    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());
    if (operands.size() != 20)
      return rewriter.notifyMatchFailure(op, "expected 20 operands");

    Value src = operands[0];
    Value embedDimVal = operands[1];
    Value numHeadsVal = operands[2];
    Value qkvWeight = operands[3];
    Value qkvBias = operands[4];
    Value projWeight = operands[5];
    Value projBias = operands[6];
    Value useGelu = operands[7];
    Value normFirst = operands[8];
    Value eps = operands[9];
    Value norm1Weight = operands[10];
    Value norm1Bias = operands[11];
    Value norm2Weight = operands[12];
    Value norm2Bias = operands[13];
    Value ffn1Weight = operands[14];
    Value ffn1Bias = operands[15];
    Value ffn2Weight = operands[16];
    Value ffn2Bias = operands[17];
    Value mask = operands[18];
    Value maskType = operands[19];

    int64_t embedDim;
    int64_t numHeads;
    if (!matchPattern(embedDimVal, m_TorchConstantInt(&embedDim)) ||
        !matchPattern(numHeadsVal, m_TorchConstantInt(&numHeads))) {
      return rewriter.notifyMatchFailure(
          op, "embed_dim and num_heads must be constant integers");
    }
    if (numHeads == 0 || embedDim % numHeads != 0) {
      return rewriter.notifyMatchFailure(
          op, "embedding dimension must be divisible by number of heads");
    }

    if (!isa<Torch::NoneType>(mask.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "attention masks are not supported");
    }
    if (!isa<Torch::NoneType>(maskType.getType())) {
      int64_t maskTypeValue;
      if (!matchPattern(maskType, m_TorchConstantInt(&maskTypeValue)) ||
          maskTypeValue != 0) {
        return rewriter.notifyMatchFailure(
            op, "mask_type must be None or the constant 0");
      }
    }

    FailureOr<ValueTensorType> srcType =
        expectRankedTensor(src, 3, "src", rewriter, op);
    if (failed(srcType)) {
      return failure();
    }
    ArrayRef<int64_t> srcSizes = srcType->getSizes();
    if (srcSizes[0] == Torch::kUnknownSize ||
        srcSizes[1] == Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "src must have static batch and sequence dimensions");
    }
    if (srcSizes[2] != Torch::kUnknownSize && srcSizes[2] != embedDim) {
      return rewriter.notifyMatchFailure(
          op, "embedding dimension must match the last dimension of src");
    }

    bool useGeluBool;
    if (!matchPattern(useGelu, m_TorchConstantBool(&useGeluBool))) {
      return rewriter.notifyMatchFailure(op, "use_gelu must be constant");
    }

    bool normFirstBool;
    if (!matchPattern(normFirst, m_TorchConstantBool(&normFirstBool))) {
      return rewriter.notifyMatchFailure(op, "norm_first must be constant");
    }

    if (failed(checkTensorShape(qkvWeight, {3 * embedDim, embedDim},
                                "qkv_weight", rewriter, op)) ||
        failed(checkTensorShape(qkvBias, {3 * embedDim}, "qkv_bias", rewriter,
                                op)) ||
        failed(checkTensorShape(projWeight, {embedDim, embedDim}, "proj_weight",
                                rewriter, op)) ||
        failed(checkTensorShape(projBias, {embedDim}, "proj_bias", rewriter,
                                op)) ||
        failed(checkTensorShape(norm1Weight, {embedDim}, "norm1_weight",
                                rewriter, op)) ||
        failed(checkTensorShape(norm1Bias, {embedDim}, "norm1_bias", rewriter,
                                op)) ||
        failed(checkTensorShape(norm2Weight, {embedDim}, "norm2_weight",
                                rewriter, op)) ||
        failed(checkTensorShape(norm2Bias, {embedDim}, "norm2_bias", rewriter,
                                op))) {
      return failure();
    }

    auto ffn1WeightType =
        checkTensorShape(ffn1Weight, {Torch::kUnknownSize, embedDim},
                         "ffn1_weight", rewriter, op);
    if (failed(ffn1WeightType)) {
      return failure();
    }
    auto ffn1BiasType = checkTensorShape(ffn1Bias, {Torch::kUnknownSize},
                                         "ffn1_bias", rewriter, op);
    if (failed(ffn1BiasType)) {
      return failure();
    }
    auto ffn2WeightType =
        checkTensorShape(ffn2Weight, {embedDim, Torch::kUnknownSize},
                         "ffn2_weight", rewriter, op);
    if (failed(ffn2WeightType)) {
      return failure();
    }
    if (failed(checkTensorShape(ffn2Bias, {embedDim}, "ffn2_bias", rewriter,
                                op))) {
      return failure();
    }

    int64_t hiddenDim = (*ffn1WeightType).getSizes()[0];
    auto enforceHidden = [&](int64_t candidate,
                             StringRef what) -> LogicalResult {
      if (candidate == Torch::kUnknownSize)
        return success();
      if (hiddenDim == Torch::kUnknownSize) {
        hiddenDim = candidate;
        return success();
      }
      if (hiddenDim != candidate) {
        return rewriter.notifyMatchFailure(
            op, Twine("inconsistent hidden dimension inferred from ") + what);
      }
      return success();
    };
    if (failed(enforceHidden((*ffn1BiasType).getSizes()[0], "ffn1_bias")) ||
        failed(enforceHidden((*ffn2WeightType).getSizes()[1], "ffn2_weight"))) {
      return failure();
    }
    if (hiddenDim == Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "unable to infer feed-forward hidden dimension");
    }

    Location loc = op.getLoc();

    auto buildLayerNorm = [&](Value input, Value weight,
                              Value bias) -> FailureOr<Value> {
      auto inputTensorType = cast<ValueTensorType>(input.getType());
      Value normalizedShape = createIntList(rewriter, loc, {embedDim});
      // Upstream aten._transformer_encoder_layer_fwd always sets
      // cudnn_enable=true for the layer-norm calls
      // (see aten/src/ATen/native/transformers/transformer.cpp), so mirror that
      // behavior here.
      Value cudnnEnable = createBoolConstant(rewriter, loc, true);
      Value ln = AtenLayerNormOp::create(rewriter, loc, inputTensorType, input,
                                         normalizedShape, weight, bias, eps,
                                         cudnnEnable);
      return ln;
    };

    Value attentionInput = src;
    if (normFirstBool) {
      auto normed = buildLayerNorm(src, norm1Weight, norm1Bias);
      if (failed(normed))
        return failure();
      attentionInput = *normed;
    }

    auto attentionInputType =
        expectRankedTensor(attentionInput, 3, "attention input", rewriter, op);
    if (failed(attentionInputType))
      return failure();

    FailureOr<QkvProjections> projections =
        buildQkvProjections(rewriter, loc, attentionInput, qkvWeight, qkvBias,
                            embedDim, numHeads, *attentionInputType);
    if (failed(projections))
      return failure();

    Type elementType = srcType->getOptionalDtype();
    if (!elementType)
      elementType = rewriter.getF32Type();
    int64_t headDim = embedDim / numHeads;

    Value permIdx = createIntList(rewriter, loc, {0, 1, 3, 2});
    SmallVector<int64_t, 4> keyTShape = {(*srcType).getSizes()[0], numHeads,
                                         headDim, (*srcType).getSizes()[1]};
    ValueTensorType keyTType = cast<ValueTensorType>(
        srcType->getWithSizesAndDtype(keyTShape, srcType->getOptionalDtype()));
    Value keyT = AtenPermuteOp::create(rewriter, loc, keyTType,
                                       projections->key, permIdx);

    SmallVector<int64_t, 4> scoreShape = {(*srcType).getSizes()[0], numHeads,
                                          (*srcType).getSizes()[1],
                                          (*srcType).getSizes()[1]};
    ValueTensorType scoreType = cast<ValueTensorType>(
        srcType->getWithSizesAndDtype(scoreShape, srcType->getOptionalDtype()));
    Value scores = AtenMatmulOp::create(rewriter, loc, scoreType,
                                        projections->query, keyT);

    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    Value scaleConst = createFloatConstant(rewriter, loc, scale, elementType);
    scores =
        AtenMulScalarOp::create(rewriter, loc, scoreType, scores, scaleConst);

    Value dimLast = createIntConstant(rewriter, loc, -1);
    Value halfToFloat = createBoolConstant(rewriter, loc, false);
    Value attnWeights = Aten_SoftmaxOp::create(rewriter, loc, scoreType, scores,
                                               dimLast, halfToFloat);

    SmallVector<int64_t, 4> contextShape = {(*srcType).getSizes()[0], numHeads,
                                            (*srcType).getSizes()[1], headDim};
    ValueTensorType contextType =
        cast<ValueTensorType>(srcType->getWithSizesAndDtype(
            contextShape, srcType->getOptionalDtype()));
    Value context = AtenMatmulOp::create(rewriter, loc, contextType,
                                         attnWeights, projections->value);

    Value mergeIdx = createIntList(rewriter, loc, {0, 2, 1, 3});
    SmallVector<int64_t, 4> mergedPermShape = {
        (*srcType).getSizes()[0], (*srcType).getSizes()[1], numHeads, headDim};
    ValueTensorType mergedPermType =
        cast<ValueTensorType>(srcType->getWithSizesAndDtype(
            mergedPermShape, srcType->getOptionalDtype()));
    Value merged =
        AtenPermuteOp::create(rewriter, loc, mergedPermType, context, mergeIdx);

    SmallVector<int64_t, 3> mergedViewShape = {
        (*srcType).getSizes()[0], (*srcType).getSizes()[1], embedDim};
    Value mergedView = AtenViewOp::create(
        rewriter, loc, src.getType(), merged,
        createIntList(rewriter, loc,
                      {adaptSizeForView((*srcType).getSizes()[0]),
                       adaptSizeForView((*srcType).getSizes()[1]), embedDim}));

    Value attnOutput = AtenLinearOp::create(rewriter, loc, *srcType, mergedView,
                                            projWeight, projBias);

    Value oneScalar = createIntConstant(rewriter, loc, 1);
    Value attnResidual = AtenAddTensorOp::create(rewriter, loc, *srcType, src,
                                                 attnOutput, oneScalar);

    Value postAttn;
    if (normFirstBool) {
      postAttn = attnResidual;
    } else {
      auto normed = buildLayerNorm(attnResidual, norm1Weight, norm1Bias);
      if (failed(normed))
        return failure();
      postAttn = *normed;
    }

    Value feedForwardInput = postAttn;
    if (normFirstBool) {
      auto normed = buildLayerNorm(attnResidual, norm2Weight, norm2Bias);
      if (failed(normed))
        return failure();
      feedForwardInput = *normed;
    }

    auto buildFeedForward = [&](Value input) -> FailureOr<Value> {
      SmallVector<int64_t, 3> hiddenShape = {
          (*srcType).getSizes()[0], (*srcType).getSizes()[1], hiddenDim};
      ValueTensorType hiddenType =
          cast<ValueTensorType>(srcType->getWithSizesAndDtype(
              hiddenShape, srcType->getOptionalDtype()));
      Value ff1 = AtenLinearOp::create(rewriter, loc, hiddenType, input,
                                       ffn1Weight, ffn1Bias);
      Value activated;
      if (useGeluBool) {
        Value approx = Torch::ConstantStrOp::create(
            rewriter, loc, Torch::StringType::get(rewriter.getContext()),
            rewriter.getStringAttr("none"));
        activated = AtenGeluOp::create(rewriter, loc, hiddenType, ff1, approx);
      } else {
        activated = AtenReluOp::create(rewriter, loc, hiddenType, ff1);
      }
      Value ff2 = AtenLinearOp::create(rewriter, loc, *srcType, activated,
                                       ffn2Weight, ffn2Bias);
      return ff2;
    };

    FailureOr<Value> feedForwardOut = buildFeedForward(feedForwardInput);
    if (failed(feedForwardOut))
      return failure();

    Value result;
    if (normFirstBool) {
      result = AtenAddTensorOp::create(rewriter, loc, *srcType, attnResidual,
                                       *feedForwardOut, oneScalar);
    } else {
      Value secondResidual = AtenAddTensorOp::create(
          rewriter, loc, *srcType, postAttn, *feedForwardOut, oneScalar);
      auto normed = buildLayerNorm(secondResidual, norm2Weight, norm2Bias);
      if (failed(normed))
        return failure();
      result = *normed;
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace mlir::torch::Torch {

void populateTransformerEncoderPatterns(RewritePatternSet &patterns,
                                        const llvm::StringSet<> &legalOpsSet) {
  MLIRContext *context = patterns.getContext();
  // Torch::OperatorOp is the root op, so the semantic name lives on the
  // operation attribute. Respect the same legality contract as other patterns
  // by checking the operator name variants directly.
  bool shouldAddPattern = false;
  for (StringRef semanticName :
       {StringRef("aten._transformer_encoder_layer_fwd"),
        StringRef("aten._transformer_encoder_layer_fwd.default")}) {
    if (!legalOpsSet.contains(semanticName)) {
      shouldAddPattern = true;
      break;
    }
  }
  if (!shouldAddPattern)
    return;
  patterns.add<DecomposeTransformerEncoderLayerFwdOperatorOp>(context);
}

} // namespace mlir::torch::Torch
