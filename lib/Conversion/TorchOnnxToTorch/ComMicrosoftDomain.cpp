//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

void mlir::torch::onnx_c::populateComMicrosoftDomain(
    OnnxCustomOpConversionPattern &patterns) {
  patterns.onOp(
      "RotaryEmbedding", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        int64_t interleaved, isPackedBatching, numHeads, rotaryEmbeddingDim;
        float scale;
        Value input, positionIds, cosCache, sinCache;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(positionIds, 1) ||
            binder.tensorOperandAtIndex(cosCache, 2) ||
            binder.tensorOperandAtIndex(sinCache, 3) ||
            binder.s64IntegerAttr(interleaved, "interleaved", 0) ||
            binder.s64IntegerAttr(isPackedBatching, "is_packed_batching", 0) ||
            binder.s64IntegerAttr(numHeads, "num_heads", 0) ||
            binder.s64IntegerAttr(rotaryEmbeddingDim, "rotary_embedding_dim",
                                  0) ||
            binder.f32FloatAttr(scale, "scale", 1.0)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to get required inputs");
        }

        Torch::ValueTensorType resultType;
        if (binder.tensorResultType(resultType)) {
          return rewriter.notifyMatchFailure(binder.op,
                                             "result type bind failure");
        }

        Value cstInterleaved = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(interleaved));
        Value cstIsPackedBatching = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(isPackedBatching));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstRotaryEmbeddingDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(rotaryEmbeddingDim));
        Value cstScale = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(scale));

        rewriter.replaceOpWithNewOp<Torch::OnnxVariantRotaryEmbeddingOp>(
            binder.op, resultType, input, positionIds, cosCache, sinCache,
            cstInterleaved, cstIsPackedBatching, cstNumHeads,
            cstRotaryEmbeddingDim, cstScale);
        return success();
      });
  patterns.onOp(
      "SimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        Value input, scale;
        float epsilon;
        int64_t axis;
        SmallVector<Type> resultTypes;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(scale, 1) ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5f) ||
            binder.s64IntegerAttr(axis, "axis", -1) ||
            binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to bind inputs/attrs");

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have dtype");
        if (!inputType.hasSizes())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have sizes");

        // Get tensor rank to normalize axis
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked input tensor");
        unsigned inputRank = *maybeRank;
        if (inputRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "scalar input not supported");

        // Normalize negative axis
        axis = Torch::toPositiveDim(axis, inputRank);

        // Build normalized_shape: [inputShape[axis], ..., inputShape[-1]]
        ArrayRef<int64_t> inputShape = inputType.getSizes();
        SmallVector<Value> normalizedShapeValues;
        for (int64_t n = axis; n < static_cast<int64_t>(inputRank); n++) {
          normalizedShapeValues.push_back(Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(inputShape[n])));
        }
        Value normalizedShape = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            normalizedShapeValues);

        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));

        // Emit aten.rms_norm
        Value output = Torch::AtenRmsNormOp::create(
            rewriter, loc, resultTypes[0], input, normalizedShape, scale,
            cstEpsilon);

        // Return outputs (may have optional inv_std_var output)
        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else if (resultTypes.size() == 2) {
          // Second output is inv_std_var = rsqrt(mean(x^2) + eps)
          // Need to compute this manually since aten.rms_norm only returns
          // the normalized output.
          Value cstOne = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(1.0));
          Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
          Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

          Value inputSquared = Torch::AtenMulTensorOp::create(
              rewriter, loc, inputType, input, input);

          // Build dim list for mean: [axis, axis+1, ..., rank-1]
          SmallVector<Value> dimValues;
          for (int64_t n = axis; n < static_cast<int64_t>(inputRank); n++) {
            dimValues.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(n)));
          }
          Value dimList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              dimValues);

          // inv_std_var type: same as resultTypes[1]
          Torch::ValueTensorType invStdVarType =
              cast<Torch::ValueTensorType>(resultTypes[1]);

          Value meanSquared = Torch::AtenMeanDimOp::create(
              rewriter, loc, invStdVarType, inputSquared, dimList, cstTrue,
              cstNone);
          Value meanPlusEpsilon = Torch::AtenAddScalarOp::create(
              rewriter, loc, invStdVarType, meanSquared, cstEpsilon, cstOne);
          Value invStdVar = Torch::AtenRsqrtOp::create(rewriter, loc,
                                                        invStdVarType,
                                                        meanPlusEpsilon);
          rewriter.replaceOp(binder.op, {output, invStdVar});
        } else {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 1-2 result types");
        }

        return success();
      });
  patterns.onOp(
      "SkipSimplifiedLayerNormalization", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();

        Value input, skip, gamma;
        float epsilon;
        SmallVector<Type> resultTypes;
        if (binder.tensorOperandAtIndex(input, 0) ||
            binder.tensorOperandAtIndex(skip, 1) ||
            binder.tensorOperandAtIndex(gamma, 2) ||
            binder.f32FloatAttr(epsilon, "epsilon", 1e-5f) ||
            binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "Failed to bind inputs/attrs");

        // Optional bias (index 3)
        Value bias;
        bool hasBias = !binder.tensorOperandAtIndex(bias, 3);

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have dtype");
        if (!inputType.hasSizes())
          return rewriter.notifyMatchFailure(binder.op,
                                             "input should have sizes");

        // Get tensor rank to compute last dimension
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked input tensor");
        unsigned inputRank = *maybeRank;
        if (inputRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "scalar input not supported");

        Value cstOne = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value cstEpsilon = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(epsilon));

        // Step 1: Compute s = input + skip + bias (if present)
        Value s = Torch::AtenAddTensorOp::create(rewriter, loc, inputType,
                                                 input, skip, cstOne);

        if (hasBias) {
          s = Torch::AtenAddTensorOp::create(rewriter, loc, inputType, s, bias,
                                             cstOne);
        }

        // Build normalized_shape for last dimension: [inputShape[-1]]
        ArrayRef<int64_t> inputShape = inputType.getSizes();
        Value cstLastDimSize = Torch::ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(inputShape[inputRank - 1]));
        Value normalizedShape = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            SmallVector<Value>{cstLastDimSize});

        // Emit aten.rms_norm
        Value output = Torch::AtenRmsNormOp::create(
            rewriter, loc, resultTypes[0], s, normalizedShape, gamma,
            cstEpsilon);

        // Compute mean and inv_std_var when needed for optional outputs
        Value meanSquared, invStdVar;
        if (resultTypes.size() >= 3) {
          Value cstTrue = Torch::ConstantBoolOp::create(rewriter, loc, true);
          Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

          Value sSquared =
              Torch::AtenMulTensorOp::create(rewriter, loc, inputType, s, s);

          Value cstLastDim = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(inputRank - 1));
          Value dimList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              Torch::ListType::get(
                  Torch::IntType::get(binder.op->getContext())),
              SmallVector<Value>{cstLastDim});

          Torch::ValueTensorType meanType =
              cast<Torch::ValueTensorType>(resultTypes[1]);

          meanSquared = Torch::AtenMeanDimOp::create(
              rewriter, loc, meanType, sSquared, dimList, cstTrue, cstNone);

          Torch::ValueTensorType invStdVarType =
              cast<Torch::ValueTensorType>(resultTypes[2]);
          Value meanPlusEpsilon = Torch::AtenAddScalarOp::create(
              rewriter, loc, invStdVarType, meanSquared, cstEpsilon, cstOne);
          invStdVar = Torch::AtenRsqrtOp::create(rewriter, loc, invStdVarType,
                                                  meanPlusEpsilon);
        }

        // Output order per spec: (output, mean, inv_std_var,
        // input_skip_bias_sum). For 2-output case, return (output,
        // input_skip_bias_sum) per ORT convention.
        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else if (resultTypes.size() == 2) {
          rewriter.replaceOp(binder.op, {output, s});
        } else if (resultTypes.size() == 3) {
          rewriter.replaceOp(binder.op, {output, meanSquared, invStdVar});
        } else if (resultTypes.size() == 4) {
          rewriter.replaceOp(binder.op, {output, meanSquared, invStdVar, s});
        } else {
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 1-4 result types");
        }

        return success();
      });
  patterns.onOp(
      "GroupQueryAttention", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        SmallVector<Value> operands;
        SmallVector<Type> resultTypes;
        int64_t doRotary, kvNumHeads, localWindowSize, numHeads,
            rotaryInterleaved, smoothSoftmax;
        float scale, softcap;
        if (binder.tensorOperandsList(operands))
          return rewriter.notifyMatchFailure(binder.op,
                                             "operands bind failure");

        if (binder.tensorResultTypes(resultTypes))
          return rewriter.notifyMatchFailure(binder.op,
                                             "result types bind failure");

        if (resultTypes.size() != 3)
          return rewriter.notifyMatchFailure(binder.op,
                                             "expected 3 result types");

        if (binder.s64IntegerAttr(doRotary, "do_rotary") ||
            binder.s64IntegerAttr(kvNumHeads, "kv_num_heads") ||
            binder.s64IntegerAttr(localWindowSize, "local_window_size", -1) ||
            binder.s64IntegerAttr(numHeads, "num_heads") ||
            binder.s64IntegerAttr(rotaryInterleaved, "rotary_interleaved") ||
            binder.f32FloatAttr(scale, "scale") ||
            binder.s64IntegerAttr(smoothSoftmax, "smooth_softmax") ||
            binder.f32FloatAttr(softcap, "softcap"))
          return rewriter.notifyMatchFailure(binder.op,
                                             "op attributes bind failure");

        // This lowering excepts input operands to be either 7 or 9 based on the
        // `do_rotary` attribute. If it's false, then the input operands can be
        // 7 but if it's true then the operands has to be 9 including cos_cache
        // and sin_cache for rotary_embedding.
        if (!((operands.size() == 9) || (!doRotary && operands.size() == 7)))
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented:  excepted input operands to be either "
                         "7 or 9 based on the `do_rotary` attribute");

        if (kvNumHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "kv_num_heads is a required attribute and should be non-zero");

        if (localWindowSize != -1)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: local_window_size attribute is not supported, "
              "hence it should have default value equal to -1");

        if (numHeads == 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "num_heads is a required attribute and should be non-zero");

        if (smoothSoftmax != 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: smooth_softmax attribute is not supported, hence "
              "it should have default value equal to 0");

        if (softcap != 0.0f)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: softcap attribute is not supported, "
                         "hence it should have default value equal to 0.0");

        // TODO: Add support for packed_qkv.

        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        Value query = operands[0];
        Value key = operands[1];
        Value value = operands[2];
        Value pastKey = operands[3];
        Value pastValue = operands[4];
        Value seqlensK = operands[5];
        Value totalSequenceLength = operands[6];
        Value cosCache, sinCache;
        if (doRotary) {
          cosCache = operands[7];
          sinCache = operands[8];
        }

        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!(queryType.hasSizes() && queryType.areAllSizesKnown()))
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected `query` input to have statically known sizes");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        int64_t batchSize = queryDims[0];
        int64_t sequenceLength = queryDims[1];
        int64_t hiddenSize = queryDims[2];
        int64_t headSize = hiddenSize / numHeads;

        Value cstBatchSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(batchSize));
        Value cstSequenceLength = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(),
            rewriter.getI64IntegerAttr(sequenceLength));
        Value cstHiddenSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(hiddenSize));
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(numHeads));
        Value cstKVNumHeads = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(kvNumHeads));

        // Reshape Query, Key and Value as follows:
        // Query: (batch_size, sequence_length, hidden_size)
        //     -> (batch_size, num_heads, sequence_length, head_size)
        // Key: (batch_size, kv_sequence_length, kv_hidden_size)
        //   -> (batch_size, kv_num_heads, sequence_length, head_size)
        // Value: (batch_size, kv_sequence_length, kv_hidden_size)
        //     -> (batch_size, kv_num_heads, sequence_length, head_size)

        // Reshaping query.
        SmallVector<int64_t> queryReshapeSizesInt{batchSize, numHeads,
                                                  sequenceLength, headSize};
        Value queryReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstNumHeads,
                                     cstSequenceLength, cstHeadSize});
        Value qInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryReshapeSizesInt,
                                           queryType.getOptionalDtype()),
            query, queryReshapeSizesList);

        // Reshaping key.
        SmallVector<int64_t> kvReshapeSizesInt{batchSize, kvNumHeads,
                                               sequenceLength, headSize};
        Value kvReshapeSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(query.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstKVNumHeads,
                                     cstSequenceLength, cstHeadSize});
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvReshapeSizesInt,
                                         keyType.getOptionalDtype()),
            key, kvReshapeSizesList);

        // Reshaping value.
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vInput = Torch::AtenReshapeOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvReshapeSizesInt,
                                           valueType.getOptionalDtype()),
            value, kvReshapeSizesList);

        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);

        Value qRotary = qInput, kRotary = kInput;
        if (doRotary) {
          // `totalSequenceLength` is a scalar tensor.
          Value scalarTotalSeqLens = Torch::AtenItemOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(),
              totalSequenceLength);
          Value cstIntOne = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Type boolTy = rewriter.getType<Torch::BoolType>();
          Value condA = Torch::AtenGtIntOp::create(
              rewriter, loc, boolTy, cstSequenceLength, cstIntOne);
          Value condB = Torch::AtenNeIntOp::create(
              rewriter, loc, boolTy, cstSequenceLength, scalarTotalSeqLens);
          //   if (sequence_length > 1 && sequence_length !=
          //   total_sequence_length)
          //         is_subsequent_prompt = false;  // Subsequent prompt
          Value isSubsequentPrompt = Torch::Aten__And__BoolOp::create(
              rewriter, loc, boolTy, condA, condB);

          // Generating position_ids for rotary_embedding as follows:
          //   pos_ids_a = torch.zeros((batch_size, seq_len), dtype=torch.int64)
          //
          //   total_seqlens = seqlens_k + 1
          //   past_seqlens = total_seqlens - sequence_length
          //   pos_ids = torch.arange(sequence_length,
          //             dtype=torch.int64).repeat(batch_size, 1)
          //   pos_ids = pos_ids + past_seqlens.view(-1, 1)
          //   cond = pos_ids < total_seqlens.view(-1, 1)
          //   one_tensor = torch.tensor(1, dtype=torch.int64)
          //   pos_ids_b = torch.where(cond, pos_ids, one_tensor)
          //
          //  if subsequent_prompt:
          //      pos_ids = pos_ids_b
          //  else:
          //      pos_ids = pos_ids_a
          SmallVector<int64_t> positionIdsSizeInt{batchSize, sequenceLength};
          Torch::ValueTensorType positionIdsType = Torch::ValueTensorType::get(
              context, positionIdsSizeInt,
              IntegerType::get(context, 64, IntegerType::Signed));
          Value cstInt64Dtype = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(
                  (int)torch_upstream::ScalarType::Long));

          Value cstInterleaved = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(),
              rewriter.getI64IntegerAttr(rotaryInterleaved));
          Value cstIntZero = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(0));
          Value cstFloatOne = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(1.0));

          Value positionIdsA, positionIdsB;

          Value posIdsSizeList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{cstBatchSize, cstSequenceLength});
          positionIdsA = Torch::AtenZerosOp::create(
              rewriter, loc, positionIdsType, /*size=*/posIdsSizeList,
              /*dtype=*/cstInt64Dtype,
              /*layout=*/cstNone, /*device=*/cstNone,
              /*pin_memory=*/cstNone);

          // Convert seqlens_k which is a tensor of type si32 to si64.
          Torch::ValueTensorType seqLensKType =
              cast<Torch::ValueTensorType>(seqlensK.getType());
          seqlensK = Torch::AtenToDtypeOp::create(
              rewriter, loc,
              seqLensKType.getWithSizesAndDtype(
                  std::nullopt,
                  rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
              seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
              /*copy=*/cstFalse, /*memory_format=*/cstNone);
          Value totalSeqLens = Torch::AtenAddScalarOp::create(
              rewriter, loc, seqlensK.getType(), /*self=*/seqlensK,
              /*other=*/cstIntOne,
              /*alpha=*/cstIntOne);
          Value pastSeqLens = Torch::AtenSubScalarOp::create(
              rewriter, loc, totalSeqLens.getType(), /*self=*/totalSeqLens,
              /*other=*/cstSequenceLength, /*alpha=*/cstIntOne);
          Torch::ValueTensorType initPosIdsType = Torch::ValueTensorType::get(
              context, {sequenceLength},
              IntegerType::get(context, 64, IntegerType::Signed));
          Value initPosIds = Torch::AtenArangeOp::create(
              rewriter, loc, initPosIdsType, cstSequenceLength, cstInt64Dtype,
              /*layout=*/cstNone,
              /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value repeatValuesList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{cstBatchSize, cstIntOne});
          positionIdsB = Torch::AtenRepeatOp::create(
              rewriter, loc, positionIdsType, initPosIds,
              /*repeats=*/repeatValuesList);

          Value cstIntMinusOne = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(1));
          Value viewSizeList = Torch::PrimListConstructOp::create(
              rewriter, binder.getLoc(),
              Torch::ListType::get(Torch::IntType::get(context)),
              llvm::SmallVector<Value>{cstIntMinusOne, cstIntOne});

          Torch::ValueTensorType seqLensViewType = Torch::ValueTensorType::get(
              context, llvm::SmallVector<int64_t>{batchSize, 1},
              IntegerType::get(context, 64, IntegerType::Signed));
          pastSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, pastSeqLens, viewSizeList);

          positionIdsB = Torch::AtenAddTensorOp::create(
              rewriter, loc, positionIdsType, positionIdsB, pastSeqLens,
              /*alpha=*/cstIntOne);

          totalSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, totalSeqLens, viewSizeList);
          Value cond = Torch::AtenLtTensorOp::create(
              rewriter, loc,
              positionIdsType.getWithSizesAndDtype(positionIdsType.getSizes(),
                                                   rewriter.getI1Type()),
              positionIdsB, totalSeqLens);

          Value cstOneTensorDataList = Torch::PrimListConstructOp::create(
              rewriter, loc,
              rewriter.getType<Torch::ListType>(
                  rewriter.getType<Torch::IntType>()),
              SmallVector<Value>{cstIntOne});
          Value cstOneTensor = Torch::AtenTensorOp::create(
              rewriter, loc,
              Torch::ValueTensorType::get(
                  context, {},
                  IntegerType::get(context, 64, IntegerType::Signed)),
              cstOneTensorDataList, /*dtype=*/cstInt64Dtype,
              /*layout=*/cstNone, /*requires_grad=*/cstFalse);

          positionIdsB = Torch::AtenWhereSelfOp::create(
              rewriter, loc, positionIdsType, cond, positionIdsB, cstOneTensor);

          isSubsequentPrompt = Torch::AtenIntBoolOp::create(
              rewriter, loc, rewriter.getType<Torch::IntType>(),
              isSubsequentPrompt);
          isSubsequentPrompt = Torch::AtenFullOp::create(
              rewriter, loc,
              Torch::ValueTensorType::get(context, positionIdsSizeInt,
                                          rewriter.getI1Type()),
              /*size=*/posIdsSizeList, /*fill_value=*/isSubsequentPrompt,
              /*dtype=*/
              Torch::ConstantIntOp::create(
                  rewriter, binder.getLoc(),
                  rewriter.getI64IntegerAttr(
                      (int)torch_upstream::ScalarType::Bool)),
              /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value positionIds = Torch::AtenWhereSelfOp::create(
              rewriter, loc, positionIdsType, isSubsequentPrompt, positionIdsB,
              positionIdsA);

          // Performing RotaryEmbedding over Query and Key.
          qRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), qInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);

          kRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), kInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);
        }

        // Do attention.
        Value cstEnableGQA = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value cstScale = cstNone;
        if (scale != 0.0f)
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, binder.getLoc(), rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qRotary.getType(), qRotary, kRotary, vInput,
            /*attn_mask=*/cstNone,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/cstFalse, cstScale,
            cstEnableGQA);
        // Reshaping the attention result from:
        //    (batch_size, num_heads, sequence_length, head_size)
        // -> (batch_size, sequence_length, hidden_size)
        Value attentionResultSizesList = Torch::PrimListConstructOp::create(
            rewriter, binder.getLoc(),
            Torch::ListType::get(Torch::IntType::get(attention.getContext())),
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstHiddenSize});
        attention = Torch::AtenReshapeOp::create(
            rewriter, loc, resultTypes[0], attention, attentionResultSizesList);

        // Compute 2nd and 3rd result: present_key, present_value.
        // present_key = torch.cat([past_key, key], dim=2) or past_key
        // present_value = torch.cat([past_value, value], dim=2) or past_value
        Value presentKey = pastKey, presentValue = pastValue;
        if (!llvm::equal(
                cast<Torch::ValueTensorType>(pastKey.getType()).getSizes(),
                cast<Torch::ValueTensorType>(resultTypes[1]).getSizes())) {
          Value cstConcatDim = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(2));
          Type kvListElemType = keyType.getWithSizesAndDtype(
              /*optionalSizes=*/std::nullopt,
              /*optionalDtype=*/nullptr);
          Type kvListType = Torch::ListType::get(kvListElemType);
          Value keyList = Torch::PrimListConstructOp::create(
              rewriter, loc, kvListType, SmallVector<Value>{pastKey, kRotary});
          presentKey = Torch::AtenCatOp::create(rewriter, loc, resultTypes[1],
                                                keyList, cstConcatDim);
        }

        if (!llvm::equal(
                cast<Torch::ValueTensorType>(pastValue.getType()).getSizes(),
                cast<Torch::ValueTensorType>(resultTypes[2]).getSizes())) {
          Value cstConcatDim = Torch::ConstantIntOp::create(
              rewriter, binder.getLoc(), rewriter.getI64IntegerAttr(2));
          Type kvListElemType = keyType.getWithSizesAndDtype(
              /*optionalSizes=*/std::nullopt,
              /*optionalDtype=*/nullptr);
          Type kvListType = Torch::ListType::get(kvListElemType);
          Value valueList = Torch::PrimListConstructOp::create(
              rewriter, loc, kvListType, SmallVector<Value>{pastValue, vInput});
          presentValue = Torch::AtenCatOp::create(rewriter, loc, resultTypes[2],
                                                  valueList, cstConcatDim);
        }

        rewriter.replaceOp(binder.op, {attention, presentKey, presentValue});
        return success();
      });
  patterns.onOp(
      "QLinearAdd", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, aScale, aZp, b, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the result of "Add".
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value alpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64FloatAttr(1.0));
        Value c = Torch::AtenAddTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b, alpha);

        // Quantizing the result of "Add" operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
  patterns.onOp(
      "QLinearLeakyRelu", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        float alpha;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.f32FloatAttr(alpha, "alpha"))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the LeakyRelu result.
        Value constAlpha = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr((double)alpha));
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y =
            Torch::AtenLeakyReluOp::create(rewriter, loc, yTy, x, constAlpha);

        // Quantizing the result of LeakyRelu op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearConcat", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        SmallVector<Value> operands;
        int64_t axis;
        if (binder.tensorOperandsList(operands) ||
            binder.s64IntegerAttr(axis, "axis") ||
            binder.tensorResultType(resultType))
          return failure();

        SmallVector<Value> inputs, inputScales, inputZeroPoints;
        for (unsigned i = 2; i < operands.size(); i = i + 3) {
          inputs.push_back(operands[i]);
          inputScales.push_back(operands[i + 1]);
          inputZeroPoints.push_back(operands[i + 2]);
        }

        unsigned numInputs = (operands.size() - 2) / 3;
        if (!(llvm::all_equal({inputs.size(), inputScales.size(),
                               inputZeroPoints.size()}) &&
              inputs.size() == numInputs))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible number of input operands, scales and/or "
                         "zero-points");

        // Preparing the dequantized inputs.
        SmallVector<Value> dequantizedInputs;
        for (unsigned i = 0; i < numInputs; i++) {
          Value scale, zeroPoint;
          if (failed(extractPerTensorQuantizationArguments(
                  rewriter, loc, /*scale=*/inputScales[i],
                  /*zero_point=*/inputZeroPoints[i], scale, zeroPoint)))
            return rewriter.notifyMatchFailure(
                binder.op, "Incompatible scale and zero-points argument for "
                           "per-tensor quantization");

          Value dequantizedInput;
          if (failed(createDequantizeTensor(rewriter, loc, inputs[i], scale,
                                            zeroPoint,
                                            /*output=*/dequantizedInput)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to dequantize the input tensor because of "
                           "missing sizes");

          dequantizedInputs.push_back(dequantizedInput);
        }

        // Concatenating the inputs.
        Type listElemType =
            cast<Torch::BaseTensorType>(dequantizedInputs[0].getType())
                .getWithSizesAndDtype(/*optionalSizes=*/std::nullopt,
                                      /*optionalDtype=*/nullptr);
        Type listType = Torch::ListType::get(listElemType);
        Value tensorList = Torch::PrimListConstructOp::create(
            rewriter, binder.op->getLoc(), listType, dequantizedInputs);
        Value cstAxis = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(axis));
        auto concatTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value concat = Torch::AtenCatOp::create(rewriter, loc, concatTy,
                                                tensorList, cstAxis);

        // Quantizing the result of concatenated inputs.
        Value yScale, yZp;
        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[0],
                /*zero_point=*/operands[1], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible scale and zero-points argument for "
                         "per-tensor quantization");
        Torch::ValueTensorType yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        Value result = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, concat, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          result);
        return success();
      });
  patterns.onOp(
      "QLinearGlobalAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperands(operands, 5) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        auto xTy = dyn_cast<Torch::ValueTensorType>(operands[0].getType());
        if (!xTy || !xTy.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "Expected input argument `x` to have sizes");
        ArrayRef<int64_t> inputShape = xTy.getSizes();

        if (!resultType || !resultType.hasSizes()) {
          return rewriter.notifyMatchFailure(
              binder.op, "Expected result type having sizes");
        }
        ArrayRef<int64_t> resultShape = resultType.getSizes();

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the AvgPool result.
        SmallVector<Value> cstKernel, cstPadding, cstStrides;
        Value cstZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        unsigned inputRank = inputShape.size();
        for (unsigned i = 2; i < inputRank; i++) {
          if (inputShape[i] == Torch::kUnknownSize) {
            Value dim = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(i));
            Value inputDimSize =
                Torch::AtenSizeIntOp::create(rewriter, loc, x, dim);
            cstKernel.push_back(inputDimSize);
          } else {
            int64_t kernelSize = inputShape[i] - resultShape[i] + 1;
            cstKernel.push_back(Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(kernelSize)));
          }
          cstPadding.push_back(cstZero);
          cstStrides.push_back(cstOne);
        }
        Value kernelSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstKernel);
        Value paddingList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstPadding);
        Value stridesList = Torch::PrimListConstructOp::create(
            rewriter, loc,
            Torch::ListType::get(Torch::IntType::get(binder.op->getContext())),
            cstStrides);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstCeilMode = cstFalse;
        Value cstCountIncludePad = cstFalse;
        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultShape, rewriter.getF32Type());
        Value avgpool;
        if (inputRank == 3) {
          avgpool = Torch::AtenAvgPool1dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad);
        } else if (inputRank == 4) {
          avgpool = Torch::AtenAvgPool2dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else if (inputRank == 5) {
          avgpool = Torch::AtenAvgPool3dOp::create(
              rewriter, loc, yTy, x, kernelSizeList, stridesList, paddingList,
              cstCeilMode, cstCountIncludePad,
              /*divisor_override=*/cstNone);
        } else {
          return failure();
        }

        // Quantizing the result of AvgPool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        avgpool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, avgpool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          avgpool);
        return success();
      });
  patterns.onOp(
      "QLinearSigmoid", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Computing the Sigmoid result.
        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value y = Torch::AtenSigmoidOp::create(rewriter, loc, yTy, x);

        // Quantizing the result of Sigmoid op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        y = Torch::AtenQuantizePerTensorOp::create(rewriter, loc, yTy, y,
                                                   yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          y);
        return success();
      });
  patterns.onOp(
      "QLinearAveragePool", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        int64_t channelsLast;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType) ||
            binder.s64IntegerAttr(channelsLast, "channels_last"))
          return failure();

        // TODO: Add support for channels_last attribute.
        if (channelsLast)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: support not present for channels_last attribute");

        if (operands.size() != 5)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 5 input operands");

        Value x, xScale, xZp, yScale, yZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], xScale, xZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[3],
                /*zero_point=*/operands[4], yScale, yZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/xScale, /*zero_point=*/xZp,
                                          /*output=*/x)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `x` because of "
                         "missing sizes");

        // Creating Onnx.AveragePool op.
        llvm::SmallVector<Value> newOperands = {x};
        llvm::SmallVector<NamedAttribute> newAttributes;
        newAttributes.push_back(rewriter.getNamedAttr(
            "name", rewriter.getStringAttr("onnx.AveragePool")));
        for (auto namedAttr : binder.op->getAttrDictionary()) {
          if (namedAttr.getName().getValue().compare("name") == 0)
            continue;
          newAttributes.push_back(namedAttr);
        }

        auto yTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value averagePool = Torch::OperatorOp::create(
                                rewriter, binder.getLoc(), yTy, newOperands,
                                newAttributes, binder.op->getRegions().size())
                                .getResult(0);

        // Quantizing the result of AveragePool op.
        yTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(yTy.getDtype()))));
        averagePool = Torch::AtenQuantizePerTensorOp::create(
            rewriter, loc, yTy, averagePool, yScale, yZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          averagePool);
        return success();
      });
  patterns.onOp(
      "FusedMatMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Torch::ValueTensorType resultType;
        Value lhs, rhs;
        int64_t transA, transB, transBatchA, transBatchB;
        if (binder.tensorOperands(lhs, rhs) ||
            binder.s64IntegerAttr(transA, "transA", 0) ||
            binder.s64IntegerAttr(transB, "transB", 0) ||
            binder.s64IntegerAttr(transBatchA, "transBatchA", 0) ||
            binder.s64IntegerAttr(transBatchB, "transBatchB", 0) ||
            binder.tensorResultType(resultType))
          return failure();

        // Transposing the LHS argument.
        Value transposedLhs = lhs;
        if (transA) {
          // Determine the rank of lhs tensor.
          std::optional<unsigned> maybeRank = Torch::getTensorRank(lhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked lhs tensor");
          unsigned lhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), lhs,
                  /*dimA=*/lhsRank - 2, /*dimB=*/lhsRank - 1, transposedLhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for lhs");
        }

        // Transposing the RHS argument.
        Value transposedRhs = rhs;
        if (transB) {
          std::optional<unsigned> maybeRank = Torch::getTensorRank(rhs);
          if (!maybeRank)
            return rewriter.notifyMatchFailure(
                binder.op, "Unimplemented: unranked rhs tensor");
          unsigned rhsRank = *maybeRank;
          if (failed(createTorchTransposeOp(
                  rewriter, binder.getLoc(), rhs,
                  /*dimA=*/rhsRank - 2, /*dimB=*/rhsRank - 1, transposedRhs)))
            return rewriter.notifyMatchFailure(
                binder.op, "Failed to create TorchTranspose op for rhs");
        }

        // TODO: Add support for `transBatchA` and `transBatchB`
        // attribute.
        if (transBatchA || transBatchB)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: support not present for "
                         "transBatchA and transBatchB attribute");

        rewriter.replaceOpWithNewOp<Torch::AtenMatmulOp>(
            binder.op, resultType, transposedLhs, transposedRhs);
        return success();
      });
  patterns.onOp(
      "QLinearMul", 1,
      [](OpBinder binder, ConversionPatternRewriter &rewriter) {
        Location loc = binder.getLoc();
        Torch::ValueTensorType resultType;
        llvm::SmallVector<Value> operands;
        if (binder.tensorOperandsList(operands) ||
            binder.tensorResultType(resultType))
          return failure();

        if (operands.size() != 8)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: expected 8 input operands");

        Value a, b, aScale, aZp, bScale, bZp, cScale, cZp;

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[1],
                /*zero_point=*/operands[2], aScale, aZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[4],
                /*zero_point=*/operands[5], bScale, bZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(extractPerTensorQuantizationArguments(
                rewriter, loc, /*scale=*/operands[6],
                /*zero_point=*/operands[7], cScale, cZp)))
          return rewriter.notifyMatchFailure(
              binder.op, "Incompatible arguments for per-tensor quantization");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[0],
                                          /*scale=*/aScale, /*zero_point=*/aZp,
                                          /*output=*/a)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `a` because of "
                         "missing sizes");

        if (failed(createDequantizeTensor(rewriter, loc, /*input=*/operands[3],
                                          /*scale=*/bScale, /*zero_point=*/bZp,
                                          /*output=*/b)))
          return rewriter.notifyMatchFailure(
              binder.op, "Failed to dequantize the input tensor `b` because of "
                         "missing sizes");

        // Computing the Mul result.
        auto cTy = rewriter.getType<Torch::ValueTensorType>(
            resultType.getOptionalSizes(), rewriter.getF32Type());
        Value c = Torch::AtenMulTensorOp::create(rewriter, binder.getLoc(), cTy,
                                                 a, b);

        // Quantizing the result of Mul operation.
        cTy = dyn_cast<Torch::ValueTensorType>(
            getQTorchTypeFromTorchIntType(resultType));
        Value dtyVal = Torch::ConstantIntOp::create(
            rewriter, binder.getLoc(), rewriter.getType<Torch::IntType>(),
            rewriter.getIntegerAttr(
                rewriter.getIntegerType(64),
                static_cast<int64_t>(
                    Torch::getScalarTypeForType(cTy.getDtype()))));
        c = Torch::AtenQuantizePerTensorOp::create(rewriter, binder.getLoc(),
                                                   cTy, c, cScale, cZp, dtyVal);
        rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(binder.op, resultType,
                                                          c);
        return success();
      });
}
