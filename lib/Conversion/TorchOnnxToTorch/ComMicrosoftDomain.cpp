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

        if (resultTypes.size() != 1)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unsupported number of results");

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype() || !inputType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "input should have dtype and sizes");

        // Get tensor rank to normalize axis
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank || *maybeRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked or scalar input tensor");
        unsigned inputRank = *maybeRank;

        // Build normalized_shape: [inputShape[axis], ..., inputShape[-1]]
        axis = Torch::toPositiveDim(axis, inputRank);
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
        Value output =
            Torch::AtenRmsNormOp::create(rewriter, loc, resultTypes[0], input,
                                         normalizedShape, scale, cstEpsilon);
        rewriter.replaceOp(binder.op, {output});
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

        if (resultTypes.size() > 2)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unsupported number of results");

        // Optional bias (index 3)
        Value bias;
        bool hasBias = !binder.tensorOperandAtIndex(bias, 3);

        // Get input type to determine shapes and dtype
        Torch::ValueTensorType inputType =
            cast<Torch::ValueTensorType>(input.getType());
        if (!inputType.hasDtype() || !inputType.hasSizes())
          return rewriter.notifyMatchFailure(
              binder.op, "input should have dtype and sizes");

        // Get tensor rank to compute last dimension
        std::optional<unsigned> maybeRank = Torch::getTensorRank(input);
        if (!maybeRank || *maybeRank == 0)
          return rewriter.notifyMatchFailure(binder.op,
                                             "unranked or scalar input tensor");
        unsigned inputRank = *maybeRank;

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
        Value output =
            Torch::AtenRmsNormOp::create(rewriter, loc, resultTypes[0], s,
                                         normalizedShape, gamma, cstEpsilon);

        if (resultTypes.size() == 1) {
          rewriter.replaceOp(binder.op, {output});
        } else {
          // 2-output: (output, input_skip_bias_sum)
          rewriter.replaceOp(binder.op, {output, s});
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

        // This lowering supports two input formats:
        // 1. Separate Q, K, V inputs (9 operands with rotary, 7 without):
        //    query, key, value, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        // 2. Packed QKV input (7 operands with rotary, 5 without):
        //    packed_qkv, past_key, past_value, seqlens_k, total_seq_len,
        //    [cos_cache, sin_cache]
        bool isPackedQKV = false;
        if (doRotary) {
          if (operands.size() == 7) {
            isPackedQKV = true;
          } else if (operands.size() != 9) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected 7 operands (packed QKV) or 9 operands (separate Q, "
                "K, V) when do_rotary is enabled");
          }
        } else {
          if (operands.size() == 5) {
            isPackedQKV = true;
          } else if (operands.size() != 7) {
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected 5 operands (packed QKV) or 7 operands (separate Q, "
                "K, V) when do_rotary is disabled");
          }
        }

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

        if (smoothSoftmax > 0)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Unimplemented: smooth_softmax attribute is not supported, hence "
              "it should have a value <= 0 (disabled)");

        if (softcap != 0.0f)
          return rewriter.notifyMatchFailure(
              binder.op, "Unimplemented: softcap attribute is not supported, "
                         "hence it should have default value equal to 0.0");

        Location loc = binder.getLoc();
        MLIRContext *context = binder.op->getContext();
        Value query, key, value, pastKey, pastValue, seqlensK;
        Value cosCache, sinCache;

        if (isPackedQKV) {
          // Packed QKV mode: first operand contains Q, K, V concatenated
          Value packedQKV = operands[0];
          pastKey = operands[1];
          pastValue = operands[2];
          seqlensK = operands[3];
          if (doRotary) {
            cosCache = operands[5];
            sinCache = operands[6];
          }

          // Split packed QKV into separate Q, K, V tensors
          // packed_qkv shape: [batch, seq, q_hidden + k_hidden + v_hidden]
          // where q_hidden = num_heads * head_size
          //       k_hidden = kv_num_heads * head_size
          //       v_hidden = kv_num_heads * head_size
          Torch::ValueTensorType packedType =
              cast<Torch::ValueTensorType>(packedQKV.getType());
          if (!packedType.hasSizes() || packedType.getSizes().size() != 3)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected packed QKV input to have 3 dimensions");

          SmallVector<int64_t> packedDims{packedType.getSizes()};
          int64_t batchSize = packedDims[0];        // may be dynamic
          int64_t sequenceLength = packedDims[1];   // may be dynamic
          int64_t packedHiddenSize = packedDims[2]; // must be static

          if (packedHiddenSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op,
                "Expected packed QKV hidden dimension (dim 2) to be static");

          // Calculate head_size from past_key shape: [batch, kv_num_heads,
          // past_seq, head_size]
          Torch::ValueTensorType pastKeyType =
              cast<Torch::ValueTensorType>(pastKey.getType());
          if (!(pastKeyType.hasSizes() && pastKeyType.getSizes().size() == 4))
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key to have 4 dimensions");

          int64_t headSize = pastKeyType.getSizes()[3];
          if (headSize == Torch::kUnknownSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Expected past_key head_size (dim 3) to be static");

          int64_t qHiddenSize = numHeads * headSize;
          int64_t kvHiddenSize = kvNumHeads * headSize;

          // Validate packed hidden size
          if (packedHiddenSize != qHiddenSize + 2 * kvHiddenSize)
            return rewriter.notifyMatchFailure(
                binder.op, "Packed QKV hidden size mismatch: expected " +
                               std::to_string(qHiddenSize + 2 * kvHiddenSize) +
                               " but got " + std::to_string(packedHiddenSize));

          Value cstOne = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(1));
          Value cstTwo = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(2));
          Value cstZero = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(0));
          Value cstQHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(qHiddenSize));
          Value cstQPlusKVHidden = Torch::ConstantIntOp::create(
              rewriter, loc,
              rewriter.getI64IntegerAttr(qHiddenSize + kvHiddenSize));
          Value cstPackedHidden = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(packedHiddenSize));

          // Slice Q: packed_qkv[:, :, 0:q_hidden]
          // batch and seq dimensions may be dynamic
          SmallVector<int64_t> querySizes{batchSize, sequenceLength,
                                          qHiddenSize};
          Torch::ValueTensorType queryType = Torch::ValueTensorType::get(
              context, querySizes, packedType.getOptionalDtype());
          query = Torch::AtenSliceTensorOp::create(
              rewriter, loc, queryType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstZero, /*end=*/cstQHidden,
              /*step=*/cstOne);

          // Slice K: packed_qkv[:, :, q_hidden:q_hidden+kv_hidden]
          SmallVector<int64_t> kvSizes{batchSize, sequenceLength, kvHiddenSize};
          Torch::ValueTensorType keyType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          key = Torch::AtenSliceTensorOp::create(rewriter, loc, keyType,
                                                 packedQKV,
                                                 /*dim=*/cstTwo,
                                                 /*start=*/cstQHidden,
                                                 /*end=*/cstQPlusKVHidden,
                                                 /*step=*/cstOne);

          // Slice V: packed_qkv[:, :, q_hidden+kv_hidden:]
          Torch::ValueTensorType valueType = Torch::ValueTensorType::get(
              context, kvSizes, packedType.getOptionalDtype());
          value = Torch::AtenSliceTensorOp::create(
              rewriter, loc, valueType, packedQKV,
              /*dim=*/cstTwo, /*start=*/cstQPlusKVHidden,
              /*end=*/cstPackedHidden,
              /*step=*/cstOne);
        } else {
          // Separate Q, K, V mode
          query = operands[0];
          key = operands[1];
          value = operands[2];
          pastKey = operands[3];
          pastValue = operands[4];
          seqlensK = operands[5];
          if (doRotary) {
            cosCache = operands[7];
            sinCache = operands[8];
          }
        }

        Torch::ValueTensorType queryType =
            cast<Torch::ValueTensorType>(query.getType());
        if (!queryType.hasSizes() || queryType.getSizes().size() != 3)
          return rewriter.notifyMatchFailure(
              binder.op, "Expected `query` input to have 3 dimensions");

        SmallVector<int64_t> queryDims{queryType.getSizes()};
        int64_t batchSize = queryDims[0];      // may be dynamic
        int64_t sequenceLength = queryDims[1]; // may be dynamic
        int64_t hiddenSize = queryDims[2];     // must be static
        if (hiddenSize == Torch::kUnknownSize)
          return rewriter.notifyMatchFailure(
              binder.op,
              "Expected `query` hidden dimension (dim 2) to be static");
        int64_t headSize = hiddenSize / numHeads;

        // For dynamic dimensions, use aten.size.int to get runtime values
        Type intType = rewriter.getType<Torch::IntType>();

        Value cstZeroDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstBatchSize = rewriter.createOrFold<Torch::AtenSizeIntOp>(
            loc, intType, query, cstZeroDim);
        Value cstOneDim = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstSequenceLength = rewriter.createOrFold<Torch::AtenSizeIntOp>(
            loc, intType, query, cstOneDim);

        Value cstHiddenSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(hiddenSize));
        Value cstHeadSize = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(headSize));
        Value cstNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(numHeads));
        Value cstKVNumHeads = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(kvNumHeads));

        Value cstNone = Torch::ConstantNoneOp::create(rewriter, loc);
        Value cstFalse = Torch::ConstantBoolOp::create(rewriter, loc, false);
        Value cstIntZero = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(0));
        Value cstIntOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstIntMinusOne = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(-1));
        Value cstDim1 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstDim2 = Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(2));
        Value cstInt64Dtype = Torch::ConstantIntOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr((int)torch_upstream::ScalarType::Long));

        Type intListType = Torch::ListType::get(Torch::IntType::get(context));

        // Convert seqlens_k to int64 unconditionally (may be si32 from ONNX).
        Torch::ValueTensorType seqLensKType =
            cast<Torch::ValueTensorType>(seqlensK.getType());
        if (seqLensKType.getOptionalDtype() &&
            seqLensKType.getOptionalDtype().isInteger(32)) {
          seqlensK = Torch::AtenToDtypeOp::create(
              rewriter, loc,
              seqLensKType.getWithSizesAndDtype(
                  seqLensKType.getOptionalSizes(),
                  rewriter.getIntegerType(/*width=*/64, /*isSigned=*/true)),
              seqlensK, cstInt64Dtype, /*non_blocking=*/cstFalse,
              /*copy=*/cstFalse, /*memory_format=*/cstNone);
        }

        // Reshape Q/K/V from [batch, seq, hidden] to [batch, heads, seq,
        // head_size]. This requires:
        // 1. Reshape to [batch, seq, heads, head_size]
        // 2. Transpose dims 1 and 2 to get [batch, heads, seq, head_size]
        // A direct reshape would incorrectly interleave heads and sequence
        // positions.

        // Reshaping query: [batch, seq, hidden] -> [batch, seq, num_heads,
        // head_size]
        SmallVector<int64_t> queryIntermediateSizesInt{
            batchSize, sequenceLength, numHeads, headSize};
        Value queryIntermediateSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstNumHeads, cstHeadSize});
        Value qIntermediate = Torch::AtenReshapeOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryIntermediateSizesInt,
                                           queryType.getOptionalDtype()),
            query, queryIntermediateSizesList);

        // Transpose query: [batch, seq, num_heads, head_size] -> [batch,
        // num_heads, seq, head_size]
        SmallVector<int64_t> queryFinalSizesInt{batchSize, numHeads,
                                                sequenceLength, headSize};
        Value qInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            queryType.getWithSizesAndDtype(queryFinalSizesInt,
                                           queryType.getOptionalDtype()),
            qIntermediate, cstDim1, cstDim2);

        // Reshaping key: [batch, seq, kv_hidden] -> [batch, seq, kv_num_heads,
        // head_size]
        SmallVector<int64_t> kvIntermediateSizesInt{batchSize, sequenceLength,
                                                    kvNumHeads, headSize};
        Value kvIntermediateSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstKVNumHeads, cstHeadSize});
        Torch::ValueTensorType keyType =
            cast<Torch::ValueTensorType>(key.getType());
        Value kIntermediate = Torch::AtenReshapeOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvIntermediateSizesInt,
                                         keyType.getOptionalDtype()),
            key, kvIntermediateSizesList);

        // Transpose key: [batch, seq, kv_num_heads, head_size] -> [batch,
        // kv_num_heads, seq, head_size]
        SmallVector<int64_t> kvFinalSizesInt{batchSize, kvNumHeads,
                                             sequenceLength, headSize};
        Value kInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            keyType.getWithSizesAndDtype(kvFinalSizesInt,
                                         keyType.getOptionalDtype()),
            kIntermediate, cstDim1, cstDim2);

        // Reshaping value: [batch, seq, kv_hidden] -> [batch, seq,
        // kv_num_heads, head_size]
        Torch::ValueTensorType valueType =
            cast<Torch::ValueTensorType>(value.getType());
        Value vIntermediate = Torch::AtenReshapeOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvIntermediateSizesInt,
                                           valueType.getOptionalDtype()),
            value, kvIntermediateSizesList);

        // Transpose value: [batch, seq, kv_num_heads, head_size] -> [batch,
        // kv_num_heads, seq, head_size]
        Value vInput = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            valueType.getWithSizesAndDtype(kvFinalSizesInt,
                                           valueType.getOptionalDtype()),
            vIntermediate, cstDim1, cstDim2);

        Value qRotary = qInput, kRotary = kInput;
        if (doRotary) {

          // Generating position_ids for rotary_embedding:
          //   total_seqlens = seqlens_k + 1
          //   past_seqlens = total_seqlens - sequence_length
          //   pos_ids = torch.arange(sequence_length).repeat(batch_size, 1)
          //   pos_ids = pos_ids + past_seqlens.view(-1, 1)
          //
          // This works for all cases:
          // - Pure prefill: past_seqlens = 0, positions = [0, 1, ..., seq-1]
          // - Single decode: past_seqlens = past_len, positions = [past_len]
          // - Multi-token with past: positions = [past_len, ...,
          // past_len+seq-1]
          SmallVector<int64_t> positionIdsSizeInt{batchSize, sequenceLength};
          Torch::ValueTensorType positionIdsType = Torch::ValueTensorType::get(
              context, positionIdsSizeInt,
              IntegerType::get(context, 64, IntegerType::Signed));

          Value cstInterleaved = Torch::ConstantIntOp::create(
              rewriter, loc, rewriter.getI64IntegerAttr(rotaryInterleaved));
          Value cstFloatOne = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(1.0));

          // total_seqlens = seqlens_k + 1 (per ONNX spec, seqlens_k = total -
          // 1)
          Value totalSeqLens = Torch::AtenAddScalarOp::create(
              rewriter, loc, seqlensK.getType(), /*self=*/seqlensK,
              /*other=*/cstIntOne,
              /*alpha=*/cstIntOne);

          // past_seqlens = total_seqlens - sequence_length
          Value pastSeqLens = Torch::AtenSubScalarOp::create(
              rewriter, loc, totalSeqLens.getType(), /*self=*/totalSeqLens,
              /*other=*/cstSequenceLength, /*alpha=*/cstIntOne);

          // Create position IDs: arange(seq_len).repeat(batch, 1) +
          // past_seqlens
          Torch::ValueTensorType initPosIdsType = Torch::ValueTensorType::get(
              context, {sequenceLength},
              IntegerType::get(context, 64, IntegerType::Signed));
          Value initPosIds = Torch::AtenArangeOp::create(
              rewriter, loc, initPosIdsType, cstSequenceLength, cstInt64Dtype,
              /*layout=*/cstNone,
              /*device=*/cstNone, /*pin_memory=*/cstNone);
          Value repeatValuesList = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              llvm::SmallVector<Value>{cstBatchSize, cstIntOne});
          Value positionIds = Torch::AtenRepeatOp::create(
              rewriter, loc, positionIdsType, initPosIds,
              /*repeats=*/repeatValuesList);

          // Reshape past_seqlens to [batch, 1] for broadcasting
          Value viewSizeList = Torch::PrimListConstructOp::create(
              rewriter, loc, intListType,
              llvm::SmallVector<Value>{cstIntMinusOne, cstIntOne});
          Torch::ValueTensorType seqLensViewType = Torch::ValueTensorType::get(
              context, llvm::SmallVector<int64_t>{batchSize, 1},
              IntegerType::get(context, 64, IntegerType::Signed));
          pastSeqLens = Torch::AtenViewOp::create(
              rewriter, loc, seqLensViewType, pastSeqLens, viewSizeList);

          // Add past_seqlens to get final position IDs
          positionIds = Torch::AtenAddTensorOp::create(
              rewriter, loc, positionIdsType, positionIds, pastSeqLens,
              /*alpha=*/cstIntOne);

          // Performing RotaryEmbedding over Query and Key.
          qRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, qInput.getType(), qInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);

          kRotary = Torch::OnnxVariantRotaryEmbeddingOp::create(
              rewriter, loc, kInput.getType(), kInput, positionIds, cosCache,
              sinCache, cstInterleaved, /*is_packed_batching=*/cstIntZero,
              /*num_heads=*/cstIntZero, /*rotary_embedding_dim=*/cstIntZero,
              /*scale=*/cstFloatOne);
        }

        // Build present_key/present_value by padding past with zeros, then
        // scattering current K/V into the correct per-batch position.
        //
        // Why pad instead of cat? With cat(past, current), the current token
        // ends up at position past_seq_len. With variable seqlens_k, ORT places
        // the current token at seqlens_k[b] and leaves position past_seq_len as
        // zero/uninitialized. Using cat+scatter leaves a stale copy of the
        // current token at past_seq_len which doesn't match ORT's output.
        // Padding with zeros then scattering avoids this.
        //
        // constant_pad_nd pads innermost dims first: [0, 0, 0, seq_len]
        //   dim 3 (head_size): [0, 0]  -- no padding
        //   dim 2 (seq):       [0, seq_len] -- extend by seq_len on the right
        Value padList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstIntZero, cstIntZero, cstIntZero,
                               cstSequenceLength});
        Value cstFloatZeroPad = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value presentKey = Torch::AtenConstantPadNdOp::create(
            rewriter, loc, resultTypes[1], pastKey, padList, cstFloatZeroPad);
        Value presentValue = Torch::AtenConstantPadNdOp::create(
            rewriter, loc, resultTypes[2], pastValue, padList, cstFloatZeroPad);

        // Scatter current K/V into the padded buffer at position pastLen[b]+q.
        // pastLen = seqlens_k + 1 - seq_len
        Value totalSeqForScatter = Torch::AtenAddScalarOp::create(
            rewriter, loc, seqlensK.getType(), seqlensK, cstIntOne, cstIntOne);
        Value pastLen = Torch::AtenSubScalarOp::create(
            rewriter, loc, totalSeqForScatter.getType(), totalSeqForScatter,
            cstSequenceLength, cstIntOne);

        // pastLen -> [B, 1, 1, 1] for 4D scatter index broadcasting
        Value scatterViewList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstIntMinusOne, cstIntOne, cstIntOne,
                               cstIntOne});
        SmallVector<int64_t> pastLenView4dSizes{batchSize, 1, 1, 1};
        Torch::ValueTensorType pastLenView4dType = Torch::ValueTensorType::get(
            context, pastLenView4dSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value pastLenView4d = Torch::AtenViewOp::create(
            rewriter, loc, pastLenView4dType, pastLen, scatterViewList);

        // qRange for scatter: arange(seq_len) -> reshape to [1, 1, seq, 1]
        Torch::ValueTensorType scatterQRangeType = Torch::ValueTensorType::get(
            context, {sequenceLength},
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterQRange = Torch::AtenArangeOp::create(
            rewriter, loc, scatterQRangeType, cstSequenceLength, cstInt64Dtype,
            /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);
        Value scatterQViewList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstIntOne, cstIntOne, cstIntMinusOne,
                               cstIntOne});
        SmallVector<int64_t> scatterQViewSizes{1, 1, sequenceLength, 1};
        Torch::ValueTensorType scatterQViewType = Torch::ValueTensorType::get(
            context, scatterQViewSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterQRangeView = Torch::AtenViewOp::create(
            rewriter, loc, scatterQViewType, scatterQRange, scatterQViewList);

        // scatterIdxBase = pastLen[B,1,1,1] + qRange[1,1,seq,1]
        //               -> [B, 1, seq, 1]
        SmallVector<int64_t> scatterIdxBaseSizes{batchSize, 1, sequenceLength,
                                                 1};
        Torch::ValueTensorType scatterIdxBaseType = Torch::ValueTensorType::get(
            context, scatterIdxBaseSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterIdxBase = Torch::AtenAddTensorOp::create(
            rewriter, loc, scatterIdxBaseType, pastLenView4d, scatterQRangeView,
            cstIntOne);

        // Expand to [B, kv_heads, seq, head_size] to match current K/V shape
        SmallVector<int64_t> scatterExpandSizes{batchSize, kvNumHeads,
                                                sequenceLength, headSize};
        Torch::ValueTensorType scatterIdxType = Torch::ValueTensorType::get(
            context, scatterExpandSizes,
            rewriter.getIntegerType(64, /*isSigned=*/true));
        Value scatterExpandSizeList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            SmallVector<Value>{cstBatchSize, cstKVNumHeads, cstSequenceLength,
                               cstHeadSize});
        Value scatterIdx = Torch::AtenExpandOp::create(
            rewriter, loc, scatterIdxType, scatterIdxBase,
            scatterExpandSizeList, /*implicit=*/cstFalse);

        // Scatter current K/V into buffer at position pastLen[b] + q
        presentKey = Torch::AtenScatterSrcOp::create(
            rewriter, loc, resultTypes[1], presentKey, cstDim2, scatterIdx,
            kRotary);
        presentValue = Torch::AtenScatterSrcOp::create(
            rewriter, loc, resultTypes[2], presentValue, cstDim2, scatterIdx,
            vInput);

        // Generate causal attention mask.
        // With scatter, KV layout matches ORT: current at pastLen[b].
        // Simple mask: k <= pastLen[b] + q
        // Mask shape: [batch, 1, seqLen, kvSeqLen] where masked = -inf.
        Value attnMask = cstNone;

        // Get the KV sequence length from presentKey shape
        Torch::ValueTensorType presentKeyType =
            cast<Torch::ValueTensorType>(presentKey.getType());
        if (presentKeyType.hasSizes() &&
            presentKeyType.getSizes().size() == 4) {
          int64_t kvSeqLen = presentKeyType.getSizes()[2];

          // Only generate mask if KV sequence length is dynamic or > 0
          // For dynamic shapes or non-trivial sequences, we need to mask
          if (kvSeqLen == Torch::kUnknownSize || kvSeqLen > 0) {
            // Get KV sequence dimension size
            Value kvSeqLenVal = Torch::AtenSizeIntOp::create(
                rewriter, loc, rewriter.getType<Torch::IntType>(), presentKey,
                cstDim2);

            // kRange: [0, 1, 2, ..., kvSeqLen-1] shape [kvSeqLen]
            Torch::ValueTensorType rangeType = Torch::ValueTensorType::get(
                context, {kvSeqLen},
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value kRange = Torch::AtenArangeOp::create(
                rewriter, loc, rangeType, kvSeqLenVal, cstInt64Dtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // qRange: [0, 1, 2, ..., seqLen-1] shape [seqLen]
            Torch::ValueTensorType qRangeType = Torch::ValueTensorType::get(
                context, {sequenceLength},
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value qRange = Torch::AtenArangeOp::create(
                rewriter, loc, qRangeType, cstSequenceLength, cstInt64Dtype,
                /*layout=*/cstNone, /*device=*/cstNone, /*pin_memory=*/cstNone);

            // Reshape for broadcasting:
            // pastLen: [batch] -> [batch, 1, 1]
            // qRange: [seqLen] -> [1, seqLen, 1]
            // kRange: [kvSeqLen] -> [1, 1, kvSeqLen]

            // pastLen -> [batch, 1, 1]
            Value seqlensViewList = Torch::PrimListConstructOp::create(
                rewriter, loc, intListType,
                SmallVector<Value>{cstIntMinusOne, cstIntOne, cstIntOne});
            SmallVector<int64_t> seqlensViewSizes{batchSize, 1, 1};
            Torch::ValueTensorType seqlensViewType =
                Torch::ValueTensorType::get(
                    context, seqlensViewSizes,
                    rewriter.getIntegerType(64, /*isSigned=*/true));
            Value pastLenView = Torch::AtenViewOp::create(
                rewriter, loc, seqlensViewType, pastLen, seqlensViewList);

            // qRange -> [1, seqLen, 1]
            Value qViewList = Torch::PrimListConstructOp::create(
                rewriter, loc, intListType,
                SmallVector<Value>{cstIntOne, cstIntMinusOne, cstIntOne});
            SmallVector<int64_t> qViewSizes{1, sequenceLength, 1};
            Torch::ValueTensorType qViewType = Torch::ValueTensorType::get(
                context, qViewSizes,
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value qRangeView = Torch::AtenViewOp::create(
                rewriter, loc, qViewType, qRange, qViewList);

            // kRange -> [1, 1, kvSeqLen]
            Value kViewList = Torch::PrimListConstructOp::create(
                rewriter, loc, intListType,
                SmallVector<Value>{cstIntOne, cstIntOne, cstIntMinusOne});
            SmallVector<int64_t> kViewSizes{1, 1, kvSeqLen};
            Torch::ValueTensorType kViewType = Torch::ValueTensorType::get(
                context, kViewSizes,
                rewriter.getIntegerType(64, /*isSigned=*/true));
            Value kRangeView = Torch::AtenViewOp::create(
                rewriter, loc, kViewType, kRange, kViewList);

            // Causal mask: k <= pastLen + q
            // pastLenView[batch,1,1] + qRangeView[1,seqLen,1]
            // -> [batch, seqLen, 1]
            SmallVector<int64_t> pastLenPlusQSizes{batchSize, sequenceLength,
                                                   1};
            Torch::ValueTensorType pastLenPlusQType =
                Torch::ValueTensorType::get(
                    context, pastLenPlusQSizes,
                    rewriter.getIntegerType(64, /*isSigned=*/true));
            Value pastLenPlusQ = Torch::AtenAddTensorOp::create(
                rewriter, loc, pastLenPlusQType, pastLenView, qRangeView,
                cstIntOne);

            // kRangeView[1,1,kvSeqLen] <= pastLenPlusQ[batch,seqLen,1]
            // -> [batch, seqLen, kvSeqLen]
            SmallVector<int64_t> maskBoolSizes{batchSize, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType maskBoolType = Torch::ValueTensorType::get(
                context, maskBoolSizes, rewriter.getI1Type());
            Value causalMask = Torch::AtenLeTensorOp::create(
                rewriter, loc, maskBoolType, kRangeView, pastLenPlusQ);

            // Convert bool mask to float mask: True -> 0, False -> -inf
            Value cstZeroFloat = Torch::ConstantFloatOp::create(
                rewriter, loc, rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(0.0));
            Value cstNegInf = Torch::ConstantFloatOp::create(
                rewriter, loc, rewriter.getType<Torch::FloatType>(),
                rewriter.getF64FloatAttr(
                    -std::numeric_limits<double>::infinity()));

            // Derive mask dtype from query to avoid type mismatch (e.g. f16
            // inputs must produce f16 mask, not f32).
            Type maskElementType = queryType.getOptionalDtype();
            if (!maskElementType)
              maskElementType = rewriter.getF32Type(); // fallback
            int64_t maskScalarType = static_cast<int64_t>(
                Torch::getScalarTypeForType(maskElementType));

            Value cstFloatDtype = Torch::ConstantIntOp::create(
                rewriter, loc, rewriter.getI64IntegerAttr(maskScalarType));
            Torch::ValueTensorType scalarTensorType =
                Torch::ValueTensorType::get(context, {}, maskElementType);
            Value emptyShapeList = Torch::PrimListConstructOp::create(
                rewriter, loc, intListType, SmallVector<Value>{});
            auto makeScalarTensor = [&](Value fillValue) {
              return Torch::AtenFullOp::create(
                  rewriter, loc, scalarTensorType, /*size=*/emptyShapeList,
                  /*fill_value=*/fillValue, /*dtype=*/cstFloatDtype,
                  /*layout=*/cstNone, /*device=*/cstNone,
                  /*pin_memory=*/cstNone);
            };
            Value zeroTensor = makeScalarTensor(cstZeroFloat);
            Value negInfTensor = makeScalarTensor(cstNegInf);

            // mask_float = where(causalMask, 0, -inf)
            SmallVector<int64_t> maskFloatSizes{batchSize, sequenceLength,
                                                kvSeqLen};
            Torch::ValueTensorType maskFloatType = Torch::ValueTensorType::get(
                context, maskFloatSizes, maskElementType);
            Value maskFloat = Torch::AtenWhereSelfOp::create(
                rewriter, loc, maskFloatType, causalMask, zeroTensor,
                negInfTensor);

            // Reshape to [batch, 1, seqLen, kvSeqLen] for SDPA
            Value maskReshapeSizeList = Torch::PrimListConstructOp::create(
                rewriter, loc, intListType,
                SmallVector<Value>{cstBatchSize, cstIntOne, cstSequenceLength,
                                   kvSeqLenVal});
            SmallVector<int64_t> attnMaskSizes{batchSize, 1, sequenceLength,
                                               kvSeqLen};
            Torch::ValueTensorType attnMaskType = Torch::ValueTensorType::get(
                context, attnMaskSizes, maskElementType);
            attnMask = Torch::AtenReshapeOp::create(
                rewriter, loc, attnMaskType, maskFloat, maskReshapeSizeList);
          }
        }

        // Do attention with full KV cache (past + current) and mask.
        Value cstEnableGQA = Torch::ConstantBoolOp::create(rewriter, loc, true);
        Value cstFloatZero = Torch::ConstantFloatOp::create(
            rewriter, loc, rewriter.getType<Torch::FloatType>(),
            rewriter.getF64FloatAttr(0.0));
        Value cstScale = cstNone;
        if (scale != 0.0f)
          cstScale = Torch::ConstantFloatOp::create(
              rewriter, loc, rewriter.getType<Torch::FloatType>(),
              rewriter.getF64FloatAttr(scale));

        // Use presentKey/presentValue (full KV cache) for attention, not just
        // the current token's K/V. This is essential for proper KV caching.
        Value attention = Torch::AtenScaledDotProductAttentionOp::create(
            rewriter, loc, qRotary.getType(), qRotary, presentKey, presentValue,
            /*attn_mask=*/attnMask,
            /*dropout_p=*/cstFloatZero, /*is_causal=*/cstFalse, cstScale,
            cstEnableGQA);

        // Reshaping the attention result from:
        //    (batch_size, num_heads, sequence_length, head_size)
        // -> (batch_size, sequence_length, hidden_size)
        // This requires:
        // 1. Transpose dims 1 and 2: [batch, num_heads, seq, head_size]
        //                         -> [batch, seq, num_heads, head_size]
        // 2. Reshape: [batch, seq, num_heads, head_size] -> [batch, seq,
        // hidden]
        // Transpose: [batch, num_heads, seq, head_size] -> [batch, seq,
        // num_heads, head_size]
        SmallVector<int64_t> attnTransposedSizes{batchSize, sequenceLength,
                                                 numHeads, headSize};
        Torch::ValueTensorType attnType =
            cast<Torch::ValueTensorType>(attention.getType());
        Value attnTransposed = Torch::AtenTransposeIntOp::create(
            rewriter, loc,
            attnType.getWithSizesAndDtype(attnTransposedSizes,
                                          attnType.getOptionalDtype()),
            attention, cstDim1, cstDim2);

        // Reshape: [batch, seq, num_heads, head_size] -> [batch, seq, hidden]
        Value attentionResultSizesList = Torch::PrimListConstructOp::create(
            rewriter, loc, intListType,
            llvm::SmallVector<Value>{cstBatchSize, cstSequenceLength,
                                     cstHiddenSize});
        attention = Torch::AtenReshapeOp::create(rewriter, loc, resultTypes[0],
                                                 attnTransposed,
                                                 attentionResultSizesList);

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
