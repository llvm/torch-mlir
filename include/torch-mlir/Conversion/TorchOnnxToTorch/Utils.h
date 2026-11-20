//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H
#define TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include <utility>

class Endian {
private:
  static constexpr uint32_t uint32_ = 0x01020304;
  static constexpr uint8_t magic_ = (const uint8_t &)uint32_;

public:
  static constexpr bool little = magic_ == 0x04;
  static constexpr bool big = magic_ == 0x01;
  static_assert(little || big, "Cannot determine endianness!");

private:
  Endian() = delete;
};

namespace mlir::torch::onnx_c {

// Default alpha and beta attributes of the ONNX activation operators, also
// used for the gate activations of LSTM, GRU and RNN of the same name.
constexpr float kLeakyReluAlpha = 0.01f;
constexpr float kEluAlpha = 1.0f;
constexpr float kThresholdedReluAlpha = 1.0f;
constexpr float kHardSigmoidAlpha = 0.2f;
constexpr float kHardSigmoidBeta = 0.5f;

// Builders for ONNX activations, shared by the standalone ONNX operators and
// the gate activations of LSTM, GRU and RNN. Each returns `input` activated,
// with type `resultType`.
Value createLeakyRelu(OpBuilder &b, Location loc, Type resultType, Value input,
                      float alpha);
Value createElu(OpBuilder &b, Location loc, Type resultType, Value input,
                float alpha);
Value createThresholdedRelu(OpBuilder &b, Location loc, Type resultType,
                            Value input, float alpha);
Value createHardSigmoid(OpBuilder &b, Location loc, Type resultType,
                        Value input, float alpha, float beta);
Value createSoftsign(OpBuilder &b, Location loc, Type resultType, Value input);
Value createSoftplus(OpBuilder &b, Location loc, Type resultType, Value input);

// A gate activation of an ONNX LSTM, GRU or RNN, with its alpha and beta
// resolved from the op's activation_alpha and activation_beta attributes.
struct RecurrentActivation {
  enum class Kind {
    Relu,
    Tanh,
    Sigmoid,
    Affine,
    LeakyRelu,
    ThresholdedRelu,
    ScaledTanh,
    HardSigmoid,
    Elu,
    Softsign,
    Softplus,
  };
  Kind kind;
  float alpha = 0.0f;
  float beta = 0.0f;
};

// Pairs each activation in `names` with its alpha and beta, taken from the
// recurrent op's activation_alpha and activation_beta attributes.
//
// Each list is read front to back, and a value goes only to an activation that
// uses that parameter. For example, with names [HardSigmoid, Tanh, LeakyRelu]
// and activation_alpha [a0, a1], HardSigmoid gets a0, Tanh gets none and
// LeakyRelu gets a1.
//
// Once a list runs out, an activation uses the default of the ONNX operator of
// the same name, e.g. alpha 0.01 for LeakyRelu. Affine and ScaledTanh have no
// such operator, so a missing alpha or beta is 0, as in onnxruntime.
//
// Reports a match failure on the op through `rewriter`, and returns failure,
// if:
// - a name is not one of the activations in RecurrentActivation::Kind,
//   compared case-insensitively as in onnxruntime, or
// - activation_alpha or activation_beta is not a list of f32 values.
FailureOr<SmallVector<RecurrentActivation>>
resolveRecurrentActivations(OpBinder binder,
                            ConversionPatternRewriter &rewriter,
                            ArrayRef<std::string> names);

// Builds `activation` applied to `input`.
Value createRecurrentActivation(ImplicitLocOpBuilder &b,
                                const RecurrentActivation &activation,
                                Value input);

Value createConstantIntList(OpBinder binder,
                            ConversionPatternRewriter &rewriter,
                            ArrayRef<int64_t> cstInput);

inline bool isSupportedQuantizeDequantizeLinearDtype(Type type) {
  // TODO: Support uint16 once Torch::getScalarTypeForType can map it.
  return type.isUnsignedInteger(8) || type.isSignedInteger(8) ||
         type.isSignedInteger(16) || type.isSignedInteger(32) ||
         isa<Float4E2M1FNType, Float8E4M3FNType, Float8E4M3FNUZType,
             Float8E5M2Type, Float8E5M2FNUZType>(type);
}

std::pair<Value, Value>
createQuantizationBounds(ConversionPatternRewriter &rewriter, Location loc,
                         IntegerType dtype);

template <typename T>
Value getItemOp(OpBinder binder, ConversionPatternRewriter &rewriter,
                Value &ofItem) {
  return Torch::AtenItemOp::create(rewriter, binder.getLoc(),
                                   rewriter.getType<T>(), ofItem);
}

LogicalResult OnnxLstmExpander(OpBinder binder,
                               ConversionPatternRewriter &rewriter);
LogicalResult OnnxGruExpander(OpBinder binder,
                              ConversionPatternRewriter &rewriter);
LogicalResult OnnxRnnExpander(OpBinder binder,
                              ConversionPatternRewriter &rewriter);

bool areAllElementsDistinct(SmallVector<int64_t> array);

namespace detail {
/// Matches the constant integers stored in a `onnx.Constant`.
struct onnx_list_of_constant_ints_op_binder {
  SmallVectorImpl<int64_t> &bind_values;

  /// Creates a matcher instance that binds the value to bvs if match succeeds.
  onnx_list_of_constant_ints_op_binder(SmallVectorImpl<int64_t> &bvs)
      : bind_values(bvs) {}

  bool match(Operation *op) {
    auto constOp = dyn_cast<Torch::OperatorOp>(op);
    if (!constOp || !(constOp.getName() == "onnx.Constant"))
      return false;

    if (DenseResourceElementsAttr attr =
            dyn_cast_or_null<DenseResourceElementsAttr>(
                constOp->getAttr("torch.onnx.value"))) {
      // Bytes are stored in little endian order. Big endian support will
      // require swizzling.
      if (!Endian::little) {
        op->emitError("unimplemented: importing on big endian systems");
        return false;
      }

      auto ty = cast<ShapedType>(attr.getType());
      ElementsAttr denseAttr;
      auto blobPtr = attr.getRawHandle().getBlob();
      if (!blobPtr)
        return false;
      denseAttr = DenseElementsAttr::getFromRawBuffer(ty, blobPtr->getData());
      for (auto axis : denseAttr.getValues<llvm::APInt>()) {
        bind_values.push_back(axis.getSExtValue());
      }
      return true;
    }
    if (ElementsAttr attr = dyn_cast_or_null<ElementsAttr>(
            constOp->getAttr("torch.onnx.value"))) {
      for (auto axis : attr.getValues<llvm::APInt>())
        bind_values.push_back(axis.getSExtValue());
      return true;
    }
    return false;
  }
};
} // namespace detail

/// Matches the constant integers stored in a `onnx.Constant`.
inline detail::onnx_list_of_constant_ints_op_binder
m_OnnxListOfConstantInts(SmallVectorImpl<int64_t> &bind_values) {
  return detail::onnx_list_of_constant_ints_op_binder(bind_values);
}

std::optional<int64_t> onnxDtypeIntToTorchDtypeInt(int64_t dtypeIntOnnx);

LogicalResult createTorchTransposeOp(ConversionPatternRewriter &rewriter,
                                     Location loc, Value input, int64_t dimA,
                                     int64_t dimB, Value &transposed);

LogicalResult createTorchPermuteOp(OpBinder binder,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, Value input,
                                   SmallVector<int64_t> permuteDims,
                                   Value &permuted);

/// This utility checks the compatibility for scale and zero_point value and
/// extracts the scalar value from it used for per-tensor quantization.
/// `inZeroPoint` may be null (an omitted optional ONNX input), in which case
/// `outZeroPoint` is set to the constant integer 0.
LogicalResult extractPerTensorQuantizationArguments(
    ConversionPatternRewriter &rewriter, Location loc, Value inScale,
    Value inZeroPoint, Value &outScale, Value &outZeroPoint);

/// This utility takes as input an integer tensor along with its per-tensor
/// affine quantization parameters (scale, zero point) and dequantizes it to
/// a floating point tensor using `quantized_decomposed.dequantize_per_tensor`
/// (PT2E representation). `input` must be a `ValueTensorType` with a known
/// integer dtype and known sizes. If `outFloatTy` is set, it is passed as the
/// `out_dtype` and used as the result element type; otherwise the result is
/// f32.
LogicalResult createDequantizeTensor(ConversionPatternRewriter &rewriter,
                                     Location loc, Value input, Value scale,
                                     Value zeroPoint, Value &output,
                                     FloatType outFloatTy = nullptr);

/// This utility takes as input a floating point tensor along with per-tensor
/// affine quantization parameters (scale, zero point) and quantizes it to
/// `resultType`'s integer element type using
/// `quantized_decomposed.quantize_per_tensor` (PT2E representation).
/// `resultType` must have known sizes and an integer dtype.
LogicalResult createQuantizeTensor(ConversionPatternRewriter &rewriter,
                                   Location loc, Value input, Value scale,
                                   Value zeroPoint,
                                   Torch::ValueTensorType resultType,
                                   Value &output);

/// This utility takes as input an integer tensor along with per-channel
/// quantization parameters (1-D scales tensor, optional 1-D zero-points
/// tensor or `torch.none`) and dequantizes it to a floating point tensor
/// using `quantized_decomposed.dequantize_per_channel` (PT2E representation).
/// `input` must be a `ValueTensorType` with known integer dtype and sizes.
/// `outFloatTy` behaves as in `createDequantizeTensor`.
LogicalResult
createDequantizePerChannelTensor(ConversionPatternRewriter &rewriter,
                                 Location loc, Value input, Value scales,
                                 Value zeroPoints, int64_t axis, Value &output,
                                 FloatType outFloatTy = nullptr);

/// This utility dequantizes an integer tensor `input` (rank >= 2) whose
/// quantization parameters vary along the batch dims (all dims before
/// rank-2) and along dim rank-2, which must have a static size. `scale` and
/// `zeroPoint` must have the same shape and rank as `input`, size 1 along the
/// last dim, and batch dims that are either 1 (broadcast) or equal to the
/// corresponding dim of `input`, e.g. (B, X, 1) for an input of shape
/// (B, X, K). A `quantized_decomposed.dequantize_per_token` op (PT2E
/// representation) is emitted. Broadcast batch dims of the parameters are
/// expanded, since the op requires them to match the input. Callers whose
/// parameters vary along the last dim have to transpose the last two dims of
/// the (integer) input and of the parameters first.
/// `outFloatTy` behaves as in `createDequantizeTensor`, except that
/// `out_dtype` is not optional for this op: it is always passed, as f32 if
/// `outFloatTy` is null.
LogicalResult createDequantizePerTokenTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value input, Value scale,
    Value zeroPoint, Value &output, FloatType outFloatTy = nullptr);

} // namespace mlir::torch::onnx_c

#endif // TORCHMLIR_CONVERSION_TORCHONNXTOTORCH_UTILS_H
