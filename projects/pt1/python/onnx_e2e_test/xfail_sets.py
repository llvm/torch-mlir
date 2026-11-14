# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Expected-failure sets for the ONNX-native e2e test suite.

Named ``ONNX_E2E_<CONFIG>_<KIND>_SET`` where CONFIG is LINALG or TOSA:

- ``*_XFAIL_SET`` — tests known to fail (XFAIL); unexpected passes are errors.
- ``*_CRASHING_SET`` — tests that crash the process; they are skipped entirely.

Sets are per-config: a name belongs in whichever config's set it fails under.
Add a test's ``unique_name`` (the decorated function name) to the appropriate
set when filing a bug, and remove it when the bug is fixed.
"""

# Tests expected to fail.  Add names here when a bug is filed.
ONNX_E2E_LINALG_XFAIL_SET = {
    # QLinearAveragePool_*: off by one quantized step in a few elements
    # versus the onnxruntime golden (ORT's integer QLinearAveragePool kernel
    # rounds differently from the dequantize/avg_pool/quantize lowering).
    "QLinearAveragePool_ui8_basic",
    "QLinearAveragePool_i8_basic",
}

# Tests that crash the interpreter and cannot be XFAILed safely.
# These are skipped (not attempted) by the runner.
ONNX_E2E_LINALG_CRASHING_SET = set()

# --- TOSA backend (onnx_tosa config) ---
# TOSA has different op coverage than linalg; tests unsupported on the
# torch -> tosa path go here.
ONNX_E2E_TOSA_XFAIL_SET = {
    # TOSA per-tensor DQ gives incorrect numerical results for uint8 inputs
    # above 127. Keep the full unsigned range in these tests to expose it.
    "OnnxMatMulIntegerToFloat_aZeroPoint",
    "OnnxMatMulIntegerToFloat_bZeroPoint",
    # TOSA cannot lower aten.item on runtime qparams.
    "OnnxMatMulIntegerToFloat_runtimeQParams",
    # TOSA has no first-class PT2E per-channel DQ lowering.
    "OnnxMatMulIntegerToFloat_perColumn",
    "OnnxMatMulIntegerToFloat_mixedQParams",
    "OnnxMatMulIntegerToFloat_batchedWeights",
    # TOSA cannot lower dynamic size.int / per-channel DQ here.
    "OnnxMatMulIntegerToFloat_dynamicColumns",
    # TOSA has no first-class PT2E per-channel-group DQ lowering.
    "OnnxMatMulNBits_4bit",
    "OnnxMatMulNBits_8bit_zeroPointBias",
    "OnnxMatMulNBits_raggedK",
    "OnnxMatMulNBits_floatZeroPoint",
    "OnnxMatMulNBits_accuracyLevel4",
    "OnnxMatMulNBits_runtimeWeights",
    # Recurrent ops: LSTM, GRU and RNN lower to torch.prim.Loop, which has no
    # torch -> tosa lowering.
    "OnnxLSTM_forward_basic",
    "OnnxLSTM_hardsigmoid_gates",
    "OnnxGRU_hardsigmoid_gates",
    "OnnxRNN_relu",
    "OnnxRNN_sigmoid",
    "OnnxRNN_affine",
    "OnnxRNN_leakyrelu",
    "OnnxRNN_thresholdedrelu",
    "OnnxRNN_scaledtanh",
    "OnnxRNN_hardsigmoid",
    "OnnxRNN_elu",
    "OnnxRNN_softsign",
    "OnnxRNN_softplus",
    "OnnxRNN_affine_alpha_only",
    "OnnxRNN_affine_beta_only",
    "OnnxRNN_affine_no_parameters",
    "OnnxRNN_scaledtanh_no_parameters",
    "OnnxLSTM_alpha_order",
    "OnnxLSTM_bidirectional_reverse_alpha",
    "OnnxRNN_thresholdedrelu_at_alpha",
    # OnnxSTFT_onesided_basic: STFT has no torch -> tosa lowering.
    "OnnxSTFT_onesided_basic",
    # OnnxAveragePool2d_countIncludePad_asymmetricPadsDilated: tosa.avg_pool2d
    # has no dilation, so the torch -> tosa pooling lowering rejects it.
    "OnnxAveragePool2d_countIncludePad_asymmetricPadsDilated",
    # QLinear*_ui8: The torch→tosa uses tosa.cast. For ui types, the
    # tosa.cast is incompatible with PyTorch/ONNX "round half to even",
    # producing off-by-one results.
    "QLinearAdd_ui8_basic",
    "QLinearMul_ui8_basic",
    "QLinearLeakyRelu_ui8_basic",
    "QLinearSigmoid_ui8_basic",
    "QLinearConcat_ui8_basic",
    "QLinearGlobalAveragePool_ui8_basic",
    "QLinearAveragePool_ui8_basic",
    "QLinearAddQLinearSigmoidFused_ui8_basic",
    # QLinearAveragePool_i8_basic: off by one quantized step versus the
    # onnxruntime golden (same rounding difference as on linalg).
    "QLinearAveragePool_i8_basic",
    # OnnxConvInteger_*: ConvInteger is lowered with an f64 convolution so the
    # int32 accumulator stays exact, but TOSA has no f64 support (tosa.conv2d
    # rejects acc_type = f64).
    "OnnxConvInteger_i8_basic",
    "OnnxConvInteger_i8_noWeightZp",
    "OnnxConvInteger_i8_withPadding",
    "OnnxConvInteger_ui8_basic",
    "OnnxConvInteger_ui8_noWeightZp",
    "OnnxConvInteger_ui8_withPadding",
    # OnnxConvInteger_*_perChannelWeightZp: TOSA has no lowering for
    # torch.quantized_decomposed.dequantize_per_channel. Once it is added,
    # these will hit the f64 issue above.
    "OnnxConvInteger_ui8_perChannelWeightZp",
    "OnnxConvInteger_i8_perChannelWeightZp",
    # DynamicQuantizeLinear_*: the lowering computes the zero point at runtime
    # and TOSA cannot lower the resulting torch.aten.item.
    "DynamicQuantizeLinear_basic",
    "DynamicQuantizeLinear_positiveRange",
    "DynamicQuantizeLinear_negativeRange",
    "DynamicQuantizeLinear_rank1",
    "DynamicQuantizeLinear_dynamicShape",
    # OnnxQLinearConv_perChannel*: TOSA has no lowering for
    # torch.quantized_decomposed.dequantize_per_channel.
    "OnnxQLinearConv_perChannel",
    "OnnxQLinearConv_perChannelBias",
    # OnnxQLinearConv_ui8_basic: TOSA mishandles uint8 inputs above 127.
    # Keep the full unsigned range in these tests to expose it.
    "OnnxQLinearConv_ui8_basic",
}


ONNX_E2E_TOSA_CRASHING_SET = set()
