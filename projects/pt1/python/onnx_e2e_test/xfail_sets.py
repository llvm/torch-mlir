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
    # OnnxLSTM_forward_basic: RefBackend bufferization fails with
    # "Yield operand #1 is not equivalent to the corresponding iter bbArg"
    # during linalg-on-tensors -> LLVM lowering.
    "OnnxLSTM_forward_basic",
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
    # OnnxLSTM_forward_basic: LSTM has no torch -> tosa lowering.
    "OnnxLSTM_forward_basic",
    # OnnxSTFT_onesided_basic: STFT has no torch -> tosa lowering.
    "OnnxSTFT_onesided_basic",
    # OnnxAdd_f32_dynamicInput: TOSA -> linalg -> RefBackend fails to bufferize
    # a dynamic-shaped tensor ("bufferization.dealloc ... marked illegal").
    "OnnxAdd_f32_dynamicInput",
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
}


ONNX_E2E_TOSA_CRASHING_SET = set()
