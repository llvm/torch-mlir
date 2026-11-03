# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native tests for Microsoft-domain normalization ops."""

import torch
import onnx.helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# com.microsoft.SkipSimplifiedLayerNormalization
#
# RMS-style layer norm over the last dimension of the residual sum
# s = input + skip (+ bias). Up to four outputs, in order:
#   0: output              = s * inv_std_var * gamma
#   1: mean                (vestigial for the simplified variant; zeros)
#   2: inv_std_var         = 1 / sqrt(mean(s^2, axis=-1) + epsilon)
#   3: input_skip_bias_sum = s
# The reference implementation lives in onnx_e2e_test.reference.
#
# All four outputs are graph outputs and are compared against ONNX Runtime.
# ONNX Runtime computes mean and inv_std_var on the CPU kernel only since
# version 1.30.0; older versions return None, which the reference runner
# rejects with an explicit error.
# ==============================================================================

_SSLN_OUTPUTS = ["output", "mean", "inv_std_var", "input_skip_bias_sum"]


def _ssln_node(inputs, epsilon=1.0e-5):
    return onnx.helper.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs,
        _SSLN_OUTPUTS,
        domain="com.microsoft",
        epsilon=epsilon,
    )


@register_onnx_test
class OnnxSkipSimplifiedLayerNormalization_f32_basic(OnnxTestCase):
    # f32 inputs: no upcast path.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 4, 8]),
            ("skip", torch.float32, [2, 4, 8]),
            ("gamma", torch.float32, [8]),
        ]
    )
    def graph(self):
        return [_ssln_node(["input", "skip", "gamma"])], _SSLN_OUTPUTS


@register_onnx_test
class OnnxSkipSimplifiedLayerNormalization_f32_bias(OnnxTestCase):
    # Supplies the optional bias input (index 3) so the hasBias path of the
    # lowering, s = input + skip + bias, is exercised end to end.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 4, 8]),
            ("skip", torch.float32, [2, 4, 8]),
            ("gamma", torch.float32, [8]),
            ("bias", torch.float32, [8]),
        ]
    )
    def graph(self):
        return [_ssln_node(["input", "skip", "gamma", "bias"])], _SSLN_OUTPUTS


@register_onnx_test
class OnnxSkipSimplifiedLayerNormalization_f32_statistics(OnnxTestCase):
    # Mirrors ONNX Runtime's SkipSimplifiedLayerNormStatistics test shape
    # ([1, 1, 4] input, [1, 1, 1] statistics), with a wider value range so the
    # inv_std_var output is not dominated by epsilon.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [1, 1, 4], (-4.0, 4.0)),
            ("skip", torch.float32, [1, 1, 4], (-1.0, 1.0)),
            ("gamma", torch.float32, [4]),
        ]
    )
    def graph(self):
        return [_ssln_node(["input", "skip", "gamma"], epsilon=1.0e-12)], _SSLN_OUTPUTS


@register_onnx_test
class OnnxSkipSimplifiedLayerNormalization_f16(OnnxTestCase):
    # f16 inputs exercise the upcast path: the lowering defaults stash_type to
    # 1 (f32), so the residual sum and normalization are computed in f32 and
    # cast back to f16.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float16, [2, 4, 8]),
            ("skip", torch.float16, [2, 4, 8]),
            ("gamma", torch.float16, [8]),
        ]
    )
    def graph(self):
        return [_ssln_node(["input", "skip", "gamma"])], _SSLN_OUTPUTS
