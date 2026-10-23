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
#   0: output             = s * inv_std_var * gamma
#   1: mean               (vestigial for the simplified variant; zeros)
#   2: inv_std_var        = 1 / sqrt(mean(s^2, axis=-1) + epsilon)
#   3: input_skip_bias_sum = s
# The reference implementation lives in onnx_e2e_test.reference.
#
# Note: the ONNX Runtime CPU kernel used as the reference does not compute the
# mean and inv_std_var outputs (it returns None for them even when requested),
# and its schema has no stash_type attribute. The e2e tests therefore declare
# all four node outputs but only compare output and input_skip_bias_sum.
# inv_std_var is covered by the lit tests in simple_ops_q_to_z.mlir.
# ==============================================================================

_SSLN_NODE_OUTPUTS = ["output", "mean", "inv_std_var", "input_skip_bias_sum"]
_SSLN_GRAPH_OUTPUTS = ["output", "input_skip_bias_sum"]


@register_onnx_test
class OnnxSkipSimplifiedLayerNormalization_f32_basic(OnnxTestCase):
    # f32 inputs: no upcast path. The node declares all four outputs so the
    # multi-output lowering is exercised end to end.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 4, 8]),
            ("skip", torch.float32, [2, 4, 8]),
            ("gamma", torch.float32, [8]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["input", "skip", "gamma"],
            _SSLN_NODE_OUTPUTS,
            domain="com.microsoft",
            epsilon=1.0e-5,
        )
        return [node], _SSLN_GRAPH_OUTPUTS


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
        node = onnx.helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["input", "skip", "gamma", "bias"],
            _SSLN_NODE_OUTPUTS,
            domain="com.microsoft",
            epsilon=1.0e-5,
        )
        return [node], _SSLN_GRAPH_OUTPUTS


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
        node = onnx.helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["input", "skip", "gamma"],
            _SSLN_NODE_OUTPUTS,
            domain="com.microsoft",
            epsilon=1.0e-5,
        )
        return [node], _SSLN_GRAPH_OUTPUTS
