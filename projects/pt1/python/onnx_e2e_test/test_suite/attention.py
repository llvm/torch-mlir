# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native end-to-end tests for the com.microsoft::Attention operator.

The fused QKV weights and bias are embedded as ONNX initializers; only the
activation tensors (input, and the optional runtime mask / past KV cache) are
annotated runtime inputs.  Each case is checked against the ONNX Runtime
contrib kernels wired into the harness's reference evaluator.
"""

import numpy as np

import onnx.helper
import onnx.numpy_helper
import torch

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# Shared tiny dimensions: batch=1, seq=4, input_hidden=8, num_heads=2,
# head_size=4, qkv_hidden=24 (8/8/8).
_BATCH = 1
_SEQ = 4
_INPUT_HIDDEN = 8
_NUM_HEADS = 2
_QKV_HIDDEN = 24


def _weights(seed, rows, cols):
    return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)


# ==============================================================================
# Attention: input + weights + bias, explicit scale.
#
#   input   [1, 4, 8]  --\
#   weights [8, 24]       >--> Attention --> output [1, 4, 8]
#   bias    [24]       --/
# ==============================================================================


@register_onnx_test
class OnnxAttention_basic(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(1, _INPUT_HIDDEN, _QKV_HIDDEN)
        self._b = _weights(2, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention without the optional bias input.
# ==============================================================================


@register_onnx_test
class OnnxAttention_noBias(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(3, _INPUT_HIDDEN, _QKV_HIDDEN)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
        )
        initializers = [onnx.numpy_helper.from_array(self._W, name="weights")]
        return [node], ["output"], initializers


# ==============================================================================
# Attention with scale omitted: defaults to 1/sqrt(head_size).
# ==============================================================================


@register_onnx_test
class OnnxAttention_scaleDefault(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(4, _INPUT_HIDDEN, _QKV_HIDDEN)
        self._b = _weights(5, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention with unidirectional (causal) masking.
# ==============================================================================


@register_onnx_test
class OnnxAttention_unidirectional(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(6, _INPUT_HIDDEN, _QKV_HIDDEN)
        self._b = _weights(7, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
            unidirectional=1,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention with a 2D raw mask_index [batch, total_seq] passed at runtime.
# ==============================================================================


@register_onnx_test
class OnnxAttention_maskIndex2d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(8, _INPUT_HIDDEN, _QKV_HIDDEN)
        self._b = _weights(9, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs(
        [
            ("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN]),
            ("mask_index", torch.int32, [_BATCH, _SEQ], (0, 2)),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias", "mask_index"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention where V has a larger head size than Q/K (qkv_hidden_sizes
# = [8, 8, 16]): output hidden is num_heads * v_head_size = 16.
# ==============================================================================


@register_onnx_test
class OnnxAttention_vHeadSize(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(10, _INPUT_HIDDEN, 32)
        self._b = _weights(11, 1, 32).reshape(32)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 16],
            scale=0.5,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention with do_rotary=1 (RoPE applied to Q and K).
# ==============================================================================


@register_onnx_test
class OnnxAttention_doRotary(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(12, _INPUT_HIDDEN, _QKV_HIDDEN)
        self._b = _weights(13, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs([("input", torch.float32, [_BATCH, _SEQ, _INPUT_HIDDEN])])
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias"],
            ["output"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
            do_rotary=1,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output"], initializers


# ==============================================================================
# Attention with a past KV cache (two outputs: output + present).
#   input [1, 2, 16], weights [16, 24], past [2, 1, 2, 3, 4]
#   -> output [1, 2, 8], present [2, 1, 2, 5, 4]
# ==============================================================================


@register_onnx_test
class OnnxAttention_pastPresent(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    def __init__(self):
        self._W = _weights(14, 16, _QKV_HIDDEN)
        self._b = _weights(15, 1, _QKV_HIDDEN).reshape(_QKV_HIDDEN)

    @annotate_inputs(
        [
            ("input", torch.float32, [1, 2, 16]),
            ("past", torch.float32, [2, 1, 2, 3, 4]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node(
            "Attention",
            ["input", "weights", "bias", "", "past"],
            ["output", "present"],
            domain="com.microsoft",
            num_heads=_NUM_HEADS,
            qkv_hidden_sizes=[8, 8, 8],
            scale=0.5,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="weights"),
            onnx.numpy_helper.from_array(self._b, name="bias"),
        ]
        return [node], ["output", "present"], initializers
