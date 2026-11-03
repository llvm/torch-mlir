# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native end-to-end tests for com.microsoft::MultiHeadAttention.

Golden values come from the ONNX Runtime CPU contrib kernel. Projection biases
are initializers; activations, masks, attention bias and past KV are runtime
inputs. Packed QKV / KV layouts are not covered here because the ORT CPU
provider does not implement them; they are covered by the lit tests only.

Notation: B batch, S query seq, L kv seq, P past seq, N heads, H head size.
"""

import numpy as np

import onnx.helper
import onnx.numpy_helper
import torch

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

_N = 2
_RANGE = (-1.0, 1.0)


def _bias(seed, size, name="bias"):
    values = np.random.default_rng(seed).standard_normal(size).astype(np.float32)
    return onnx.numpy_helper.from_array(values, name=name)


def _mha(inputs, outputs=("output",), num_heads=_N, **attrs):
    return onnx.helper.make_node(
        "MultiHeadAttention",
        list(inputs),
        list(outputs),
        domain="com.microsoft",
        num_heads=num_heads,
        **attrs,
    )


def _f32(name, shape):
    return (name, torch.float32, shape, _RANGE)


# Cross-length 3D Q/K/V with an explicit scale: S = 3, L = 5, H = 4.
@register_onnx_test
class OnnxMultiHeadAttention_basic(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [1, 3, 8]), _f32("k", [1, 5, 8]), _f32("v", [1, 5, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"], scale=0.5)], ["output"]


# N = 4, H = 2: head split independent of the default N.
@register_onnx_test
class OnnxMultiHeadAttention_fourHeads(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [2, 3, 8]), _f32("k", [2, 5, 8]), _f32("v", [2, 5, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"], num_heads=4)], ["output"]


# scale omitted: 1 / sqrt(H).
@register_onnx_test
class OnnxMultiHeadAttention_scaleDefault(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [2, 4, 8]), _f32("k", [2, 4, 8]), _f32("v", [2, 4, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"])], ["output"]


# scale = 0 is the ORT spelling of the default.
@register_onnx_test
class OnnxMultiHeadAttention_scaleZero(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [1, 4, 8]), _f32("k", [1, 4, 8]), _f32("v", [1, 4, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"], scale=0.0)], ["output"]


# Bias [Q | K | V] added to the 3D projections.
@register_onnx_test
class OnnxMultiHeadAttention_bias(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [1, 3, 8]), _f32("k", [1, 5, 8]), _f32("v", [1, 5, 8])])
    def graph(self):
        return [_mha(["q", "k", "v", "bias"])], ["output"], [_bias(1, 24)]


# v_hidden != hidden (Hv = 6) with bias of length 8 + 8 + 12.
@register_onnx_test
class OnnxMultiHeadAttention_vHidden(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [_f32("q", [1, 3, 8]), _f32("k", [1, 5, 8]), _f32("v", [1, 5, 12])]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "bias"])], ["output"], [_bias(2, 28)]


# Causal self-attention.
@register_onnx_test
class OnnxMultiHeadAttention_unidirectional(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [1, 4, 8]), _f32("k", [1, 4, 8]), _f32("v", [1, 4, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"], unidirectional=1)], ["output"]


# [B, T] key_padding_mask (0 = padded) with a custom mask_filter_value.
@register_onnx_test
class OnnxMultiHeadAttention_keyPaddingMask2d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 3, 8]),
            _f32("k", [2, 5, 8]),
            _f32("v", [2, 5, 8]),
            ("mask", torch.int32, [2, 5], (0, 2)),
        ]
    )
    def graph(self):
        node = _mha(["q", "k", "v", "", "mask"], mask_filter_value=-1000.0)
        return [node], ["output"]


# [B, S, T] key_padding_mask.
@register_onnx_test
class OnnxMultiHeadAttention_keyPaddingMask3d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 3, 8]),
            _f32("k", [2, 5, 8]),
            _f32("v", [2, 5, 8]),
            ("mask", torch.int32, [2, 3, 5], (0, 2)),
        ]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "", "mask"])], ["output"]


# [B] key lengths: positions >= length are padded.
@register_onnx_test
class OnnxMultiHeadAttention_keyLengths1d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 3, 8]),
            _f32("k", [2, 5, 8]),
            _f32("v", [2, 5, 8]),
            ("lengths", torch.int32, [2], (1, 6)),
        ]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "", "lengths"])], ["output"]


# attention_bias [1, N, S, T], broadcast over the batch.
@register_onnx_test
class OnnxMultiHeadAttention_attentionBias(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 3, 8]),
            _f32("k", [2, 5, 8]),
            _f32("v", [2, 5, 8]),
            _f32("attn_bias", [1, 2, 3, 5]),
        ]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "", "", "attn_bias"])], ["output"]


# Dynamic B / S / L with [B] key lengths: the mask length is checked against
# B at runtime; dims are 2 at runtime.
@register_onnx_test
class OnnxMultiHeadAttention_dynamicKeyLengths1d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [-1, -1, 8]),
            _f32("k", [-1, -1, 8]),
            _f32("v", [-1, -1, 8]),
            ("lengths", torch.int32, [-1], (1, 3)),
        ]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "", "lengths"])], ["output"]


# Causal + [B] key lengths + attention_bias, summed into one additive mask.
@register_onnx_test
class OnnxMultiHeadAttention_combinedMasks(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 4, 8]),
            _f32("k", [2, 4, 8]),
            _f32("v", [2, 4, 8]),
            ("lengths", torch.int32, [2], (1, 5)),
            _f32("attn_bias", [2, 1, 4, 4]),
        ]
    )
    def graph(self):
        node = _mha(["q", "k", "v", "", "lengths", "attn_bias"], unidirectional=1)
        return [node], ["output"]


# past_key / past_value with all three outputs: present = cat(past, current).
@register_onnx_test
class OnnxMultiHeadAttention_pastPresent(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [1, 3, 8]),
            _f32("k", [1, 3, 8]),
            _f32("v", [1, 3, 8]),
            _f32("past_k", [1, 2, 2, 4]),
            _f32("past_v", [1, 2, 2, 4]),
        ]
    )
    def graph(self):
        outputs = ["output", "present_k", "present_v"]
        node = _mha(["q", "k", "v", "", "", "", "past_k", "past_v"], outputs)
        return [node], outputs


# past + causal mask offset by P, with bias.
@register_onnx_test
class OnnxMultiHeadAttention_pastPresentUnidirectional(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [2, 3, 8]),
            _f32("k", [2, 3, 8]),
            _f32("v", [2, 3, 8]),
            _f32("past_k", [2, 2, 4, 4]),
            _f32("past_v", [2, 2, 4, 4]),
        ]
    )
    def graph(self):
        outputs = ["output", "present_k", "present_v"]
        node = _mha(
            ["q", "k", "v", "bias", "", "", "past_k", "past_v"],
            outputs,
            unidirectional=1,
        )
        return [node], outputs, [_bias(3, 24)]


# Explicit non-default scale (0.25 != 1/sqrt(4)) together with a mask.
@register_onnx_test
class OnnxMultiHeadAttention_maskedExplicitScale(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs([_f32("q", [1, 4, 8]), _f32("k", [1, 4, 8]), _f32("v", [1, 4, 8])])
    def graph(self):
        return [_mha(["q", "k", "v"], unidirectional=1, scale=0.25)], ["output"]


# Cross-attention with pre-projected [B, N, L, H] key / value. ORT requires
# the full [N*H + N*H + N*Hv] bias (24 here) for every layout and rejects a
# query-only one; with BNSH key / value only the Q slice [0:8] is applied.
@register_onnx_test
class OnnxMultiHeadAttention_crossBnsh(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [_f32("q", [2, 3, 8]), _f32("k", [2, 2, 6, 4]), _f32("v", [2, 2, 6, 4])]
    )
    def graph(self):
        return [_mha(["q", "k", "v", "bias"])], ["output"], [_bias(4, 24)]


# Dynamic B / S / P with past and causal mask; dims are 2 at runtime.
@register_onnx_test
class OnnxMultiHeadAttention_dynamicPastUnidirectional(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            _f32("q", [-1, -1, 8]),
            _f32("k", [-1, -1, 8]),
            _f32("v", [-1, -1, 8]),
            _f32("past_k", [-1, 2, -1, 4]),
            _f32("past_v", [-1, 2, -1, 4]),
        ]
    )
    def graph(self):
        outputs = ["output", "present_k", "present_v"]
        node = _mha(
            ["q", "k", "v", "", "", "", "past_k", "past_v"],
            outputs,
            unidirectional=1,
        )
        return [node], outputs


# f16 causal self-attention; the causal fill is the lowest finite f16.
@register_onnx_test
class OnnxMultiHeadAttention_f16Unidirectional(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("q", torch.float16, [1, 4, 8], _RANGE),
            ("k", torch.float16, [1, 4, 8], _RANGE),
            ("v", torch.float16, [1, 4, 8], _RANGE),
        ]
    )
    def graph(self):
        return [_mha(["q", "k", "v"], unidirectional=1)], ["output"]
