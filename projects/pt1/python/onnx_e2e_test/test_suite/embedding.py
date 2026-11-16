# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native tests for Microsoft-domain embedding ops."""

import torch
import onnx.helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# com.microsoft.RotaryEmbedding
#
# Rotates each head of `input` by the angles in `cos_cache` and `sin_cache` at
# `position_ids`. A 3-D input is laid out as (batch, sequence, num_heads *
# head_size), a 4-D input as (batch, num_heads, sequence, head_size).
# ==============================================================================


def _rotary_embedding_node(**attributes):
    return onnx.helper.make_node(
        "RotaryEmbedding",
        ["input", "position_ids", "cos_cache", "sin_cache"],
        ["output"],
        domain="com.microsoft",
        **attributes,
    )


@register_onnx_test
class OnnxRotaryEmbedding_3d_dynamic(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [-1, -1, 8]),
            ("position_ids", torch.int64, [-1, -1], (0, 8)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        return [_rotary_embedding_node()], ["output"]


@register_onnx_test
class OnnxRotaryEmbedding_4d(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 2, 3, 4]),
            ("position_ids", torch.int64, [2, 3], (0, 8)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        return [_rotary_embedding_node()], ["output"]


@register_onnx_test
class OnnxRotaryEmbedding_interleaved(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 3, 8]),
            ("position_ids", torch.int64, [2, 3], (0, 8)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        return [_rotary_embedding_node(interleaved=1)], ["output"]


@register_onnx_test
class OnnxRotaryEmbedding_partialRotaryDim(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 3, 16]),
            ("position_ids", torch.int64, [2, 3], (0, 8)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        node = _rotary_embedding_node(num_heads=2, rotary_embedding_dim=4)
        return [node], ["output"]


@register_onnx_test
class OnnxRotaryEmbedding_positionOffset(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 3, 8]),
            ("position_ids", torch.int64, [1], (0, 5)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        return [_rotary_embedding_node()], ["output"]


@register_onnx_test
class OnnxRotaryEmbedding_scale(OnnxTestCase):
    # ONNX Runtime ignores `scale`.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("input", torch.float32, [2, 3, 8]),
            ("position_ids", torch.int64, [2, 3], (0, 8)),
            ("cos_cache", torch.float32, [8, 2]),
            ("sin_cache", torch.float32, [8, 2]),
        ]
    )
    def graph(self):
        return [_rotary_embedding_node(scale=2.0)], ["output"]
