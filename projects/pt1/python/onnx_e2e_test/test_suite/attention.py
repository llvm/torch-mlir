# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native tests for Microsoft-domain attention ops."""

import numpy as np
import onnx.helper
import onnx.numpy_helper
import torch

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# com.microsoft.GroupQueryAttention
#
# Inputs (positional): query, key, value, past_key, past_value, seqlens_k,
# total_sequence_length, cos_cache, sin_cache. With packed QKV, key and value
# are empty input names and query holds Q, K and V concatenated along the
# hidden dimension. ONNX Runtime detects packed QKV by checking whether key is
# absent; these tests compare the torch-mlir lowering against that behavior.
#
# Shapes: batch=1, num_heads=4, kv_num_heads=2, head_size=16, so the packed
# hidden size is 4*16 + 2*2*16 = 128. head_size must be a multiple of 16
# because the ONNX Runtime CPU kernel rejects other sizes when do_rotary is
# set. seqlens_k and total_sequence_length are initializers because they must
# be consistent with the past KV length.
# ==============================================================================

_NUM_HEADS = 4
_KV_NUM_HEADS = 2
_HEAD_SIZE = 16
_Q_HIDDEN = _NUM_HEADS * _HEAD_SIZE
_KV_HIDDEN = _KV_NUM_HEADS * _HEAD_SIZE
_PACKED_HIDDEN = _Q_HIDDEN + 2 * _KV_HIDDEN
_PROMPT_LEN = 4
_GQA_OUTPUTS = ["output", "present_key", "present_value"]


def _gqa_node(inputs, do_rotary):
    return onnx.helper.make_node(
        "GroupQueryAttention",
        inputs,
        _GQA_OUTPUTS,
        domain="com.microsoft",
        num_heads=_NUM_HEADS,
        kv_num_heads=_KV_NUM_HEADS,
        do_rotary=do_rotary,
    )


def _seq_initializers(past_len, seq_len):
    # seqlens_k is total_sequence_length - 1 per batch entry.
    total = past_len + seq_len
    return [
        onnx.numpy_helper.from_array(
            np.array([total - 1], dtype=np.int32), name="seqlens_k"
        ),
        onnx.numpy_helper.from_array(
            np.array([total], dtype=np.int32), name="total_seq_len"
        ),
    ]


def _empty_past_initializers():
    empty = np.zeros((1, _KV_NUM_HEADS, 0, _HEAD_SIZE), dtype=np.float32)
    return [
        onnx.numpy_helper.from_array(empty, name="past_key"),
        onnx.numpy_helper.from_array(empty.copy(), name="past_value"),
    ]


@register_onnx_test
class OnnxGroupQueryAttention_packedQkvRotary(OnnxTestCase):
    # Packed QKV with empty key/value and rotary embedding (9 positional
    # inputs). The prompt has several tokens so that attention mixes different
    # keys and the query and its rotary affect the compared output.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, _PROMPT_LEN, _PACKED_HIDDEN]),
            ("cos_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
        ]
    )
    def graph(self):
        node = _gqa_node(
            [
                "packed_qkv",
                "",  # key (omitted)
                "",  # value (omitted)
                "past_key",
                "past_value",
                "seqlens_k",
                "total_seq_len",
                "cos_cache",
                "sin_cache",
            ],
            do_rotary=1,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS, initializers


@register_onnx_test
class OnnxGroupQueryAttention_packedQkvTrailingNone(OnnxTestCase):
    # Packed QKV with rotary and the 12-input layout that the
    # onnxruntime-genai model builder emits since v0.13.0: empty key, value,
    # position_ids, attention_bias and head_sink.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, _PROMPT_LEN, _PACKED_HIDDEN]),
            ("cos_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
        ]
    )
    def graph(self):
        node = _gqa_node(
            [
                "packed_qkv",
                "",  # key (omitted)
                "",  # value (omitted)
                "past_key",
                "past_value",
                "seqlens_k",
                "total_seq_len",
                "cos_cache",
                "sin_cache",
                "",  # position_ids (omitted)
                "",  # attention_bias (omitted)
                "",  # head_sink (omitted)
            ],
            do_rotary=1,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS, initializers


@register_onnx_test
class OnnxGroupQueryAttention_packedQkvNoRotary(OnnxTestCase):
    # Packed QKV with empty key/value and no rotary (7 positional inputs).
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, _PROMPT_LEN, _PACKED_HIDDEN]),
        ]
    )
    def graph(self):
        node = _gqa_node(
            [
                "packed_qkv",
                "",  # key (omitted)
                "",  # value (omitted)
                "past_key",
                "past_value",
                "seqlens_k",
                "total_seq_len",
            ],
            do_rotary=0,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS, initializers


@register_onnx_test
class OnnxGroupQueryAttention_packedQkvPastKv(OnnxTestCase):
    # Packed QKV decoding one token on top of a non-empty past KV cache, so
    # the KV-cache scatter and causal mask are compared against ONNX Runtime.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, 1, _PACKED_HIDDEN]),
            ("past_key", torch.float32, [1, _KV_NUM_HEADS, 3, _HEAD_SIZE]),
            ("past_value", torch.float32, [1, _KV_NUM_HEADS, 3, _HEAD_SIZE]),
        ]
    )
    def graph(self):
        node = _gqa_node(
            [
                "packed_qkv",
                "",  # key (omitted)
                "",  # value (omitted)
                "past_key",
                "past_value",
                "seqlens_k",
                "total_seq_len",
            ],
            do_rotary=0,
        )
        return [node], _GQA_OUTPUTS, _seq_initializers(3, 1)


@register_onnx_test
class OnnxGroupQueryAttention_separateQkvRotary(OnnxTestCase):
    # Baseline: separate Q, K and V inputs with rotary embedding.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("query", torch.float32, [1, _PROMPT_LEN, _Q_HIDDEN]),
            ("key", torch.float32, [1, _PROMPT_LEN, _KV_HIDDEN]),
            ("value", torch.float32, [1, _PROMPT_LEN, _KV_HIDDEN]),
            ("cos_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [_PROMPT_LEN, _HEAD_SIZE // 2]),
        ]
    )
    def graph(self):
        node = _gqa_node(
            [
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_seq_len",
                "cos_cache",
                "sin_cache",
            ],
            do_rotary=1,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS, initializers
