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


def _gqa_node(inputs, outputs=_GQA_OUTPUTS, **attributes):
    return onnx.helper.make_node(
        "GroupQueryAttention",
        inputs,
        outputs,
        domain="com.microsoft",
        num_heads=_NUM_HEADS,
        kv_num_heads=_KV_NUM_HEADS,
        **attributes,
    )


def _length_initializers(seqlens_k, total):
    # seqlens_k is the total length - 1 of each batch entry, and
    # total_sequence_length is the batch maximum.
    return [
        onnx.numpy_helper.from_array(
            np.array(seqlens_k, dtype=np.int32), name="seqlens_k"
        ),
        onnx.numpy_helper.from_array(
            np.array([total], dtype=np.int32), name="total_seq_len"
        ),
    ]


def _seq_initializers(past_len, seq_len):
    total = past_len + seq_len
    return _length_initializers([total - 1], total)


def _empty_past_initializers(batch=1):
    empty = np.zeros((batch, _KV_NUM_HEADS, 0, _HEAD_SIZE), dtype=np.float32)
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


@register_onnx_test
class OnnxGroupQueryAttention_fixedCapacityDecode(OnnxTestCase):
    # Decoding into a fixed-capacity cache: past and present keep 8 rows, rows
    # 0-3 hold tokens and the new token goes to row 4. ONNX Runtime zeroes the
    # rows that hold no token.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, 1, _PACKED_HIDDEN]),
            ("past_key", torch.float32, [1, _KV_NUM_HEADS, 8, _HEAD_SIZE]),
            ("past_value", torch.float32, [1, _KV_NUM_HEADS, 8, _HEAD_SIZE]),
            ("cos_cache", torch.float32, [8, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [8, _HEAD_SIZE // 2]),
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
        return [node], _GQA_OUTPUTS, _length_initializers([4], 5)


@register_onnx_test
class OnnxGroupQueryAttention_rightPaddedPrompt(OnnxTestCase):
    # A batched first prompt whose second entry has 2 tokens followed by 2
    # padding tokens. ONNX Runtime starts both entries at position 0.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("query", torch.float32, [2, _PROMPT_LEN, _Q_HIDDEN]),
            ("key", torch.float32, [2, _PROMPT_LEN, _KV_HIDDEN]),
            ("value", torch.float32, [2, _PROMPT_LEN, _KV_HIDDEN]),
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
        initializers = _empty_past_initializers(batch=2) + _length_initializers(
            [3, 1], _PROMPT_LEN
        )
        return [node], _GQA_OUTPUTS, initializers


@register_onnx_test
class OnnxGroupQueryAttention_slidingWindowCache(OnnxTestCase):
    # A sliding-window cache of capacity 6 with a window of 4 holds positions
    # 0-5. Three new tokens make 9, so the cache drops its 3 oldest rows as one
    # block and keeps positions 3-8. attention_bias columns are absolute
    # positions, so the cache reads them from position 0 of its first row.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, 3, _PACKED_HIDDEN]),
            ("past_key", torch.float32, [1, _KV_NUM_HEADS, 6, _HEAD_SIZE]),
            ("past_value", torch.float32, [1, _KV_NUM_HEADS, 6, _HEAD_SIZE]),
            ("cos_cache", torch.float32, [16, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [16, _HEAD_SIZE // 2]),
            ("attention_bias", torch.float32, [1, _NUM_HEADS, 3, 9], (-1.0, 1.0)),
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
                "attention_bias",
            ],
            do_rotary=1,
            local_window_size=4,
            sliding_window_cache=1,
        )
        return [node], _GQA_OUTPUTS, _length_initializers([8], 9)


def _kv_scale_initializers(per_channel, k_scale, v_scale):
    shape = (_KV_NUM_HEADS * _HEAD_SIZE,) if per_channel else (1,)
    return [
        onnx.numpy_helper.from_array(
            np.full(shape, k_scale, dtype=np.float32), name="k_scale"
        ),
        onnx.numpy_helper.from_array(
            np.linspace(v_scale, 2 * v_scale, num=shape[0], dtype=np.float32),
            name="v_scale",
        ),
    ]


_QUANTIZED_INPUTS = [
    "packed_qkv",
    "",  # key (omitted)
    "",  # value (omitted)
    "past_key",
    "past_value",
    "seqlens_k",
    "total_seq_len",
    "",  # cos_cache (omitted)
    "",  # sin_cache (omitted)
    "",  # position_ids (omitted)
    "",  # attention_bias (omitted)
    "",  # head_sink (omitted)
    "k_scale",
    "v_scale",
]


@register_onnx_test
class OnnxGroupQueryAttention_int8KvCache(OnnxTestCase):
    # A per-tensor int8 cache. The new rows are quantized into the present
    # cache, some of them saturating, and attention reads the dequantized
    # cache.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, 1, _PACKED_HIDDEN], (-1.0, 1.0)),
            ("past_key", torch.int8, [1, _KV_NUM_HEADS, 3, _HEAD_SIZE], (-128, 128)),
            ("past_value", torch.int8, [1, _KV_NUM_HEADS, 3, _HEAD_SIZE], (-128, 128)),
        ]
    )
    def graph(self):
        node = _gqa_node(
            _QUANTIZED_INPUTS,
            k_quant_type="PER_TENSOR",
            v_quant_type="PER_TENSOR",
            kv_cache_bit_width=8,
        )
        initializers = _kv_scale_initializers(False, 0.005, 0.01)
        return [node], _GQA_OUTPUTS, initializers + _length_initializers([3], 4)


@register_onnx_test
class OnnxGroupQueryAttention_int4KvCache(OnnxTestCase):
    # A per-channel int4 cache stores two values per byte, low nibble first,
    # biased by 8. ONNX Runtime reads the quantization types in any case.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, 1, _PACKED_HIDDEN], (-1.0, 1.0)),
            (
                "past_key",
                torch.uint8,
                [1, _KV_NUM_HEADS, 3, _HEAD_SIZE // 2],
                (0, 256),
            ),
            (
                "past_value",
                torch.uint8,
                [1, _KV_NUM_HEADS, 3, _HEAD_SIZE // 2],
                (0, 256),
            ),
        ]
    )
    def graph(self):
        node = _gqa_node(
            _QUANTIZED_INPUTS,
            k_quant_type="per_channel",
            v_quant_type="per_channel",
            kv_cache_bit_width=4,
        )
        initializers = _kv_scale_initializers(True, 0.1, 0.1)
        return [node], _GQA_OUTPUTS, initializers + _length_initializers([3], 4)


@register_onnx_test
class OnnxGroupQueryAttention_scoreFeatures(OnnxTestCase):
    # softcap, a per-head softmax sink and output_qk need the attention scores,
    # so the lowering computes them instead of using
    # scaled_dot_product_attention. An attention bias and a local window of 2
    # apply on top.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("packed_qkv", torch.float32, [1, _PROMPT_LEN, _PACKED_HIDDEN]),
            (
                "attention_bias",
                torch.float32,
                [1, _NUM_HEADS, _PROMPT_LEN, _PROMPT_LEN],
                (-1.0, 1.0),
            ),
            ("head_sink", torch.float32, [_NUM_HEADS], (-1.0, 1.0)),
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
                "",  # cos_cache (omitted)
                "",  # sin_cache (omitted)
                "",  # position_ids (omitted)
                "attention_bias",
                "head_sink",
            ],
            _GQA_OUTPUTS + ["output_qk"],
            softcap=2.0,
            local_window_size=2,
            qk_output=1,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS + ["output_qk"], initializers


@register_onnx_test
class OnnxGroupQueryAttention_bidirectionalPositionIds(OnnxTestCase):
    # causal 0 with a custom scale, and position_ids, of which a first prompt
    # reads only the first element.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("query", torch.float32, [1, _PROMPT_LEN, _Q_HIDDEN]),
            ("key", torch.float32, [1, _PROMPT_LEN, _KV_HIDDEN]),
            ("value", torch.float32, [1, _PROMPT_LEN, _KV_HIDDEN]),
            ("cos_cache", torch.float32, [16, _HEAD_SIZE // 2]),
            ("sin_cache", torch.float32, [16, _HEAD_SIZE // 2]),
            ("position_ids", torch.int64, [1, _PROMPT_LEN], (0, 8)),
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
                "position_ids",
            ],
            do_rotary=1,
            causal=0,
            scale=0.3,
        )
        initializers = _empty_past_initializers() + _seq_initializers(0, _PROMPT_LEN)
        return [node], _GQA_OUTPUTS, initializers


@register_onnx_test
class OnnxGroupQueryAttention_noPastOutputOnly(OnnxTestCase):
    # Without a past cache and with only the attention output.
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
                "",  # past_key (omitted)
                "",  # past_value (omitted)
                "seqlens_k",
                "total_seq_len",
            ],
            ["output"],
        )
        return [node], ["output"], _seq_initializers(0, _PROMPT_LEN)


@register_onnx_test
class OnnxGroupQueryAttention_sharedKvDecode(OnnxTestCase):
    # An empty key and value reuse the past as a shared KV cache: the past
    # buffer of 8 rows holds all 5 tokens, and the query adds none.
    extra_opsets = {"com.microsoft": 1}

    @annotate_inputs(
        [
            ("query", torch.float32, [1, 1, _Q_HIDDEN]),
            ("past_key", torch.float32, [1, _KV_NUM_HEADS, 8, _HEAD_SIZE]),
            ("past_value", torch.float32, [1, _KV_NUM_HEADS, 8, _HEAD_SIZE]),
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
            ]
        )
        empty = np.zeros((1, 0, _KV_HIDDEN), dtype=np.float32)
        initializers = [
            onnx.numpy_helper.from_array(empty, name="key"),
            onnx.numpy_helper.from_array(empty.copy(), name="value"),
        ]
        return [node], _GQA_OUTPUTS, initializers + _length_initializers([4], 5)
