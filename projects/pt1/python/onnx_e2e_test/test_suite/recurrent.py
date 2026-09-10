# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Recurrent ONNX op tests (LSTM)."""

import numpy as np
import torch
import onnx.helper
import onnx.numpy_helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# LSTM — single-direction; seq=5, batch=1, input_size=4, hidden_size=8.
# X is the only runtime input; W/R/B are baked in as initializers.
#   Outputs: Y [seq, num_dir, batch, H] and Y_h [num_dir, batch, H].
# ==============================================================================


@register_onnx_test
class OnnxLSTM_forward_basic(OnnxTestCase):
    _SEQ, _BATCH, _INPUT, _HIDDEN = 5, 1, 4, 8
    _NUM_DIR = 1

    def __init__(self):
        rng = np.random.default_rng(7)
        self._W = rng.standard_normal(
            (self._NUM_DIR, 4 * self._HIDDEN, self._INPUT)
        ).astype(np.float32)
        self._R = rng.standard_normal(
            (self._NUM_DIR, 4 * self._HIDDEN, self._HIDDEN)
        ).astype(np.float32)
        self._B = rng.standard_normal((self._NUM_DIR, 8 * self._HIDDEN)).astype(
            np.float32
        )

    @annotate_inputs([("X", torch.float32, [_SEQ, _BATCH, _INPUT])])
    def graph(self):
        node = onnx.helper.make_node(
            "LSTM",
            ["X", "W", "R", "B"],
            ["Y", "Y_h"],
            hidden_size=self._HIDDEN,
            direction="forward",
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="W"),
            onnx.numpy_helper.from_array(self._R, name="R"),
            onnx.numpy_helper.from_array(self._B, name="B"),
        ]
        return [node], ["Y", "Y_h"], initializers
