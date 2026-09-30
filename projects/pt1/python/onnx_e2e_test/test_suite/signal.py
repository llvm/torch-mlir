# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Signal-processing ONNX op tests (STFT)."""

import numpy as np
import torch
import onnx.helper
import onnx.numpy_helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# STFT — opset 17
#
# signal is the only runtime input; frame_step, window and frame_length are
# baked in as initializers.
#   signal_length = 64, frame_step = 8, window_length = 16
#   frames = (64 - 16) // 8 + 1 = 7 ; freq_bins = 16 // 2 + 1 = 9
# ==============================================================================


@register_onnx_test
class OnnxSTFT_onesided_basic(OnnxTestCase):
    _SIGNAL_LEN = 64
    _FRAME_STEP = 8
    _WIN_LEN = 16
    _BATCH = 1

    def __init__(self):
        self._window = np.hanning(self._WIN_LEN).astype(np.float32)
        self._frame_step_arr = np.array(self._FRAME_STEP, dtype=np.int64)
        self._frame_length_arr = np.array(self._WIN_LEN, dtype=np.int64)

    @annotate_inputs([("signal", torch.float32, [_BATCH, _SIGNAL_LEN, 1])])
    def graph(self):
        node = onnx.helper.make_node(
            "STFT",
            ["signal", "frame_step", "window", "frame_length"],
            ["output"],
            onesided=1,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._frame_step_arr, name="frame_step"),
            onnx.numpy_helper.from_array(self._window, name="window"),
            onnx.numpy_helper.from_array(self._frame_length_arr, name="frame_length"),
        ]
        return [node], ["output"], initializers
