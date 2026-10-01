# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Pooling ONNX op tests."""

import torch
import onnx.helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================


@register_onnx_test
class OnnxAveragePool2d_countIncludePad_asymmetricPadsDilated(OnnxTestCase):
    # AveragePool gained `dilations` in opset 19.
    opset = 19

    # pads = [h_begin=1, w_begin=0, h_end=2, w_end=1], so the trailing window
    # in each dim has a tap in the end padding, which count_include_pad=1 must
    # count. Dilation is what makes the TorchToLinalg lowering compute the
    # divisor per window instead of using the kernel volume.
    @annotate_inputs([("x", torch.float32, [1, 2, 6, 6])])
    def graph(self):
        node = onnx.helper.make_node(
            "AveragePool",
            ["x"],
            ["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            dilations=[2, 2],
            pads=[1, 0, 2, 1],
            count_include_pad=1,
        )
        return [node], ["y"]
