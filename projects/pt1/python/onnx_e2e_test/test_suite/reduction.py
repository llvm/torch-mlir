# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Reduction ONNX op tests (HardMax)."""

import torch
import onnx.helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# HardMax — opset 13+
#
# HardMax(x, axis=1) returns a one-hot tensor where 1.0 marks the argmax
# position along the given axis.
# ==============================================================================


@register_onnx_test
class OnnxHardmax_axis1_basic(OnnxTestCase):
    @annotate_inputs([("x", torch.float32, [3, 5])])
    def graph(self):
        node = onnx.helper.make_node("Hardmax", ["x"], ["y"], axis=1)
        return [node], ["y"]
