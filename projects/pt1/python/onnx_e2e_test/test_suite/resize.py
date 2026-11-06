# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Resize ONNX op tests."""

import numpy as np
import torch
import onnx.helper
import onnx.numpy_helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# Resize, mode=nearest, coordinate_transformation_mode=asymmetric,
# nearest_mode=floor
#
# This is what the PyTorch ONNX exporter emits for
# F.interpolate(mode="nearest"). The source index is floor(dst * in / out), so
# an output position whose source coordinate ends in .5 must read the lower
# input element. x is the only runtime input; scales or sizes is baked in as an
# initializer.
# ==============================================================================


class _ResizeNearestFloor(OnnxTestCase):
    # Name of the Resize input slot (scales or sizes) and its initializer value.
    resize_input = None
    resize_value = None

    def graph(self):
        node_inputs = ["x", ""]
        if self.resize_input == "sizes":
            node_inputs.append("")
        node_inputs.append(self.resize_input)
        node = onnx.helper.make_node(
            "Resize",
            node_inputs,
            ["y"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        initializers = [
            onnx.numpy_helper.from_array(self.resize_value, name=self.resize_input)
        ]
        return [node], ["y"], initializers


@register_onnx_test
class OnnxResize_nearestAsymmetricFloor_scales(_ResizeNearestFloor):
    # 2x upsampling maps every odd output row and column to a .5 coordinate.
    resize_input = "scales"
    resize_value = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    @annotate_inputs([("x", torch.float32, [1, 2, 4, 5])])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxResize_nearestAsymmetricFloor_sizes(_ResizeNearestFloor):
    # Upsampling 4x5 to 6x7 gives ratios that are not integers.
    resize_input = "sizes"
    resize_value = np.array([1, 2, 6, 7], dtype=np.int64)

    @annotate_inputs([("x", torch.float32, [1, 2, 4, 5])])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxResize_nearestAsymmetricFloor_downsample(_ResizeNearestFloor):
    # Downsampling 6x6 to 4x4 maps output index 1 to source coordinate 1.5.
    resize_input = "sizes"
    resize_value = np.array([1, 2, 4, 4], dtype=np.int64)

    @annotate_inputs([("x", torch.float32, [1, 2, 6, 6])])
    def graph(self):
        return super().graph()
