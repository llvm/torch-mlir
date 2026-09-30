# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Elementwise ONNX op tests."""

import torch
import onnx.helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================


@register_onnx_test
class OnnxAdd_f32_basic(OnnxTestCase):
    @annotate_inputs(
        [
            ("x", torch.float32, [4, 4]),
            ("y", torch.float32, [4, 4]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node("Add", ["x", "y"], ["z"])
        return [node], ["z"]


# ==============================================================================


@register_onnx_test
class OnnxRelu_f32_basic(OnnxTestCase):
    @annotate_inputs([("x", torch.float32, [3, 4])])
    def graph(self):
        node = onnx.helper.make_node("Relu", ["x"], ["y"])
        return [node], ["y"]


# ==============================================================================


@register_onnx_test
class OnnxRelu_negrange_basic(OnnxTestCase):
    # Sample from [-1, 1) so Relu sees negative inputs to clamp.
    @annotate_inputs([("x", torch.float32, [3, 4], (-1.0, 1.0))])
    def graph(self):
        node = onnx.helper.make_node("Relu", ["x"], ["y"])
        return [node], ["y"]


# ==============================================================================


@register_onnx_test
class OnnxAdd_i64_basic(OnnxTestCase):
    @annotate_inputs(
        [
            ("x", torch.int64, [4, 4]),
            ("y", torch.int64, [4, 4]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node("Add", ["x", "y"], ["z"])
        return [node], ["z"]


# ==============================================================================


@register_onnx_test
class OnnxSplit_f32_twoOutputs(OnnxTestCase):
    # Split along axis 0 into two equal halves -> two graph outputs, exercising
    # the multi-output trace/compare path.
    @annotate_inputs([("x", torch.float32, [4, 4])])
    def graph(self):
        node = onnx.helper.make_node("Split", ["x"], ["a", "b"], axis=0)
        return [node], ["a", "b"]


# ==============================================================================


@register_onnx_test
class OnnxAnd_bool_basic(OnnxTestCase):
    # Bool inputs exercise the boolean branch of materialize_inputs (the only
    # input-generation path that samples with rng.integers(0, 2) and ignores
    # any value_range).
    @annotate_inputs(
        [
            ("x", torch.bool, [4, 4]),
            ("y", torch.bool, [4, 4]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node("And", ["x", "y"], ["z"])
        return [node], ["z"]


# ==============================================================================


@register_onnx_test
class OnnxAdd_i32_customRange(OnnxTestCase):
    # Integer input with an explicit (low, high) range exercises the custom
    # value_range branch for integers in materialize_inputs (elsewhere only
    # floats carry a custom range).
    @annotate_inputs(
        [
            ("x", torch.int32, [4, 4], (-5, 5)),
            ("y", torch.int32, [4, 4], (-5, 5)),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node("Add", ["x", "y"], ["z"])
        return [node], ["z"]


# ==============================================================================


@register_onnx_test
class OnnxAdd_f32_dynamicInput(OnnxTestCase):
    # A -1 dim marks a dynamic INPUT: it imports as `?` and materialize_inputs
    # substitutes a concrete size at runtime for both golden and SUT.  The
    # output value_info is still concrete (pass-2 infers it from the probe run),
    # so this exercises dynamic-input handling in the lowering, not a dynamic
    # result dim.
    @annotate_inputs(
        [
            ("x", torch.float32, [-1, 4]),
            ("y", torch.float32, [-1, 4]),
        ]
    )
    def graph(self):
        node = onnx.helper.make_node("Add", ["x", "y"], ["z"])
        return [node], ["z"]
