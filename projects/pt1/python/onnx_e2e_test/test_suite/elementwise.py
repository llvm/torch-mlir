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


# ==============================================================================


class _FastGelu(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}
    optional_inputs = ()

    def graph(self):
        node = onnx.helper.make_node(
            "FastGelu", ["x", *self.optional_inputs], ["y"], domain="com.microsoft"
        )
        return [node], ["y"]


@register_onnx_test
class OnnxFastGelu_f32_basic(_FastGelu):
    @annotate_inputs([("x", torch.float32, [3, 4], (-3.0, 3.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f32_noneBias(_FastGelu):
    optional_inputs = ("",)

    @annotate_inputs([("x", torch.float32, [3, 4], (-3.0, 3.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f32_bias(_FastGelu):
    optional_inputs = ("bias",)

    @annotate_inputs(
        [
            ("x", torch.float32, [2, 3, 4], (-3.0, 3.0)),
            ("bias", torch.float32, [4], (-1.0, 1.0)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f32_dynamicBias(_FastGelu):
    optional_inputs = ("bias",)
    # Exercise dynamic broadcasting away from tanh saturation, where small
    # differences between CPU approximations exceed the harness tolerances.

    @annotate_inputs(
        [
            ("x", torch.float32, [-1, 4], (-2.0, 2.0)),
            ("bias", torch.float32, [4], (-1.0, 1.0)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f64_basic(_FastGelu):
    @annotate_inputs([("x", torch.float64, [3, 4], (-3.0, 3.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f16_basic(_FastGelu):
    @annotate_inputs([("x", torch.float16, [8, 8], (-4.0, 4.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f16_bias(_FastGelu):
    optional_inputs = ("bias",)

    @annotate_inputs(
        [
            ("x", torch.float16, [2, 3, 4], (-3.0, 3.0)),
            ("bias", torch.float16, [4], (-1.0, 1.0)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxFastGelu_f16_rounding(_FastGelu):
    # At x = -3.09765625, f16 intermediates lose accuracy in the negative
    # tail. FP32 computation followed by a single f16 rounding gives
    # -0.00264739990234375.
    @annotate_inputs([("x", torch.float16, [1], (-3.09765625, -3.09765625))])
    def graph(self):
        return super().graph()
