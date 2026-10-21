# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native tests for quantized ops in PT2E QDQ form."""

import numpy as np
import onnx.helper
import onnx.numpy_helper
import torch

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test


# ==============================================================================


class _MatMulIntegerToFloat(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}
    optional_inputs = ()
    a_scale = np.array(0.25, dtype=np.float32)
    b_scale = np.array(0.125, dtype=np.float32)
    a_zero_point = np.array(17, dtype=np.int8)
    b_zero_point = np.array(-11, dtype=np.int8)
    bias = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32)
    b = np.arange(-7, 8, dtype=np.int8).reshape(3, 5)

    def graph(self):
        inputs = ["a", "b", "a_scale", "b_scale", *self.optional_inputs]
        runtime_inputs = {annotation[0] for annotation in self._input_annotations()}
        # The same graph builder covers initializer and runtime qparams/weights.
        # Empty names keep omitted zero-point slots ahead of bias in place.
        initializers = [
            onnx.numpy_helper.from_array(getattr(self, name), name=name)
            for name in inputs
            if name and name not in runtime_inputs
        ]
        node = onnx.helper.make_node(
            "MatMulIntegerToFloat", inputs, ["y"], domain="com.microsoft"
        )
        return [node], ["y"], initializers


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_missingOptional(_MatMulIntegerToFloat):
    @annotate_inputs(
        [
            ("a", torch.int8, [4, 3], (-128, 128)),
            ("b", torch.int8, [3, 5], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_aZeroPoint(_MatMulIntegerToFloat):
    optional_inputs = ("a_zero_point",)
    a_zero_point = np.array(200, dtype=np.uint8)

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            ("b", torch.uint8, [3, 5], (128, 256)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_bZeroPoint(_MatMulIntegerToFloat):
    optional_inputs = ("", "b_zero_point")

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            # Setting RHS with reduced range to avoid saturation issues on x86-64 machines (with AVX2/AVX512).
            # ONNX Runtime uses the VPMADDUBSW instruction for U8S8 for performance but that may cause the input to be 16-bit clamped (saturated).
            # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
            # for more details.
            ("b", torch.int8, [3, 5], (-64, 64)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_runtimeQParams(_MatMulIntegerToFloat):
    optional_inputs = ("a_zero_point", "b_zero_point")

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            # Setting RHS with reduced range to avoid saturation issues on x86-64 machines (with AVX2/AVX512).
            # ONNX Runtime uses the VPMADDUBSW instruction for U8S8 for performance but that may cause the input to be 16-bit clamped (saturated).
            # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
            # for more details.
            ("b", torch.int8, [3, 5], (-64, 64)),
            ("a_scale", torch.float32, [], (0.125, 0.5)),
            ("b_scale", torch.float32, [1], (0.125, 0.5)),
            ("a_zero_point", torch.uint8, [], (128, 256)),
            ("b_zero_point", torch.int8, [1], (-16, 16)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_bias(_MatMulIntegerToFloat):
    optional_inputs = ("", "", "bias")

    @annotate_inputs(
        [
            ("a", torch.int8, [4, 3], (-128, 128)),
            ("b", torch.int8, [3, 5], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_perColumn(_MatMulIntegerToFloat):
    optional_inputs = ("a_zero_point", "b_zero_point")
    b_scale = np.array([0.5, 0.125, 2.0, 0.25, 1.0], dtype=np.float32)
    # ONNX Runtime supports per-column qparams for B and scalar qparams for A.
    a_zero_point = np.array(200, dtype=np.uint8)
    b_zero_point = np.array([-128, -3, 0, 7, 127], dtype=np.int8)

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            # Setting RHS with reduced range to avoid saturation issues on x86-64 machines (with AVX2/AVX512).
            # ONNX Runtime uses the VPMADDUBSW instruction for U8S8 for performance but that may cause the input to be 16-bit clamped (saturated).
            # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
            # for more details.
            ("b", torch.int8, [3, 5], (-64, 64)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_mixedQParams(_MatMulIntegerToFloat):
    # B broadcasts its scalar scale alongside per-column zero points.
    optional_inputs = ("a_zero_point", "b_zero_point")
    b_zero_point = np.array([-128, -3, 0, 7, 127], dtype=np.int8)

    @annotate_inputs(
        [
            ("a", torch.int8, [4, 3], (-128, 128)),
            ("b", torch.int8, [3, 5], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_batchedWeights(_MatMulIntegerToFloat):
    # A [2,4,3] @ B [3,5] -> Y [2,4,5], with B shared across batches.
    # B's vector scale and omitted zero point exercise per-channel DQ defaults.
    b_scale = np.array([0.5, 0.125, 2.0, 0.25, 1.0], dtype=np.float32)

    @annotate_inputs([("a", torch.int8, [2, 4, 3], (-128, 128))])
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxMatMulIntegerToFloat_dynamicColumns(_MatMulIntegerToFloat):
    # The harness materializes dynamic dimensions as 2. Qparam lengths and
    # matrix column counts are unknown to the compiler and agree at runtime.
    optional_inputs = ("a_zero_point", "b_zero_point")

    @annotate_inputs(
        [
            ("a", torch.int8, [-1, -1], (-128, 128)),
            ("b", torch.int8, [-1, -1], (-128, 128)),
            ("a_scale", torch.float32, [], (0.125, 1.0)),
            ("b_scale", torch.float32, [-1], (0.125, 1.0)),
            ("a_zero_point", torch.int8, [], (-16, 16)),
            ("b_zero_point", torch.int8, [-1], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================


@register_onnx_test
class OnnxConvInteger_basic(OnnxTestCase):
    opset = 10

    def __init__(self):
        rng = np.random.default_rng(42)
        self._w = rng.integers(0, 10, size=(1, 1, 2, 2), dtype=np.int8)
        self._x_zp = np.array(1, dtype=np.int8)
        self._w_zp = np.array(0, dtype=np.int8)

    @annotate_inputs([("x", torch.int8, [1, 1, 3, 3], (0, 20))])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvInteger",
            ["x", "w", "x_zero_point", "w_zero_point"],
            ["y"],
        )
        initializers = [
            onnx.numpy_helper.from_array(self._w, name="w"),
            onnx.numpy_helper.from_array(self._x_zp, name="x_zero_point"),
            onnx.numpy_helper.from_array(self._w_zp, name="w_zero_point"),
        ]
        return [node], ["y"], initializers


# ==============================================================================


@register_onnx_test
class OnnxConvInteger_noWeightZp(OnnxTestCase):
    opset = 10

    def __init__(self):
        rng = np.random.default_rng(43)
        self._w = rng.integers(0, 10, size=(1, 1, 2, 2), dtype=np.int8)
        self._x_zp = np.array(2, dtype=np.int8)

    @annotate_inputs([("x", torch.int8, [1, 1, 3, 3], (0, 20))])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvInteger",
            ["x", "w", "x_zero_point"],
            ["y"],
        )
        initializers = [
            onnx.numpy_helper.from_array(self._w, name="w"),
            onnx.numpy_helper.from_array(self._x_zp, name="x_zero_point"),
        ]
        return [node], ["y"], initializers


# ==============================================================================


@register_onnx_test
class OnnxConvInteger_withPadding(OnnxTestCase):
    opset = 10

    def __init__(self):
        rng = np.random.default_rng(44)
        self._w = rng.integers(0, 10, size=(1, 1, 2, 2), dtype=np.int8)
        self._x_zp = np.array(0, dtype=np.int8)
        self._w_zp = np.array(0, dtype=np.int8)

    @annotate_inputs([("x", torch.int8, [1, 1, 3, 3], (0, 20))])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvInteger",
            ["x", "w", "x_zero_point", "w_zero_point"],
            ["y"],
            pads=[1, 1, 1, 1],
        )
        initializers = [
            onnx.numpy_helper.from_array(self._w, name="w"),
            onnx.numpy_helper.from_array(self._x_zp, name="x_zero_point"),
            onnx.numpy_helper.from_array(self._w_zp, name="w_zero_point"),
        ]
        return [node], ["y"], initializers


# ==============================================================================


@register_onnx_test
class OnnxConvInteger_perChannelWeightZp(OnnxTestCase):
    opset = 10

    def __init__(self):
        rng = np.random.default_rng(45)
        self._w = rng.integers(0, 10, size=(3, 2, 2, 2), dtype=np.int8)
        self._x_zp = np.array(1, dtype=np.int8)
        self._w_zp = rng.integers(0, 5, size=(3,), dtype=np.int8)

    @annotate_inputs([("x", torch.int8, [1, 2, 3, 3], (0, 20))])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvInteger",
            ["x", "w", "x_zero_point", "w_zero_point"],
            ["y"],
        )
        initializers = [
            onnx.numpy_helper.from_array(self._w, name="w"),
            onnx.numpy_helper.from_array(self._x_zp, name="x_zero_point"),
            onnx.numpy_helper.from_array(self._w_zp, name="w_zero_point"),
        ]
        return [node], ["y"], initializers
