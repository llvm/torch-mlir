# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""ONNX-native tests for Microsoft-domain quantized ops in PT2E QDQ form."""

import numpy as np
import onnx.helper
import onnx.numpy_helper
import torch

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# Helpers
# ==============================================================================

_EXTRA_OPSETS = {"com.microsoft": 1}

_TORCH_TO_NP_DTYPE = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
}

_QUANT_RANGE = {
    torch.uint8: (0, 256),
    torch.int8: (-128, 128),
}

def _zp_initializer(name: str, value: int, dtype: torch.dtype):
    np_dtype = _TORCH_TO_NP_DTYPE[dtype]
    return onnx.numpy_helper.from_array(np_dtype(value), name=name)

def _scale_initializer(name: str, value: float):
    return onnx.numpy_helper.from_array(np.float32(value), name=name)


# ==============================================================================
# MatMulIntegerToFloat
# ==============================================================================


class _MatMulIntegerToFloat(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
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


@register_onnx_test
class OnnxMatMulIntegerToFloat_bZeroPoint(_MatMulIntegerToFloat):
    optional_inputs = ("", "b_zero_point")

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            ("b", torch.int8, [3, 5], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulIntegerToFloat_runtimeQParams(_MatMulIntegerToFloat):
    optional_inputs = ("a_zero_point", "b_zero_point")

    @annotate_inputs(
        [
            ("a", torch.uint8, [4, 3], (128, 256)),
            ("b", torch.int8, [3, 5], (-128, 128)),
            ("a_scale", torch.float32, [], (0.125, 0.5)),
            ("b_scale", torch.float32, [1], (0.125, 0.5)),
            ("a_zero_point", torch.uint8, [], (128, 256)),
            ("b_zero_point", torch.int8, [1], (-16, 16)),
        ]
    )
    def graph(self):
        return super().graph()


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
            ("b", torch.int8, [3, 5], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


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


@register_onnx_test
class OnnxMatMulIntegerToFloat_batchedWeights(_MatMulIntegerToFloat):
    # A [2,4,3] @ B [3,5] -> Y [2,4,5], with B shared across batches.
    # B's vector scale and omitted zero point exercise per-channel DQ defaults.
    b_scale = np.array([0.5, 0.125, 2.0, 0.25, 1.0], dtype=np.float32)

    @annotate_inputs([("a", torch.int8, [2, 4, 3], (-128, 128))])
    def graph(self):
        return super().graph()


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
# QLinearAdd
# ==============================================================================


class _QLinearAdd(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [2, 3]
    a_scale, a_zp = 0.037, 5
    b_scale, b_zp = 0.029, 9
    c_scale, c_zp = 0.041, 3

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearAdd",
            ["A", "a_scale", "a_zp", "B", "b_scale", "b_zp", "c_scale", "c_zp"],
            ["C"],
            domain="com.microsoft",
        )
        initializers = [
            _scale_initializer("a_scale", self.a_scale),
            _zp_initializer("a_zp", self.a_zp, self.dtype),
            _scale_initializer("b_scale", self.b_scale),
            _zp_initializer("b_zp", self.b_zp, self.dtype),
            _scale_initializer("c_scale", self.c_scale),
            _zp_initializer("c_zp", self.c_zp, self.dtype),
        ]
        return [node], ["C"], initializers


@register_onnx_test
class QLinearAdd_ui8_basic(_QLinearAdd):
    dtype = torch.uint8
    shape = [2, 3]

    @annotate_inputs(
        [
            ("A", torch.uint8, [2, 3], (0, 256)),
            ("B", torch.uint8, [2, 3], (0, 256)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class QLinearAdd_i8_basic(_QLinearAdd):
    dtype = torch.int8
    shape = [2, 3]
    a_scale, a_zp = 0.037, -5
    b_scale, b_zp = 0.029, 9
    c_scale, c_zp = 0.041, -3

    @annotate_inputs(
        [
            ("A", torch.int8, [2, 3], (-128, 128)),
            ("B", torch.int8, [2, 3], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearMul
# ==============================================================================


class _QLinearMul(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [1, 16]
    a_scale, a_zp = 0.013, 11
    b_scale, b_zp = 0.021, 4
    c_scale, c_zp = 0.0067, 6

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearMul",
            ["A", "a_scale", "a_zp", "B", "b_scale", "b_zp", "c_scale", "c_zp"],
            ["C"],
            domain="com.microsoft",
        )
        initializers = [
            _scale_initializer("a_scale", self.a_scale),
            _zp_initializer("a_zp", self.a_zp, self.dtype),
            _scale_initializer("b_scale", self.b_scale),
            _zp_initializer("b_zp", self.b_zp, self.dtype),
            _scale_initializer("c_scale", self.c_scale),
            _zp_initializer("c_zp", self.c_zp, self.dtype),
        ]
        return [node], ["C"], initializers


@register_onnx_test
class QLinearMul_ui8_basic(_QLinearMul):
    @annotate_inputs(
        [
            ("A", torch.uint8, [1, 16], (0, 256)),
            ("B", torch.uint8, [1, 16], (0, 256)),
        ]
    )
    def graph(self):
        return super().graph()


@register_onnx_test
class QLinearMul_i8_basic(_QLinearMul):
    dtype = torch.int8
    a_scale, a_zp = 0.013, -11
    b_scale, b_zp = 0.021, 4
    c_scale, c_zp = 0.0067, -6

    @annotate_inputs(
        [
            ("A", torch.int8, [1, 16], (-128, 128)),
            ("B", torch.int8, [1, 16], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearLeakyRelu
# ==============================================================================


class _QLinearLeakyRelu(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [1, 8, 4, 4]
    x_scale, x_zp = 0.019, 7
    y_scale, y_zp = 0.027, 2
    alpha = 0.1

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearLeakyRelu",
            ["X", "x_scale", "x_zp", "y_scale", "y_zp"],
            ["Y"],
            domain="com.microsoft",
            alpha=self.alpha,
        )
        initializers = [
            _scale_initializer("x_scale", self.x_scale),
            _zp_initializer("x_zp", self.x_zp, self.dtype),
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
        ]
        return [node], ["Y"], initializers


@register_onnx_test
class QLinearLeakyRelu_ui8_basic(_QLinearLeakyRelu):
    @annotate_inputs([("X", torch.uint8, [1, 8, 4, 4], (0, 256))])
    def graph(self):
        return super().graph()


@register_onnx_test
class QLinearLeakyRelu_i8_basic(_QLinearLeakyRelu):
    dtype = torch.int8
    x_scale, x_zp = 0.019, -7
    y_scale, y_zp = 0.027, -2

    @annotate_inputs([("X", torch.int8, [1, 8, 4, 4], (-128, 128))])
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearSigmoid
# ==============================================================================


class _QLinearSigmoid(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [3, 5]
    x_scale, x_zp = 0.023, 13
    y_scale, y_zp = 0.0039, 1

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearSigmoid",
            ["X", "x_scale", "x_zp", "y_scale", "y_zp"],
            ["Y"],
            domain="com.microsoft",
        )
        initializers = [
            _scale_initializer("x_scale", self.x_scale),
            _zp_initializer("x_zp", self.x_zp, self.dtype),
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
        ]
        return [node], ["Y"], initializers


@register_onnx_test
class QLinearSigmoid_ui8_basic(_QLinearSigmoid):
    @annotate_inputs([("X", torch.uint8, [3, 5], (0, 256))])
    def graph(self):
        return super().graph()


@register_onnx_test
class QLinearSigmoid_i8_basic(_QLinearSigmoid):
    dtype = torch.int8
    x_scale, x_zp = 0.023, -5
    y_scale, y_zp = 0.0039, -120

    @annotate_inputs([("X", torch.int8, [3, 5], (-128, 128))])
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearConcat
# ==============================================================================


class _QLinearConcat(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [1, 2, 4, 4]
    axis = 1
    x0_scale, x0_zp = 0.017, 6
    x1_scale, x1_zp = 0.029, 10
    y_scale, y_zp = 0.021, 4

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearConcat",
            [
                "y_scale",
                "y_zp",
                "X0",
                "x0_scale",
                "x0_zp",
                "X1",
                "x1_scale",
                "x1_zp",
            ],
            ["Y"],
            domain="com.microsoft",
            axis=self.axis,
        )
        initializers = [
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
            _scale_initializer("x0_scale", self.x0_scale),
            _zp_initializer("x0_zp", self.x0_zp, self.dtype),
            _scale_initializer("x1_scale", self.x1_scale),
            _zp_initializer("x1_zp", self.x1_zp, self.dtype),
        ]
        return [node], ["Y"], initializers


@register_onnx_test
class QLinearConcat_ui8_basic(_QLinearConcat):
    @annotate_inputs(
        [
            ("X0", torch.uint8, [1, 2, 4, 4], (0, 256)),
            ("X1", torch.uint8, [1, 2, 4, 4], (0, 256)),
        ]
    )
    def graph(self):
        return _QLinearConcat.graph(self)


@register_onnx_test
class QLinearConcat_i8_basic(_QLinearConcat):
    dtype = torch.int8
    x0_scale, x0_zp = 0.017, -6
    x1_scale, x1_zp = 0.029, 10
    y_scale, y_zp = 0.021, -4

    @annotate_inputs(
        [
            ("X0", torch.int8, [1, 2, 4, 4], (-128, 128)),
            ("X1", torch.int8, [1, 2, 4, 4], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearGlobalAveragePool
# ==============================================================================


class _QLinearGlobalAveragePool(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [1, 4, 5, 5]
    x_scale, x_zp = 0.016, 8
    y_scale, y_zp = 0.011, 5

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearGlobalAveragePool",
            ["X", "x_scale", "x_zp", "y_scale", "y_zp"],
            ["Y"],
            domain="com.microsoft",
            channels_last=0,
        )
        initializers = [
            _scale_initializer("x_scale", self.x_scale),
            _zp_initializer("x_zp", self.x_zp, self.dtype),
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
        ]
        return [node], ["Y"], initializers


@register_onnx_test
class QLinearGlobalAveragePool_ui8_basic(_QLinearGlobalAveragePool):
    @annotate_inputs([("X", torch.uint8, [1, 4, 5, 5], (0, 256))])
    def graph(self):
        return _QLinearGlobalAveragePool.graph(self)


@register_onnx_test
class QLinearGlobalAveragePool_i8_basic(_QLinearGlobalAveragePool):
    dtype = torch.int8
    x_scale, x_zp = 0.016, -8
    y_scale, y_zp = 0.011, -5

    @annotate_inputs([("X", torch.int8, [1, 4, 5, 5], (-128, 128))])
    def graph(self):
        return _QLinearGlobalAveragePool.graph(self)


# ==============================================================================
# QLinearAveragePool
# ==============================================================================


class _QLinearAveragePool(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [1, 4, 8, 8]
    kernel = 2
    stride = 2
    x_scale, x_zp = 0.014, 9
    y_scale, y_zp = 0.018, 3

    def graph(self):
        node = onnx.helper.make_node(
            "QLinearAveragePool",
            ["X", "x_scale", "x_zp", "y_scale", "y_zp"],
            ["Y"],
            domain="com.microsoft",
            auto_pad="NOTSET",
            ceil_mode=0,
            count_include_pad=0,
            kernel_shape=[self.kernel, self.kernel],
            pads=[0, 0, 0, 0],
            strides=[self.stride, self.stride],
        )
        initializers = [
            _scale_initializer("x_scale", self.x_scale),
            _zp_initializer("x_zp", self.x_zp, self.dtype),
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
        ]
        return [node], ["Y"], initializers


@register_onnx_test
class QLinearAveragePool_ui8_basic(_QLinearAveragePool):
    @annotate_inputs([("X", torch.uint8, [1, 4, 8, 8], (0, 256))])
    def graph(self):
        return super().graph()


@register_onnx_test
class QLinearAveragePool_i8_basic(_QLinearAveragePool):
    dtype = torch.int8
    x_scale, x_zp = 0.014, -9
    y_scale, y_zp = 0.018, -3

    @annotate_inputs([("X", torch.int8, [1, 4, 8, 8], (-128, 128))])
    def graph(self):
        return super().graph()


# ==============================================================================
# QLinearAddQLinearSigmoidFused
# ==============================================================================


class _QLinearAddQLinearSigmoidFused(OnnxTestCase):
    extra_opsets = _EXTRA_OPSETS
    dtype = torch.uint8
    shape = [2, 4]
    a_scale, a_zp = 0.031, 4
    b_scale, b_zp = 0.027, 8
    mid_scale, mid_zp = 0.044, 6
    y_scale, y_zp = 0.0039, 2

    def graph(self):
        add_node = onnx.helper.make_node(
            "QLinearAdd",
            [
                "A",
                "a_scale",
                "a_zp",
                "B",
                "b_scale",
                "b_zp",
                "mid_scale",
                "mid_zp",
            ],
            ["Mid"],
            domain="com.microsoft",
        )
        sigmoid_node = onnx.helper.make_node(
            "QLinearSigmoid",
            ["Mid", "mid_scale", "mid_zp", "y_scale", "y_zp"],
            ["Y"],
            domain="com.microsoft",
        )
        initializers = [
            _scale_initializer("a_scale", self.a_scale),
            _zp_initializer("a_zp", self.a_zp, self.dtype),
            _scale_initializer("b_scale", self.b_scale),
            _zp_initializer("b_zp", self.b_zp, self.dtype),
            _scale_initializer("mid_scale", self.mid_scale),
            _zp_initializer("mid_zp", self.mid_zp, self.dtype),
            _scale_initializer("y_scale", self.y_scale),
            _zp_initializer("y_zp", self.y_zp, self.dtype),
        ]
        return [add_node, sigmoid_node], ["Y"], initializers


@register_onnx_test
class QLinearAddQLinearSigmoidFused_ui8_basic(_QLinearAddQLinearSigmoidFused):
    @annotate_inputs(
        [
            ("A", torch.uint8, [2, 4], (0, 256)),
            ("B", torch.uint8, [2, 4], (0, 256)),
        ]
    )
    def graph(self):
        return _QLinearAddQLinearSigmoidFused.graph(self)


@register_onnx_test
class QLinearAddQLinearSigmoidFused_i8_basic(_QLinearAddQLinearSigmoidFused):
    dtype = torch.int8
    a_scale, a_zp = 0.031, -4
    b_scale, b_zp = 0.027, 8
    mid_scale, mid_zp = 0.044, -6
    y_scale, y_zp = 0.0039, -2

    @annotate_inputs(
        [
            ("A", torch.int8, [2, 4], (-128, 128)),
            ("B", torch.int8, [2, 4], (-128, 128)),
        ]
    )
    def graph(self):
        return super().graph()
