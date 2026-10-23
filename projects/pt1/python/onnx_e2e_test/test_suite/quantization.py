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
from onnx_e2e_test.utils import numpy_dtype_for_torch

# ==============================================================================
# Helpers
# ==============================================================================

_EXTRA_OPSETS = {"com.microsoft": 1}


def _zp_initializer(name: str, value: int, dtype: torch.dtype):
    np_dtype = numpy_dtype_for_torch(dtype)
    return onnx.numpy_helper.from_array(np.array(value, dtype=np_dtype), name=name)


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
            # Setting RHS with reduced range to avoid saturation issues on x86-64 machines (with AVX2/AVX512).
            # ONNX Runtime uses the VPMADDUBSW instruction for U8S8 for performance but that may cause the input to be 16-bit clamped (saturated).
            # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
            # for more details.
            ("b", torch.int8, [3, 5], (-64, 64)),
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
            # Setting RHS with reduced range to avoid saturation issues on x86-64 machines (with AVX2/AVX512).
            # ONNX Runtime uses the VPMADDUBSW instruction for U8S8 for performance but that may cause the input to be 16-bit clamped (saturated).
            # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
            # for more details.
            ("b", torch.int8, [3, 5], (-64, 64)),
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
# QLinear binary elementwise ops (QLinearAdd, QLinearMul)
# ==============================================================================


class _QLinearBinary(OnnxTestCase):
    """Base for com.microsoft QLinear binary ops: C = op(A, B).

    Subclasses set ``op_type``, ``dtype`` and the a/b/c scale and zero-point
    attributes.
    """

    extra_opsets = _EXTRA_OPSETS
    op_type: str

    def graph(self):
        node = onnx.helper.make_node(
            self.op_type,
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


class _QLinearAdd(_QLinearBinary):
    op_type = "QLinearAdd"
    a_scale, a_zp = 0.037, 5
    b_scale, b_zp = 0.029, 9
    c_scale, c_zp = 0.041, 3


@register_onnx_test
class QLinearAdd_ui8_basic(_QLinearAdd):
    dtype = torch.uint8

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


class _QLinearMul(_QLinearBinary):
    op_type = "QLinearMul"
    a_scale, a_zp = 0.013, 11
    b_scale, b_zp = 0.021, 4
    c_scale, c_zp = 0.0067, 6


@register_onnx_test
class QLinearMul_ui8_basic(_QLinearMul):
    dtype = torch.uint8

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
    dtype = torch.uint8

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
    dtype = torch.uint8

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
    dtype = torch.uint8

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
    dtype = torch.uint8

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
    dtype = torch.uint8

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
    dtype = torch.uint8

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


class _MatMulNBits(OnnxTestCase):
    extra_opsets = {"com.microsoft": 1}
    k = 64
    n = 5
    bits = 4
    block_size = 32
    accuracy_level = 0
    # "none", "packed" (bit-packed uint8) or "float".
    zero_points = "none"
    # Flattened [N * k_blocks] scales and zero points, accepted for
    # compatibility with older models.
    flat_qparams = False
    bias = False

    def graph(self):
        rng = np.random.default_rng(0)
        k_blocks = -(-self.k // self.block_size)
        values_per_byte = 8 // self.bits
        tensors = {
            "b": rng.integers(
                0,
                256,
                (self.n, k_blocks, self.block_size // values_per_byte),
                dtype=np.uint8,
            ),
            # Power-of-two scales and quarter-step zero points keep the
            # dequantized weights exact in f32.
            "scales": 2.0
            ** rng.integers(-6, -2, (self.n, k_blocks)).astype(np.float32),
        }
        if self.zero_points == "packed":
            tensors["zero_points"] = rng.integers(
                0, 256, (self.n, -(-k_blocks // values_per_byte)), dtype=np.uint8
            )
        elif self.zero_points == "float":
            tensors["zero_points"] = (
                rng.integers(0, 4 * (1 << self.bits), (self.n, k_blocks)) / 4
            ).astype(np.float32)
        if self.flat_qparams:
            for name in ("scales", "zero_points"):
                if name in tensors:
                    tensors[name] = tensors[name].reshape(-1)
        if self.bias:
            tensors["bias"] = np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32)

        inputs = ["a", "b", "scales"]
        inputs.append("zero_points" if "zero_points" in tensors else "")
        if self.bias:
            # An empty name keeps the deprecated g_idx slot ahead of bias.
            inputs += ["", "bias"]
        runtime_inputs = {annotation[0] for annotation in self._input_annotations()}
        initializers = [
            onnx.numpy_helper.from_array(tensors[name], name=name)
            for name in inputs
            if name and name not in runtime_inputs
        ]
        node = onnx.helper.make_node(
            "MatMulNBits",
            inputs,
            ["y"],
            domain="com.microsoft",
            K=self.k,
            N=self.n,
            bits=self.bits,
            block_size=self.block_size,
            accuracy_level=self.accuracy_level,
        )
        return [node], ["y"], initializers


@register_onnx_test
class OnnxMatMulNBits_4bit(_MatMulNBits):
    @annotate_inputs([("a", torch.float32, [3, 64], (-2.0, 2.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulNBits_8bit_zeroPointBias(_MatMulNBits):
    bits = 8
    zero_points = "packed"
    bias = True

    @annotate_inputs([("a", torch.float32, [2, 3, 64], (-2.0, 2.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulNBits_raggedK(_MatMulNBits):
    # K = 40 leaves the last of three blocks padded, and the three 4-bit
    # zero points of a row occupy two bytes with one padding slot.
    k = 40
    block_size = 16
    zero_points = "packed"
    flat_qparams = True

    @annotate_inputs([("a", torch.float32, [3, 40], (-2.0, 2.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulNBits_floatZeroPoint(_MatMulNBits):
    zero_points = "float"
    flat_qparams = True

    @annotate_inputs([("a", torch.float32, [3, 64], (-2.0, 2.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulNBits_accuracyLevel4(_MatMulNBits):
    # Dynamic int8 quantization of A per K-block, with a padded last block.
    k = 40
    block_size = 16
    accuracy_level = 4
    zero_points = "packed"

    @annotate_inputs([("a", torch.float32, [-1, 40], (-2.0, 2.0))])
    def graph(self):
        return super().graph()


@register_onnx_test
class OnnxMatMulNBits_runtimeWeights(_MatMulNBits):
    @annotate_inputs(
        [
            ("a", torch.float32, [3, 64], (-2.0, 2.0)),
            ("b", torch.uint8, [5, 2, 16], (0, 256)),
        ]
    )
    def graph(self):
        return super().graph()
