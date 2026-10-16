# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Reference implementations for operators outside ONNX's standard domains."""

import numpy as np
import onnx
import onnx.helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun

# ---------------------------------------------------------------------------
# Shared quantize / dequantize helpers for the com.microsoft QLinear* ops.
# ---------------------------------------------------------------------------


def _dequantize(x: np.ndarray, scale, zero_point=None) -> np.ndarray:
    """Dequantizes an integer tensor `x` to float32: (x - zero_point) * scale."""
    x_f = x.astype(np.float32)
    if zero_point is not None:
        x_f = x_f - zero_point.astype(np.float32)
    return x_f * scale


def _quantize(x_f32: np.ndarray, scale, zero_point, dtype: np.dtype) -> np.ndarray:
    """Quantizes a float32 tensor to `dtype`, round-half-to-even, saturating."""
    zp = zero_point.astype(np.float32) if zero_point is not None else np.float32(0)
    q = np.round(x_f32 / scale) + zp
    qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
    return np.clip(q, qmin, qmax).astype(dtype)


class QLinearAdd(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        c_scale,
        c_zero_point,
    ):
        a_f = _dequantize(a, a_scale, a_zero_point)
        b_f = _dequantize(b, b_scale, b_zero_point)
        c_f = a_f + b_f
        return (_quantize(c_f, c_scale, c_zero_point, a.dtype),)


class QLinearMul(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        c_scale,
        c_zero_point,
    ):
        a_f = _dequantize(a, a_scale, a_zero_point)
        b_f = _dequantize(b, b_scale, b_zero_point)
        c_f = a_f * b_f
        return (_quantize(c_f, c_scale, c_zero_point, a.dtype),)


class QLinearLeakyRelu(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, x_scale, x_zero_point, y_scale, y_zero_point, alpha=0.01):
        x_f = _dequantize(x, x_scale, x_zero_point)
        y_f = np.where(x_f >= 0, x_f, x_f * alpha)
        return (_quantize(y_f, y_scale, y_zero_point, x.dtype),)


class QLinearSigmoid(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, x_scale, x_zero_point, y_scale, y_zero_point):
        x_f = _dequantize(x, x_scale, x_zero_point)
        y_f = 1.0 / (1.0 + np.exp(-x_f))
        return (_quantize(y_f, y_scale, y_zero_point, x.dtype),)


class QLinearConcat(OpRun):
    op_domain = "com.microsoft"

    def _run(self, y_scale, y_zero_point, *rest, axis=0):
        # `rest` is a flattened repetition of (tensor, scale, zero_point)
        # triples, one per concatenated input, per the com.microsoft spec.
        dtype = None
        dequantized = []
        for i in range(0, len(rest), 3):
            x, x_scale, x_zero_point = rest[i], rest[i + 1], rest[i + 2]
            dtype = x.dtype
            dequantized.append(_dequantize(x, x_scale, x_zero_point))
        y_f = np.concatenate(dequantized, axis=axis)
        return (_quantize(y_f, y_scale, y_zero_point, dtype),)


class QLinearGlobalAveragePool(OpRun):
    op_domain = "com.microsoft"

    def _run(self, x, x_scale, x_zero_point, y_scale, y_zero_point, channels_last=0):
        x_f = _dequantize(x, x_scale, x_zero_point)
        if channels_last:
            axes = tuple(range(1, x_f.ndim - 1))
        else:
            axes = tuple(range(2, x_f.ndim))
        y_f = np.mean(x_f, axis=axes, keepdims=True)
        return (_quantize(y_f, y_scale, y_zero_point, x.dtype),)


def _avg_pool_dequantized(
    x_f: np.ndarray,
    auto_pad,
    ceil_mode,
    count_include_pad,
    kernel_shape,
    pads,
    strides,
) -> np.ndarray:
    kwargs = {"kernel_shape": list(kernel_shape)}
    if auto_pad is not None:
        kwargs["auto_pad"] = auto_pad
    if ceil_mode is not None:
        kwargs["ceil_mode"] = ceil_mode
    if count_include_pad is not None:
        kwargs["count_include_pad"] = count_include_pad
    if pads is not None:
        kwargs["pads"] = list(pads)
    if strides is not None:
        kwargs["strides"] = list(strides)

    node = onnx.helper.make_node("AveragePool", ["x"], ["y"], **kwargs)
    graph = onnx.helper.make_graph(
        [node],
        "avg_pool_sub_graph",
        [onnx.helper.make_empty_tensor_value_info("x")],
        [onnx.helper.make_empty_tensor_value_info("y")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 19)]
    )
    (result,) = ReferenceEvaluator(model).run(None, {"x": x_f})
    return result


class QLinearAveragePool(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        auto_pad=None,
        ceil_mode=None,
        count_include_pad=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        x_f = _dequantize(x, x_scale, x_zero_point)
        y_f = _avg_pool_dequantized(
            x_f, auto_pad, ceil_mode, count_include_pad, kernel_shape, pads, strides
        )
        return (_quantize(y_f, y_scale, y_zero_point, x.dtype),)


def create_reference_evaluator(model):
    return ReferenceEvaluator(
        model,
        new_ops=[
            QLinearAdd,
            QLinearMul,
            QLinearLeakyRelu,
            QLinearSigmoid,
            QLinearConcat,
            QLinearGlobalAveragePool,
            QLinearAveragePool,
        ],
    )
