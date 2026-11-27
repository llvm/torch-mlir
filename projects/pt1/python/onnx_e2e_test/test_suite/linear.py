# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Linear / weight-carrying ONNX op tests.

Exercises the shared-weights design: concrete weight and bias tensors are
embedded as ONNX initializers (returned as the third element of ``graph``),
not runtime inputs.  Only the activation ``x`` is an annotated input.
"""

import numpy as np
import torch
import onnx.helper
import onnx.numpy_helper

from onnx_e2e_test.framework import OnnxTestCase, annotate_inputs
from onnx_e2e_test.registry import register_onnx_test

# ==============================================================================
# Gemm with embedded weight + bias initializers
#
#   x [4, 8]  --\
#   W [8, 16]    >--> Gemm --> y [4, 16]
#   b [16]    --/
# ==============================================================================


@register_onnx_test
class OnnxGemm_withWeights_basic(OnnxTestCase):
    def __init__(self):
        self._W = np.random.default_rng(42).standard_normal((8, 16)).astype(np.float32)
        self._b = np.random.default_rng(43).standard_normal((16,)).astype(np.float32)

    @annotate_inputs([("x", torch.float32, [4, 8])])
    def graph(self):
        node = onnx.helper.make_node(
            "Gemm",
            ["x", "W", "b"],
            ["y"],
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        )
        initializers = [
            onnx.numpy_helper.from_array(self._W, name="W"),
            onnx.numpy_helper.from_array(self._b, name="b"),
        ]
        return [node], ["y"], initializers


# ==============================================================================
# Gemm with non-unit alpha and no bias (C omitted): y = alpha * (x @ W)
#
#   x [4, 8]  --\
#                >--> Gemm(alpha=0.5) --> y [4, 16]
#   W [8, 16] --/
# ==============================================================================


@register_onnx_test
class OnnxGemm_alphaNoBias_basic(OnnxTestCase):
    def __init__(self):
        self._W = np.random.default_rng(44).standard_normal((8, 16)).astype(np.float32)

    @annotate_inputs([("x", torch.float32, [4, 8])])
    def graph(self):
        node = onnx.helper.make_node(
            "Gemm",
            ["x", "W"],
            ["y"],
            alpha=0.5,
            transA=0,
            transB=0,
        )
        initializers = [onnx.numpy_helper.from_array(self._W, name="W")]
        return [node], ["y"], initializers


# ==============================================================================
# Gemm with non-unit alpha and C passed as an empty optional input ("").
# The importer materializes the empty name as torch.constant.none.
# ==============================================================================


@register_onnx_test
class OnnxGemm_alphaEmptyBias_basic(OnnxTestCase):
    def __init__(self):
        self._W = np.random.default_rng(45).standard_normal((8, 16)).astype(np.float32)

    @annotate_inputs([("x", torch.float32, [4, 8])])
    def graph(self):
        node = onnx.helper.make_node(
            "Gemm",
            ["x", "W", ""],
            ["y"],
            alpha=0.5,
            transA=0,
            transB=0,
        )
        initializers = [onnx.numpy_helper.from_array(self._W, name="W")]
        return [node], ["y"], initializers


# ==============================================================================
# ConvTranspose with an output_shape one smaller than the full output
#
#   x [1, 8, 4, 13] --> ConvTranspose(W [8, 4, 3, 3]) --> y [1, 4, 8, 26]
# ==============================================================================


@register_onnx_test
class OnnxConvTranspose2d_f32_tfSameOutputShape(OnnxTestCase):
    # A TensorFlow Conv2DTranspose with SAME padding as exported by tf2onnx:
    # output_shape is twice the input and the bias is a separate op.
    def __init__(self):
        self._W = (
            np.random.default_rng(46).standard_normal((8, 4, 3, 3)).astype(np.float32)
        )

    @annotate_inputs([("x", torch.float32, [1, 8, 4, 13])])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvTranspose",
            ["x", "W"],
            ["y"],
            dilations=[1, 1],
            kernel_shape=[3, 3],
            output_shape=[8, 26],
            strides=[2, 2],
        )
        initializers = [onnx.numpy_helper.from_array(self._W, name="W")]
        return [node], ["y"], initializers


# ==============================================================================
# ConvTranspose with a larger begin pad on H and a larger end pad on W
#
#   x [1, 2, 4, 4] --> ConvTranspose(W [2, 3, 3, 3]) --> y [1, 3, 8, 8]
# ==============================================================================


@register_onnx_test
class OnnxConvTranspose2d_f32_asymmetricPads(OnnxTestCase):
    def __init__(self):
        self._W = (
            np.random.default_rng(47).standard_normal((2, 3, 3, 3)).astype(np.float32)
        )

    @annotate_inputs([("x", torch.float32, [1, 2, 4, 4])])
    def graph(self):
        node = onnx.helper.make_node(
            "ConvTranspose",
            ["x", "W"],
            ["y"],
            pads=[1, 0, 0, 1],
            strides=[2, 2],
        )
        initializers = [onnx.numpy_helper.from_array(self._W, name="W")]
        return [node], ["y"], initializers
