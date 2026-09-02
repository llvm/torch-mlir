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
