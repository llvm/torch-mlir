# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Reference implementations for operators outside ONNX's standard domains."""

import numpy as np
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


class MatMulIntegerToFloat(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self, A, B, a_scale, b_scale, a_zero_point=None, b_zero_point=None, bias=None
    ):
        # A_f = (float(A) - a_zero_point) * a_scale, and similarly for B.
        # Cast before subtraction to avoid int8/uint8 wraparound. NumPy's last
        # axis broadcasting implements the spec's per-column qparams.
        a_float = A.astype(np.float32)
        b_float = B.astype(np.float32)
        if a_zero_point is not None:
            a_float = a_float - a_zero_point.astype(np.float32)
        if b_zero_point is not None:
            b_float = b_float - b_zero_point.astype(np.float32)
        result = np.matmul(a_float * a_scale, b_float * b_scale)
        if bias is not None:
            result = result + bias
        return (result.astype(np.float32),)


def create_reference_evaluator(model):
    # The same operator set must be used for output-shape probing and goldens.
    return ReferenceEvaluator(model, new_ops=[MatMulIntegerToFloat])
