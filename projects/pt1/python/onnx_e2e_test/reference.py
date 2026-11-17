# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Shared reference evaluation for output-shape probing and golden values."""

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_conv_integer import ConvInteger as _OnnxConvInteger
from onnx.reference.ops.op_qlinear_conv import QLinearConv as _OnnxQLinearConv


# ReferenceEvaluator keys `new_ops` by class name, so this must be named after
# the ONNX op it overrides.
class ConvInteger(_OnnxConvInteger):
    """ConvInteger with per-channel ``w_zero_point`` support.

    The onnx 1.17 reference evaluates ``if w_zero_point:``, which raises for a
    1-D zero point. This mirrors the upstream fix in onnx/onnx commit 6318ddb
    (https://github.com/onnx/onnx/pull/7165, first released in onnx 1.20.0)
    and can be dropped once the pinned onnx version is >= 1.20.0.
    """

    op_domain = ""

    def _run(self, X, W, x_zero_point=None, w_zero_point=None, **kwargs):
        if w_zero_point is not None and w_zero_point.ndim == 1:
            W = W.astype(np.int32) - np.expand_dims(
                w_zero_point.astype(np.int32), tuple(range(1, W.ndim))
            )
            w_zero_point = None
        return super()._run(X, W, x_zero_point, w_zero_point, **kwargs)


class QLinearConv(_OnnxQLinearConv):
    """QLinearConv with per-channel ``w_scale`` support.

    The onnx 1.17 reference multiplies the convolution result by a 1-D
    ``w_scale`` without reshaping it to the output-channel axis, which fails
    to broadcast. This mirrors the upstream fix in onnx/onnx
    (https://github.com/onnx/onnx/pull/6460, first released in onnx 1.18.0)
    and can be dropped once the pinned onnx version is >= 1.18.0.
    """

    op_domain = ""

    def _run(self, x, x_scale, x_zero_point, w, w_scale, *args, **kwargs):
        if np.ndim(w_scale) == 1 and np.size(w_scale) > 1:
            w_scale = np.expand_dims(w_scale, (0, *range(2, w.ndim)))
        return super()._run(x, x_scale, x_zero_point, w, w_scale, *args, **kwargs)


_NEW_OPS = [ConvInteger, QLinearConv]


# Model metadata key a test sets (through
# ``OnnxTestCase.use_onnxruntime_reference``) to be evaluated with ONNX Runtime.
ONNXRUNTIME_REFERENCE_KEY = "onnx_e2e_test.reference"


def mark_onnxruntime_reference(model):
    """Mark *model* to be evaluated with ONNX Runtime instead of onnx.reference."""
    prop = model.metadata_props.add()
    prop.key = ONNXRUNTIME_REFERENCE_KEY
    prop.value = "onnxruntime"


def create_reference_evaluator(model):
    # ONNX Runtime supplies the actual Microsoft contrib kernels. Use it for
    # the entire graph so mixed standard/contrib graphs share one evaluator.
    # Tests whose ops onnx.reference does not implement correctly ask for it
    # through the model metadata.
    wants_onnxruntime = any(
        prop.key == ONNXRUNTIME_REFERENCE_KEY and prop.value == "onnxruntime"
        for prop in model.metadata_props
    )
    if wants_onnxruntime or any(
        opset.domain == "com.microsoft" for opset in model.opset_import
    ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        return ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return ReferenceEvaluator(model, new_ops=_NEW_OPS)


def run_reference(model, feed):
    """Run *model* through the reference evaluator and return its outputs.

    Raises if the reference returns None for any declared graph output.
    Older ONNX Runtime versions skip optional outputs (e.g. mean and
    inv_std_var of SkipSimplifiedLayerNormalization before 1.30.0), which
    would otherwise silently drop them from the comparison.
    """
    outputs = create_reference_evaluator(model).run(None, feed)
    missing = [
        info.name for info, out in zip(model.graph.output, outputs) if out is None
    ]
    if missing:
        raise RuntimeError(
            "Reference evaluator returned None for output(s) "
            f"{missing}; the installed reference (e.g. onnxruntime) may be "
            "too old to compute them."
        )
    return outputs
