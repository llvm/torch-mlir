# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Shared reference evaluation for output-shape probing and golden values."""

from onnx.reference import ReferenceEvaluator


def create_reference_evaluator(model):
    # ONNX Runtime supplies the actual Microsoft contrib kernels. Use it for
    # the entire graph so mixed standard/contrib graphs share one evaluator.
    if any(opset.domain == "com.microsoft" for opset in model.opset_import):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        return ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return ReferenceEvaluator(model)
