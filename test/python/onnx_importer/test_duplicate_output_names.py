# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

"""Regression test: ONNX graphs with duplicate output names must preserve all outputs.

ONNX permits the same tensor name to appear multiple times in graph.output
(meaning "return this value in multiple output positions"). The importer must
not collapse these into fewer func results.
"""

import onnx
from onnx import TensorProto, helper

from _torch_mlir_config import configure_context, ir, onnx_importer


def _duplicate_output_model() -> onnx.ModelProto:
    """Graph with 4 outputs but only 2 unique names (each repeated once)."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    n1 = helper.make_node("Identity", ["x"], ["a"])
    n2 = helper.make_node("Relu", ["x"], ["b"])

    out_a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
    out_b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [n1, n2],
        "duplicate_outputs",
        [inp],
        [out_a, out_a, out_b, out_b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


model = _duplicate_output_model()
ctx = ir.Context()
configure_context(ctx)
mi = onnx_importer.ModelInfo(model)
m = mi.create_module(context=ctx).operation
onnx_importer.NodeImporter.define_function(mi.main_graph, m).import_all()
print(m)

# The func should have 4 results (not 2) and the return should have 4 operands.
# CHECK-LABEL: func.func @duplicate_outputs
# CHECK-SAME: -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>)
# CHECK: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>
