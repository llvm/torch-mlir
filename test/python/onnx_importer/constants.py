# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s %t.onnx
# RUN: %PYTHON -m torch_mlir.tools.import_onnx %t.onnx > %t.mlir
# RUN: FileCheck %s < %t.mlir

import onnx
from onnx.helper import make_graph, make_tensor, make_tensor_value_info

graph = make_graph(
    name="graph",
    inputs=[],
    nodes=[],
    outputs=[],
    initializer=[
        # CHECK{LITERAL}: torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[true, false], [false, true]]> : tensor<2x2xi1>} : () -> !torch.vtensor<[2,2],i1>
        make_tensor(
            "bool_tensor",
            onnx.TensorProto.BOOL,
            dims=[2, 2],
            vals=[True, False, False, True],
        )
    ],
)
model = onnx.helper.make_model(graph)

import sys

out_file_path = sys.argv[1]
onnx.save(model, out_file_path)
