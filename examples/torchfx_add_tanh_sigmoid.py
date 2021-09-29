# -*- Python -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""
Example of taking a moduled traced by TorchFX and compiling it using torch-mlir.

To run the example, make sure the following are in your PYTHONPATH:
    1. /path/to/torch-mlir/examples
    2. /path/to/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir

then, simply call `python torchfx_add_tanh_sigmoid.py`.
"""

import torch
import numpy as np
from torch.fx.experimental.fx_acc import acc_tracer

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend \
    import RefBackendLinalgOnTensorsBackend
from torch_mlir.passmanager import PassManager

from torchfx.builder import build_module
from utils.annotator import annotate_forward_args
from utils.torch_mlir_types import TorchTensorType


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # TODO: Debug issue with RefBackend
        #return torch.tanh(x) + torch.sigmoid(y)
        return torch.tanh(x)


module = MyModule()
traced_module = acc_tracer.trace(module, [torch.Tensor(2,2),
                                          torch.Tensor(2,2)])

print("TRACE")
arg_type = TorchTensorType(shape=[None, None], dtype=torch.float)
traced_module = annotate_forward_args(traced_module, [arg_type, arg_type])
print(traced_module.graph)
mlir_module = build_module(traced_module)

print("\n\nTORCH MLIR")
mlir_module.dump()
print(mlir_module.operation.verify())

with mlir_module.context:
    pm = PassManager.parse('torchscript-module-to-linalg-on-tensors-backend-pipeline')
pm.run(mlir_module)

print("\n\nLOWERED MLIR")
mlir_module.dump()

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(mlir_module)
jit_module = backend.load(compiled)

print("\n\nRunning Forward Function")
np_t = np.random.rand(2, 2).astype(dtype=np.float32)
t = torch.tensor(np_t, dtype=torch.float)
print("Compiled result:\n", jit_module.forward(np_t, np_t))
print("\nExpected result:\n", module.forward(t, t))
