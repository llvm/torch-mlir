# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# From the torch-mlir root, run with:
# `python -m examples.torchfx.examples.example_add_tanh_sigmoid`
# (after setting up python environment with write_env_file.sh)

import torch
from torch.fx.experimental.fx_acc import acc_tracer
import torch_mlir
from torch_mlir.dialects.torch import register_dialect
from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from ..builder import build_module
from ..annotator import annotate_forward_args
from ..torch_mlir_types import TorchTensorType


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.tanh(x) + torch.sigmoid(y)


module = MyModule()
traced_module = acc_tracer.trace(module, [torch.Tensor(2,2),
                                          torch.Tensor(2,2)])

print("TRACE")
arg_type = TorchTensorType(shape=[None, None], dtype=torch.float)
traced_module = annotate_forward_args(traced_module, [arg_type, arg_type])
print(traced_module.graph)
torch_mlir_module = build_module(traced_module)

print("\n\nTORCH MLIR")
torch_mlir_module.dump()
print(torch_mlir_module.operation.verify())

with torch_mlir_module.context:
    pm = PassManager.parse('torchscript-module-to-linalg-on-tensors-backend-pipeline')
    pm.run(torch_mlir_module)

print("\n\nLOWERED MLIR")
torch_mlir_module.dump()

backend = RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(torch_mlir_module)
jit_module = backend.load(compiled)

print("\n\nRunning Forward Function")
t = torch.rand((2, 2), dtype=torch.float)
print("Compiled result:\n", jit_module.forward(t.numpy(), t.numpy()))
print("\nExpected result:\n", module.forward(t, t))
