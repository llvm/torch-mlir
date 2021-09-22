# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
from torch.fx.experimental.fx_acc import acc_tracer
import npcomp
from npcomp.compiler.pytorch.backend import refbackend
from npcomp.passmanager import PassManager

from torchfx.builder import build_module
from torchfx.annotator import annotate_forward_args
from torchfx.torch_mlir_types import TorchTensorType


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

with npcomp.ir.Context() as ctx:
    npcomp.register_all_dialects(ctx)
    lowered_mlir_module = npcomp.ir.Module.parse(str(torch_mlir_module))
    pm = PassManager.parse('torchscript-to-npcomp-backend-pipeline')
pm.run(lowered_mlir_module)

print("\n\nLOWERED MLIR")
lowered_mlir_module.dump()

backend = refbackend.RefBackendNpcompBackend()
compiled = backend.compile(lowered_mlir_module)
jit_module = backend.load(compiled)

print("\n\nRunning Forward Function")
t = torch.rand((2, 2), dtype=torch.float)
print("Compiled result:\n", jit_module.forward(t.numpy(), t.numpy()))
print("\nExpected result:\n", module.forward(t, t))
