# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import sys
import numpy as np
import torch
import torch_mlir

import npcomp
from npcomp.compiler.pytorch.backend.refjit import *
from npcomp.compiler.utils import logging

logging.enable()

torch.manual_seed(0)
lhs = torch.rand(2, 3)
rhs = torch.rand(3, 4)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mm", [lhs, rhs]) as f:
  result = torch.mm(lhs, rhs)
  f.returns([result])

backend = CompilerBackend()
jit_module = backend.load(backend.compile(mb.module))

jit_result = jit_module.mm(lhs.numpy(), rhs.numpy())

print(f"PyTorch Result = {result.numpy()}", file=sys.stderr)
print(f"JIT Result = {jit_result}", file=sys.stderr)

np.testing.assert_allclose(result.numpy(), jit_result)
