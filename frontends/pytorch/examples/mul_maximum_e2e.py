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

lhs = torch.ones((4, 6, 1))
rhs = torch.ones((1, 1, 3)) * 0.6
bias = torch.ones((1, 1, 3)) * 0.2
threshold = torch.tensor((0.75, 0.25, 0.10))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mul_maximum", [lhs, rhs, threshold, bias]) as f:
  result = torch.maximum(lhs * rhs, threshold)
  result = result + bias
  f.returns([result])

backend = CompilerBackend()
jit_module = backend.load(backend.compile(mb.module))

jit_result = jit_module.mul_maximum(lhs.numpy(), rhs.numpy(), threshold.numpy(),
                                    bias.numpy())

print(f"PyTorch Result = {result.numpy()}", file=sys.stderr)
print(f"JIT Result = {jit_result}", file=sys.stderr)

np.testing.assert_allclose(result.numpy(), jit_result)
