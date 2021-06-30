# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

import npcomp
from npcomp.compiler.pytorch.backend import refjit, frontend_lowering
from npcomp.compiler.utils import logging

import test_utils

logging.enable()

torch.manual_seed(0)
lhs = torch.rand(2, 3)
rhs = torch.rand(3, 4)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mm", [lhs, rhs]) as f:
  result = torch.mm(lhs, rhs)
  f.returns([result])

backend = iree.IreeNpcompBackend()
jit_module = backend.load(backend.compile(frontend_lowering.lower_module(mb.module)))

test_utils.compare_outputs(torch.mm, jit_module.mm, lhs, rhs)
test_utils.compare_outputs(torch.mm, jit_module.mm, lhs + 1, rhs - 1)
