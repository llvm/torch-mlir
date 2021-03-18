# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

import npcomp
from npcomp.compiler.pytorch.backend import refjit
from npcomp.compiler.utils import logging

import test_utils

logging.enable()

torch.manual_seed(0)

arg0 = torch.ones(2, 2)
arg1 = torch.ones(2, 2)

def fun(a, b):
  return a.div_(b)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("test", [arg0, arg1]) as f:
  f.returns([fun(arg0, arg1)])

backend = refjit.CompilerBackend()
jit_module = backend.load(backend.compile(mb.module))

test_utils.compare_outputs(torch.mm, jit_module.test, arg0, arg1)
test_utils.compare_outputs(torch.mm, jit_module.test, arg0 + 1, arg1 + 1)
