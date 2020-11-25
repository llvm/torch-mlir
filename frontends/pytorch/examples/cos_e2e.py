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
input = torch.rand(2, 3)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("cos", [input]) as f:
  result = torch.cos(input)
  f.returns([result])

backend = refjit.CompilerBackend()
jit_module = backend.load(backend.compile(mb.module))

logging.debug(f"Executing jit_module.cos")
test_utils.compare_outputs(torch.cos, jit_module.cos, input)

# This fails because the "Initial PyTorch IR" hardcodes the input value and
# ignores %arg0.
test_utils.compare_outputs(torch.cos, jit_module.cos, input + 1)
