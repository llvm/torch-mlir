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

lhs = torch.ones((4, 6, 1))
rhs = torch.ones((1, 1, 3)) * 0.6
bias = torch.ones((1, 1, 3)) * 0.2
threshold = torch.tensor((0.75, 0.25, 0.10))


def mul_maximum(lhs, rhs, threshold, bias):
  return torch.maximum(lhs * rhs, threshold) + bias


mb = torch_mlir.ModuleBuilder()
with mb.capture_function("mul_maximum", [lhs, rhs, threshold, bias]) as f:
  result = mul_maximum(lhs, rhs, threshold, bias)
  f.returns([result])

backend = iree.IreeNpcompBackend()
jit_module = backend.load(backend.compile(frontend_lowering.lower_module(mb.module)))

test_utils.compare_outputs(mul_maximum, jit_module.mul_maximum, lhs, rhs,
                           threshold, bias)
test_utils.compare_outputs(mul_maximum, jit_module.mul_maximum, lhs + 1,
                           rhs + 2, threshold, bias)
