# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s

@torch.jit.script
class ExampleClass:
  def __init__(self, x):
    self.x = x


mb = torch_mlir.ModuleBuilder()

# For now, TorchScript classes are wholly unsupported, so use it to test
# type conversion errors.
try:
  @mb.import_function
  @torch.jit.script
  def import_class(c: ExampleClass):
    return c.x
except RuntimeError as e:
  # TODO: Once diagnostics are enabled, verify the actual error emitted.
  assert str(e) == "could not convert function input type"
else:
  assert False, "Expected exception"
