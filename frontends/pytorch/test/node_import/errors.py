# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s

mb = torch_mlir.ModuleBuilder()

# To test errors, use a type that we don't support yet.
try:
  @mb.import_function
  @torch.jit.script
  def import_class(x: typing.Any):
    return x
except Exception as e:
  # TODO: Once diagnostics are enabled, verify the actual error emitted.
  assert str(e) == "unsupported type in function schema: 'Any'"
else:
  assert False, "Expected exception"
