# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class Submodule(torch.nn.Module):
  def forward(self, x):
    return x

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.s = Submodule()
  def forward(self, x, y):
    return x * y

# The symbol name of the function is NOT load-bearing and cannot be relied upon.
# However, we do make an attempt to ensure that the names are debuggable.
#
# The names have the following structure:
# - `__npcomp_priv_fn`: marker that this symbol name is private to npcomp
# - `__torch__.Submodule.forward`: the name that TorchScript gives the function
#   - For those curious, the `__torch__` would be the Python module name, but in
#     the case that the name is `__main__` Torch replaces it with `__torch__` to
#     avoid collisions.

# CHECK-DAG: func private @__npcomp_priv_fn.__torch__.TestModule.forward
# CHECK=DAG: func private @__npcomp_priv_fn.__torch__.Submodule.forward


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
