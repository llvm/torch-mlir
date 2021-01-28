# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x, y):
    # CHECK-LABEL: torch.nn_module
    # CHECK: loc("{{.*}}methods-locations.py":[[@LINE+1]]
    return x * y

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print(enable_debug_info=True)
