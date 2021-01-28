# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x, y):
    return x * y

# The symbol name of the function is NOT load-bearing and cannot be relied upon.

# CHECK-LABEL:   func private
# CHECK-SAME:                 @[[SYMNAME:.*]](
# CHECK-SAME:                                 %[[SELF:.*]]: !torch.nn.Module,
# CHECK-SAME:                                 %[[X:.*]]: !numpy.ndarray<*:!numpy.any_dtype>,
# CHECK-SAME:                                 %[[Y:.*]]: !numpy.ndarray<*:!numpy.any_dtype>) -> !numpy.ndarray<*:!numpy.any_dtype> {
# CHECK:           %[[RET:.*]] = torch.kernel_call "aten::mul" %[[X]], %[[Y]]
# CHECK:           return %[[RET]] : !numpy.ndarray<*:!numpy.any_dtype>

# CHECK:         %[[ROOT:.*]] = torch.nn_module  {
# CHECK:           torch.method "forward", @[[SYMNAME]]
# CHECK:         }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
