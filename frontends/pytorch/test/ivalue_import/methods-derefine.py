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

  # CHECK-LABEL:   func private @__torch__.TestModule.forward(
  # CHECK-SAME:                                               %[[SELF:.*]]: !torch.nn.Module<"__torch__.TestModule">) -> !torch.optional<i64> {
  # CHECK:           %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
  # CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[NONE]] : !basicpy.NoneType to !torch.optional<i64>
  # CHECK:           %[[RET:.*]] = torch.prim.CallMethod %[[SELF]]["callee"] (%[[DEREFINED]]) : !torch.nn.Module<"__torch__.TestModule">, (!torch.optional<i64>) -> !torch.optional<i64>
  # CHECK:           return %[[RET]] : !torch.optional<i64>
  def forward(self):
    return self.callee(None)
  def callee(self, o: typing.Optional[int]):
    return o

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
