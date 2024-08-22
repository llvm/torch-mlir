# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.ones(1)
        self.t2 = torch.ones(1)

    # CHECK-LABEL:   func.func private @__torch__.TestModule.forward(
    # CHECK-SAME:         %[[SELF:.*]]: !torch.nn.Module<"{{.*}}">) -> !torch.none {
    def forward(self):
        # CHECK: %[[T2:.*]] = torch.prim.GetAttr %[[SELF]]["t2"]
        # CHECK: torch.prim.SetAttr %[[SELF]]["t1"] = %[[T2]]
        self.t1 = self.t2
        # CHECK: torch.prim.CallMethod %[[SELF]]["callee"] (%{{.*}}, %{{.*}})
        self.callee(self.t1, self.t2)

    # CHECK-LABEL:   func.func private @__torch__.TestModule.callee(
    # CHECK-SAME:         %[[SELF:.*]]: !torch.nn.Module<"{{.*}}">,
    # CHECK-SAME:         %[[X:.*]]: !torch.tensor,
    # CHECK-SAME:         %[[Y:.*]]: !torch.tensor
    def callee(self, x, y):
        pass


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
