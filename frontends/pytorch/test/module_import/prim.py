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
        self.t1 = torch.ones(1)
        self.t2 = torch.ones(1)

    # CHECK-LABEL:   func{{.*}}TestModule.forward{{.*}}(
    # CHECK-SAME:         %[[SELF:.*]]: !torch.nn.Module) -> !basicpy.NoneType {
    def forward(self):
        # CHECK: %[[T2:.*]] = torch.prim.GetAttr %[[SELF]]["t2"]
        # CHECK: torch.prim.SetAttr %[[SELF]]["t1"] = %[[T2]]
        self.t1 = self.t2
        # CHECK: torch.prim.CallMethod %arg0["callee"]
        self.callee()
    def callee(self):
        pass

test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
