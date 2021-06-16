# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import typing

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

class Submodule(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def forward(self):
       return self.n

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s1 = Submodule(1)
        self.s2 = Submodule(2)

    # CHECK-LABEL: func private @{{.*}}TestModule.forward
    def forward(self, b: bool):
        # Modules with the same class can be selected between.
        # CHECK: %[[MOD:.*]] = torch.prim.If
        s = self.s1 if b else self.s2
        # CHECK: %[[N:.*]] = torch.prim.CallMethod %[[MOD]]["forward"] ()
        # CHECK: return %[[N]]
        return s.forward()


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
