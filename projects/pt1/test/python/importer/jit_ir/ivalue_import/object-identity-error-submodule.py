# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: not %PYTHON %s 2>&1 | FileCheck %s

mb = ModuleBuilder()


class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.tensor([10.0, 20.0])
        # Test a nontrivial recursive case of the diagnostic.
        # CHECK: Unhandled tensor that shares storage with another tensor.
        # CHECK-NEXT: Found at path '<root>.m.t2' from root object '__torch__.TestModule'
        self.t2 = self.t1[0]


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Submodule()


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
