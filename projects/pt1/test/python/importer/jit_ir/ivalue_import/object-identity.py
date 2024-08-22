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
        # CHECK: %[[T:.*]] = torch.tensor.literal
        # CHECK: torch.nn_module {
        # CHECK:   torch.slot "t1", %[[T]]
        # CHECK:   torch.slot "t2", %[[T]]
        self.t1 = self.t2 = torch.tensor([10.0, 20.0])


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
