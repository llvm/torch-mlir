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
        # CHECK: %[[A:.*]] = numpy.create_array_from_tensor
        # CHECK: torch.nn_module {
        # CHECK:   torch.slot "t1", %[[A]]
        # CHECK:   torch.slot "t2", %[[A]]
        self.t1 = self.t2 = torch.tensor([10., 20.])


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
