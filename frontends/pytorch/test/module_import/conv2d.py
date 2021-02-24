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

        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3))

    def forward(self, x):
        return self.conv1(x)

# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# TODO: Don't lose element type.
# CHECK:   torch.attr "l" : !basicpy.ListType
# CHECK: }
# CHECK: %[[N1:.*]] = basicpy.numeric_constant 1 : i64
# CHECK: %[[N2:.*]] = basicpy.numeric_constant 2 : i64
# CHECK: %[[LIST:.*]] = basicpy.build_list %[[N1]], %[[N2]] : (i64, i64) -> !basicpy.ListType
# CHECK: torch.nn_module  {
# CHECK:   torch.slot "l", %[[LIST]] : !basicpy.ListType
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE]]">


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
