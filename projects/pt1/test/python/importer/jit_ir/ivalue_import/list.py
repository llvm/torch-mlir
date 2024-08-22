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
        self.l = [1, 2]


# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# CHECK:   torch.attr "l" : !torch.list<int>
# CHECK: }
# CHECK: %[[N1:.*]] = torch.constant.int 1
# CHECK: %[[N2:.*]] = torch.constant.int 2
# CHECK: %[[LIST:.*]] = torch.prim.ListConstruct %[[N1]], %[[N2]] : (!torch.int, !torch.int) -> !torch.list<int>
# CHECK: torch.nn_module  {
# CHECK:   torch.slot "l", %[[LIST]] : !torch.list<int>
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE]]">


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
