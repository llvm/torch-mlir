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
        self.i = 3
        self.f = 42.5


# CHECK: torch.class_type @[[CLASSTYPE:.*]] {
# CHECK:   torch.attr "training" : !torch.bool
# CHECK:   torch.attr "i" : !torch.int
# CHECK:   torch.attr "f" : !torch.float
# CHECK: }
# CHECK: %[[TRUE:.*]] = torch.constant.bool true
# CHECK: %[[N3:.*]] = torch.constant.int 3
# CHECK: %[[N42:.*]] = torch.constant.float 4.250000e+01
# CHECK: %[[MODULE:.*]] = torch.nn_module  {
# Note: for some reason, Torch always adds a "training" property to all modules.
# CHECK:   torch.slot "training", %[[TRUE]] : !torch.bool
# CHECK:   torch.slot "i", %[[N3]] : !torch.int
# CHECK:   torch.slot "f", %[[N42]] : !torch.float
# CHECK: } : !torch.nn.Module<"[[CLASSTYPE:.*]]">


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
