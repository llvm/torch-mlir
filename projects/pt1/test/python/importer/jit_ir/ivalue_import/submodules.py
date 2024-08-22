# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class Submodule(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s0 = Submodule(0)
        self.s1 = Submodule(1)


# CHECK-LABEL: torch.class_type @__torch__.TestModule {
# CHECK:         %[[T:.*]] = torch.constant.bool true

# CHECK:         %[[N0:.*]] = torch.constant.int 0
# CHECK:         %[[S0:.*]] = torch.nn_module  {
# CHECK:           torch.slot "training", %[[T]] : !torch.bool
# CHECK:           torch.slot "n", %[[N0]] : !torch.int
# CHECK:         }

# CHECK:         %[[N1:.*]] = torch.constant.int 1
# CHECK:         %[[S1:.*]] = torch.nn_module  {
# CHECK:           torch.slot "training", %[[T]] : !torch.bool
# CHECK:           torch.slot "n", %[[N1]] : !torch.int
# CHECK:         }

# CHECK:        %[[ROOT:.*]] = torch.nn_module  {
# CHECK:           torch.slot "training", %[[T]] : !torch.bool
# CHECK:           torch.slot "s0", %[[S0]] : !torch.nn.Module
# CHECK:           torch.slot "s1", %[[S1]] : !torch.nn.Module
# CHECK:         }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
