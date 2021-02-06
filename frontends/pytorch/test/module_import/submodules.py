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

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s0 = Submodule(0)
        self.s1 = Submodule(1)

# CHECK:         %[[T:.*]] = basicpy.bool_constant true

# CHECK:         %[[N0:.*]] = basicpy.numeric_constant 0 : i64
# CHECK:         %[[S0:.*]] = torch.nn_module  {
# CHECK:           torch.attr "training", %[[T]] : !basicpy.BoolType
# CHECK:           torch.attr "n", %[[N0]] : i64
# CHECK:         }

# CHECK:         %[[N1:.*]] = basicpy.numeric_constant 1 : i64
# CHECK:         %[[S1:.*]] = torch.nn_module  {
# CHECK:           torch.attr "training", %[[T]] : !basicpy.BoolType
# CHECK:           torch.attr "n", %[[N1]] : i64
# CHECK:         }

# CHECK:        %[[ROOT:.*]] = torch.nn_module  {
# CHECK:           torch.attr "training", %[[T]] : !basicpy.BoolType
# CHECK:           torch.attr "s0", %[[S0]] : !torch.nn.Module
# CHECK:           torch.attr "s1", %[[S1]] : !torch.nn.Module
# CHECK:         }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
