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
        self.i = 3
        self.f = 42.5

# CHECK: %[[TRUE:.*]] = basicpy.bool_constant true
# CHECK: %[[N3:.*]] = basicpy.numeric_constant 3 : i64
# CHECK: %[[N42:.*]] = basicpy.numeric_constant 4.250000e+01 : f64
# CHECK: %[[MODULE:.*]] = torch.nn_module  {
# Note: for some reason, Torch always adds a "training" property to all modules.
# CHECK:   torch.attr "training", %[[TRUE]] : !basicpy.BoolType
# CHECK:   torch.attr "i", %[[N3]] : i64
# CHECK:   torch.attr "f", %[[N42]] : f64


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
