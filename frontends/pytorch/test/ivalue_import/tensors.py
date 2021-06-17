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
        # TODO: Test (and make work) tensors that alias each other.
        self.ones = torch.ones(1)
        self.arange = torch.nn.Parameter(torch.arange(3.0))

# CHECK: %[[ARANGE:.*]] = torch.tensor.literal(dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>) : !torch.tensor<[3],f32>
# CHECK: %[[ONES:.*]] = torch.tensor.literal(dense<1.000000e+00> : tensor<1xf32>) : !torch.tensor<[1],f32>
# CHECK: %[[ROOT:.*]] = torch.nn_module  {
# CHECK:   torch.slot "arange", %[[ARANGE]] : !torch.tensor<[3],f32>
# CHECK:   torch.slot "ones", %[[ONES]] : !torch.tensor<[1],f32>
# CHECK: }


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
