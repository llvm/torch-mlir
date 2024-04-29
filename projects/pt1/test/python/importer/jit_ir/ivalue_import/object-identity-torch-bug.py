# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# TorchScript doesn't model object identity correctly!!!
#
# As of February 2021, the way that TorchScript imports modules does not
# respect object identity. It happens to work for Tensor because a
# `torch.Tensor` is just a pointer to a TensorImpl under the hood, and so
# naively duplicating a Tensor retains the identity of the TensorImpl.


# CHECK-LABEL: torch.class_type @__torch__.TestModule {
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # CHECK: %[[L2:.*]] = torch.prim.ListConstruct
        # CHECK: %[[L1:.*]] = torch.prim.ListConstruct
        # CHECK: torch.nn_module {
        # CHECK:   torch.slot "l2", %[[L2]]
        # CHECK:   torch.slot "l1", %[[L1]]
        self.l2 = self.l1 = [1]

    # This can be uncommented when the graph importer supports it.
    # def forward(self):
    #     self.l1.append(2)
    #     self.l2.append(3)
    #     print('l1', self.l1) # TorchScript prints [1, 2]
    #     print('l2', self.l2) # TorchScript prints [1, 3] (should be [1, 2, 3])


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
