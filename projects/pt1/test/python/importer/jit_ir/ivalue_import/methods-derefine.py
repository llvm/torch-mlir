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

    # CHECK-LABEL:   func.func private @__torch__.TestModule.forward(
    # CHECK-SAME:                                               %[[SELF:.*]]: !torch.nn.Module<"__torch__.TestModule">) -> !torch.optional<int> {
    # CHECK:           %[[NONE:.*]] = torch.constant.none
    # CHECK:           %[[DEREFINED:.*]] = torch.derefine %[[NONE]] : !torch.none to !torch.optional<int>
    # CHECK:           %[[RET:.*]] = torch.prim.CallMethod %[[SELF]]["callee"] (%[[DEREFINED]]) : !torch.nn.Module<"__torch__.TestModule">, (!torch.optional<int>) -> !torch.optional<int>
    # CHECK:           return %[[RET]] : !torch.optional<int>
    def forward(self):
        return self.callee(None)

    def callee(self, o: typing.Optional[int]):
        return o


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
