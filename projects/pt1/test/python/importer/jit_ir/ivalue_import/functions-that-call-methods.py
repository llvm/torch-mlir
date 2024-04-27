# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# Interesting test case, where a function calls a method.

# CHECK-LABEL:     func.func private @__torch__.TestModule.forward
# CHECK-SAME:        (%[[ARG0:.*]]: !torch.nn.Module<"__torch__.TestModule">, %[[ARG1:.*]]: !torch.tensor) -> !torch.none {
# CHECK:             %[[F:.*]] = constant @__torch__.calls_method : (!torch.nn.Module<"__torch__.TestModule">, !torch.tensor) -> !torch.none
# CHECK:             %[[RET:.*]] = call_indirect %[[F]](%[[ARG0]], %[[ARG1]]) : (!torch.nn.Module<"__torch__.TestModule">, !torch.tensor) -> !torch.none
# CHECK:             return %[[RET]] : !torch.none
# CHECK:           }
# CHECK-LABEL:     func.func private @__torch__.TestModule.method
# CHECK-SAME:        (%[[ARG0:.*]]: !torch.nn.Module<"__torch__.TestModule">, %[[ARG1:.*]]: !torch.tensor) -> !torch.none {
# CHECK:             %[[RET:.*]] = torch.constant.none
# CHECK:             return %[[RET]] : !torch.none
# CHECK:           }
# CHECK-LABEL:     func.func private @__torch__.calls_method
# CHECK-SAME:        (%[[ARG0:.*]]: !torch.nn.Module<"__torch__.TestModule">, %[[ARG1:.*]]: !torch.tensor) -> !torch.none {
# CHECK:             %[[RET:.*]] = torch.prim.CallMethod %[[ARG0]]["method"] (%[[ARG1]]) : !torch.nn.Module<"__torch__.TestModule">, (!torch.tensor) -> !torch.none
# CHECK:             return %[[RET]] : !torch.none
# CHECK:           }


def calls_method(c: "TestModule", x):
    return c.method(x)


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return calls_method(self, x)

    @torch.jit.export  # Needed so that scripting sees it.
    def method(self, x):
        return


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
