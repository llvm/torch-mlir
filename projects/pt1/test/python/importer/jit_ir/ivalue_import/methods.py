# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


# Function names in the Torch compilation unit are systematic -- they
# are effectively Python dotted paths. E.g. a Python module "foo" with a class
# "bar" with a method "baz" will result in a function in the compilation unit
# called "foo.bar.baz" when it gets `torch.jit.script`'ed.
# (with the exception that `__main__` is replaced with `__torch__`).
#
# Given how systematic this is, we don't treat the symbol names as opaque (i.e.
# we don't need to capture their names when FileCheck testing).

# CHECK-LABEL:     func.func private @__torch__.TestModule.forward
# CHECK-SAME:        (%[[SELF:.*]]: !torch.nn.Module<"__torch__.TestModule">, %[[X:.*]]: !torch.tensor) -> !torch.tensor {
# CHECK:             return %[[X]] : !torch.tensor
# CHECK:           }
#
# CHECK-LABEL:   torch.class_type @__torch__.TestModule  {
# CHECK:           torch.method "forward", @__torch__.TestModule.forward
# CHECK:         }


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


test_module = TestModule()
recursivescriptmodule = torch.jit.script(test_module)
# TODO: Automatically handle unpacking Python class RecursiveScriptModule into the underlying ScriptModule.
mb.import_module(recursivescriptmodule._c)
mb.module.operation.print()
