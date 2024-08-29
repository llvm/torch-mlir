# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import typing

import torch
from torch._C import CompilationUnit
from torch_mlir.jit_ir_importer import ModuleBuilder

import typing

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


class BasicClass:
    def __init__(self, x: int):
        self.x = x


# CHECK-LABEL:   func.func @__torch__.prim_CreateObject(
# CHECK-SAME:                                      %[[ARG0:.*]]: !torch.int) -> !torch.nn.Module<"__torch__.BasicClass"> {
# CHECK:           %[[OBJECT:.*]] = torch.prim.CreateObject !torch.nn.Module<"__torch__.BasicClass">
# CHECK:           %[[NONE:.*]] = torch.prim.CallMethod %[[OBJECT]]["__init__"] (%[[ARG0]]) : !torch.nn.Module<"__torch__.BasicClass">, (!torch.int) -> !torch.none
# CHECK:           return %[[OBJECT]] : !torch.nn.Module<"__torch__.BasicClass">
@mb.import_function
@torch.jit.script
def prim_CreateObject(i: int):
    return BasicClass(i)


mb.module.operation.print()
print()
