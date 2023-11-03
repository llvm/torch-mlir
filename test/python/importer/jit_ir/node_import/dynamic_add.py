# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

import typing

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# CHECK-LABEL:   func.func @__torch__.add
# CHECK-SAME:    %[[ARG0:.*]]: !torch.tensor
# CHECK-DAG:     %[[INT1:.*]] = torch.constant.int 1
# CHECK-DAG:     %[[INT20:.*]] = torch.constant.int 20
# CHECK:         %[[RES:.*]] = torch.aten.add.Scalar %[[ARG0]], %[[INT20]], %[[INT1]] : !torch.tensor, !torch.int, !torch.int -> !torch.tensor 
# CHECK:         return %[[RES]]
@mb.import_function
@torch.jit.script
def add(x: torch.Tensor):
    return torch.add(x, 20)

x = add(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
x = add(torch.tensor([1, 2, 3, 4, 5]))

mb.module.operation.print()
print()
