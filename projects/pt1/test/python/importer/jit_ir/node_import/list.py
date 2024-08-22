# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()

# CHECK-LABEL:   func.func @__torch__.f(
# CHECK-SAME:            %[[T0:.*]]: !torch.tensor,
# CHECK-SAME:            %[[T1:.*]]: !torch.tensor) -> !torch.list<tensor> {
# CHECK:           %[[RET:.*]] = torch.prim.ListConstruct %[[T0]], %[[T1]] : (!torch.tensor, !torch.tensor) -> !torch.list<tensor>
# CHECK:           return %[[RET]] : !torch.list<tensor>


@mb.import_function
@torch.jit.script
def f(t0, t1):
    return [t0, t1]


assert isinstance(f, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
