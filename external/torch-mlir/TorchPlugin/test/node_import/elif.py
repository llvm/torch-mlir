# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: @__torch__.f
@mb.import_function
@torch.jit.script
def f(b: bool, i: int):
    # elif is modeled as a nested if, so we only need to do cursory checking.
    # CHECK: torch.prim.If {{.*}} {
    # CHECK: } else {
    # CHECK:   torch.prim.If {{.*}} {
    # CHECK:   } else {
    # CHECK:   }
    # CHECK: }

    if b:
        return i + i
    elif i:
        return i + i * i
    else:
        return i * i

assert isinstance(f, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
