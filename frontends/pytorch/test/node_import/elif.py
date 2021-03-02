# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: @__torch__.f
@mb.import_function
@torch.jit.script
def f(b: bool, i: int):
    # elif is modeled as a nested if
    # CHECK: scf.if{{.*}}{
    # CHECK: } else {
    # CHECK:   scf.if{{.*}}{
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
