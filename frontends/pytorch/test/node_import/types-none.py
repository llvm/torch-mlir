# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK: @returns_none
@mb.import_function
@torch.jit.script
def returns_none():
    # CHECK-NEXT: %[[NONE:.*]] = basicpy.singleton : !basicpy.NoneType
    # CHECK-NEXT: return %[[NONE]]
    pass

assert isinstance(returns_none, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
