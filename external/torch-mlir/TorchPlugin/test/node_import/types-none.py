# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK: @__torch__.returns_none
@mb.import_function
@torch.jit.script
def returns_none():
    # CHECK-NEXT: %[[NONE:.*]] = torch.constant.none
    # CHECK-NEXT: return %[[NONE]]
    pass

assert isinstance(returns_none, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
