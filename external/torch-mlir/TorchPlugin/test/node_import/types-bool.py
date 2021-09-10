# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK: @__torch__.returns_bool
@mb.import_function
@torch.jit.script
def returns_bool():
    # CHECK-NEXT: %[[T:.*]] = torch.constant.bool true
    # CHECK-NEXT: return %[[T]]
    return True

assert isinstance(returns_bool, torch.jit.ScriptFunction)
mb.module.operation.print()
print()
