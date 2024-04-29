# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


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
