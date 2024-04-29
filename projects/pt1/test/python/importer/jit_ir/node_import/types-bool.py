# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = ModuleBuilder()


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
