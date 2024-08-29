# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.jit_ir_importer import ModuleBuilder

# RUN: %PYTHON %s | FileCheck %s

mb = ModuleBuilder()


# CHECK-LABEL: func.func @__torch__.add3
# Note that line-level debug information for parts unannotated in the Torch
# graph are ascribed to the first op that carries source information. Presently
# this includes naked constants, return and the function itself. This heuristic
# likely needs to be improved and this test should be reworked when it is.
@mb.import_function
@torch.jit.script
def add3(t0, t1, t2):
    # CHECK-DAG: torch.aten.add.Tensor {{.*}} loc("aten::add"({{.*}}debug-info.py":[[# @LINE + 1]]
    intermediate = t0 + t1
    # CHECK-DAG: torch.aten.mul.Tensor {{.*}} loc("aten::mul"({{.*}}debug-info.py":[[# @LINE + 1]]
    return intermediate * t2


# Verify again with debug info present. Just checking that it makes it in there.
mb.module.operation.print(enable_debug_info=True, use_local_scope=True)
print()
