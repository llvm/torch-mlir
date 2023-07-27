# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE.pytorch for license information.

import torch
from torch_mlir.dialects.torch.importer.jit_ir import ModuleBuilder

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
  # TODO: Checks for debug info are quite hard with the new trailing debug
  # attribute print. See if this can be improved.
  # CHECK: torch.aten.add.Tensor {{.*}} loc(#[[LOC1:loc.*]])
  # CHECK: torch.aten.add.Tensor {{.*}} loc(#[[LOC2:loc.*]])
  # CHECK-DAG: #[[OP_NAME:loc.*]] = loc("aten::add")
  # CHECK-DAG: #[[LOC1]] = loc(fused[#[[FLC_1:loc.*]], #[[OP_NAME]]])
  # CHECK-DAG: #[[LOC2]] = loc(fused[#[[FLC_2:loc.*]], #[[OP_NAME]]])
  # CHECK-DAG: #[[FLC_1]] = loc({{.*}}debug-info.py":[[# @LINE + 1]]
  intermediate = t0 + t1
  # CHECK-DAG: #[[FLC_2]] = loc({{.*}}debug-info.py":[[# @LINE + 1]]
  final = intermediate + t2
  return final

# Verify again with debug info present. Just checking that it makes it in there.
mb.module.operation.print(enable_debug_info=True)
print()
