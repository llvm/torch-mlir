# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: func @__torch__.add3
# Note that line-level debug information for parts unannotated in the Torch
# graph are ascribed to the first op that carries source information. Presently
# this includes naked constants, return and the function itself. This heuristic
# likely needs to be improved and this test should be reworked when it is.
@mb.import_function
@torch.jit.script
def add3(t0, t1, t2):
  # TODO: Checks for debug info are quite hard with the new trailing debug
  # attribute print. See if this can be improved.
  # CHECK: loc({{.*}}debug-info.py":[[# @LINE + 1]]
  intermediate = t0 + t1
  # CHECK: loc({{.*}}debug-info.py":[[# @LINE + 1]]
  final = intermediate + t2
  return final

# Verify again with debug info present. Just checking that it makes it in there.
mb.module.operation.print(enable_debug_info=True)
print()
