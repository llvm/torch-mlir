# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

mb = torch_mlir.ModuleBuilder()

# CHECK-LABEL: func @add3$generic
# Note that line-level debug information for parts unannotated in the Torch
# graph are ascribed to the first op that carries source information. Presently
# this includes naked constants, return and the function itself. This heuristic
# likely needs to be improved and this test should be reworked when it is.
@mb.import_function
@torch.jit.script
def add3(t0, t1, t2):
  # CHECK: constant 1{{.*}}loc({{.*}}test_script_debug_info.py":[[# @LINE + 2]]
  # CHECK: aten::add{{.*}}loc({{.*}}test_script_debug_info.py":[[# @LINE + 1]]
  intermediate = t0 + t1
  # CHECK: aten::add{{.*}}loc({{.*}}test_script_debug_info.py":[[# @LINE + 1]]
  final = intermediate + t2
  # CHECK: return{{.*}}loc({{.*}}test_script_debug_info.py":[[# @LINE - 3]]
  return final
  # CHECK: }{{.*}}loc({{.*}}test_script_debug_info.py":[[# @LINE - 5]]

# Verify again with debug info present. Just checking that it makes it in there.
mb.module.operation.print(enable_debug_info=True)
print()
