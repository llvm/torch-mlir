# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | FileCheck %s

@torch.jit.script
def add3(t0, t1, t2):
  intermediate = t0 + t1
  final = intermediate + t2
  return final

mb = torch_mlir.ModuleBuilder()
mb.import_function(add3)

# Verify again with debug info present. Just checking that it makes it in there.
# CHECK-LABEL: func @add3$generic
# CHECK: constant 1{{.*}}loc({{.*}}test_script_debug_info.py
# CHECK: aten::add{{.*}}loc({{.*}}test_script_debug_info.py
# CHECK: return{{.*}}loc({{.*}}test_script_debug_info.py
# CHECK: }{{.*}}loc({{.*}}test_script_debug_info.py
mb.module.operation.print(enable_debug_info=True)
print()
