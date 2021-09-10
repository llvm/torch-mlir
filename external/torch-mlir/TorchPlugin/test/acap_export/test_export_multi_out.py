# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

t0 = torch.randn(4)
t1 = torch.randn(4)
t2 = torch.randn(4)

with mb.capture_function("multi_output", [t0, t1, t2]) as f:
  t4 = t0 + t1 + t2
  t5 = t4 + t1
  t6 = t5 + t4
  f.returns([t4, t5, t6])

# CHECK-LABEL: func @multi_output
# CHECK: %[[ADD0:.*]] = torch.operator "aten.add.out"(%arg0
# CHECK: %[[ADD1:.*]] = torch.operator "aten.add.out"(%[[ADD0]]
# CHECK: %[[ADD2:.*]] = torch.operator "aten.add.out"(%[[ADD1]]
# CHECK: %[[ADD3:.*]] = torch.operator "aten.add.out"(%[[ADD2]]
# CHECK: return %[[ADD1]], %[[ADD2]], %[[ADD3]]
print(mb.module)
