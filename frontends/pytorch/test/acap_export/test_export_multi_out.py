# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

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
# CHECK: %[[ADD0:.*]] = torch.kernel_call "aten::add" %arg0
# CHECK: %[[ADD1:.*]] = torch.kernel_call "aten::add" %[[ADD0]]
# CHECK: %[[ADD2:.*]] = torch.kernel_call "aten::add" %[[ADD1]]
# CHECK: %[[ADD3:.*]] = torch.kernel_call "aten::add" %[[ADD2]]
# CHECK: return %[[ADD1]], %[[ADD2]], %[[ADD3]]
print(mb.module)

