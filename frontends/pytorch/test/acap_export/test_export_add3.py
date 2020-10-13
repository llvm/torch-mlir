# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

t0 = torch.randn((1,2,3,4))
t1 = torch.randn((1,2,3,4))
t2 = torch.randn((1,2,3,4))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("add3", [t0, t1, t2]) as f:
  t3 = t0 + t1 + t2
  f.returns([t3])

# CHECK-LABEL: func @add3
# CHECK:   %[[CST_1A:.*]] = constant 1 : i64
# CHECK:   %[[CST_1B:.*]] = constant 1 : i64
# CHECK:   %[[ADD0:.*]] = torch.kernel_call "aten::add" %arg0, %arg1, %[[CST_1A]]
# CHECK:   %[[ADD1:.*]] = torch.kernel_call "aten::add" %[[ADD0]], %arg2, %[[CST_1B]]
# CHECK:   return %[[ADD1]]
print(mb.module)
