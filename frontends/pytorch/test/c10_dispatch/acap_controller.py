# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.
# RUN: python %s | FileCheck %s

import torch
import _torch_mlir as m

t0 = torch.randn((4,4))
t1 = torch.randn((4,4))

with m.c10.AcapController() as c:
  result = t0 + t1

result = result * t0

# NOTE: Ops involved with printing throw RuntimeError about calling a kernel
# from an unboxed API.
print(result)

# CHECK: CAPTURE: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
# CHECK-NOT: CAPTURE: aten::mul
log = c.get_debug_log()
for line in log: print(line)
