# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.
# RUN: python %s | FileCheck %s

import torch
import _torch_mlir as m

t0 = torch.randn((4,4))
t1 = torch.randn((4,4))

mb = m.ModuleBuilder()
with mb.capture_function("foobar") as c:
  result = t0 + t1

# CHECK: module {
# CHECK:   func @foobar() {
# CHECK:   }
# CHECK: }
print(mb)

# CHECK: CAPTURE: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> (Tensor)
for line in c.get_debug_log(): print(line)
