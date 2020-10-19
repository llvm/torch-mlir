# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.
# RUN: %PYTHON %s | FileCheck %s

# TODO: Once stabilized, expand tests to include all argument dtypes.

import torch
import torch_mlir

t0 = torch.randn((1,4))
t1 = torch.randn((4,1))

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("foobar", [t0, t1]) as f:
  result = t0 + t1
  f.returns([result])

# CHECK: module {
# CHECK:   func @foobar(%arg0: !numpy.ndarray<[1,4]:f32>, %arg1: !numpy.ndarray<[4,1]:f32>) -> !numpy.ndarray<[4,4]:f32> {
# CHECK:     %c1_i64 = constant 1 : i64
# CHECK:     %0 = torch.kernel_call "aten::add" %arg0, %arg1, %c1_i64 : (!numpy.ndarray<[1,4]:f32>, !numpy.ndarray<[4,1]:f32>, i64) -> !numpy.ndarray<[4,4]:f32>
# CHECK:     return %0 : !numpy.ndarray<[4,4]:f32>
# CHECK:   }
# CHECK: }
print(mb.module)

# CHECK: CAPTURE: aten::add
for line in f.get_debug_log(): print(line)
