# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# XFAIL: *
# TODO: https://github.com/llvm/mlir-npcomp/issues/86
# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

mb = torch_mlir.ModuleBuilder()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

model = torch.nn.Conv2d(Cin, Cout, (3,3))
ref_model = torch.nn.Conv2d(Cin, Cout, (3,3))

ref_model.weight.data = model.weight.clone()
ref_model.bias.data = model.bias.clone()

softmax = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()

tensor = torch.randn(N, Cin, h, w)

with mb.capture_function("conv2d_fwd", [tensor]) as f:
  result = model(tensor)
  f.returns([result])

# CHECK-LABEL: func @conv2d_fwd
# CHECK-SAME: (%arg0: !numpy.ndarray<[3,16,10,10]:f32>) -> !numpy.ndarray<[3,4,8,8]:f32> {
# CHECK: %[[P1:.*]] = numpy.create_array_from_tensor %cst : (tensor<4x16x3x3xf32>) -> !numpy.ndarray<[4,16,3,3]:f32>
# CHECK: %[[P2:.*]] = numpy.create_array_from_tensor %cst_0 : (tensor<4xf32>) -> !numpy.ndarray<[4]:f32>
# CHECK: %[[R:.*]] = torch.kernel_call "aten::conv2d" %arg0, %[[P1]], %[[P2]], %0, %1, %2, %c1_i64_5 : (!numpy.ndarray<[3,16,10,10]:f32>, !numpy.ndarray<[4,16,3,3]:f32>, !numpy.ndarray<[4]:f32>, !basicpy.ListType, !basicpy.ListType, !basicpy.ListType, i64) -> !numpy.ndarray<[3,4,8,8]:f32>
# CHECK: return %[[R]] : !numpy.ndarray<[3,4,8,8]:f32>
print(mb.module)
