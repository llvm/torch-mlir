# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch_mlir

# TODO: Fix https://github.com/llvm/mlir-npcomp/issues/80
# XFAIL: *
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

with mb.capture_function("@conv2d_fwd", [tensor]) as f:
  result = model(tensor)
  f.returns([result])

target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, Cout)
ref_target = target.clone()

with mb.capture_function("@conv2d_backward", [result, target]) as f:
  test_loss = loss(softmax(result), target)
  f.returns([test_loss.backward()])

# CHECK-LABEL: func @conv2d_fwd
# TODO: Add checks when passing

# CHECK-LABEL: func @conv2d_backward
# TODO: Update checks when passing
# NO-CHECK: aten.convolution_overrideable
# NO-CHECK: aten._log_softmax
# NO-CHECK: aten.nll_loss2d_forward
print(mb.module)
