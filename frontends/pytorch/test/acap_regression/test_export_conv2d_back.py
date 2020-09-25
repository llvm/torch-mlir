# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import npcomp.frontends.pytorch as torch_mlir

# RUN: python %s | FileCheck %s

dev = torch_mlir.mlir_device()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

model = torch.nn.Conv2d(Cin, Cout, (3,3))
ref_model = torch.nn.Conv2d(Cin, Cout, (3,3))

ref_model.weight.data = model.weight.clone()
ref_model.bias.data = model.bias.clone()

model = model.to(dev)

softmax = torch.nn.LogSoftmax(dim=1)
loss = torch.nn.NLLLoss()

tensor = torch.randn(N, Cin, h, w, device=dev)
result = model(tensor)

# CHECK-LABEL: test_export_conv2d
#   CHECK: aten.convolution_overrideable
print("test_export_conv2d")
print(torch_mlir.get_mlir( result ))

target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, Cout)
ref_target = target.clone()
target = target.to(dev)

test_loss = loss( softmax(result), target )
test_loss.backward()

# CHECK-LABEL: test_export_conv2d_back
# CHECK: aten.convolution_overrideable
# CHECK: aten._log_softmax
# CHECK: aten.nll_loss2d_forward
print("test_export_conv2d_back")
print(torch_mlir.get_mlir( test_loss ))
