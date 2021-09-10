# -*- Python -*-
# This file is licensed under a pytorch-style license
# See LICENSE for license information.

# RUN: %PYTHON %s | torch-mlir-opt | FileCheck %s

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir

torch_mlir.debug_trace_to_stderr()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

class Net(nn.Module):
    def __init__(self, Cin, Cout):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(Cin, Cout, (3,3))
    def forward(self, x):
      x0 = self.conv1(x)
      x1 = self.conv1(x)
      z = torch.cat([x0, x1])
      output = F.log_softmax(z, dim=1)
      return output

model = Net(Cin, Cout)
inputs = torch.ones((N,Cin,h,w))
loss = torch.nn.NLLLoss()
target = torch.empty(2*N, 8, 8, dtype=torch.long).random_(0, Cout)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("conv_cat", [inputs, target]) as f:
  result = loss(model(inputs), target)
  f.returns([result])

# CHECK: "aten.convolution"
# CHECK: "aten.convolution"
# CHECK: torch.prim.ListConstruct
# CHECK: "aten._cat"
# CHECK: "aten._log_softmax.out"
# CHECK: "aten.nll_loss2d_forward"
mb.module.operation.print(large_elements_limit=2)
