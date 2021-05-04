# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

# RUN: %PYTHON %s | npcomp-opt | FileCheck %s

import torch
from torch.autograd import Variable
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
      x = self.conv1(x)
      output = F.log_softmax(x, dim=1)
      return output

model = Net(Cin, Cout)

inputs = torch.ones((N,Cin,h,w))
loss = torch.nn.NLLLoss()
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, Cout)

mb = torch_mlir.ModuleBuilder()
with mb.capture_function("resa", [inputs, target]) as f:
  result = loss(model(inputs), target)
  result.backward()
  f.returns([result] + [p.grad for p in model.parameters()])

# CHECK: torch.operator "aten.convolution"
# CHECK: torch.operator "aten._log_softmax"
# CHECK: %[[FWD:.*]]:2 = torch.operator "aten.nll_loss2d_forward"
# CHECK: torch.operator "aten.nll_loss2d_backward"
# CHECK: torch.operator "aten._log_softmax_backward_data"
# CHECK: %[[BWD_CONV:.*]]:3 = torch.operator "aten.convolution_backward_overrideable"
# CHECK: %[[BWD_CONV_WEIGHTS:.*]] = torch.operator "aten.copy_"{{.*}}%[[BWD_CONV]]#1
# CHECK: %[[BWD_CONV_BIAS:.*]] = torch.operator "aten.copy_"{{.*}}%[[BWD_CONV]]#2
# CHECK: return %[[FWD]]#0, %[[BWD_CONV_WEIGHTS]], %[[BWD_CONV_BIAS]]
mb.module.operation.print(large_elements_limit=2)
