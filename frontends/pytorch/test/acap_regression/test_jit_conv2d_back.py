# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch.nn as nn
import torch.nn.functional as F
import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

dev = torch_mlir.mlir_device()

N = 3
Cin = 16
Cout = 4
w = 10
h = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout, (3,3))

    def forward(self, x):
        x = self.conv1(x)
        output = F.log_softmax(x, dim=1)
        return output

model = Net()
tensor = torch.randn(N, Cin, h, w)

# CHECK: PASS! fwd check
fwd_path = test.check_ref(model, tensor)

loss = torch.nn.NLLLoss()
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, Cout)

# CHECK: PASS! back check
test.check_back(fwd_path, target, loss)

# CHECK: PASS! weight_grad check
test.compare(model.conv1.weight.grad, fwd_path[0].conv1.weight.grad, "weight_grad")
# CHECK: PASS! bias_grad check
test.compare(model.conv1.bias.grad, fwd_path[0].conv1.bias.grad, "bias_grad")
