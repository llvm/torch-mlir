# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: %PYTHON %s | FileCheck %s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        #self.maxpool2d = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(9216*4, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.maxpool2d(x)
        x = x.view((64,9216*4))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    model = Net()
    tensor = torch.randn((64, 1, 28, 28), requires_grad=True)

    # CHECK: PASS! fwd check
    fwd_path = test.check_fwd(model, tensor)

    target = torch.ones((64), dtype=torch.long)
    loss = F.nll_loss

    # CHECK: PASS! back check
    test.check_back(fwd_path, target, loss)

    # CHECK: PASS! weight_grad check
    test.compare(model.conv2.weight.grad,
                 fwd_path[0].conv2.weight.grad, "weight_grad")
    # CHECK: PASS! bias_grad check
    test.compare(model.conv2.bias.grad,
                 fwd_path[0].conv2.bias.grad, "bias_grad")
    # CHECK: PASS! fc1_weight_grad check
    test.compare(model.fc1.weight.grad,
                 fwd_path[0].fc1.weight.grad, "fc1_weight_grad")

if __name__ == '__main__':
    main()
