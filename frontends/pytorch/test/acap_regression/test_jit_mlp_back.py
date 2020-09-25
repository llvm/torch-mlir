# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import npcomp.frontends.pytorch as torch_mlir
import npcomp.frontends.pytorch.test as test

# RUN: python %s | FileCheck %s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    
def main():
    device = torch_mlir.mlir_device()
    model = Net()
    tensor = torch.randn((64, 1, 28, 28),requires_grad=True)
    # CHECK: PASS! fwd check
    fwd_path = test.check_ref(model, tensor)
    
    target = torch.ones((64), dtype=torch.long)
    loss = F.nll_loss
        
    # CHECK: PASS! back check
    test.check_back(fwd_path, target, loss)
    
    # CHECK: PASS! fc1_weight_grad check
    test.compare(model.fc1.weight.grad, fwd_path[0].fc1.weight.grad, "fc1_weight_grad")

if __name__ == '__main__':
    main()
