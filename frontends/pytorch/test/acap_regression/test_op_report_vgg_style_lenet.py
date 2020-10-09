# -*- Python -*-
# This file is licensed under a pytorch-style license
# See frontends/pytorch/LICENSE for license information.

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import npcomp.frontends.pytorch as torch_mlir
import json

# RUN: %PYTHON %s | FileCheck %s

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=0)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.maxpool1(x)
        print(x.shape)

        x = self.conv2(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = self.maxpool2(x)
        print(x.shape)
        x = x.view(8, 6*6*16)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output

def main():

    test_status = "PASS!"

    # CHECK-LABEL: test_op_report_vgg_style_lenet
    # CHECK:       PASS!
    print("test_op_report_vgg_style_lenet")

    device = torch_mlir.mlir_device()

    model = Net().to(device)
    ref_tensor = torch.randn((8, 1, 30, 30))
    tensor = ref_tensor.clone().to(device)

    result = model(tensor)
    target = torch.ones((8), dtype=torch.long).to(device)
    loss = F.nll_loss(result, target)
    loss.backward()

    mlir0 = torch_mlir.get_mlir(model.conv1.weight.grad)
    print(mlir0)
    report = torch_mlir.op_report(mlir0)
    print(report)

    report_dict = report
    expected = 32
    if (len(report_dict) != expected):
        print("### ERROR: Expecting",expected,"items in the report, but got ",len(report_dict))
        test_status = "FAIL!"

    # Every item should have a read and a write
    for key, value in report_dict.items():
        if not 'reads' in value:
            print(f"### ERROR: {key} does not contain the required reads field")
            test_status = "FAIL!"
        if not 'writes' in value:
            print(f"### ERROR: {key} does not contain the required writes field")
            test_status = "FAIL!"
        if "convolution" in key:
            if not 'ops:MAC' in value:
                print(f"### ERROR: convolution {key} does not contain the required MAC field")
                test_status = "FAIL!"
        if "mm" in key:
            if not 'ops:MAC' in value:
                print(f"### ERROR: mm {key} does not contain the required MAC field")
                test_status = "FAIL!"


    print(test_status)

if __name__ == '__main__':
    main()
