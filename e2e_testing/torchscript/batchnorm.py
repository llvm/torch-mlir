#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================
class BatchNorm1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1d = torch.nn.BatchNorm1d(4)
        self.bn1d.eval()
        self.bn1d.running_mean = torch.tensor([0.5, 0.4, 0.3, 0.6])
        self.bn1d.running_var = torch.tensor([3.0, 2.0, 4.0, 5.0])
        self.bn1d.weight = torch.nn.Parameter(torch.tensor([3.0, 2.0, 4.0, 5.0]))
        self.bn1d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4, 0.3, 0.6]))
    @export
    @annotate_args([
        None,
        ([10, 4, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn1d(x)

@register_test_case(module_factory=lambda: BatchNorm1DModule())
def BatchNorm1DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 3))

# ==============================================================================
class BatchNorm2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn2d = torch.nn.BatchNorm2d(2)
        self.bn2d.eval()
        self.bn2d.running_mean = torch.tensor([0.5, 0.4])
        self.bn2d.running_var = torch.tensor([3.0, 2.0])
        self.bn2d.weight = torch.nn.Parameter(torch.tensor([3.0, 2.0]))
        self.bn2d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4]))
    @export
    @annotate_args([
        None,
        ([10, 2, 3, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn2d(x)

@register_test_case(module_factory=lambda: BatchNorm2DModule())
def BatchNorm2DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 2, 3, 3))

# ==============================================================================
class BatchNorm3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn3d = torch.nn.BatchNorm3d(5)
        self.bn3d.eval()
        self.bn3d.running_mean = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.4])
        self.bn3d.running_var = torch.tensor([3.0, 2.0, 4.0, 2.0, 3.0])
        self.bn3d.weight = torch.nn.Parameter(torch.tensor([3.0, 2.0, 4.0, 2.0, 3.0]))
        self.bn3d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4, 0.3, 0.2, 0.4]))
    @export
    @annotate_args([
        None,
        ([2, 5, 3, 6, 4], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn3d(x)

@register_test_case(module_factory=lambda: BatchNorm3DModule())
def BatchNorm3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 3, 6, 4))
