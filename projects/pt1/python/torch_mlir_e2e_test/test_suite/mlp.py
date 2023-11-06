# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch.nn as nn

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

# Multi-layer perceptron (MLP) models.

class Mlp1LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = nn.Linear(3, 5)
        self.tanh0 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.tanh0(self.fc0(x))

@register_test_case(module_factory=lambda: Mlp1LayerModule())
def Mlp1LayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class Mlp2LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        N_HIDDEN = 5
        self.fc0 = nn.Linear(3, N_HIDDEN)
        self.tanh0 = nn.Tanh()
        self.fc1 = nn.Linear(N_HIDDEN, 2)
        self.tanh1 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.tanh0(self.fc0(x))
        x = self.tanh1(self.fc1(x))
        return x

@register_test_case(module_factory=lambda: Mlp2LayerModule())
def Mlp2LayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class Mlp2LayerModuleNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        N_HIDDEN = 5
        self.fc0 = nn.Linear(3, N_HIDDEN, bias=False)
        self.tanh0 = nn.Tanh()
        self.fc1 = nn.Linear(N_HIDDEN, 2, bias=False)
        self.tanh1 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.tanh0(self.fc0(x))
        x = self.tanh1(self.fc1(x))
        return x

@register_test_case(module_factory=lambda: Mlp2LayerModuleNoBias())
def Mlp2LayerModuleNoBias_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class BatchMlpLayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = nn.Linear(3, 5)
        self.tanh0 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.tanh0(self.fc0(x))

@register_test_case(module_factory=lambda: BatchMlpLayerModule())
def BatchMlpLayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(7, 5, 3))
