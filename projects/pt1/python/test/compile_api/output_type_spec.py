# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir import torchscript

class TanhModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ops.aten.tanh(x)

tanh_example_input = torch.ones(2, 3)

print(torchscript.compile(TanhModule(), tanh_example_input, output_type=torchscript.OutputType.TORCH))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
print(torchscript.compile(TanhModule(), tanh_example_input, output_type="torch"))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
