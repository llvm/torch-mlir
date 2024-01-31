# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

from torch_mlir import torchscript

class AddmmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        return torch.ops.aten.addmm(x, y, z)

example_args = 3 * [torchscript.TensorPlaceholder([-1, -1], torch.float32)]

print(torchscript.compile(AddmmModule(), example_args,
      output_type="torch", backend_legal_ops=["aten.addmm"]))
# CHECK-LABEL: @forward
# CHECK: torch.aten.addmm
