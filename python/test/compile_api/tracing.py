# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

import torch_mlir

class TanhModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.ops.aten.tanh(x)

tanh_example_input = torch.ones(2, 3)

# Simplest case: One example argument.
print(torch_mlir.compile(TanhModule(), tanh_example_input, use_tracing=True))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>

# Simplest case: Passed as a tuple.
print(torch_mlir.compile(TanhModule(), (tanh_example_input,), use_tracing=True))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>

# Simplest case: Passed as a list.
print(torch_mlir.compile(TanhModule(), [tanh_example_input], use_tracing=True))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
