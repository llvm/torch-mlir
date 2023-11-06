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
print(torch_mlir.compile(TanhModule(), tanh_example_input))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>

# Use a TensorPlaceholder to represent dynamic axes.
placeholder = torch_mlir.TensorPlaceholder.like(tanh_example_input, dynamic_axes=[1])
print(torch_mlir.compile(TanhModule(), placeholder))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,?],f32> -> !torch.vtensor<[2,?],f32>

# Explicitly construct a TensorPlaceholder.
placeholder = torch_mlir.TensorPlaceholder([-1, 2], torch.float32)
print(torch_mlir.compile(TanhModule(), placeholder))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[?,2],f32> -> !torch.vtensor<[?,2],f32>

# Basic smoke test for the raw output type.
print(torch_mlir.compile(TanhModule(), tanh_example_input, output_type=torch_mlir.OutputType.RAW))
# CHECK: torch.nn_module {
# CHECK: } : !torch.nn.Module<"{{.*}}.TanhModule">

class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, lhs, rhs  ):
        return torch.ops.aten.mm(lhs, rhs)

# N > 1 inputs.
mm_example_inputs = [torch.ones(2, 3), torch.ones(3, 4)]
print(torch_mlir.compile(MmModule(), mm_example_inputs))
# CHECK-LABEL: @forward
# CHECK: torch.aten.mm %{{.*}}, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[2,4],f32>

# Mixes Tensor's and TensorPlaceholder's.
mm_dynamic_inputs = [mm_example_inputs[0], torch_mlir.TensorPlaceholder.like(mm_example_inputs[1], dynamic_axes=[1])]
print(torch_mlir.compile(MmModule(), mm_dynamic_inputs))
# CHECK-LABEL: @forward
# CHECK: torch.aten.mm %{{.*}}, %{{.*}} : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,?],f32> -> !torch.vtensor<[2,?],f32>
