# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch

import torch_mlir


class TanhModule(torch.nn.Module):
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

# TensorPlaceholder support.
placeholder = torch_mlir.TensorPlaceholder.like(
    tanh_example_input, dynamic_axes=[1])
print(torch_mlir.compile(TanhModule(), [placeholder],
                         use_tracing=True, ignore_traced_shapes=True))
# CHECK-LABEL: @forward
# CHECK: torch.aten.tanh %{{.*}} : !torch.vtensor<[2,?],f32> -> !torch.vtensor<[2,?],f32>

try:
    # CHECK: `ignore_traced_shapes` requires `use_tracing`
    torch_mlir.compile(TanhModule(), [placeholder], ignore_traced_shapes=True)
except Exception as e:
    print(e)


try:
    # CHECK: TensorPlaceholder can only be used with tracing when `ignore_traced_shapes=True`
    torch_mlir.compile(TanhModule(), [placeholder], use_tracing=True)
except Exception as e:
    print(e)


class DictModule(torch.nn.Module):
    def forward(self, x):
        return x['a'] * 2.0


try:
    # CHECK: Only Tensors, TensorPlaceholders, or a sequences of Tensors and TensorPlaceholders are supported as inputs.
    torch_mlir.compile(DictModule(), {'a': torch.tensor(3.0)}, use_tracing=True)
except Exception as e:
    print(e)


try:
    # CHECK: Only Tensors, TensorPlaceholders, or a sequences of Tensors and TensorPlaceholders are supported as inputs.
    torch_mlir.compile(DictModule(), [{'a': torch.tensor(3.0)}], use_tracing=True)
except Exception as e:
    print(e)