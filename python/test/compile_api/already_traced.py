# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch_mlir

class BasicModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.sin(x)

example_arg = torch.ones(2, 3)
example_args = torch_mlir.ExampleArgs.get(example_arg)

traced = torch.jit.trace(BasicModule(), example_arg)
print(torch_mlir.compile(traced, example_args))
# CHECK: module
# CHECK-DAG: func.func @forward

traced = torch.jit.trace(BasicModule(), example_arg)
try:
    # CHECK: Model does not have exported method 'nonexistent', requested in `example_args`. Consider adding `@torch.jit.export` to the method definition.
    torch_mlir.compile(traced, torch_mlir.ExampleArgs().add_method("nonexistent", example_arg))
except Exception as e:
    print(e)
