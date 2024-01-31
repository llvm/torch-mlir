# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
from torch_mlir import torchscript

class BasicModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.sin(x)

example_arg = torch.ones(2, 3)
example_args = torchscript.ExampleArgs.get(example_arg)

traced = torch.jit.trace(BasicModule(), example_arg)
print(torchscript.compile(traced, example_args))
# CHECK: module
# CHECK-DAG: func.func @forward

traced = torch.jit.trace(BasicModule(), example_arg)
try:
    # CHECK: Model does not have exported method 'nonexistent', requested in `example_args`. Consider adding `@torch.jit.export` to the method definition.
    torchscript.compile(traced, torchscript.ExampleArgs().add_method("nonexistent", example_arg))
except Exception as e:
    print(e)
