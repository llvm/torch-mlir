# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
from torch_mlir import torchscript


class BasicModule(torch.nn.Module):
    @torch.jit.export
    def sin(self, x):
        return torch.ops.aten.sin(x)


example_args = torchscript.ExampleArgs()
example_args.add_method("sin", torch.ones(2, 3))

scripted = torch.jit.script(BasicModule())
print(torchscript.compile(scripted, example_args))
# CHECK: module
# CHECK-DAG: func.func @sin

scripted = torch.jit.script(BasicModule())
try:
    # CHECK: Model does not have exported method 'nonexistent', requested in `example_args`. Consider adding `@torch.jit.export` to the method definition.
    torchscript.compile(scripted, torchscript.ExampleArgs().add_method("nonexistent", torch.ones(2, 3)))
except Exception as e:
    print(e)
