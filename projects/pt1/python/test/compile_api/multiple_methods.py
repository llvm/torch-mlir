# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
from torch_mlir import torchscript


class TwoMethodsModule(torch.nn.Module):
    def sin(self, x):
        return torch.ops.aten.sin(x)

    def cos(self, x):
        return torch.ops.aten.cos(x)


example_args = torchscript.ExampleArgs()
example_args.add_method("sin", torch.ones(2, 3))
example_args.add_method("cos", torch.ones(2, 4))

# Note: Due to https://github.com/pytorch/pytorch/issues/88735 we need to
# check the `use_tracing` case first.

print(torchscript.compile(TwoMethodsModule(), example_args, use_tracing=True))
# CHECK: module
# CHECK-DAG: func.func @sin
# CHECK-DAG: func.func @cos

# As a convenience, we do the equivalent of calling `torch.jit.export` on all
# methods indicated in `example_args` before calling `torch.jit.script`.
# Otherwise the user would have to do this manually, which is tedious. This
# technically mutates the user input model which is not great but probably okay
# for this kind of API sugar. Users can always take full control of the process
# by scripting the model themselves before passing it to `torchscript.compile`.
print(torchscript.compile(TwoMethodsModule(), example_args))
# CHECK: module
# CHECK-DAG: func.func @sin
# CHECK-DAG: func.func @cos
