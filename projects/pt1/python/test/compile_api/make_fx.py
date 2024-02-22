# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import functorch
import torch

from torch_mlir import torchscript

def simple(x):
    return x * x

example_input = torch.randn(1,)
graph = functorch.make_fx(simple)(torch.randn(1,))

# Simplest case: One example argument.
print(torchscript.compile(graph, example_input))
# CHECK-LABEL: @forward
# CHECK: torch.aten.mul.Tensor %{{.*}} : !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32> -> !torch.vtensor<[1],f32>