# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.nn as nn
from torch.export import Dim
from torch.library import Library, impl, impl_abstract

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_tanh_sigmoid_cat_custom_op
def test_tanh_sigmoid_cat_custom_op():
    m = Library("my_custom_library", "DEF")
    m.define("custom_op(Tensor x, Tensor y, Tensor z) -> Tensor")

    @impl(m, "custom_op", "CompositeExplicitAutograd")
    def custom_op(x, y, z):
        a = torch.tanh(x)
        b = torch.sigmoid(y)
        return torch.cat((a, a, b, z), dim=1)

    @impl_abstract("my_custom_library::custom_op")
    def custom_op_meta(x, y, z):
        result = custom_op(x, y, z)
        return torch.empty_like(result)

    class TanhSigmoidCatCustomOp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            return torch.ops.my_custom_library.custom_op(x, y, z)

    # Sample inputs
    x = torch.randn(5, 2, 3)
    y = torch.randn(5, 6, 3)
    z = torch.randn(5, 4, 3)

    # Dynamic dim constraints
    dim_n = Dim("n", min=5, max=10)
    dim_x1 = Dim("x1", max=100)
    dim_y1 = Dim("y1", max=50)
    dim_z1 = Dim("z1")
    dynamic_shapes = {
        "x": {0: dim_n, 1: dim_x1},
        "y": {0: dim_n, 1: dim_y1},
        "z": {0: dim_n, 1: dim_z1},
    }

    m = fx.export_and_import(
        TanhSigmoidCatCustomOp(),
        x,
        y,
        z,
        dynamic_shapes=dynamic_shapes,
    )
    print(m)
