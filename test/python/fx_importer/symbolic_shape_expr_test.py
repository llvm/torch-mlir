# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s
# This file contains tests of various op special forms that the fx_importer
# handles.

from typing import Optional

import torch
import torch.export
import torch.nn as nn

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_tanh_sigmoid_cat_shape_expr_import
def test_tanh_sigmoid_cat_shape_expr_import():
    class Basic(nn.Module):
        def forward(self, x, y):
            a = torch.tanh(x)
            b = torch.sigmoid(y)
            return torch.cat((a, a, b), dim=-1)

    x, y = torch.randn(5, 3, 2), torch.randn(5, 3, 5)

    dim_n = torch.export.Dim("n", min=5, max=10)
    dim_x2 = torch.export.Dim("x2", max=100)
    dim_y2 = torch.export.Dim("y2", max=50)

    # CHECK: torch.
    m = fx.export_and_import(
        Basic(),
        x,
        y,
        dynamic_shapes={
            "x": {0: dim_n, 2: dim_x2},
            "y": {0: dim_n, 2: dim_y2},
        },
    )
    print(m)


@run
# CHECK-LABEL: test_range_constraint_relation
def test_range_constraint_relation():
    class Basic(torch.nn.Module):
        def forward(self, x, y):
            return x + y[1:]

    x, y = torch.randn(5), torch.randn(6)
    dimx = torch.export.Dim("dimx", min=3, max=6)
    dimy = dimx + 1

    # CHECK: torch.
    m = fx.export_and_import(
        Basic(),
        x,
        y,
        dynamic_shapes={
            "x": {0: dimx},
            "y": {0: dimy},
        },
        experimental_support_mutation=True,
    )
    print(m)
