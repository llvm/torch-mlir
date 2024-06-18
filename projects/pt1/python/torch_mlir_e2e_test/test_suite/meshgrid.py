# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class MeshgridIndexingIJ(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3], torch.int64, True),
            ([4], torch.int64, True),
            ([5], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        x1, y1, z1 = torch.meshgrid(x, y, z, indexing="ij")
        return x1, y1, z1


@register_test_case(module_factory=lambda: MeshgridIndexingIJ())
def MeshgridIndexingIJ_basic(module, tu: TestUtils):
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6, 7])
    z = torch.tensor([8, 9, 10, 11, 12])
    module.forward(x, y, z)


class MeshgridIndexingXY(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3], torch.int64, True),
            ([4], torch.int64, True),
            ([5], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        x1, y1, z1 = torch.meshgrid(x, y, z, indexing="xy")
        return x1, y1, z1


@register_test_case(module_factory=lambda: MeshgridIndexingXY())
def MeshgridIndexingXY_basic(module, tu: TestUtils):
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6, 7])
    z = torch.tensor([8, 9, 10, 11, 12])
    module.forward(x, y, z)


class Meshgrid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3], torch.int64, True),
            ([4], torch.int64, True),
        ]
    )
    def forward(self, x, y):
        x1, y1 = torch.meshgrid(x, y)
        return x1, y1


@register_test_case(module_factory=lambda: Meshgrid())
def Meshgrid_basic(module, tu: TestUtils):
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6, 7])
    module.forward(x, y)
