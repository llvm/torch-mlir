# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class GridSamplerBasic1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([7, 8, 12, 4], torch.float32, True),
            ([7, 11, 13, 2], torch.float32, True),
        ]
    )
    def forward(self, x, g):
        interpolation_mode = (0,)
        padding_mode = (0,)
        align_corners = (True,)
        tRes = torch.ops.aten.grid_sampler(
            x, g, interpolation_mode[0], padding_mode[0], align_corners[0]
        )
        return tRes


@register_test_case(module_factory=lambda: GridSamplerBasic1())
def GridSamplerBasic1_basic(module, tu: TestUtils):
    inp = torch.rand(7, 8, 12, 4)
    grd = torch.rand(7, 11, 13, 2) * 2.0 - 1.0
    module.forward(inp, grd)


class GridSamplerBasic2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [None, ([1, 1, 4, 4], torch.float32, True), ([1, 1, 3, 2], torch.float32, True)]
    )
    def forward(self, x, g):
        interpolation_mode = (0,)
        padding_mode = (0,)
        align_corners = (True,)
        tRes = torch.ops.aten.grid_sampler(
            x, g, interpolation_mode[0], padding_mode[0], align_corners[0]
        )
        return tRes


@register_test_case(module_factory=lambda: GridSamplerBasic2())
def GridSamplerBasic2_basic(module, tu: TestUtils):
    inp = torch.tensor(
        [
            [
                [
                    [0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185],
                ]
            ]
        ]
    ).type(torch.FloatTensor)
    grd = torch.tensor(
        [[[[-0.3498, -0.8196], [-0.2127, 0.2138], [-0.6515, -0.0513]]]]
    ).type(torch.FloatTensor)
    module.forward(inp, grd)


class GridSamplerBasic3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [None, ([1, 1, 4, 4], torch.float32, True), ([1, 1, 3, 2], torch.float32, True)]
    )
    def forward(self, x, g):
        interpolation_mode = (0,)
        padding_mode = (0,)
        align_corners = (False,)
        tRes = torch.ops.aten.grid_sampler(
            x, g, interpolation_mode[0], padding_mode[0], align_corners[0]
        )
        return tRes


@register_test_case(module_factory=lambda: GridSamplerBasic3())
def GridSamplerBasic3_basic(module, tu: TestUtils):
    inp = torch.tensor(
        [
            [
                [
                    [0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185],
                ]
            ]
        ]
    ).type(torch.FloatTensor)
    grd = torch.tensor(
        [[[[-0.3498, -0.8196], [-0.2127, 0.2138], [-0.6515, -0.0513]]]]
    ).type(torch.FloatTensor)
    module.forward(inp, grd)


class GridSamplerBasic4(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [None, ([1, 1, 4, 4], torch.float32, True), ([1, 1, 3, 2], torch.float32, True)]
    )
    def forward(self, x, g):
        interpolation_mode = (1,)
        padding_mode = (0,)
        align_corners = (False,)
        tRes = torch.ops.aten.grid_sampler(
            x, g, interpolation_mode[0], padding_mode[0], align_corners[0]
        )
        return tRes


@register_test_case(module_factory=lambda: GridSamplerBasic4())
def GridSamplerBasic4_basic(module, tu: TestUtils):
    inp = torch.tensor(
        [
            [
                [
                    [0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185],
                ]
            ]
        ]
    ).type(torch.FloatTensor)
    grd = torch.tensor(
        [[[[-0.3498, -0.8196], [-0.2127, 0.2138], [-0.6515, -0.0513]]]]
    ).type(torch.FloatTensor)
    module.forward(inp, grd)
