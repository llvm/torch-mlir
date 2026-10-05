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


class GridSamplerNearestZeros(torch.nn.Module):
    def __init__(self, align_corners):
        super().__init__()
        self.align_corners = align_corners

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, 2], torch.float32, True),
        ]
    )
    def forward(self, x, grid):
        return torch.ops.aten.grid_sampler(x, grid, 1, 0, self.align_corners)


def _grid_sampler_nearest_zeros_inputs(module, align_corners):
    # Use exactly representable normalized coordinates. Distinct nonzero values
    # in each batch/channel distinguish zero padding from a sample at the origin.
    size = 5 if align_corners else 4
    x = torch.arange(1, 4 * size * size + 1, dtype=torch.float32).reshape(
        2, 2, size, size
    )
    pixels = torch.tensor(
        [
            [-1.5, 0],
            [-0.5, 0],
            [0.5, 0],
            [1.5, 0],
            [2.5, 0],
            [3.5, 0],
            [4.5, 0],
            [0, 1.5],
            [0, 2.5],
            [0, -0.5],
            [0, -1.5],
            [100, 100],
            [-100, -100],
        ],
        dtype=torch.float32,
    )
    grid = 2 * pixels / (size - 1) - 1 if align_corners else (2 * pixels + 1) / size - 1
    grid = torch.cat(
        [
            grid,
            torch.nextafter(grid, torch.full_like(grid, -torch.inf)),
            torch.nextafter(grid, torch.full_like(grid, torch.inf)),
            torch.tensor([[3.0e38, -3.0e38]], dtype=torch.float32),
        ]
    )
    module.forward(x, grid.reshape(1, 1, -1, 2).repeat(2, 1, 1, 1))
    # With align_corners=False, reassociating unnormalization moves this
    # coordinate off the x=2.5 tie.
    x = torch.arange(1, 16, dtype=torch.float32).reshape(1, 1, 3, 5)
    grid = torch.tensor([[[[0.20000006258487701, 0.0]]]], dtype=torch.float32)
    module.forward(x, grid)
    # Exercise dynamic sizes and singleton spatial dimensions as well.
    singleton = torch.tensor([[[[7.0]]]])
    grid = torch.tensor(
        [[[[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [-100.0, -100.0], [100.0, 100.0]]]]
    )
    grid = torch.cat(
        [grid, torch.nextafter(grid, torch.full_like(grid, torch.inf))], dim=2
    )
    module.forward(singleton, grid)


@register_test_case(module_factory=lambda: GridSamplerNearestZeros(False))
def GridSamplerNearestZeros_basic(module, tu: TestUtils):
    _grid_sampler_nearest_zeros_inputs(module, False)


@register_test_case(module_factory=lambda: GridSamplerNearestZeros(True))
def GridSamplerNearestZeros_align_corners(module, tu: TestUtils):
    _grid_sampler_nearest_zeros_inputs(module, True)
