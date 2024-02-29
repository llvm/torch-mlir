# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class GridSamplerBasic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 10, 10, 4], torch.float32, True),
        ([4, 6, 8, 2], torch.float32, True)
    ])
    def forward(self, x, g):
        interpolation_mode=0,
        padding_mode=0,
        align_corners=True,
        tRes = torch.ops.aten.grid_sampler(x, g, interpolation_mode[0], padding_mode[0], align_corners[0])
        #print(tRes)
        return 1
    
@register_test_case(
    module_factory=lambda: GridSamplerBasic())
def GridSamplerBasic_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(4, 10, 10, 4), tu.rand(4,6,8,2,low=-1.0, high=1.0))


class GridSamplerBasic2(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 4, 4], torch.float32, True),
        ([1, 1, 3, 2], torch.float32, True),
        ([1, 1, 1, 3], torch.float32, True)
    ])
    def forward(self, x, g, e):
        interpolation_mode=0,
        padding_mode=0,
        align_corners=True,
        tRes = torch.ops.aten.grid_sampler(x, g, interpolation_mode[0], padding_mode[0], align_corners[0])
        return 1

@register_test_case(
    module_factory=lambda: GridSamplerBasic2())
def GridSamplerBasic2_basic(
        module, tu: TestUtils):
    inp = torch.tensor([[[[0.4963, 0.7682, 0.0885, 0.1320],
          [0.3074, 0.6341, 0.4901, 0.8964],
          [0.4556, 0.6323, 0.3489, 0.4017],
          [0.0223, 0.1689, 0.2939, 0.5185]]]]).type(torch.FloatTensor)
    grd = torch.tensor([[[[-0.3498, -0.8196],[-0.2127,  0.2138],[-0.6515, -0.0513]]]]).type(torch.FloatTensor)
    exp = torch.tensor([[[[0.72483041723, 0.58586429802, 0.5077060201]]]]).type(torch.FloatTensor)
    module.forward(inp, grd, exp)


class GridSamplerBasic3(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 10, 10, 4], torch.float32, True),
        ([4, 6, 8, 2], torch.float32, True)
    ])
    def forward(self, x, g):
        tRes = torch.nn.functional.grid_sample(x, g, align_corners=True)
        return tRes
    
@register_test_case(
    module_factory=lambda: GridSamplerBasic3())
def GridSamplerBasic3_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(4, 10, 10, 4), tu.rand(4,6,8,2,low=-1.0, high=1.0))
