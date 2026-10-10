# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class DepthToSpaceCRDBlockSize3Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 18, 2, 2], torch.float32, True),
    ])
    def forward(self, x):
        # pixel_shuffle natively uses CRD mode permutation logic
        # upscale_factor=3 translates to blocksize=3
        return torch.nn.functional.pixel_shuffle(x, upscale_factor=3)


@register_test_case(module_factory=lambda: DepthToSpaceCRDBlockSize3Module())
def DepthToSpaceCRDBlockSize3_basic(module, tu: TestUtils):
    # Input channels must be divisible by blocksize^2
    module.forward(tu.rand(1, 18, 2, 2))
