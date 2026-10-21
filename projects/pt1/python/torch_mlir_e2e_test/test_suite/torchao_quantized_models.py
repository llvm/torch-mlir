# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torchao

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class TorchaoQuantizeAffinePerToken(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([2, 16, 64], torch.float32, True),
            ([2, 16, 1], torch.float32, True),
            ([2, 16, 1], torch.int8, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.torchao.quantize_affine.default(
            x, [1, 1, 64], scale, zero_point, torch.int8
        )


@register_test_case(module_factory=lambda: TorchaoQuantizeAffinePerToken())
def TorchaoQuantizeAffinePerToken_basic(module, tu: TestUtils):
    x = 4 * tu.rand(2, 16, 64) - 2
    module.forward(
        x,
        x.abs().amax(dim=-1, keepdim=True) / 127,
        tu.randint(2, 16, 1, low=-10, high=10).to(torch.int8),
    )


# ==============================================================================


class TorchaoDequantizeAffinePerGroup(torch.nn.Module):
    @export
    @annotate_args(
        [
            None,
            ([4, 16], torch.int8, True),
            ([4, 2], torch.float32, True),
            ([4, 2], torch.float32, True),
        ]
    )
    def forward(self, x, scale, zero_point):
        return torch.ops.torchao.dequantize_affine.default(
            x, [1, 8], scale.to(torch.bfloat16), zero_point, torch.int8, -128, 127
        )


@register_test_case(module_factory=lambda: TorchaoDequantizeAffinePerGroup())
def TorchaoDequantizeAffinePerGroup_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 16, low=-128, high=128).to(torch.int8),
        tu.rand(4, 2) + 0.01,
        20 * tu.rand(4, 2) - 10,
    )
