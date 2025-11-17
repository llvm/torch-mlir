# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.annotations import annotate_args, export
from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case


class TransformerEncoderModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([1, 4, 8], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.layer(x)


@register_test_case(module_factory=lambda: TransformerEncoderModule())
def TransformerEncoderModule_basic(module, tu: TestUtils):
    x = tu.rand(1, 4, 8)
    module.forward(x)
