# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class AtenHannWindowPeriodicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.aten.hann_window(20, False)


@register_test_case(module_factory=lambda: AtenHannWindowPeriodicModule())
def AtenHannWindowPeriodicModule_basic(module, tu: TestUtils):
    module.forward()
