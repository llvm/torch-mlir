#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_torchscript.e2e_test.framework import TestUtils
from torch_mlir_torchscript.e2e_test.registry import register_test_case
from torch_mlir_torchscript.annotations import annotate_args, export

# ==============================================================================


class ListLiteralModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    def forward(self, x: int):
        return [x, x]


@register_test_case(module_factory=lambda: ListLiteralModule())
def ListLiteralModule_basic(module, tu: TestUtils):
    module.forward(3)
