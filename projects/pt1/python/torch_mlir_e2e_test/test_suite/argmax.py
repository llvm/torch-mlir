#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ArgmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.argmax(a)


@register_test_case(module_factory=lambda: ArgmaxModule())
def ArgmaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class ArgmaxWithDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.argmax(a, dim=1)

@register_test_case(module_factory=lambda: ArgmaxWithDimModule())
def ArgmaxModule_with_dim(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ArgmaxKeepDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None, 
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.argmax(a, 0, True)

@register_test_case(module_factory=lambda: ArgmaxKeepDimsModule())
def ArgmaxModule_keepDim(module, tu: TestUtils):
    module.forward(tu.rand(4, 6))
