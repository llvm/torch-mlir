#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ArgminModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten.argmin(a)


@register_test_case(module_factory=lambda: ArgminModule())
def ArgminModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

@register_test_case(module_factory=lambda: ArgminModule())
def ArgminModule_multiple_mins(module, tu: TestUtils):
    """To cover the special case that the minimal value occurs more than once.
    The pytorch convention is here to consider the first occurence as the argmax.
    """
    module.forward(torch.full((3,4), 0.0))

# ==============================================================================

class ArgminWithDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.argmin(a, dim=1)

@register_test_case(module_factory=lambda: ArgminWithDimModule())
def ArgminModule_with_dim(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class ArgminKeepDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None, 
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.argmin(a, 0, True)

@register_test_case(module_factory=lambda: ArgminKeepDimsModule())
def ArgminModule_keepDim(module, tu: TestUtils):
    module.forward(tu.rand(4, 6))
