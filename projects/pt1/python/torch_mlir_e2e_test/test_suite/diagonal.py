#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class DiagonalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, a):
        return torch.ops.aten.diagonal(a)


@register_test_case(module_factory=lambda: DiagonalModule())
def DiagonalModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))

@register_test_case(module_factory=lambda: DiagonalModule())
def DiagonalModule_nonsquare(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class DiagonalTransposedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.diagonal(a, dim1=1, dim2=0)

@register_test_case(module_factory=lambda: DiagonalTransposedModule())
def DiagonalModule_transposed(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))

# ==============================================================================

class DiagonalWithDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.diagonal(a, dim1=0, dim2=1)

@register_test_case(module_factory=lambda: DiagonalWithDimsModule())
def DiagonalModule_with_dims(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class DiagonalWithNegativeDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.diagonal(a, dim1=-2, dim2=-1)

@register_test_case(module_factory=lambda: DiagonalWithNegativeDimsModule())
def DiagonalModule_with_negative_dims(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))

# ==============================================================================

class DiagonalWithOffsetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.diagonal(a, offset=1)

@register_test_case(module_factory=lambda: DiagonalWithOffsetModule())
def DiagonalModule_with_offset(module, tu: TestUtils):
    module.forward(tu.rand(4, 6))

# ==============================================================================

class DiagonalWithDimsOffsetModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.diagonal(a, dim1=0, dim2=1, offset=-1)

@register_test_case(module_factory=lambda: DiagonalWithDimsOffsetModule())
def DiagonalModule_with_dims_and_offset(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))
