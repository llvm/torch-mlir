# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class MatmulDot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulDot())
def Matmul_dot(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3))
    
# ==============================================================================

class Matmul2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul2D())
def Matmul_2d(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(4, 5))
    
# ==============================================================================

class MatmulVecMat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulVecMat())
def Matmul_vecmat(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4, 5))
    
# ==============================================================================

class MatmulMatVec(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulMatVec())
def Matmul_matvec(module, tu: TestUtils):
    module.forward(tu.rand(4, 5), tu.rand(5))
    
# ==============================================================================

class Matmul3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul3D())
def Matmul_3d(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 5, 4))
    
# ==============================================================================

class Matmul4d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul4d())
def Matmul_4d(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))
    
# ==============================================================================

class Matmul4dStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5, 6, 7], torch.float32, True),
        ([4, 5, 7, 6], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: Matmul4dStatic())
def Matmul4dStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))

# ==============================================================================

class MatmulStaticBroadcast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 1, 6, 7], torch.float32, True),
        ([8, 1, 5, 7, 6], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulStaticBroadcast())
def MatmulStaticBroadcast_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 1, 6, 7), tu.rand(8, 1, 5, 7, 6))

# ==============================================================================

class MatmulSingleDynamicBatchDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, -1, -1, -1], torch.float32, True),
        ([4, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulSingleDynamicBatchDim())
def MatmulSingleDynamicBatchDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(4, 5, 7, 6))
    
# ==============================================================================

class MatmulBroadcastBatchDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


@register_test_case(module_factory=lambda: MatmulBroadcastBatchDim())
def MatmulBroadcastBatchDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5, 6, 7), tu.rand(5, 7, 6))
    
# ==============================================================================

class Mv(torch.nn.Module):

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, m, v):
        return torch.mv(m, v)


@register_test_case(module_factory=lambda: Mv())
def Mv_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2), tu.rand(2))