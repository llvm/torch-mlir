# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class DeterminantModule(torch.nn.Module):
    @export
    @annotate_args([None, [(4, 4), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.det(A)


@register_test_case(module_factory=lambda: DeterminantModule())
def DeterminantModule_F32(module, tu: TestUtils):
    A = tu.rand(4, 4).to(dtype=torch.float32)
    module.forward(A)


class DeterminantBatchedModule(torch.nn.Module):
    @export
    @annotate_args([None, [(3, 4, 4), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.det(A)


@register_test_case(module_factory=lambda: DeterminantBatchedModule())
def DeterminantBatchedModule_F32(module, tu: TestUtils):
    A = tu.rand(3, 4, 4).to(dtype=torch.float32)
    module.forward(A)


class DeterminantDynamicModule(torch.nn.Module):
    @export
    @annotate_args([None, [(-1, -1, -1), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.det(A)


@register_test_case(module_factory=lambda: DeterminantBatchedModule())
def DeterminantDynamicModule_F32(module, tu: TestUtils):
    A = tu.rand(3, 4, 4).to(dtype=torch.float32)
    module.forward(A)


# ==============================================================================


class SignAndLogarithmOfDeterminantModule(torch.nn.Module):
    @export
    @annotate_args([None, [(4, 4), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.slogdet(A)


@register_test_case(module_factory=lambda: SignAndLogarithmOfDeterminantModule())
def SignAndLogarithmOfDeterminantModule_F32(module, tu: TestUtils):
    A = tu.rand(4, 4).to(dtype=torch.float32)
    module.forward(A)


class SignAndLogarithmOfDeterminantBatchedModule(torch.nn.Module):
    @export
    @annotate_args([None, [(3, 4, 4), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.slogdet(A)


@register_test_case(module_factory=lambda: SignAndLogarithmOfDeterminantBatchedModule())
def SignAndLogarithmOfDeterminantBatchedModule_F32(module, tu: TestUtils):
    A = tu.rand(3, 4, 4).to(dtype=torch.float32)
    module.forward(A)


class SignAndLogarithmOfDeterminantDynamicModule(torch.nn.Module):
    @export
    @annotate_args([None, [(-1, -1, -1), torch.float32, True]])
    def forward(self, A):
        return torch.linalg.slogdet(A)


@register_test_case(module_factory=lambda: SignAndLogarithmOfDeterminantBatchedModule())
def SignAndLogarithmOfDeterminantDynamicModule_F32(module, tu: TestUtils):
    A = tu.rand(3, 4, 4).to(dtype=torch.float32)
    module.forward(A)
