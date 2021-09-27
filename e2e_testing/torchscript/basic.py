#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


@register_test_case(module_factory=lambda: MmModule())
def MmModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


# TODO: Investigate why RefBackend sometimes can't handle two calls in a row in
# the trace.
# It actually works, if MmModule_chained is run by itself, but if other tests
# are mixed with it, it fails with a mysterious-sounding low level ctypes error
# that exceeds my current ability to debug.
#
@register_test_case(module_factory=lambda: MmModule())
def MmModule_chained(module, tu: TestUtils):
    res = module.forward(tu.rand(4, 4), tu.rand(4, 4))
    module.forward(res, res)

# ==============================================================================


class BmmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


@register_test_case(module_factory=lambda: BmmModule())
def BmmModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 5, 4))


# ==============================================================================


# A subgraph with multiple mm ops.
class MmDagModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 4], torch.float32, True),
        ([4, 4], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.mm(lhs, torch.mm(lhs, rhs))


@register_test_case(module_factory=lambda: MmDagModule())
def MmDagModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 4))


# ==============================================================================


class MmTanhModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs, rhs):
        return torch.tanh(self.matmul(lhs, rhs))

    def matmul(self, lhs, rhs):
        return torch.mm(lhs, rhs)


@register_test_case(module_factory=lambda: MmTanhModule())
def MmTanhModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 2), tu.rand(2, 4))


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool2dModule())
def AdaptiveAvgPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9))


class FlattenStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(2, 4)

    @export
    @annotate_args([
        None,
        ([10, 3, 8, 9, 3, 4], torch.float32, True),
    ])
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenStaticModule())
def FlattenStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9, 3, 4))


class FlattenRank0Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(-1, -1)

    @export
    @annotate_args([
        None,
        ([], torch.float32, True),
    ])
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenRank0Module())
def FlattenRank0Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor(4.0))


class FlattenDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(2, 4)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, 9, 3, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenDynamicModule())
def FlattenDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9, 3, 4))


class MaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       dilation=2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dModule())
def MaxPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20) - 0.5)


class TransposeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True),
    ])
    def forward(self, x):
        return torch.transpose(x, 0, 1)


@register_test_case(module_factory=lambda: TransposeIntModule())
def TransposeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


class TensorsConcatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, y, z):
        return torch.cat([x, y, z], 1)


@register_test_case(module_factory=lambda: TensorsConcatModule())
def TensorsConcatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


class GatherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, tensor, indices):
        return torch.gather(tensor, 2, indices)


@register_test_case(module_factory=lambda: GatherModule())
def GatherModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), torch.tensor([[[1, 2, 3], [1, 2, 3]]]))
