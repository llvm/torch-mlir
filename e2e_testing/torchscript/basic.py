# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

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

class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True)
    ])
    def forward(self, x):
        return x.permute(0, 2, 1)

@register_test_case(module_factory=lambda: PermuteModule())
def PermuteModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))

class TransposeIntNegDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True),
    ])
    def forward(self, x):
        return torch.transpose(x, -1, -2)


@register_test_case(module_factory=lambda: TransposeIntNegDimsModule())
def TransposeIntNegDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


class PermuteNegativeIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True)
    ])
    def forward(self, x):
        return x.permute(0, -1, 1)

@register_test_case(module_factory=lambda: PermuteNegativeIndexModule())
def PermuteNegativeIndexModule_basic(module, tu: TestUtils):
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

class AddSizeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        # This is a workaround for not supporting scalar arguments.
        # TODO: pass in dim as an argument to the forward method when scalar
        # arguments are supported.
        return tensor.add(tensor, alpha=tensor.size(1))


@register_test_case(module_factory=lambda: AddSizeIntModule())
def AddSizeIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3))


class AddSizeIntNegDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        # This is a workaround for not supporting scalar arguments.
        # TODO: pass in dim as an argument to the forward method when scalar
        # arguments are supported.
        return tensor.add(tensor, alpha=tensor.size(-2))


@register_test_case(module_factory=lambda: AddSizeIntNegDimModule())
def AddSizeIntNegDimModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 3))

class EmbeddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(num_embeddings=100,
                                        embedding_dim=50,
                                        padding_idx=4)

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModule())
def EmbeddingModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(100, (3, 3)))


class SoftmaxIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.softmax = torch.nn.Softmax(2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntModule())
def SoftmaxIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4))


class SoftmaxIntNegDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.softmax = torch.nn.Softmax(-2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntNegDimModule())
def SoftmaxIntNegDimModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4))


class SoftmaxIntArgTypeF64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.softmax = torch.nn.Softmax(2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return self.softmax.forward(tensor)

@register_test_case(module_factory=lambda: SoftmaxIntArgTypeF64Module())
def SoftmaxIntArgTypeF64Module_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4).double())

class BroadcastToModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, 1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.broadcast_to(x, [1, -1, -1, 4])


@register_test_case(module_factory=lambda: BroadcastToModule())
def BroadcastToModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 1))

class ExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, 1], torch.float32, True),
    ])
    def forward(self, x):
        return x.expand([1, -1, -1, 4])


@register_test_case(module_factory=lambda: ExpandModule())
def ExpandModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 1))

class OnesModuleInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ones(3, 4, dtype=torch.int64)

@register_test_case(module_factory=lambda: OnesModuleInt())
def OnesModuleInt_basic(module, tu: TestUtils):
    module.forward()

class OnesModuleFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ones(3, 4, dtype=torch.float32)

@register_test_case(module_factory=lambda: OnesModuleFloat())
def OnesModuleFloat_basic(module, tu: TestUtils):
    module.forward()

class ContiguousModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return x.contiguous()


@register_test_case(module_factory=lambda: ContiguousModule())
def ContiguousModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1))
    
class TensorToInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, x):
        return int(x)

@register_test_case(module_factory=lambda: TensorToInt())
def TensorToInt_basic(module, tu: TestUtils):
    module.forward(torch.randint(10,[]))
    
class LogSoftmaxIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return self.log_softmax.forward(tensor)

@register_test_case(module_factory=lambda: LogSoftmaxIntModule())
def LogSoftmaxIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4).double())
