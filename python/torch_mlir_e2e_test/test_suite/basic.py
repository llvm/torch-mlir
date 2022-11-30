# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import functorch
import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

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


class IsFloatingPointInt(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.is_floating_point(x)


@register_test_case(module_factory=lambda: IsFloatingPointInt())
def IsFloatingPointInt_False(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class IsFloatingPointFloat(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.is_floating_point(x)


@register_test_case(module_factory=lambda: IsFloatingPointFloat())
def IsFloatingPointFloat_True(module, tu: TestUtils):
    module.forward(tu.rand(3))


# ==============================================================================


class ContainsIntList(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @export
    @annotate_args([
        None
    ])
    def forward(self):
        return torch.ops.aten.__contains__([1,2,3], 3)
@register_test_case(module_factory=lambda: ContainsIntList())
def ContainsIntList_True(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ContainsIntListFalse(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @export
    @annotate_args([
        None
    ])
    def forward(self):
        return torch.ops.aten.__contains__([1,2,3], 4)
@register_test_case(module_factory=lambda: ContainsIntListFalse())
def ContainsIntList_False(module, tu: TestUtils):
    module.forward()


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


# ==============================================================================


class AddmmModuleFloat(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, M, mat1, mat2):
        return torch.addmm(M, mat1, mat2, beta=3.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleFloat())
def AddmmModuleFloat_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4), tu.rand(4, 2), tu.rand(2, 4))


#  ==============================================================================


class AddmmModuleBroadcastable(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, M, mat1, mat2):
        return torch.addmm(M, mat1, mat2, beta=2.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleBroadcastable())
def AddmmModule_broadcastable(module, tu: TestUtils):
    module.forward(tu.rand(1, 2), tu.rand(3, 2), tu.rand(2, 2))


#  ==============================================================================


class AddmmModuleDifferentRankBroadcastable(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, M, mat1, mat2):
        return torch.addmm(M, mat1, mat2, beta=11.0, alpha=7.0)


@register_test_case(
    module_factory=lambda: AddmmModuleDifferentRankBroadcastable())
def AddmmModule_differentRankBroadcastable(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3, 2), tu.rand(2, 3))


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


class ConstantPad2dStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.pad2d = torch.nn.ConstantPad2d((0, 1, 2, 3), -float('inf'))

    @export
    @annotate_args([
        None,
        ([1, 1, 20, 20], torch.float32, True),
    ])
    def forward(self, x):
        return self.pad2d(x)


@register_test_case(module_factory=lambda: ConstantPad2dStaticModule())
def ConstantPad2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class PadModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        pad = [0, 1, 2, 3]
        mode = "constant"
        return torch.ops.aten.pad(x, pad, mode, float(1.5))


@register_test_case(module_factory=lambda: PadModule())
def PadModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class PadWithNoneValModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        pad = [0, 1, 2, 3]
        mode = "constant"
        return torch.ops.aten.pad(x, pad, mode, None)


@register_test_case(module_factory=lambda: PadWithNoneValModule())
def PadWithNoneValModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class ConstantPadNdModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1), -float('inf'))


@register_test_case(module_factory=lambda: ConstantPadNdModule())
def ConstantPadNdModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================


class ConstantPadNdStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 20, 20, 4, 4], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1), -float('inf'))


@register_test_case(module_factory=lambda: ConstantPadNdStaticModule())
def ConstantPadNdStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================


class ConstantPadNdPartialStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 20, 20, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1, 2, 3), -float('inf'))


@register_test_case(module_factory=lambda: ConstantPadNdPartialStaticModule())
def ConstantPadNdPartialStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================


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


# ==============================================================================


class PermuteModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([3, 4, 2], torch.float32, True)])
    def forward(self, x):
        return x.permute(0, 2, 1)


@register_test_case(module_factory=lambda: PermuteModule())
def PermuteModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class PermuteNegativeIndexModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([3, 4, 2], torch.float32, True)])
    def forward(self, x):
        return x.permute(0, -1, 1)


@register_test_case(module_factory=lambda: PermuteNegativeIndexModule())
def PermuteNegativeIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class Permute0RankModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], torch.float32, True)])
    def forward(self, x):
        return x.permute([])


@register_test_case(module_factory=lambda: Permute0RankModule())
def Permute0RankModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor(3.0))


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


class TensorsConcatNegativeDimModule(torch.nn.Module):

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
        return torch.cat([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsConcatNegativeDimModule())
def TensorsConcatNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


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


# ==============================================================================


class GatherRandomIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, tensor, indices):
        return torch.gather(tensor, 1, indices)

@register_test_case(module_factory=lambda: GatherRandomIndexModule())
def GatherRandomIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), tu.randint(2, 3, 4, high=3))


# ==============================================================================


class Gather2DInputModdule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, tensor, indices):
        return torch.gather(tensor, 1, indices)

@register_test_case(module_factory=lambda: Gather2DInputModdule())
def Gather2DInputModdule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5), torch.tensor([[1, 2, 3], [4, 3, 2]]))


# ==============================================================================


class GatherStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 3, 4], torch.float32, True),
        ([1, 2, 3], torch.int64, True),
    ])
    def forward(self, tensor, indices):
        return torch.gather(tensor, 2, indices)


@register_test_case(module_factory=lambda: GatherStaticModule())
def GatherStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), torch.tensor([[[1, 2, 3], [1, 2, 3]]]))


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


class EmbeddingModuleI64(torch.nn.Module):

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


@register_test_case(module_factory=lambda: EmbeddingModuleI64())
def EmbeddingModuleI64_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100))


# ==============================================================================


class EmbeddingModuleI32(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(num_embeddings=100,
                                        embedding_dim=50,
                                        padding_idx=4)

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleI32())
def EmbeddingModuleI32_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100).to(torch.int32))


# ==============================================================================

class EmbeddingModuleI32Static(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(num_embeddings=100,
                                        embedding_dim=50,
                                        padding_idx=4)

    @export
    @annotate_args([
        None,
        ([3, 3], torch.int32, True),
    ])
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleI32Static())
def EmbeddingModuleI32Static_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100).to(torch.int32))


# ==============================================================================


class EmbeddingModule1DIndices(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(num_embeddings=100,
                                        embedding_dim=50,
                                        padding_idx=4)

    @export
    @annotate_args([
        None,
        ([-1], torch.int32, True),
    ])
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModule1DIndices())
def EmbeddingModule1DIndices_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, high=100).to(torch.int32))


# ==============================================================================


class SoftmaxIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
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


# ==============================================================================


class _SoftmaxModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten._softmax(tensor, 0, False)


@register_test_case(module_factory=lambda: _SoftmaxModule())
def _SoftmaxModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4))


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


class _LogSoftmaxModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten._log_softmax(tensor, dim=0, half_to_float=False)


@register_test_case(module_factory=lambda: _LogSoftmaxModule())
def _LogSoftmaxModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4))


# ==============================================================================


class _LogSoftmaxModuleStable(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten._log_softmax(tensor, dim=0, half_to_float=False)


@register_test_case(module_factory=lambda: _LogSoftmaxModuleStable())
def _LogSoftmaxModuleStable_basic(module, tu: TestUtils):
    # testing for numerical stability.
    # Should result in  tensor([-1e9, 0.00]) rather than tensor([-inf, 0.]).
    a = torch.tensor([0, 1e9])
    module.forward(a)


# ==============================================================================


class SoftplusModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.softplus(x)


@register_test_case(module_factory=lambda: SoftplusModule())
def SoftplusModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))


# ==============================================================================


class HardsigmoidModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardsigmoid(x)


@register_test_case(module_factory=lambda: HardsigmoidModule())
def HardsigmoidModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[4.0, -5.0, 3.0], [2.9, -1.5, -3.0]]))


# ==============================================================================


class HardsigmoidRandomModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardsigmoid(x)


@register_test_case(module_factory=lambda: HardsigmoidRandomModule())
def HardsigmoidRandomModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, low=-10, high=10))


# ==============================================================================


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


# ==============================================================================


class BroadcastToSameRankStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 1, 8], torch.float32, True),
        ([3, 1, 1], torch.float32, True),
    ])
    def forward(self, x, y):
        y = torch.broadcast_to(y, [3, 1, 8])
        return torch.ops.aten.sub(x, y)


@register_test_case(module_factory=lambda: BroadcastToSameRankStaticModule())
def BroadcastToSameRankStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 8), tu.rand(3, 1, 1))


# ==============================================================================


class BroadcastZeroRankInputStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 1, 8], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, x, y):
        y = torch.broadcast_to(y, [3, 1, 8])
        return torch.ops.aten.sub(x, y)


@register_test_case(module_factory=lambda: BroadcastZeroRankInputStaticModule())
def BroadcastZeroRankInputStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 8), tu.rand())

# ==============================================================================


class RollModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, -1, 2], torch.float32, True),
    ])
    def forward(self, x):
        return x.roll([2, -1], [0, 2])


@register_test_case(module_factory=lambda: RollModule())
def RollModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))

# ==============================================================================


class RepeatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 1, 2], torch.float32, True),
    ])
    def forward(self, x):
        return x.repeat([2, 1, 3, 4])


@register_test_case(module_factory=lambda: RepeatModule())
def RepeatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))

# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


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


# ==============================================================================


class NumToTensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.prim.NumToTensor(1)


@register_test_case(module_factory=lambda: NumToTensorIntModule())
def NumToTensorIntModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class NumToTensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.prim.NumToTensor(1.0)


@register_test_case(module_factory=lambda: NumToTensorFloatModule())
def NumToTensorFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


# This test can be removed once we have one real op returning 3 float32 tensors
class ReturnThreeTensorFloat32(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a, b, c):
        return a, b, c


@register_test_case(module_factory=lambda: ReturnThreeTensorFloat32())
def ReturnThreeTensorFloat32_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3), tu.rand(2, 3), tu.rand(2, 3))


# ==============================================================================


class AddCMulModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, tensor1, tensor2):
        return torch.addcmul(input, tensor1, tensor2, value=1.0)


@register_test_case(module_factory=lambda: AddCMulModule())
def AddCMulModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3), tu.rand(1, 3), tu.rand(1, 3))


# ==============================================================================


class AddCDivModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, input, tensor1, tensor2):
        return torch.addcdiv(input, tensor1, tensor2, value=1.0)


@register_test_case(module_factory=lambda: AddCDivModule())
def AddCDivModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 3), tu.rand(1, 3), tu.rand(1, 3))


# ==============================================================================


class tensorIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        a = 1
        return torch.tensor(a)


@register_test_case(module_factory=lambda: tensorIntModule())
def TensorIntModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class tensorFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        a = 1.0
        return torch.tensor(a)


@register_test_case(module_factory=lambda: tensorFloatModule())
def TensorFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class DropoutEvalIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.dropout(x, 0.2, train=False)


@register_test_case(module_factory=lambda: DropoutEvalIntModule())
def DropoutEvalIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=5, high=10))


# ==============================================================================


class DropoutEvalFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.dropout(x, 0.1, train=False)


@register_test_case(module_factory=lambda: DropoutEvalFloatModule())
def DropoutEvalFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class DropoutTrainModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        res = torch.dropout(x, 0.3, train=True)
        return torch.mean(res), torch.std(res)


@register_test_case(module_factory=lambda: DropoutTrainModule())
def DropoutTrainModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1536))


# ==============================================================================


class NumelModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input):
        return torch.ops.aten.numel(input)


@register_test_case(module_factory=lambda: NumelModule())
def NumelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3, 5))


# ==============================================================================


class NumelZeroRankModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, input):
        return torch.ops.aten.numel(input)


@register_test_case(module_factory=lambda: NumelZeroRankModule())
def NumelZeroRankModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=10))


# ==============================================================================


class BoolTensorReturnFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.bool, True),
    ])
    def forward(self, a):
        return a


@register_test_case(module_factory=lambda: BoolTensorReturnFalseModule())
def BoolTensorReturnFalseModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([0, 0], dtype=torch.bool))


# ==============================================================================


class BoolTensorReturnTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.bool, True),
    ])
    def forward(self, a):
        return a


@register_test_case(module_factory=lambda: BoolTensorReturnTrueModule())
def BoolTensorReturnTrueModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool))


# ==============================================================================


class BoolTensorReturnMixedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, a):
        return a


@register_test_case(module_factory=lambda: BoolTensorReturnMixedModule())
def BoolTensorReturnMixedModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[1, 0], [0, 1]], dtype=torch.bool))


# ==============================================================================


class BoolTensorHandleSignless(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.bool, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, a, b):
        return a * b


@register_test_case(module_factory=lambda: BoolTensorHandleSignless())
def BoolTensorHandleSignless_basic(module, tu: TestUtils):
    a = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
    b = torch.tensor([[0, 0], [0, 0]], dtype=torch.bool)
    module.forward(a, b)


# ==============================================================================


class TModuleRank2(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.t(lhs)


@register_test_case(module_factory=lambda: TModuleRank2())
def TModuleRank2_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class TModuleRank1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.t(lhs)


@register_test_case(module_factory=lambda: TModuleRank1())
def TModuleRank1_basic(module, tu: TestUtils):
    module.forward(tu.rand(3))


# ==============================================================================


class TModuleRank0(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.t(lhs)


@register_test_case(module_factory=lambda: TModuleRank0())
def TModuleRank0_basic(module, tu: TestUtils):
    module.forward(torch.tensor(7, dtype=torch.float32))


# ==============================================================================


class TensorLiteralModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.t = torch.randint(-5, 5, (2, 3))

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.add(self.t, self.t)


@register_test_case(module_factory=lambda: TensorLiteralModule())
def TensorLiteralModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class TensorOpaqueLiteralModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.t = torch.randint(-5, 5, (256, 1024))

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.add(self.t, self.t)


@register_test_case(module_factory=lambda: TensorOpaqueLiteralModule())
def TensorOpaqueLiteralModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ReturnTwoTensorF32I64(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a, b):
        return a, b


@register_test_case(module_factory=lambda: ReturnTwoTensorF32I64())
def ReturnTwoTensorF32I64_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3), tu.randint(2, 3, high=5))


# ==============================================================================


class IndexTensorModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index, ))


@register_test_case(module_factory=lambda: IndexTensorModule())
def IndexTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5), tu.randint(2, 3, high=4))


# ==============================================================================


class IndexTensorModule3dInput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))


@register_test_case(module_factory=lambda: IndexTensorModule3dInput())
def IndexTensorModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))


# ==============================================================================


class IndexTensorSelectDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a, ind):
        return torch.ops.aten.index(a, (None, ind, None))


@register_test_case(module_factory=lambda: IndexTensorSelectDimModule())
def IndexTensorSelectDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 6), tu.randint(2, 3, high=3))

# ==============================================================================


class IndexTensorMultiInput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([3, 3], torch.int64, True),
        ([3], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (index1, index2,))


@register_test_case(module_factory=lambda: IndexTensorMultiInput())
def IndexTensorMultiInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(3, 3, high=3), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputOneDim(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([6, 1], torch.int64, True),
        ([3], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (index1, index2,))


@register_test_case(module_factory=lambda: IndexTensorMultiInputOneDim())
def IndexTensorMultiInputOneDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputContiguousOneDimDynamic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, 1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (
            None,
            index1,
            index2,
        ))


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputContiguousOneDimDynamic())
def IndexTensorMultiInputContiguousOneDimDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4),
                   tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousOneDimDynamic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, 1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (
            index1,
            None,
            index2,
        ))


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputNonContiguousOneDimDynamic())
def IndexTensorMultiInputNonContiguousOneDimDynamic_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4),
                   tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousDynamic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, 2], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (
            index2,
            None,
            index1,
        ))


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputNonContiguousDynamic())
def IndexTensorMultiInputNonContiguousDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 2, high=2),
                   tu.randint(2, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousMultipleStaticDims(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([4, 1], torch.int64, True),
        ([1, 3], torch.int64, True),
        ([-1, 3], torch.int64, True),
    ])
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, (index1, index2, index3))


@register_test_case(module_factory=lambda:
                    IndexTensorMultiInputNonContiguousMultipleStaticDims())
def IndexTensorMultiInputNonContiguousMultipleStaticDims_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2), tu.randint(4, 1, high=3),
                   tu.randint(1, 3, high=1), tu.randint(4, 3, high=1))


# ==============================================================================


class IndexTensorMultiInputNonContiguous(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([4, 2], torch.int64, True),
        ([4, 2], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (index1, None, index2))


@register_test_case(module_factory=lambda: IndexTensorMultiInputNonContiguous())
def IndexTensorMultiInputNonContiguous_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2), tu.randint(4, 2, high=3), tu.randint(4, 2, high=1))


# ==============================================================================


class IndexTensorMultiInputThreeIndexers(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1, -1], torch.float32, True),
        ([8, 4, 2], torch.int64, True),
        ([8, 1, 1], torch.int64, True),
        ([4, 2], torch.int64, True),
    ])
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, (None, None, index1, None, index2, index3))


@register_test_case(module_factory=lambda: IndexTensorMultiInputThreeIndexers())
def IndexTensorMultiInputThreeIndexers_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 4, 4, 5, 3),
                   tu.randint(8, 4, 2, high=3),
                   tu.randint(8, 1, 1, high=4),
                   tu.randint(4, 2, high=2))


# ==============================================================================


class IndexTensorMultiInputContiguousCenter(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([2, 2], torch.int64, True),
        ([2], torch.int64, True),
    ])
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (None, index1, index2, None))


@register_test_case(module_factory=lambda: IndexTensorMultiInputContiguousCenter())
def IndexTensorMultiInputContiguousCenter_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2), tu.randint(2, 2, high=3), tu.randint(2, high=2))


# ==============================================================================


class IndexTensorHackedTwinModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, index):
        return torch.ops.aten.index(x, [index])


@register_test_case(module_factory=lambda: IndexTensorHackedTwinModule())
def IndexTensorHackedTwinModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5), tu.randint(2, 3, high=4))


# ==============================================================================


class IndexTensorHackedTwinModule3dInput(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x, index):
        return torch.ops.aten.index(x, [index])


@register_test_case(
    module_factory=lambda: IndexTensorHackedTwinModule3dInput())
def IndexTensorHackedTwinModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))


# ==============================================================================


class IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims(
        torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([4, 1], torch.int64, True),
        ([1, 3], torch.int64, True),
        ([-1, 3], torch.int64, True),
    ])
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, [index1, index2, index3])


@register_test_case(
    module_factory=lambda:
    IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims())
def IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims_basic(
        module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2), tu.randint(4, 1, high=3),
                   tu.randint(1, 3, high=1), tu.randint(4, 3, high=1))


# ==============================================================================


class SquareModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.square(x)


@register_test_case(module_factory=lambda: SquareModule())
def SquareModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class HardswishModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardswish(x)


@register_test_case(module_factory=lambda: HardswishModule())
def HardswishModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[4.0, -5.0, 3.0], [2.9, -1.5, -3.0]]))


# ==============================================================================


class HardswishRandomModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardswish(x)


@register_test_case(module_factory=lambda: HardswishRandomModule())
def HardswishRandomModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128, 128, low=-10, high=10))


# ==============================================================================


class SiluModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.silu(x)


@register_test_case(module_factory=lambda: SiluModule())
def SiluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(128, 128, low=-10, high=10))


# ==============================================================================


class HardTanhModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardtanh(x, min_val=-2, max_val=2)


@register_test_case(module_factory=lambda: HardTanhModule())
def HardTanhModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(100, 100, low=-5, high=5))


# ==============================================================================


class HardTanhIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.hardtanh(x, min_val=-2, max_val=2)


@register_test_case(module_factory=lambda: HardTanhIntModule())
def HardTanhIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(100, 100, low=-5, high=5))


# ==============================================================================


class BincountModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.bincount(x)


@register_test_case(module_factory=lambda: BincountModule())
def BincountModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1000, high=10))


# ==============================================================================


class BincountStaticSizeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([200], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.bincount(x)


@register_test_case(module_factory=lambda: BincountStaticSizeModule())
def BincountStaticSizeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(200, high=100))


# ==============================================================================


class BincountMinlengthModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ops.aten.bincount(x, minlength=600)


@register_test_case(module_factory=lambda: BincountMinlengthModule())
def BincountMinlengthModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(20, high=5))


# ==============================================================================


class ExpandAsFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, 1, 1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.expand_as(x, y)


@register_test_case(module_factory=lambda: ExpandAsFloatModule())
def ExpandAsFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 1), tu.rand(3, 4, 5))


class ExpandAsIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.expand_as(x, y)


@register_test_case(module_factory=lambda: ExpandAsIntModule())
def ExpandAsIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 1, 1, high=100),
                   tu.randint(4, 5, 6, high=200))


# ==============================================================================


class CopyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyModule())
def CopyModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))


class CopyWithDifferentSizesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, 4], torch.float32, True),
        ([-1, -1, 1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyWithDifferentSizesModule())
def CopyWithDifferentSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 1))


class CopyWithDifferentDTypesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyWithDifferentDTypesModule())
def CopyWithDifferentDTypesModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 2, 4, high=100), tu.rand(3, 2, 4))


class CopyWithDifferentDTypesAndSizesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, 4], torch.float32, True),
        ([-1, -1, 1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(
    module_factory=lambda: CopyWithDifferentDTypesAndSizesModule())
def CopyWithDifferentDTypesAndSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.randint(3, 2, 1, high=1000))


# ==============================================================================


class ToCopyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten._to_copy(x)


@register_test_case(module_factory=lambda: ToCopyModule())
def ToCopyModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyWithDTypeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=torch.int64)


@register_test_case(module_factory=lambda: ToCopyWithDTypeModule())
def ToCopyWithDTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyWithDTypeFalsePinMemoryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=torch.int64, pin_memory=False)


@register_test_case(
    module_factory=lambda: ToCopyWithDTypeFalsePinMemoryModule())
def ToCopyWithDTypeFalsePinMemoryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyBoolDTypeStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1, 1, 5, 5], torch.uint8, True),
    ])
    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=torch.bool)


@register_test_case(module_factory=lambda: ToCopyBoolDTypeStaticModule())
def ToCopyBoolDTypeStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 1, 5, 5).to(dtype=torch.uint8))


# ==============================================================================


class FlipModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.flip(x, [1, 2])


@register_test_case(module_factory=lambda: FlipModule())
def FlipModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class DetachModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.detach(x)


@register_test_case(module_factory=lambda: DetachModule())
def DetachModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class LenStrModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.str = "test"

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.len(self.str)


@register_test_case(module_factory=lambda: LenStrModule())
def LenStrModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ScalarImplicitFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float64, True),
    ])
    def forward(self, x):
        return float(torch.ops.aten.ScalarImplicit(x))


@register_test_case(module_factory=lambda: ScalarImplicitFloatModule())
def ScalarImplicitFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double())


class ScalarImplicitIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.int64, True),
    ])
    def forward(self, x):
        return int(torch.ops.aten.ScalarImplicit(x))


@register_test_case(module_factory=lambda: ScalarImplicitIntModule())
def ScalarImplicitIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100))


# ==============================================================================


class BaddbmmDynamicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmDynamicModule())
def BaddbmmDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([5, 2, 7], torch.float32, True),
        ([5, 2, 9], torch.float32, True),
        ([5, 9, 7], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmStaticModule())
def BaddbmmStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 7), tu.rand(5, 2, 9), tu.rand(5, 9, 7))


class BaddbmmDifferentDtypesModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmDifferentDtypesModule())
def BaddbmmDifferentDtypesModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=10), tu.rand(3, 4, 6),
                   tu.rand(3, 6, 5))


class BaddbmmWithAlphaModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, alpha=5)


@register_test_case(module_factory=lambda: BaddbmmWithAlphaModule())
def BaddbmmWithAlphaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmWithBetaModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, beta=0.5)


@register_test_case(module_factory=lambda: BaddbmmWithBetaModule())
def BaddbmmWithBetaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmWithAlphaBetaModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, beta=6, alpha=2.4)


@register_test_case(module_factory=lambda: BaddbmmWithAlphaBetaModule())
def BaddbmmWithAlphaBetaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmBroadcast1DInputModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([1], torch.float32, True),
        ([5, 2, 9], torch.float32, True),
        ([5, 9, 7], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmBroadcast1DInputModule())
def BaddbmmBroadcast1DInputModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1,), tu.rand(5, 2, 9), tu.rand(5, 9, 7))


class BaddbmmBroadcast2DInputModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 7], torch.float32, True),
        ([5, 2, 9], torch.float32, True),
        ([5, 9, 7], torch.float32, True),
    ])
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmBroadcast2DInputModule())
def BaddbmmBroadcast2DInputModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7), tu.rand(5, 2, 9), tu.rand(5, 9, 7))


# ==============================================================================


class NumpyTRankNStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 5, 6], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRankNStaticModule())
def NumpyTRankNStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6))


class NumpyTRankNDynamicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRankNDynamicModule())
def NumpyTRankNDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6, 2))


class NumpyTRank2Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank2Module())
def NumpyTRank2Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class NumpyTRank1Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank1Module())
def NumpyTRank1Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3))


class NumpyTRank0Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([], torch.float32, True),
    ])
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank0Module())
def NumpyTRank0Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor(7, dtype=torch.float32))

class AtenEmbeddingBagSumExample(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, weight, indices, offsets):
        return torch.ops.aten.embedding_bag(weight, indices, offsets, scale_grad_by_freq=False, mode=0, sparse=False, per_sample_weights=None, include_last_offset=False, padding_idx=None)

@register_test_case(module_factory=lambda: AtenEmbeddingBagSumExample())
def AtenEmbeddingBagSumExample_basic(module, tu: TestUtils):
    weight  = torch.rand(100, 10)
    indices = torch.LongTensor([0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54])
    offsets = torch.LongTensor([0, 3, 5, 7, 9, 10, 15])
    module.forward(weight, indices, offsets)

class Aten_EmbeddingBagExample(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, weight, indices, offsets):
        return torch.ops.aten._embedding_bag(weight, indices, offsets)

@register_test_case(module_factory=lambda: Aten_EmbeddingBagExample())
def Aten_EmbeddingBagExample_basic(module, tu: TestUtils):
    weight  = torch.rand(100, 10)
    indices = torch.LongTensor([0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54])
    offsets = torch.LongTensor([0, 3, 5, 7, 9, 10, 15])
    module.forward(weight, indices, offsets)

# ==============================================================================

class CumsumModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, val):
        return torch.ops.aten.cumsum(val, 1)

@register_test_case(module_factory=lambda: CumsumModule())
def CumsumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))

class CumsumStaticModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 7, 4], torch.float32, True),
    ])
    def forward(self, val):
        return torch.ops.aten.cumsum(val, 1)

@register_test_case(module_factory=lambda: CumsumStaticModule())
def CumsumStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))

# ==============================================================================

class AtenToDeviceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1 , -1], torch.float32, True),
    ])

    def forward(self, val):
        return torch.ops.aten.to(val, device='cpu', dtype=torch.float, non_blocking=False)

@register_test_case(module_factory=lambda: AtenToDeviceModule())
def AtenToDeviceModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(2, 4))

# ==============================================================================


class UpSampleNearest2dBackward(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float64, True),
    ])
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d_backward(input,
                                               output_size=[6, 12],
                                               input_size=[1, 1, 2, 3],
                                               scales_h=3.0,
                                               scales_w=4.0)


@register_test_case(module_factory=lambda: UpSampleNearest2dBackward())
def UpSampleNearest2dBackward_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))


class UpSampleNearest2dBackwardScalesNone(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d_backward(input,
                                               output_size=[4, 8],
                                               input_size=[1, 1, 2, 3],
                                               scales_h=None,
                                               scales_w=None)

@register_test_case(module_factory=lambda: UpSampleNearest2dBackwardScalesNone())
def UpSampleNearest2dBackwardScalesNone_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 8))


# ==============================================================================


class SortIntList(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        a = [1, 0, 3, 2]
        b = [0, 1, 2, 3]
        a.sort()
        return a == b


@register_test_case(module_factory=lambda: SortIntList())
def SortIntList_basic(module, tu: TestUtils):
    module.forward()


class SortIntListReverse(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        a = [1, 0, 3, 2]
        b = [3, 2, 1, 0]
        a.sort(reverse=True)
        return a == b


@register_test_case(module_factory=lambda: SortIntListReverse())
def SortIntListReverse_basic(module, tu: TestUtils):
    module.forward()