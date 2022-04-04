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


@register_test_case(module_factory=lambda: AddmmModuleDifferentRankBroadcastable())
def AddmmModule_differentRankBroadcastable(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3, 2), tu.rand(2, 3))

#  ==============================================================================

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

# ==============================================================================

@register_test_case(module_factory=lambda: MaxPool2dModule())
def MaxPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20) - 0.5)

class MaxPool2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[3, 3],
                                       stride=[2, 2],
                                       padding=[1, 1],
                                       dilation=[1, 1])

    @export
    @annotate_args([
        None,
        ([1, 64, 112, 112], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dStaticModule())
def MaxPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112))

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
    module.forward(tu.rand(1, 1, 20, 20) - 0.5)

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
    module.forward(tu.rand(1, 1, 20, 20, 4, 4) - 0.5)

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
    module.forward(tu.rand(1, 1, 20, 20, 4, 4) - 0.5)

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
    module.forward(tu.rand(1, 1, 20, 20, 4, 4) - 0.5)

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
    @annotate_args([
        None,
        ([3, 4, 2], torch.float32, True)
    ])
    def forward(self, x):
        return x.permute(0, 2, 1)

@register_test_case(module_factory=lambda: PermuteModule())
def PermuteModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))

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
    module.forward(tu.rand(1,3), tu.rand(1,3), tu.rand(1,3))

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
    module.forward(tu.rand(1,3), tu.rand(1,3), tu.rand(1,3))

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
    module.forward(torch.randint(5, 10, (3, 4)))

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
    module.forward(tu.rand(256, 256))

# ==============================================================================

class MeanModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4], torch.float32, True),
    ])
    def forward(self, x):
        return torch.mean(x)


@register_test_case(module_factory=lambda: MeanModule())
def MeanModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 4))

# ==============================================================================

class MeanDynamicSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.mean(x)


@register_test_case(module_factory=lambda: MeanDynamicSizesModule())
def MeanDynamicSizesModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 4))

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
        return torch.numel(input)

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
        return torch.numel(input)

@register_test_case(module_factory=lambda: NumelZeroRankModule())
def NumelZeroRankModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10,[]))

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
    module.forward(torch.tensor([[1, 0], [0,1]], dtype=torch.bool))

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
    module.forward(tu.rand(2, 3), torch.randint(5, (2, 3)))

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
        return torch.ops.aten.index(x, (index,))

@register_test_case(module_factory=lambda: IndexTensorModule())
def IndexTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5), torch.randint(4, (2, 3)))

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

class VarUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=True)

@register_test_case(module_factory=lambda: VarUnbiasedModule())
def VarUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class VarBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.var(x, unbiased=False)

@register_test_case(module_factory=lambda: VarBiasedModule())
def VarBiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdUnbiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=True)

@register_test_case(module_factory=lambda: StdUnbiasedModule())
def StdUnbiasedModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))

# ==============================================================================

class StdBiasedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.std(x, unbiased=False)

@register_test_case(module_factory=lambda: StdBiasedModule())
def StdBiasedModule_basic(module, tu: TestUtils):
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
    module.forward(torch.randint(-5, 5, (100, 100)))


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
    module.forward(torch.randint(10, (1000,)))


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
    module.forward(torch.randint(100, (200,)))


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
    module.forward(torch.randint(5, (20,)))

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
    module.forward(torch.randint(100, (1, 1, 1)), torch.randint(200, (4, 5, 6)))

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
    module.forward(torch.randint(100, (3, 2, 4)), tu.rand(3, 2, 4))


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


@register_test_case(module_factory=lambda: CopyWithDifferentDTypesAndSizesModule())
def CopyWithDifferentDTypesAndSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), torch.randint(1000, (3, 2, 1)))

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


@register_test_case(module_factory=lambda: ToCopyWithDTypeFalsePinMemoryModule())
def ToCopyWithDTypeFalsePinMemoryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class MaxPool2dWithIndicesBackwardModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[2, 2]
       stride=[2, 2] 
       padding=[0, 0]
       dilation=[1, 1]
       ceil_mode=False
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModule())
def MaxPool2dWithIndicesBackwardsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 2), tu.rand(1, 4, 4), tu.rand(1, 2, 2).long())

class MaxPool2dWithIndicesBackwardModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[2, 2]
       stride=[1, 1] 
       padding=[0, 0]
       dilation=[1, 1]
       ceil_mode=False
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModule1())
def MaxPool2dWithIndicesBackwardModule1_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 2), tu.rand(5, 5, 3), tu.rand(5, 4, 2).long())    

class MaxPool2dWithIndicesBackwardModul2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[2, 2]
       stride=[1, 1] 
       padding=[1, 1]
       dilation=[1, 1]
       ceil_mode=False
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModul2())
def MaxPool2dWithIndicesBackwardModul2_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 6, 4), tu.rand(5, 5, 3), tu.rand(5, 6, 4).long())        

class MaxPool2dWithIndicesBackwardModul3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[2, 2]
       stride=[1, 1] 
       padding=[1, 1]
       dilation=[1, 1]
       ceil_mode=True
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModul3())
def MaxPool2dWithIndicesBackwardModul3_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 6, 4), tu.rand(5, 5, 3), tu.rand(5, 6, 4).long())         

class MaxPool2dWithIndicesBackwardModul4(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[2, 2]
       stride=[1, 1] 
       padding=[0, 0]
       dilation=[1, 1]
       ceil_mode=False
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModul4())
def MaxPool2dWithIndicesBackwardModul4_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 1), tu.rand(1, 2, 2), tu.rand(1, 1, 1).long())  

class MaxPool2dWithIndicesBackwardModul5(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int, True),
    ])
    def forward(self, output, input, indices):
       kernel_size=[3, 3]
       stride=[1, 1] 
       padding=[0, 0]
       dilation=[1, 1]
       ceil_mode=False
       return torch.ops.aten.max_pool2d_with_indices_backward(output, input, kernel_size, stride, padding, dilation, ceil_mode, indices)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesBackwardModul5())
def MaxPool2dWithIndicesBackwardModul5_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 1), tu.rand(1, 3, 3), tu.rand(1, 1, 1).long())                