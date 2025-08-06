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


class ScalarConstantTupleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return (1, 2)


@register_test_case(module_factory=lambda: ScalarConstantTupleModule())
def ScalarConstantTupleModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 4))


# ==============================================================================


class MmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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


class BmmFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


@register_test_case(module_factory=lambda: BmmFloatModule())
def BmmFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 5, 4))


class BmmFloat16Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float16, True),
            ([-1, -1, -1], torch.float16, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


@register_test_case(module_factory=lambda: BmmFloat16Module())
def BmmFloat16Module_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(3, 4, 5).to(torch.float16), tu.rand(3, 5, 4).to(torch.float16)
    )


class BmmIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, lhs, rhs):
        return torch.bmm(lhs, rhs)


@register_test_case(module_factory=lambda: BmmIntModule())
def BmmIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, high=100), tu.randint(3, 5, 4, high=100))


# ==============================================================================


class IsFloatingPointInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
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
    @annotate_args([None])
    def forward(self):
        return torch.ops.aten.__contains__([1, 2, 3], 3)


@register_test_case(module_factory=lambda: ContainsIntList())
def ContainsIntList_True(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ContainsIntListFalse(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None])
    def forward(self):
        return torch.ops.aten.__contains__([1, 2, 3], 4)


@register_test_case(module_factory=lambda: ContainsIntListFalse())
def ContainsIntList_False(module, tu: TestUtils):
    module.forward()


# ==============================================================================


# A subgraph with multiple mm ops.
class MmDagModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 4], torch.float32, True),
            ([4, 4], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, M, mat1, mat2):
        return torch.addmm(M, mat1, mat2, beta=11.0, alpha=7.0)


@register_test_case(module_factory=lambda: AddmmModuleDifferentRankBroadcastable())
def AddmmModule_differentRankBroadcastable(module, tu: TestUtils):
    module.forward(tu.rand(3), tu.rand(3, 2), tu.rand(2, 3))


# ==============================================================================


class UnflattenStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 6, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.unflatten(x, 1, (2, 3))


@register_test_case(module_factory=lambda: UnflattenStaticModule())
def UnflattenStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 6, 4))


# ==============================================================================


class FlattenStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(2, 4)

    @export
    @annotate_args(
        [
            None,
            ([10, 3, 8, 9, 3, 4], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, 9, 3, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenDynamicModule())
def FlattenDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9, 3, 4))


class FlattenDynamicModuleCollapseAll(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = torch.nn.Flatten(0)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, 9, 3, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.flat(x)


@register_test_case(module_factory=lambda: FlattenDynamicModuleCollapseAll())
def FlattenDynamicModuleCollapseAll_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9, 3, 4))


# ==============================================================================


class AliasModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, inp_tensor):
        return torch.ops.aten.alias(inp_tensor)


@register_test_case(module_factory=lambda: AliasModule())
def AliasModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, low=-1))


# ==============================================================================


class ConstantPad2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad2d = torch.nn.ConstantPad2d((0, 1, 2, 3), -float("inf"))

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20, 20], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1), -float("inf"))


@register_test_case(module_factory=lambda: ConstantPadNdModule())
def ConstantPadNdModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================


class ConstantPadNdStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20, 20, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1), -float("inf"))


@register_test_case(module_factory=lambda: ConstantPadNdStaticModule())
def ConstantPadNdStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================


class ConstantPadNdPartialStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 20, 20, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.constant_pad_nd(x, (0, 1, 2, 3), -float("inf"))


@register_test_case(module_factory=lambda: ConstantPadNdPartialStaticModule())
def ConstantPadNdPartialStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20, 4, 4, low=-1))


# ==============================================================================
class ReflectionPad1dModule3dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad1d(x, (3, 1))


class ReplicationPad2dModule_basic_module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.replication_pad2d(x, (1, 2, 3, 4))


@register_test_case(module_factory=lambda: ReplicationPad2dModule_basic_module())
def ReplicationPad2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 3, 3, low=-1))


# ==============================================================================


class ReplicationPad2dModule_left0_module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.replication_pad2d(x, (0, 2, 3, 4))


@register_test_case(module_factory=lambda: ReplicationPad2dModule_left0_module())
def ReplicationPad2dModule_left0(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 3, 3, low=-1))


# ==============================================================================


class ReplicationPad2dModule_right0_module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.replication_pad2d(x, (1, 0, 3, 4))


@register_test_case(module_factory=lambda: ReplicationPad2dModule_right0_module())
def ReplicationPad2dModule_right0(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 3, 3, low=-1))


# ==============================================================================


class ReplicationPad2dModule_top0_module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.replication_pad2d(x, (1, 2, 0, 4))


@register_test_case(module_factory=lambda: ReplicationPad2dModule_top0_module())
def ReplicationPad2dModule_top0(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 3, 3, low=-1))


# ==============================================================================


class ReplicationPad2dModule_bottom0_module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 3, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.replication_pad2d(x, (1, 2, 3, 0))


@register_test_case(module_factory=lambda: ReplicationPad2dModule_bottom0_module())
def ReplicationPad2dModule_bottom0(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 3, 3, low=-1))


# ==============================================================================


@register_test_case(module_factory=lambda: ReflectionPad1dModule3dInput())
def ReflectionPad1dModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 4))


class ReflectionPad1dModule2dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad1d(x, (3, 2))


@register_test_case(module_factory=lambda: ReflectionPad1dModule2dInput())
def ReflectionPad1dModule2dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


class ReflectionPad1dModule3dInputLeft(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad1d(x, (2, 0))


@register_test_case(module_factory=lambda: ReflectionPad1dModule3dInputLeft())
def ReflectionPad1dModule3dInput_Left(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 5))


class ReflectionPad1dModule2dInputRight(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 6], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.reflection_pad1d(x, (0, 3))


@register_test_case(module_factory=lambda: ReflectionPad1dModule2dInputRight())
def ReflectionPad1dModule2dInput_Right(module, tu: TestUtils):
    module.forward(tu.rand(3, 6))


# ==============================================================================
class TransposeIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 2], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([3, 4, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.transpose(x, -1, -2)


@register_test_case(module_factory=lambda: TransposeIntNegDimsModule())
def TransposeIntNegDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class PixelShuffleModuleStaticRank4Float32(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([3, 18, 2, 2], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.pixel_shuffle(x, 3)


@register_test_case(module_factory=lambda: PixelShuffleModuleStaticRank4Float32())
def PixelShuffleModuleStaticRank4Float32_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 18, 2, 2))


# ==============================================================================


class PixelShuffleModuleStaticRank3Int64(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([12, 2, 3], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.pixel_shuffle(x, 2)


@register_test_case(module_factory=lambda: PixelShuffleModuleStaticRank3Int64())
def PixelShuffleModuleStaticRank3Int64_basic(module, tu: TestUtils):
    module.forward(tu.randint(12, 2, 3, low=0, high=100))


# ==============================================================================


class PixelShuffleModuleFullDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.pixel_shuffle(x, 2)


@register_test_case(module_factory=lambda: PixelShuffleModuleFullDynamic())
def PixelShuffleModuleFullDynamic_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 8, 3, 3, low=0, high=100))


# ==============================================================================


class PixelShuffleModuleSpatiallyDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 1, 8, -1, -1], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.pixel_shuffle(x, 2)


@register_test_case(module_factory=lambda: PixelShuffleModuleSpatiallyDynamic())
def PixelShuffleModuleSpatiallyDynamic_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 1, 8, 2, 3, low=0, high=100))


# ==============================================================================


class PixelShuffleModuleSpatiallyStatic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, 3, 1], torch.int64, True)])
    def forward(self, x):
        return torch.ops.aten.pixel_shuffle(x, 2)


@register_test_case(module_factory=lambda: PixelShuffleModuleSpatiallyStatic())
def PixelShuffleModuleSpatiallyStatic_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 2, 12, 3, 1, low=0, high=100))


# ==============================================================================


class ChannelShuffleBasic(torch.nn.Module):
    # Basic test case for ChannelShuffle operation.
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=4)

    @export
    @annotate_args(
        [
            None,
            ([1, 8, 4, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffleBasic())
def ChannelShuffleBasic_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 129, dtype=torch.float32).reshape(1, 8, 4, 4))


# ==============================================================================


class ChannelShuffleUnitaryGroup(torch.nn.Module):
    # Test case where group = 1.
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=1)

    @export
    @annotate_args(
        [
            None,
            ([3, 8, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffleUnitaryGroup())
def ChannelShuffleUnitaryGroup_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 289, dtype=torch.float32).reshape(3, 8, 3, 4))


# ==============================================================================


class ChannelShuffle1D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=3)

    @export
    @annotate_args(
        [
            None,
            ([2, 9, 3], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffle1D())
def ChannelShuffle1D_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 55, dtype=torch.float32).reshape(2, 9, 3))


# ==============================================================================


class ChannelShuffle4D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=2)

    @export
    @annotate_args(
        [
            None,
            ([1, 4, 1, 2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffle4D())
def ChannelShuffle4D_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 97, dtype=torch.float32).reshape(1, 4, 1, 2, 3, 4))


# ==============================================================================


class ChannelShuffleTrailingOnes(torch.nn.Module):
    # Test case where ChannelShuffle last dimensions are ones.
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=2)

    @export
    @annotate_args(
        [
            None,
            ([1, 8, 1, 1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffleTrailingOnes())
def ChannelShuffleTrailingOnes_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1, 1))


# ==============================================================================


class ChannelShuffleDynamicDims(torch.nn.Module):
    # Test case for dynamic dimensions in ChannelShuffle operation.
    def __init__(self):
        super().__init__()
        self.shuffle = torch.nn.ChannelShuffle(groups=4)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.shuffle(x)


@register_test_case(module_factory=lambda: ChannelShuffleDynamicDims())
def ChannelShuffleDynamicDims_basic(module, tu: TestUtils):
    module.forward(torch.arange(1, 129, dtype=torch.float32).reshape(1, 8, 4, 4))


# ==============================================================================


class TensorsConcatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.cat([x, y, z], 1)


@register_test_case(module_factory=lambda: TensorsConcatModule())
def TensorsConcatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsConcatComplex64FloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex64, True),
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float16, True),
        ]
    )
    def forward(self, a, b, c, d):
        return torch.cat([a, b, c, d], 1)


@register_test_case(module_factory=lambda: TensorsConcatComplex64FloatModule())
def TensorsConcatComplex64FloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 1, 4, low=1, high=10).to(torch.complex64),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float64),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float32),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float16),
    )


# ==============================================================================


class TensorsConcatComplex128FloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex128, True),
            ([-1, -1, -1], torch.float64, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float16, True),
        ]
    )
    def forward(self, a, b, c, d):
        return torch.cat([a, b, c, d], 1)


@register_test_case(module_factory=lambda: TensorsConcatComplex128FloatModule())
def TensorsConcatComplex128FloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 1, 4, low=1, high=10).to(torch.complex128),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float64),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float32),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.float16),
    )


# ==============================================================================


class TensorsConcatComplex128IntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex128, True),
            ([-1, -1, -1], torch.int64, True),
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, a, b, c):
        return torch.cat([a, b, c], 1)


@register_test_case(module_factory=lambda: TensorsConcatComplex128IntModule())
def TensorsConcatComplex128IntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 1, 4, low=1, high=10).to(torch.complex128),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.int64),
        tu.rand(2, 3, 4, low=1, high=10).to(torch.int32),
    )


# ==============================================================================


class TensorsConcatNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.cat([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsConcatNegativeDimModule())
def TensorsConcatNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsConcatPromoteDTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.bool, True),
            ([-1, -1, -1], torch.int32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.cat([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsConcatPromoteDTypeModule())
def TensorsConcatPromoteDTypeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(2, 2, 4, low=0, high=2).bool(),
        tu.randint(2, 1, 4, low=0, high=100).int(),
        tu.randint(2, 3, 4, low=0, high=100).long(),
    )


# ==============================================================================


class TensorsConcatStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 2, 4], torch.float32, True),
            ([2, 1, 4], torch.float32, True),
            ([2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.cat([x, y, z], dim=1)


@register_test_case(module_factory=lambda: TensorsConcatStaticModule())
def TensorsConcatStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsConcatNegativeDimStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 2, 4], torch.float32, True),
            ([2, 1, 4], torch.float32, True),
            ([2, 3, 4], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.cat([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsConcatNegativeDimStaticModule())
def TensorsConcatNegativeDimStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 4), tu.rand(2, 1, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsStackModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.stack([x, y, z], dim=1)


@register_test_case(module_factory=lambda: TensorsStackModule())
def TensorsStackModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), tu.rand(2, 3, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsStackSingleElementListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.stack([x], dim=1)


@register_test_case(module_factory=lambda: TensorsStackSingleElementListModule())
def TensorsStackSingleElementListModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 32))


# ==============================================================================


class TensorsStackNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.stack([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsStackNegativeDimModule())
def TensorsStackNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), tu.rand(2, 3, 4), tu.rand(2, 3, 4))


# ==============================================================================


class TensorsStackPromoteDTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.bool, True),
            ([-1, -1, -1], torch.int32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.stack([x, y, z], dim=-2)


@register_test_case(module_factory=lambda: TensorsStackPromoteDTypeModule())
def TensorsStackPromoteDTypeModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(2, 3, 4, low=0, high=2).bool(),
        tu.randint(2, 3, 4, low=0, high=100).int(),
        tu.randint(2, 3, 4, low=0, high=100).long(),
    )


# ==============================================================================


class HstackBasicIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.bool, True),
            ([2, 3, 4], torch.int32, True),
            ([2, 3, 4], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.ops.aten.hstack([x, y, z])


@register_test_case(module_factory=lambda: HstackBasicIntModule())
def HstackBasicIntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(2, 3, 4, low=0, high=2).bool(),
        tu.randint(2, 3, 4, low=0, high=100).int(),
        tu.randint(2, 3, 4, low=0, high=100).long(),
    )


class HstackBasicFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 6, 4], torch.int32, True),
            ([2, 3, 4], torch.float64, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.hstack([x, y])


@register_test_case(module_factory=lambda: HstackBasicFloatModule())
def HstackBasicFloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 6, 4).int(),
        tu.rand(2, 3, 4).double(),
    )


class HstackBasicIntFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.int32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.hstack([x, y])


@register_test_case(module_factory=lambda: HstackBasicIntFloatModule())
def HstackBasicIntFloatModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(4, 6, 4, 2, low=1, high=50).int(),
        tu.rand(4, 3, 4, 2),
    )


class HstackBasicComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.complex64, True),
            ([-1, -1, -1, -1], torch.complex128, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.hstack([x, y])


@register_test_case(module_factory=lambda: HstackBasicComplexModule())
def HstackBasicComplexModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(4, 6, 4, 2).type(torch.complex64),
        tu.rand(4, 3, 4, 2).type(torch.complex128),
    )


# ==============================================================================


class ColumnStackBasicIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.bool, True),
            ([2, 3, 4], torch.int32, True),
            ([2, 3, 4], torch.int64, True),
        ]
    )
    def forward(self, x, y, z):
        return torch.ops.aten.column_stack([x, y, z])


@register_test_case(module_factory=lambda: ColumnStackBasicIntModule())
def ColumnStackBasicIntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(2, 3, 4, low=0, high=2).bool(),
        tu.randint(2, 3, 4, low=0, high=100).int(),
        tu.randint(2, 3, 4, low=0, high=100).long(),
    )


# ==============================================================================


class ColumnStack1dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4], torch.float32, True),
            ([4], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.column_stack([x, y])


@register_test_case(module_factory=lambda: ColumnStack1dModule())
def ColumnStack1dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand(4))


# ==============================================================================


class ColumnStack0dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
            ([], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.column_stack([x, y])


@register_test_case(module_factory=lambda: ColumnStack0dModule())
def ColumnStack0dModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor(4.0), torch.tensor(1.0))


# ==============================================================================


class GatherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return torch.gather(tensor, 2, indices)


@register_test_case(module_factory=lambda: GatherModule())
def GatherModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), torch.tensor([[[1, 2, 3], [1, 2, 3]]]))


# ==============================================================================


class GatherNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, tensor, indices):
        return torch.gather(tensor, -1, indices)


@register_test_case(module_factory=lambda: GatherNegativeDimModule())
def GatherNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4), torch.tensor([[[1, 2, 3], [1, 2, 3]]]))


# ==============================================================================


class GatherRandomIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([2, 3, 4], torch.float32, True),
            ([1, 2, 3], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        # This is a workaround for not supporting scalar arguments.
        # TODO: pass in dim as an argument to the forward method when scalar
        # arguments are supported.
        return tensor.add(tensor, alpha=tensor.size(1))


@register_test_case(module_factory=lambda: AddSizeIntModule())
def AddSizeIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))


# ==============================================================================


class AddSizeIntNegDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        # This is a workaround for not supporting scalar arguments.
        # TODO: pass in dim as an argument to the forward method when scalar
        # arguments are supported.
        return tensor.add(tensor, alpha=tensor.size(-2))


@register_test_case(module_factory=lambda: AddSizeIntNegDimModule())
def AddSizeIntNegDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3))


# ==============================================================================


class Add_MixPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float64, True),
        ]
    )
    def forward(self, a, b):
        a += b
        return a


@register_test_case(module_factory=lambda: Add_MixPModule())
def Add_MixPModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 3), tu.rand(3, 3).double())


# ==============================================================================


class EmbeddingModuleI64(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
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
        self.embed = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleI32())
def EmbeddingModuleI32_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100).to(torch.int32))


# ==============================================================================


class EmbeddingModuleF16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        ).to(torch.half)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int32, True),
        ]
    )
    def forward(self, indices):
        return self.embed.forward(indices)


@register_test_case(module_factory=lambda: EmbeddingModuleF16())
def EmbeddingModuleF16_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 3, high=100).to(torch.int32))


# ==============================================================================


class EmbeddingModuleI32Static(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @export
    @annotate_args(
        [
            None,
            ([3, 3], torch.int32, True),
        ]
    )
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
        self.embed = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=50, padding_idx=4
        )

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.int32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntModule())
def SoftmaxIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class SoftmaxIntNonNoneDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten.softmax(tensor, dim=2, dtype=torch.float64)


@register_test_case(module_factory=lambda: SoftmaxIntNonNoneDtypeModule())
def SoftmaxIntNonNoneDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class _SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten._softmax(tensor, 0, False)


@register_test_case(module_factory=lambda: _SoftmaxModule())
def _SoftmaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class SoftmaxIntNegDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.softmax = torch.nn.Softmax(-2)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntNegDimModule())
def SoftmaxIntNegDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class SoftmaxIntArgTypeF64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.softmax = torch.nn.Softmax(2)

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, tensor):
        return self.softmax.forward(tensor)


@register_test_case(module_factory=lambda: SoftmaxIntArgTypeF64Module())
def SoftmaxIntArgTypeF64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4).double())


# ==============================================================================


class _LogSoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten._log_softmax(tensor, dim=0, half_to_float=False)


@register_test_case(module_factory=lambda: _LogSoftmaxModule())
def _LogSoftmaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class _LogSoftmaxModuleStable(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten._log_softmax(tensor, dim=0, half_to_float=False)


@register_test_case(module_factory=lambda: _LogSoftmaxModuleStable())
def _LogSoftmaxModuleStable_basic(module, tu: TestUtils):
    # testing for numerical stability.
    # Should result in  tensor([-1e9, 0.00]) rather than tensor([-inf, 0.]).
    a = torch.tensor([0, 1e9])
    module.forward(a)


# ==============================================================================


class SafeSoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten._safe_softmax(tensor, dim=0)


@register_test_case(module_factory=lambda: SafeSoftmaxModule())
def SafeSoftmaxModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class SafeSoftmaxNonNoneDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, tensor):
        return torch.ops.aten._safe_softmax(tensor, dim=2, dtype=torch.float64)


@register_test_case(module_factory=lambda: SafeSoftmaxNonNoneDtypeModule())
def SafeSoftmaxNonNoneDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class SoftplusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, 1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([3, 1, 8], torch.float32, True),
            ([3, 1, 1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([3, 1, 8], torch.float32, True),
            ([], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        y = torch.broadcast_to(y, [3, 1, 8])
        return torch.ops.aten.sub(x, y)


@register_test_case(module_factory=lambda: BroadcastZeroRankInputStaticModule())
def BroadcastZeroRankInputStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 8), tu.rand())


# ==============================================================================


class BroadcastListConstructWithMinusOneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 8], torch.float32, True),
            ([3, 1, 8], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        y = torch.broadcast_to(y, [-1, -1, -1])
        return torch.ops.aten.sub(x, y)


@register_test_case(module_factory=lambda: BroadcastListConstructWithMinusOneModule())
def BroadcastListConstructWithMinusOneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 8), tu.rand(3, 1, 8))


# ==============================================================================


class BroadcastDynamicDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, -1, 1, -1], torch.float32, True),
            ([1, -1, 1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        dim_at_index_1 = torch.ops.aten.size(x, 1)
        dim_at_index_3 = torch.ops.aten.size(x, 3)
        res = torch.ops.aten.broadcast_to(y, [1, dim_at_index_1, 1, dim_at_index_3])
        return res


@register_test_case(module_factory=lambda: BroadcastDynamicDimModule())
def BroadcastDynamicDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 1, 4), tu.rand(1, 1, 1, 1))


# ==============================================================================


class RollModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, -1, 2], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([3, 1, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.repeat([2, 1, 3, 4])


@register_test_case(module_factory=lambda: RepeatModule())
def RepeatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))


# ==============================================================================


class RepeatInterleaveSelfIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.repeat_interleave(2, 1)


@register_test_case(module_factory=lambda: RepeatInterleaveSelfIntModule())
def RepeatInterleaveSelfIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class RepeatInterleaveSelfIntNoDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 4, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.repeat_interleave(2)


@register_test_case(module_factory=lambda: RepeatInterleaveSelfIntNoDimModule())
def RepeatInterleaveSelfIntNoDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


# ==============================================================================


class TileSmallDimsSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.tile([3, 4])


@register_test_case(module_factory=lambda: TileSmallDimsSizeModule())
def TileSmallDimsSizeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))


# ==============================================================================


class TileBigDimsSizeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 2], torch.float32, True),
        ]
    )
    def forward(self, x):
        return x.tile([3, 4, 5, 6])


@register_test_case(module_factory=lambda: TileBigDimsSizeModule())
def TileBigDimsSizeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 2))


# ==============================================================================


class ExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, tensor):
        return self.log_softmax.forward(tensor)


@register_test_case(module_factory=lambda: LogSoftmaxIntModule())
def LogSoftmaxIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4).double())


# ==============================================================================


class PrimMinIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.prim.min(1, -1)


@register_test_case(module_factory=lambda: PrimMinIntModule())
def PrimMinIntModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class PrimMinIntDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.prim.min(a.size(0), a.size(1))


@register_test_case(module_factory=lambda: PrimMinIntDynamicModule())
def PrimMinIntDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5))


# ==============================================================================


class PrimMaxIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.prim.max(a.size(0), a.size(1))


@register_test_case(module_factory=lambda: PrimMaxIntModule())
def PrimMaxIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5))


# ==============================================================================


class NumToTensorIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        res = torch.dropout(x, 0.3, train=True)
        return torch.mean(res), torch.std(res)


@register_test_case(module_factory=lambda: DropoutTrainModule())
def DropoutTrainModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1536))


# ==============================================================================


class DropoutTrainStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1024, 1536], torch.float32, True),
        ]
    )
    def forward(self, x):
        res = torch.dropout(x, 0.3, train=True)
        return torch.mean(res), torch.std(res)


@register_test_case(module_factory=lambda: DropoutTrainStaticShapeModule())
def DropoutTrainStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1536))


# ==============================================================================


class NativeDropoutEvalFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.native_dropout(x, 0.1, train=False)


@register_test_case(module_factory=lambda: NativeDropoutEvalFloatModule())
def NativeDropoutEvalFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class NativeDropoutTrainModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        res = torch.native_dropout(x, 0.3, train=True)
        return (
            torch.mean(res[0]),
            torch.std(res[0]),
            torch.mean(res[1].to(torch.float32)),
            torch.std(res[1].to(torch.float32)),
        )


@register_test_case(module_factory=lambda: NativeDropoutTrainModule())
def NativeDropoutTrainModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1536))


# ==============================================================================


class NativeDropoutTrainStaticShapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1024, 1536], torch.float32, True),
        ]
    )
    def forward(self, x):
        res = torch.native_dropout(x, 0.3, train=True)
        return (
            torch.mean(res[0]),
            torch.std(res[0]),
            torch.mean(res[1].to(torch.float32)),
            torch.std(res[1].to(torch.float32)),
        )


@register_test_case(module_factory=lambda: NativeDropoutTrainStaticShapeModule())
def NativeDropoutTrainStaticShapeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1024, 1536))


# ==============================================================================


class NumelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.bool, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.bool, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.bool, True),
            ([-1, -1], torch.bool, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
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
        self.register_buffer("t", torch.randint(-5, 5, (2, 3)))

    @export
    @annotate_args(
        [
            None,
        ]
    )
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
        self.register_buffer("t", torch.randint(-5, 5, (256, 1024)))

    @export
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))


@register_test_case(module_factory=lambda: IndexTensorModule())
def IndexTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5), tu.randint(2, 3, high=4))


# ==============================================================================
class IndexTensorStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 5], torch.float32, True),
            ([2, 3], torch.int64, True),
        ]
    )
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))


@register_test_case(module_factory=lambda: IndexTensorStaticModule())
def IndexTensorStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5), tu.randint(2, 3, high=4))


# ==============================================================================


class IndexTensorMultiIndexStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 5], torch.float32, True),
            ([2, 3], torch.int64, True),
            ([2, 3], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (index1, index2))


@register_test_case(module_factory=lambda: IndexTensorMultiIndexStaticModule())
def IndexTensorMultiIndexStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 5), tu.randint(2, 3, high=4), tu.randint(2, 3, high=4))


# ==============================================================================


class IndexTensorModule3dInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, index):
        return torch.ops.aten.index(x, (index,))


@register_test_case(module_factory=lambda: IndexTensorModule3dInput())
def IndexTensorModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))


# ==============================================================================


class IndexTensorStaticContiguousWithNoneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4, 5, 32], torch.float32, True),
            ([1, 2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
        ]
    )
    def forward(self, x, index, index1):
        return torch.ops.aten.index(x, (None, index, index1, None))


@register_test_case(module_factory=lambda: IndexTensorStaticContiguousWithNoneModule())
def IndexTensorStaticContiguousWithNoneModule_basic(module, tu: TestUtils):

    module.forward(
        tu.rand(2, 3, 4, 5, 32), torch.tensor([[[0], [1]]]), torch.tensor([[0], [1]])
    )


# ==============================================================================


class IndexTensorDyanmicInputContiguousWithNoneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([1, 2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
        ]
    )
    def forward(self, x, index, index1):
        return torch.ops.aten.index(x, (None, index, index1, None))


@register_test_case(
    module_factory=lambda: IndexTensorDyanmicInputContiguousWithNoneModule()
)
def IndexTensorDyanmicInputContiguousWithNoneModule_basic(module, tu: TestUtils):

    module.forward(
        tu.rand(2, 3, 4, 5, 32), torch.tensor([[[0], [1]]]), torch.tensor([[0], [1]])
    )


# ==============================================================================


class IndexTensorStaticNonContiguousWithNoneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 4, 5, 32], torch.float32, True),
            ([1, 2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
        ]
    )
    def forward(self, x, index, index1, index2):
        return torch.ops.aten.index(x, (None, index, index1, None, index2))


@register_test_case(
    module_factory=lambda: IndexTensorStaticNonContiguousWithNoneModule()
)
def IndexTensorStaticNonContiguousWithNoneModule_basic(module, tu: TestUtils):

    module.forward(
        tu.rand(2, 3, 4, 5, 32),
        torch.tensor([[[0], [1]]]),
        torch.tensor([[0], [1]]),
        torch.tensor([[0], [1]]),
    )


# ==============================================================================


class IndexTensorDyanmicInputNonContiguousWithNoneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
            ([1, 2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
            ([2, 1], torch.int64, True),
        ]
    )
    def forward(self, x, index, index1, index2):
        return torch.ops.aten.index(x, (None, index, index1, None, index2))


@register_test_case(
    module_factory=lambda: IndexTensorDyanmicInputNonContiguousWithNoneModule()
)
def IndexTensorDyanmicInputNonContiguousWithNoneModule_basic(module, tu: TestUtils):

    module.forward(
        tu.rand(2, 3, 4, 5, 32),
        torch.tensor([[[0], [1]]]),
        torch.tensor([[0], [1]]),
        torch.tensor([[0], [1]]),
    )


# ==============================================================================


class IndexTensorSelectDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([3, 3], torch.int64, True),
            ([3], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(
            x,
            (
                index1,
                index2,
            ),
        )


@register_test_case(module_factory=lambda: IndexTensorMultiInput())
def IndexTensorMultiInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(3, 3, high=3), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputOneDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([6, 1], torch.int64, True),
            ([3], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(
            x,
            (
                index1,
                index2,
            ),
        )


@register_test_case(module_factory=lambda: IndexTensorMultiInputOneDim())
def IndexTensorMultiInputOneDim_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputContiguousOneDimDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, 1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(
            x,
            (
                None,
                index1,
                index2,
            ),
        )


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputContiguousOneDimDynamic()
)
def IndexTensorMultiInputContiguousOneDimDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousOneDimDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, 1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(
            x,
            (
                index1,
                None,
                index2,
            ),
        )


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputNonContiguousOneDimDynamic()
)
def IndexTensorMultiInputNonContiguousOneDimDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 1, high=4), tu.randint(3, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, 2], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(
            x,
            (
                index2,
                None,
                index1,
            ),
        )


@register_test_case(module_factory=lambda: IndexTensorMultiInputNonContiguousDynamic())
def IndexTensorMultiInputNonContiguousDynamic_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(6, 2, high=2), tu.randint(2, high=3))


# ==============================================================================


class IndexTensorMultiInputNonContiguousMultipleStaticDims(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([4, 1], torch.int64, True),
            ([1, 3], torch.int64, True),
            ([-1, 3], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, (index1, index2, index3))


@register_test_case(
    module_factory=lambda: IndexTensorMultiInputNonContiguousMultipleStaticDims()
)
def IndexTensorMultiInputNonContiguousMultipleStaticDims_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(5, 4, 3, 2),
        tu.randint(4, 1, high=3),
        tu.randint(1, 3, high=1),
        tu.randint(4, 3, high=1),
    )


# ==============================================================================


class IndexTensorMultiInputNonContiguous(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([4, 2], torch.int64, True),
            ([4, 2], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (index1, None, index2))


@register_test_case(module_factory=lambda: IndexTensorMultiInputNonContiguous())
def IndexTensorMultiInputNonContiguous_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(5, 4, 3, 2), tu.randint(4, 2, high=3), tu.randint(4, 2, high=1)
    )


# ==============================================================================


class IndexTensorMultiInputThreeIndexers(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1, -1], torch.float32, True),
            ([8, 4, 2], torch.int64, True),
            ([8, 1, 1], torch.int64, True),
            ([4, 2], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, (None, None, index1, None, index2, index3))


@register_test_case(module_factory=lambda: IndexTensorMultiInputThreeIndexers())
def IndexTensorMultiInputThreeIndexers_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(1, 2, 4, 4, 5, 3),
        tu.randint(8, 4, 2, high=3),
        tu.randint(8, 1, 1, high=4),
        tu.randint(4, 2, high=2),
    )


# ==============================================================================


class IndexTensorMultiInputContiguousCenter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([2, 2], torch.int64, True),
            ([2], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2):
        return torch.ops.aten.index(x, (None, index1, index2, None))


@register_test_case(module_factory=lambda: IndexTensorMultiInputContiguousCenter())
def IndexTensorMultiInputContiguousCenter_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3, 2), tu.randint(2, 2, high=3), tu.randint(2, high=2))


# ==============================================================================


class IndexTensorNegativeIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 2, 3, 2], torch.float32, True),
            ([1], torch.int64, True),
        ]
    )
    def forward(self, x, index):
        return torch.ops.aten.index(x, (None, None, index))


@register_test_case(module_factory=lambda: IndexTensorNegativeIndexModule())
def IndexTensorNegativeIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 3, 2), tu.randint(1, low=-2, high=0))


# ==============================================================================


class IndexTensorHackedTwinModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, index):
        return torch.ops.aten.index(x, [index])


@register_test_case(module_factory=lambda: IndexTensorHackedTwinModule3dInput())
def IndexTensorHackedTwinModule3dInput_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 4, 3), tu.randint(2, 3, high=3))


# ==============================================================================


class IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([4, 1], torch.int64, True),
            ([1, 3], torch.int64, True),
            ([-1, 3], torch.int64, True),
        ]
    )
    def forward(self, x, index1, index2, index3):
        return torch.ops.aten.index(x, [index1, index2, index3])


@register_test_case(
    module_factory=lambda: IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims()
)
def IndexTensorHackedTwinMultiInputNonContiguousMultipleStaticDims_basic(
    module, tu: TestUtils
):
    module.forward(
        tu.rand(5, 4, 3, 2),
        tu.randint(4, 1, high=3),
        tu.randint(1, 3, high=1),
        tu.randint(4, 3, high=1),
    )


# ==============================================================================


class SquareModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([200], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1], torch.int64, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, 1, 1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.expand_as(x, y)


@register_test_case(module_factory=lambda: ExpandAsFloatModule())
def ExpandAsFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 1), tu.rand(3, 4, 5))


class ExpandAsIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 1], torch.int64, True),
            ([-1, -1, -1], torch.int64, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.expand_as(x, y)


@register_test_case(module_factory=lambda: ExpandAsIntModule())
def ExpandAsIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(1, 1, 1, high=100), tu.randint(4, 5, 6, high=200))


# ==============================================================================


class CopyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyModule())
def CopyModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 4))


class CopyWithDifferentSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 4], torch.float32, True),
            ([-1, -1, 1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyWithDifferentSizesModule())
def CopyWithDifferentSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.rand(3, 2, 1))


class CopyWithDifferentDTypesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int64, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyWithDifferentDTypesModule())
def CopyWithDifferentDTypesModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 2, 4, high=100), tu.rand(3, 2, 4))


class CopyWithDifferentDTypesAndSizesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, 4], torch.float32, True),
            ([-1, -1, 1], torch.int64, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.copy_(x, y)


@register_test_case(module_factory=lambda: CopyWithDifferentDTypesAndSizesModule())
def CopyWithDifferentDTypesAndSizesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4), tu.randint(3, 2, 1, high=1000))


# ==============================================================================


class ToCopyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten._to_copy(x)


@register_test_case(module_factory=lambda: ToCopyModule())
def ToCopyModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyWithDTypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=torch.int64)


@register_test_case(module_factory=lambda: ToCopyWithDTypeModule())
def ToCopyWithDTypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyWithDTypeFalsePinMemoryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=torch.int64, pin_memory=False)


@register_test_case(module_factory=lambda: ToCopyWithDTypeFalsePinMemoryModule())
def ToCopyWithDTypeFalsePinMemoryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


class ToCopyBoolDTypeStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 5, 5], torch.uint8, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.flip(x, [1, 2])


@register_test_case(module_factory=lambda: FlipModule())
def FlipModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class FlipModuleStaticShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 2, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.flip(x, [1, 2])


@register_test_case(module_factory=lambda: FlipModuleStaticShape())
def FlipModuleStaticShape_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class FlipNegativeIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([3, 2, 4], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.flip(x, [-1])


@register_test_case(module_factory=lambda: FlipNegativeIndexModule())
def FlipNegativeIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 2, 4))


# ==============================================================================


class FliplrOddRankModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.fliplr(a)


@register_test_case(module_factory=lambda: FliplrOddRankModule())
def FliplrOddRankModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2))


# ==============================================================================


class FliplrEvenRankModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.fliplr(a)


@register_test_case(module_factory=lambda: FliplrEvenRankModule())
def FliplrEvenRankModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2, 4))


# ==============================================================================


class FlipudOddRankModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.flipud(a)


@register_test_case(module_factory=lambda: FlipudOddRankModule())
def FlipudOddRankModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2))


# ==============================================================================


class FlipudEvenRankModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, a):
        return torch.ops.aten.flipud(a)


@register_test_case(module_factory=lambda: FlipudEvenRankModule())
def FlipudEvenRankModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2, 4))


# ==============================================================================


class DetachModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.aten.len(self.str)


@register_test_case(module_factory=lambda: LenStrModule())
def LenStrModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class IntFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = 1.0

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return torch.ops.aten.Int(self.value)


@register_test_case(module_factory=lambda: IntFloatModule())
def IntFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class AtenSubFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value1 = 1.0
        self.value2 = 2.0

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return float(torch.ops.aten.sub(self.value1, self.value2))


@register_test_case(module_factory=lambda: AtenSubFloatModule())
def AtenSubFloatModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ScalarImplicitFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float64, True),
        ]
    )
    def forward(self, x):
        return float(torch.ops.aten.ScalarImplicit(x))


@register_test_case(module_factory=lambda: ScalarImplicitFloatModule())
def ScalarImplicitFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double())


class ScalarImplicitIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.int64, True),
        ]
    )
    def forward(self, x):
        return int(torch.ops.aten.ScalarImplicit(x))


@register_test_case(module_factory=lambda: ScalarImplicitIntModule())
def ScalarImplicitIntModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(low=-100, high=100))


# ==============================================================================


class FloatImplicitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float64, True),
        ]
    )
    def forward(self, x):
        return float(torch.ops.aten.FloatImplicit(x))


@register_test_case(module_factory=lambda: FloatImplicitModule())
def FloatImplicitModule_basic(module, tu: TestUtils):
    module.forward(tu.rand().double())


# ==============================================================================


class IntImplicitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.int64, True),
        ]
    )
    def forward(self, x):
        return float(torch.ops.aten.IntImplicit(x))


@register_test_case(module_factory=lambda: IntImplicitModule())
def IntImplicitModule_basic(module, tu: TestUtils):
    module.forward(tu.randint())


# ==============================================================================


class PowModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.pow(x, y)


@register_test_case(module_factory=lambda: PowModule())
def PowFloatFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 5))


# ==============================================================================


class PowIntFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.pow(x, y)


@register_test_case(module_factory=lambda: PowIntFloatModule())
def PowIntFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, 5, dtype=torch.int32), tu.rand(3, 4, 5))


# ==============================================================================


class PowFloatIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.pow(x, y)


@register_test_case(module_factory=lambda: PowFloatIntModule())
def PowFloatIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.randint(3, 4, 5, dtype=torch.int32))


# ==============================================================================


class PowIntIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.int32, True),
            ([-1, -1, -1], torch.int32, True),
        ]
    )
    def forward(self, x, y):
        return torch.ops.aten.pow(x, y)


@register_test_case(module_factory=lambda: PowIntIntModule())
def PowIntIntModule_basic(module, tu: TestUtils):
    module.forward(
        tu.randint(3, 4, 5, high=10, dtype=torch.int32),
        tu.randint(3, 4, 5, high=20, dtype=torch.int32),
    )


# ==============================================================================


class IsInfiniteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.isfinite(x)


@register_test_case(module_factory=lambda: IsInfiniteModule())
def IsInfiniteModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([-torch.inf, torch.inf, torch.nan, -2.3, 0.0, 1.5]))


# ==============================================================================


class BaddbmmDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmDynamicModule())
def BaddbmmDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 7], torch.float32, True),
            ([5, 2, 9], torch.float32, True),
            ([5, 9, 7], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmStaticModule())
def BaddbmmStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 7), tu.rand(5, 2, 9), tu.rand(5, 9, 7))


class BaddbmmWithAlphaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, alpha=5)


@register_test_case(module_factory=lambda: BaddbmmWithAlphaModule())
def BaddbmmWithAlphaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmWithBetaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, beta=0.5)


@register_test_case(module_factory=lambda: BaddbmmWithBetaModule())
def BaddbmmWithBetaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmWithAlphaBetaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2, beta=6, alpha=2.4)


@register_test_case(module_factory=lambda: BaddbmmWithAlphaBetaModule())
def BaddbmmWithAlphaBetaModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(3, 4, 6), tu.rand(3, 6, 5))


class BaddbmmBroadcast1DInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1], torch.float32, True),
            ([5, 2, 9], torch.float32, True),
            ([5, 9, 7], torch.float32, True),
        ]
    )
    def forward(self, input, batch1, batch2):
        return torch.ops.aten.baddbmm(input, batch1, batch2)


@register_test_case(module_factory=lambda: BaddbmmBroadcast1DInputModule())
def BaddbmmBroadcast1DInputModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(
            1,
        ),
        tu.rand(5, 2, 9),
        tu.rand(5, 9, 7),
    )


class BaddbmmBroadcast2DInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7], torch.float32, True),
            ([5, 2, 9], torch.float32, True),
            ([5, 9, 7], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([3, 4, 5, 6], torch.float32, True),
        ]
    )
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRankNStaticModule())
def NumpyTRankNStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6))


class NumpyTRankNDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRankNDynamicModule())
def NumpyTRankNDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5, 6, 2))


class NumpyTRank2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank2Module())
def NumpyTRank2Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class NumpyTRank1Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank1Module())
def NumpyTRank1Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(3))


class NumpyTRank0Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.float32, True),
        ]
    )
    def forward(self, lhs):
        return torch.ops.aten.numpy_T(lhs)


@register_test_case(module_factory=lambda: NumpyTRank0Module())
def NumpyTRank0Module_basic(module, tu: TestUtils):
    module.forward(torch.tensor(7, dtype=torch.float32))


# ==============================================================================


class AtenEmbeddingBagStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 2], torch.float32, True),
            ([3], torch.int64, True),
            ([1], torch.int64, True),
        ]
    )
    def forward(self, weight, indices, offsets):
        return torch.ops.aten.embedding_bag(
            weight,
            indices,
            offsets,
            scale_grad_by_freq=False,
            mode=0,
            sparse=False,
            per_sample_weights=None,
            include_last_offset=False,
            padding_idx=None,
        )


@register_test_case(module_factory=lambda: AtenEmbeddingBagStaticModule())
def AtenEmbeddingBagStaticModule_basic(module, tu: TestUtils):
    weight = tu.rand(4, 2)
    indices = torch.LongTensor([3, 0, 1])
    offsets = torch.LongTensor([0])
    module.forward(weight, indices, offsets)


class AtenEmbeddingBagSumExample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, weight, indices, offsets):
        return torch.ops.aten.embedding_bag(
            weight,
            indices,
            offsets,
            scale_grad_by_freq=False,
            mode=0,
            sparse=False,
            per_sample_weights=None,
            include_last_offset=False,
            padding_idx=None,
        )


@register_test_case(module_factory=lambda: AtenEmbeddingBagSumExample())
def AtenEmbeddingBagSumExample_basic(module, tu: TestUtils):
    weight = tu.rand(100, 10)
    indices = torch.LongTensor(
        [0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54]
    )
    offsets = torch.LongTensor([0, 3, 5, 7, 9, 10, 15])
    module.forward(weight, indices, offsets)


class Aten_EmbeddingBagExample(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, weight, indices, offsets):
        return torch.ops.aten._embedding_bag(weight, indices, offsets)


@register_test_case(module_factory=lambda: Aten_EmbeddingBagExample())
def Aten_EmbeddingBagExample_basic(module, tu: TestUtils):
    weight = tu.rand(100, 10)
    indices = torch.LongTensor(
        [0, 1, 2, 2, 0, 2, 1, 3, 20, 50, 99, 2, 4, 5, 6, 7, 34, 54]
    )
    offsets = torch.LongTensor([0, 3, 5, 7, 9, 10, 15])
    module.forward(weight, indices, offsets)


# ==============================================================================


class CumsumModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, val):
        # the onnx cumsum op uses a constant 1d tensor
        # to specify the dimension along which to do cumsum
        # we replicate that here to ensure that cumsum correctly
        # trigger the relevant folders and provides TMTensor
        # with a constant dimension
        ones = torch.ones([1], dtype=torch.int32)
        return torch.ops.aten.cumsum(val, ones.item())


@register_test_case(module_factory=lambda: CumsumModule())
def CumsumModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumsumStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumsum(val, 1)


@register_test_case(module_factory=lambda: CumsumStaticModule())
def CumsumStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumsumStaticNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumsum(val, dim=-1)


@register_test_case(module_factory=lambda: CumsumStaticNegativeDimModule())
def CumsumStaticNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumsumInputDtypeInt32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.int32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumsum(val, 1)


@register_test_case(module_factory=lambda: CumsumInputDtypeInt32Module())
def CumsumInputDtypeInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 7, 4).to(torch.int32))


class CumsumWithDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.bool, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumsum(val, dim=1, dtype=6)


@register_test_case(module_factory=lambda: CumsumWithDtypeModule())
def CumsumWithDtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 7, 4, low=-1, high=10).to(torch.bool))


# ==============================================================================


class LogCumsumExpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.logcumsumexp(x, dim=1)


@register_test_case(module_factory=lambda: LogCumsumExpModule())
def LogCumsumExpModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 2, 3))


class LogCumsumExpStaticNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([8, 5, 6], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.logcumsumexp(x, dim=-2)


@register_test_case(module_factory=lambda: LogCumsumExpStaticNegativeDimModule())
def LogCumsumExpStaticNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(8, 5, 6))


class LogCumsumExpStaticFloat64DtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([5, 3, 6, 9], torch.float64, True)])
    def forward(self, x):
        return torch.ops.aten.logcumsumexp(x, dim=1)


@register_test_case(module_factory=lambda: LogCumsumExpStaticFloat64DtypeModule())
def LogCumsumExpStaticFloat64DtypeModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3, 6, 9).to(torch.float64))


# ==============================================================================


class CumprodModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, val):
        ones = torch.ones([1], dtype=torch.int32)
        return torch.ops.aten.cumprod(val, ones.item())


@register_test_case(module_factory=lambda: CumprodModule())
def CumprodModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumprodStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumprod(val, 1)


@register_test_case(module_factory=lambda: CumprodStaticModule())
def CumprodStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumprodStaticNegativeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumprod(val, dim=-1)


@register_test_case(module_factory=lambda: CumprodStaticNegativeDimModule())
def CumprodStaticNegativeDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 7, 4))


class CumprodInputDtypeInt32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 7, 4], torch.int32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.cumprod(val, 1)


@register_test_case(module_factory=lambda: CumprodInputDtypeInt32Module())
def CumprodInputDtypeInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 7, 4).to(torch.int32))


# ==============================================================================


class AtenToDeviceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten.to(
            val, device="cpu", dtype=torch.float, non_blocking=False
        )


@register_test_case(module_factory=lambda: AtenToDeviceModule())
def AtenToDeviceModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class Aten_CastFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.int64, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten._cast_Float(val)


@register_test_case(module_factory=lambda: Aten_CastFloatModule())
def Aten_CastFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 4))


class Aten_CastLongModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.float32, True),
        ]
    )
    def forward(self, val):
        return torch.ops.aten._cast_Long(val)


@register_test_case(module_factory=lambda: Aten_CastLongModule())
def Aten_CastLongModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4))


# ==============================================================================


class UpSampleNearest2dBackward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d_backward(
            input,
            output_size=[6, 12],
            input_size=[1, 1, 2, 3],
            scales_h=3.0,
            scales_w=4.0,
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dBackward())
def UpSampleNearest2dBackward_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 6, 12).to(torch.float64))


class UpSampleNearest2dBackwardScalesNone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input):
        return torch.ops.aten.upsample_nearest2d_backward(
            input,
            output_size=[4, 8],
            input_size=[1, 1, 2, 3],
            scales_h=None,
            scales_w=None,
        )


@register_test_case(module_factory=lambda: UpSampleNearest2dBackwardScalesNone())
def UpSampleNearest2dBackwardScalesNone_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 4, 8))


# ==============================================================================


class SortIntList(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
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
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        a = [1, 0, 3, 2]
        b = [3, 2, 1, 0]
        a.sort(reverse=True)
        return a == b


@register_test_case(module_factory=lambda: SortIntListReverse())
def SortIntListReverse_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class SortTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, input):
        return torch.sort(input)


@register_test_case(module_factory=lambda: SortTensor())
def SortTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class SortTensorInteger(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, input):
        return torch.sort(input)


@register_test_case(module_factory=lambda: SortTensorInteger())
def SortTensorInteger_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3))


class SortTensorDescending(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, input):
        return torch.sort(input, descending=True)


@register_test_case(module_factory=lambda: SortTensorDescending())
def SortTensorDescending_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class SortTensorSpecificDimension(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, input):
        return torch.sort(input, dim=1)


@register_test_case(module_factory=lambda: SortTensorSpecificDimension())
def SortTensorSpecificDimension_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class SortTensorNegativeDimension(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, input):
        return torch.sort(input, dim=-1)


@register_test_case(module_factory=lambda: SortTensorNegativeDimension())
def SortTensorNegativeDimension_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class ArgsortTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, input):
        return torch.argsort(input)


@register_test_case(module_factory=lambda: ArgsortTensor())
def ArgsortTensor_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5))


class ArgsortTensorInteger(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.int64, True)])
    def forward(self, input):
        return torch.argsort(input)


@register_test_case(module_factory=lambda: ArgsortTensorInteger())
def ArgsortTensorInteger_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3))


# ==============================================================================


class BucketizeTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, input, boundaries):
        return torch.bucketize(input, boundaries)


@register_test_case(module_factory=lambda: BucketizeTensorModule())
def BucketizeTensorModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6]]), torch.tensor([1, 4, 6]))


class BucketizeTensorOutInt32RightModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.int64, True),
            ([-1], torch.int64, True),
        ]
    )
    def forward(self, input, boundaries):
        return torch.bucketize(input, boundaries, out_int32=True, right=True)


@register_test_case(module_factory=lambda: BucketizeTensorOutInt32RightModule())
def BucketizeTensorOutInt32RightModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6]]), torch.tensor([1, 4, 6]))


class BucketizeTensorFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1], torch.float32, True),
        ]
    )
    def forward(self, input, boundaries):
        return torch.bucketize(input, boundaries)


@register_test_case(module_factory=lambda: BucketizeTensorFloatModule())
def BucketizeTensorFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(15, 17), torch.sort(tu.rand(16)).values)


class BucketizeTensorStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4], torch.int64, True),
            ([3], torch.int64, True),
        ]
    )
    def forward(self, input, boundaries):
        return torch.bucketize(input, boundaries)


@register_test_case(module_factory=lambda: BucketizeTensorStaticModule())
def BucketizeTensorStaticModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6]]), torch.tensor([1, 4, 6]))


class BucketizeTensorStaticFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([15, 17], torch.float32, True),
            ([16], torch.float32, True),
        ]
    )
    def forward(self, input, boundaries):
        return torch.bucketize(input, boundaries)


@register_test_case(module_factory=lambda: BucketizeTensorStaticFloatModule())
def BucketizeTensorStaticFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(15, 17), torch.sort(tu.rand(16)).values)


# ==============================================================================


class AtenFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([], torch.int64, True),
        ]
    )
    def forward(self, x):
        a = torch.ops.aten.ScalarImplicit(x)
        return torch.ops.aten.Float(a)


@register_test_case(module_factory=lambda: AtenFloatScalarModule())
def AtenFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(high=5))


# ==============================================================================


class MoveDimIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.movedim(x, source=1, destination=2)  # 0, 2, 1


@register_test_case(module_factory=lambda: MoveDimIntModule())
def MoveDimIntModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2, 1))


# ==============================================================================


class MoveDimIntNegativeIndexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.movedim(x, source=-1, destination=1)


@register_test_case(module_factory=lambda: MoveDimIntNegativeIndexModule())
def MoveDimIntNegativeIndexModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


# ==============================================================================


class ScaledDotProductAttentionSameModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([1, 5, 5], torch.float32, True),
            ([1, 5, 5], torch.float32, True),
            ([1, 5, 5], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value)


@register_test_case(module_factory=lambda: ScaledDotProductAttentionSameModule())
def ScaledDotProductAttentionSameModule_basic(module, tu: TestUtils):
    query = torch.randn(1, 5, 5, dtype=torch.float32)
    key = torch.randn(1, 5, 5, dtype=torch.float32)
    value = torch.randn(1, 5, 5, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionSameDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value)


@register_test_case(module_factory=lambda: ScaledDotProductAttentionSameDynamicModule())
def ScaledDotProductAttentionSameDynamicModule_basic(module, tu: TestUtils):
    query = torch.randn(1, 5, 5, dtype=torch.float32)
    key = torch.randn(1, 5, 5, dtype=torch.float32)
    value = torch.randn(1, 5, 5, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionSameCausalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )


@register_test_case(module_factory=lambda: ScaledDotProductAttentionSameCausalModule())
def ScaledDotProductAttentionSameCausalModule_basic(module, tu: TestUtils):
    query = torch.randn(1, 5, 5, dtype=torch.float32)
    key = torch.randn(1, 5, 5, dtype=torch.float32)
    value = torch.randn(1, 5, 5, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionDifferentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 8, 16], torch.float32, True),
            ([2, 3, 12, 16], torch.float32, True),
            ([2, 3, 12, 20], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value)


@register_test_case(module_factory=lambda: ScaledDotProductAttentionDifferentModule())
def ScaledDotProductAttentionDifferentModule_basic(module, tu: TestUtils):
    query = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    key = torch.randn(2, 3, 12, 16, dtype=torch.float32)
    value = torch.randn(2, 3, 12, 20, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionDifferentCausalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 8, 16], torch.float32, True),
            ([2, 3, 12, 16], torch.float32, True),
            ([2, 3, 12, 20], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )


@register_test_case(
    module_factory=lambda: ScaledDotProductAttentionDifferentCausalModule()
)
def ScaledDotProductAttentionDifferentDynamicCausalModule_basic(module, tu: TestUtils):
    query = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    key = torch.randn(2, 3, 12, 16, dtype=torch.float32)
    value = torch.randn(2, 3, 12, 20, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionDifferentDynamicCausalModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, -1, 16], torch.float32, True),
            ([2, 3, -1, 16], torch.float32, True),
            ([2, 3, -1, 20], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )


@register_test_case(
    module_factory=lambda: ScaledDotProductAttentionDifferentDynamicCausalModule()
)
def ScaledDotProductAttentionDifferentCausalModule_basic(module, tu: TestUtils):
    query = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    key = torch.randn(2, 3, 12, 16, dtype=torch.float32)
    value = torch.randn(2, 3, 12, 20, dtype=torch.float32)
    module.forward(query, key, value)


class ScaledDotProductAttentionMaskModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 8, 16], torch.float32, True),
            ([2, 3, 12, 16], torch.float32, True),
            ([2, 3, 12, 20], torch.float32, True),
            ([2, 1, 8, 12], torch.float32, True),
        ]
    )
    def forward(self, query, key, value, mask):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value, mask)


@register_test_case(module_factory=lambda: ScaledDotProductAttentionMaskModule())
def ScaledDotProductAttentionMaskModule_basic(module, tu: TestUtils):
    query = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    key = torch.randn(2, 3, 12, 16, dtype=torch.float32)
    value = torch.randn(2, 3, 12, 20, dtype=torch.float32)
    mask = torch.randn(2, 1, 8, 12, dtype=torch.float32)
    module.forward(query, key, value, mask)


class ScaledDotProductAttentionBoolMaskModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3, 8, 16], torch.float32, True),
            ([2, 3, 12, 16], torch.float32, True),
            ([2, 3, 12, 20], torch.float32, True),
            ([2, 3, 8, 12], torch.bool, True),
        ]
    )
    def forward(self, query, key, value, mask):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value, mask)


@register_test_case(module_factory=lambda: ScaledDotProductAttentionBoolMaskModule())
def ScaledDotProductAttentionBoolMaskModule_basic(module, tu: TestUtils):
    query = torch.randn(2, 3, 8, 16, dtype=torch.float32)
    key = torch.randn(2, 3, 12, 16, dtype=torch.float32)
    value = torch.randn(2, 3, 12, 20, dtype=torch.float32)
    mask = torch.randn(2, 3, 8, 12, dtype=torch.float32) > 0.5
    module.forward(query, key, value, mask)


class ScaledDotProductAttentionGQAModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([4, 32, 3, 8], torch.float32, True),
            ([4, 8, 3, 8], torch.float32, True),
            ([4, 8, 3, 8], torch.float32, True),
        ]
    )
    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(
            query, key, value, enable_gqa=True, is_causal=True
        )


@register_test_case(module_factory=lambda: ScaledDotProductAttentionGQAModule())
def ScaledDotProductAttentionGQAModule_basic(module, tu: TestUtils):
    query = torch.randn(4, 32, 3, 8, dtype=torch.float32)
    key = torch.randn(4, 8, 3, 8, dtype=torch.float32)
    value = torch.randn(4, 8, 3, 8, dtype=torch.float32)
    module.forward(query, key, value)


# ==============================================================================


class PrimsViewOfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.prims.view_of(x)


@register_test_case(module_factory=lambda: PrimsViewOfModule())
def PrimsViewOfModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 2))


class PrimsViewOfZeroRankModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([], torch.float32, True)])
    def forward(self, x):
        return torch.ops.prims.view_of(x)


@register_test_case(module_factory=lambda: PrimsViewOfZeroRankModule())
def PrimsViewOfZeroRankModule_basic(module, tu: TestUtils):
    module.forward(tu.rand())


# ==============================================================================


class OneHotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.long, True)])
    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=5)


@register_test_case(module_factory=lambda: OneHotModule())
def OneHotModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, high=5))


# ==============================================================================


class ConstantBoolParameterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bool_tensor = torch.tensor([True, False, True, False], dtype=torch.bool)

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        return self.bool_tensor


@register_test_case(module_factory=lambda: ConstantBoolParameterModule())
def ConstantBoolParameterModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class TensorAlloc1dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 4, 6], torch.int, True),
        ]
    )
    def forward(self, x):
        res = torch.tensor([x.shape[0]])
        return res


@register_test_case(module_factory=lambda: TensorAlloc1dStaticModule())
def TensorAlloc1dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 6))


# ==============================================================================


class ScalarTensorFloat32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        scalar = torch.ops.aten.scalar_tensor(1.0, dtype=torch.float32)
        return scalar


@register_test_case(module_factory=lambda: ScalarTensorFloat32Module())
def ScalarTensorFloat32Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ScalarTensorDefaultDtypeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        scalar = torch.ops.aten.scalar_tensor(1.0)
        return scalar


@register_test_case(module_factory=lambda: ScalarTensorDefaultDtypeModule())
def ScalarTensorDefaultDtypeModule_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ScalarTensorInt64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        scalar = torch.ops.aten.scalar_tensor(1, dtype=torch.int64)
        return scalar


@register_test_case(module_factory=lambda: ScalarTensorInt64Module())
def ScalarTensorInt64Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class ScalarTensorInt32Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
        ]
    )
    def forward(self):
        scalar = torch.ops.aten.scalar_tensor(1, dtype=torch.int32)
        return scalar


@register_test_case(module_factory=lambda: ScalarTensorInt32Module())
def ScalarTensorInt32Module_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class AtenTopKModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.topk(x, k=50, dim=-1, largest=True, sorted=True)


@register_test_case(module_factory=lambda: AtenTopKModule())
def AtenTopKModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 100))


class AtenTopKSmallestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1], torch.float32, True)])
    def forward(self, x):
        return torch.ops.aten.topk(x, k=20, dim=1, largest=False, sorted=True)


@register_test_case(module_factory=lambda: AtenTopKSmallestModule())
def AtenTopKSmallestModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 40, 50))


# ==============================================================================


class AtenComplexImagModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.complex64, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.imag(x)


@register_test_case(module_factory=lambda: AtenComplexImagModule())
def AtenComplexImagModule_basic(module, tu: TestUtils):
    module.forward(torch.view_as_complex(tu.rand(5, 2)))


class AtenComplexRealModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.complex64, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.real(x)


@register_test_case(module_factory=lambda: AtenComplexRealModule())
def AtenComplexRealModule_basic(module, tu: TestUtils):
    module.forward(torch.view_as_complex(tu.rand(5, 2)))


class AtenComplex64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.complex64, True),
        ]
    )
    def forward(self, x):
        return x


@register_test_case(module_factory=lambda: AtenComplex64Module())
def AtenComplex64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2).to(torch.complex64))


class AtenComplexViewModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.view_as_complex(x)


@register_test_case(module_factory=lambda: AtenComplexViewModule())
def AtenComplexViewModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2))


# ==============================================================================
class AtenRealView128Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex128, True),
        ]
    )
    def forward(self, x):
        return torch.view_as_real(x)


@register_test_case(module_factory=lambda: AtenRealView128Module())
def AtenRealView128Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 6, 1).to(torch.complex128))


# ==============================================================================
class AtenRealView64Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.complex64, True),
        ]
    )
    def forward(self, x):
        return torch.view_as_real(x)


@register_test_case(module_factory=lambda: AtenRealView64Module())
def AtenRealView64Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 6, 1).to(torch.complex64))


# ==============================================================================


class Add_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("tensor", torch.ones(2, 3))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.add_(x, self.tensor)


@register_test_case(module_factory=lambda: Add_Module())
def Add_Module_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


# ==============================================================================


class CosineSimilarityStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([2, 3], torch.float32, True),
            ([2, 3], torch.float32, True),
        ]
    )
    def forward(self, x1, x2):
        return torch.ops.aten.cosine_similarity(x1, x2)


@register_test_case(module_factory=lambda: CosineSimilarityStaticModule())
def CosineSimilarityStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3), tu.rand(2, 3))


# ==============================================================================


class CosineSimilarityStaticBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 2, 3], torch.float32, True),
            ([4, 5, 1, 1], torch.float32, True),
        ]
    )
    def forward(self, x1, x2):
        return torch.ops.aten.cosine_similarity(x1, x2)


@register_test_case(module_factory=lambda: CosineSimilarityStaticBroadcastModule())
def CosineSimilarityStaticBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 3), tu.rand(4, 5, 1, 1))


# ==============================================================================


class CosineSimilarityModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
            ([-1, -1], torch.float32, True),
        ]
    )
    def forward(self, x1, x2):
        return torch.ops.aten.cosine_similarity(x1, x2)


@register_test_case(module_factory=lambda: CosineSimilarityModule())
def CosineSimilarityModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3), tu.rand(2, 3))


# ==============================================================================


class IscloseStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 5], torch.float32, True),
            ([5, 5], torch.float32, True),
        ]
    )
    def forward(self, x, y):
        return torch.isclose(x, y)


@register_test_case(module_factory=lambda: IscloseStaticModule())
def IscloseStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 5), tu.rand(5, 5))


# ==============================================================================


class IscloseStaticModuleTrue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("tensor", torch.ones(1))

    @export
    @annotate_args(
        [
            None,
            ([5, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.isclose(x, self.tensor)


@register_test_case(module_factory=lambda: IscloseStaticModuleTrue())
def IscloseStaticModuleTrue_basic(module, tu: TestUtils):
    module.forward(torch.ones(5, 5))


# ==============================================================================


class CloneModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([5, 5], torch.float32, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.clone(x)


@register_test_case(module_factory=lambda: CloneModule())
def CloneModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 5))


# ==============================================================================


class AtenKthvalueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 6, 3], torch.int32, True)])
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, k=4, dim=1, keepdim=False)


@register_test_case(module_factory=lambda: AtenKthvalueModule())
def AtenKthvalueModule_basic(module, tu: TestUtils):
    module.forward(torch.randperm(2 * 6 * 3, dtype=torch.int32).reshape(2, 6, 3))


# ==============================================================================


class AtenKthvalueKeepDimModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([2, 6, 3], torch.int32, True)])
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, k=4, dim=1, keepdim=True)


@register_test_case(module_factory=lambda: AtenKthvalueKeepDimModule())
def AtenKthvalueKeepDimModule_basic(module, tu: TestUtils):
    module.forward(torch.randperm(2 * 6 * 3, dtype=torch.int32).reshape(2, 6, 3))


# ==============================================================================


class AtenKthvalueDynamicDimsModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.int32, True)])
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, k=6, dim=2, keepdim=True)


@register_test_case(module_factory=lambda: AtenKthvalueDynamicDimsModule())
def AtenKthvalueDynamicDimsModule_basic(module, tu: TestUtils):
    module.forward(torch.randperm(4 * 2 * 8 * 3, dtype=torch.int32).reshape(4, 2, 8, 3))


# ==============================================================================


class AtenKthvalueFloat64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([4, 2, 8, 3], torch.float64, True)])
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, k=3, dim=0, keepdim=True)


@register_test_case(module_factory=lambda: AtenKthvalueFloat64Module())
def AtenKthvalueFloat64Module_basic(module, tu: TestUtils):
    module.forward(
        torch.randperm(4 * 2 * 8 * 3, dtype=torch.float64).reshape(4, 2, 8, 3)
    )


# ==============================================================================


class AtenKthvalueFloat64DynamicDimsModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1, -1, -1, -1], torch.float64, True)])
    def forward(self, x):
        return torch.ops.aten.kthvalue(x, k=3, dim=3, keepdim=True)


@register_test_case(module_factory=lambda: AtenKthvalueFloat64DynamicDimsModule())
def AtenKthvalueFloat64DynamicDimsModule_basic(module, tu: TestUtils):
    module.forward(
        torch.randperm(4 * 2 * 8 * 3, dtype=torch.float64).reshape(4, 2, 8, 3)
    )


# ==============================================================================


class UnfoldModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(2, 3))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, input):
        return self.unfold(input)


@register_test_case(module_factory=lambda: UnfoldModule())
def UnfoldModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 3, 4))


# ==============================================================================


class AtenPolarFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(2, 3))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float32, True),
            ([-1, -1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, abs_, angle):
        return torch.ops.aten.polar(torch.ops.aten.abs(abs_), angle)


@register_test_case(module_factory=lambda: AtenPolarFloatModule())
def AtenPolarFloatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 3, 4), tu.rand(2, 5, 3, 4))


# ==============================================================================


class AtenPolarDoubleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(2, 3))

    @export
    @annotate_args(
        [
            None,
            ([-1, -1, -1, -1], torch.float64, True),
            ([-1, -1, -1, -1], torch.float64, True),
        ]
    )
    def forward(self, abs_, angle):
        return torch.ops.aten.polar(torch.ops.aten.abs(abs_), angle)


@register_test_case(module_factory=lambda: AtenPolarDoubleModule())
def AtenPolarDoubleModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 5, 3, 4).to(torch.float64), tu.rand(2, 5, 3, 4).to(torch.float64)
    )


# ==============================================================================


class AtenNonzero1DDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args(
        [
            None,
            ([-1], torch.bool, True),
        ]
    )
    def forward(self, x):
        return torch.ops.aten.nonzero(x)


@register_test_case(module_factory=lambda: AtenNonzero1DDynamicModule())
def AtenNonzero1DDynamicModule_basic(module, tu: TestUtils):
    module.forward(torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool))


# ==============================================================================


class AtenSymConstrainRange(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.int, True)])
    def forward(self, x):
        a = x.item()
        torch.ops.aten.sym_constrain_range(a, max=5)
        return a


@register_test_case(module_factory=lambda: AtenSymConstrainRange())
def AtenSymConstrainRange_basic(module, tu: TestUtils):
    module.forward(torch.tensor(4))


# ==============================================================================


class AtenSymConstrainRangeForSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.int, True)])
    def forward(self, x):
        a = x.item()
        torch.ops.aten.sym_constrain_range_for_size(a, min=0, max=10)
        return a


@register_test_case(module_factory=lambda: AtenSymConstrainRangeForSize())
def AtenSymConstrainRangeForSize_basic(module, tu: TestUtils):
    module.forward(torch.tensor(4))


# ==============================================================================
class Aten_AssertScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([None, ([-1], torch.int, True)])
    def forward(self, x):
        a = x.item()
        assert_msg = "Assertion failed for condition x.item() > 3"
        torch.ops.aten._assert_scalar(a > 3, assert_msg)
        return a


@register_test_case(module_factory=lambda: Aten_AssertScalar())
def Aten_AssertScalar_basic(module, tu: TestUtils):
    module.forward(torch.tensor(4))
