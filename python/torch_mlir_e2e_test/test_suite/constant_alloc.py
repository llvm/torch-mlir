# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class ZerosModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4)


@register_test_case(module_factory=lambda: ZerosModuleDefaultDtype())
def ZerosModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward()


class ZerosModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4, dtype=torch.int64)


@register_test_case(module_factory=lambda: ZerosModuleInt2D())
def ZerosModuleInt2D_basic(module, tu: TestUtils):
    module.forward()


class ZerosModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4, 5, dtype=torch.int64)


@register_test_case(module_factory=lambda: ZerosModuleInt3D())
def ZerosModuleInt3D_basic(module, tu: TestUtils):
    module.forward()


class ZerosModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4, dtype=torch.float32)


@register_test_case(module_factory=lambda: ZerosModuleFloat2D())
def ZerosModuleFloat2D_basic(module, tu: TestUtils):
    module.forward()


class ZerosModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4, 5, dtype=torch.float32)


@register_test_case(module_factory=lambda: ZerosModuleFloat3D())
def ZerosModuleFloat3D_basic(module, tu: TestUtils):
    module.forward()


class ZerosModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.zeros(3, 4, dtype=torch.float32, pin_memory=False)


@register_test_case(module_factory=lambda: ZerosModuleFalsePinMemory())
def ZerosModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class OnesModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ones(3, 4)


@register_test_case(module_factory=lambda: OnesModuleDefaultDtype())
def OnesModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward()


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


class OnesModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ones(3, 4, dtype=torch.float32, pin_memory=False)


@register_test_case(module_factory=lambda: OnesModuleFalsePinMemory())
def OnesModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward()


class OnesModuleCPUDevice(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ones(3, 4, device="cpu")


@register_test_case(module_factory=lambda: OnesModuleCPUDevice())
def OnesModuleCPUDevice_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class EmptyContiguousModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.empty((3, 4),
                           memory_format=torch.contiguous_format).fill_(0)


@register_test_case(module_factory=lambda: EmptyContiguousModule())
def EmptyModule_contiguous(module, tu: TestUtils):
    module.forward()


class EmptyDefaultDtypeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.empty((3, 4)).fill_(0)


@register_test_case(module_factory=lambda: EmptyDefaultDtypeModule())
def EmptyModule_defaultDtype(module, tu: TestUtils):
    module.forward()


class EmptyIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.empty((3, 4), dtype=torch.int64).fill_(0)


@register_test_case(module_factory=lambda: EmptyIntModule())
def EmptyModule_int(module, tu: TestUtils):
    module.forward()


class EmptyFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.empty((3, 4), dtype=torch.float32).fill_(0)


@register_test_case(module_factory=lambda: EmptyFloatModule())
def EmptyModule_float(module, tu: TestUtils):
    module.forward()


class EmptyFalsePinMemoryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.empty((3, 4), dtype=torch.float32,
                           pin_memory=False).fill_(0)


@register_test_case(module_factory=lambda: EmptyFalsePinMemoryModule())
def EmptyModule_falsePinMemory(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class EmptyLikeDefaultDtypeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.empty_like(a).fill_(0)


@register_test_case(module_factory=lambda: EmptyLikeDefaultDtypeModule())
def EmptyLikeModule_defaultDtype(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


class EmptyLikeIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.empty_like(a, dtype=torch.int32).fill_(0)


@register_test_case(module_factory=lambda: EmptyLikeIntModule())
def EmptyLikeModule_int(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10))


class EmptyLikeMemoryFormatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.empty_like(a,
                                memory_format=torch.preserve_format).fill_(0)


@register_test_case(module_factory=lambda: EmptyLikeMemoryFormatModule())
def EmptyLikeMemoryFormatModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5, 2, 1))


class EmptyLikeFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.empty_like(a, dtype=torch.float32).fill_(0)


@register_test_case(module_factory=lambda: EmptyLikeFloatModule())
def EmptyLikeModule_float(module, tu: TestUtils):
    module.forward(tu.rand(4, 5))


class EmptyLikeFalsePinMemoryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.empty_like(a, dtype=torch.float64,
                                pin_memory=False).fill_(0)


@register_test_case(module_factory=lambda: EmptyLikeFalsePinMemoryModule())
def EmptyLikeModule_falsePinMemory(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class ZerosLikeDefaultDtypeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.zeros_like(a)


@register_test_case(module_factory=lambda: ZerosLikeDefaultDtypeModule())
def ZerosLikeModule_defaultDtype(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


class ZerosLikeIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.zeros_like(a, dtype=torch.int32)


@register_test_case(module_factory=lambda: ZerosLikeIntModule())
def ZerosLikeModule_int(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10))


class ZerosLikeFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.zeros_like(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: ZerosLikeFloatModule())
def ZerosLikeModule_float(module, tu: TestUtils):
    module.forward(tu.rand(4, 5))


class ZerosLikeFalsePinMemoryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.zeros_like(a, dtype=torch.float64, pin_memory=False)


@register_test_case(module_factory=lambda: ZerosLikeFalsePinMemoryModule())
def ZerosLikeModule_falsePinMemory(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class OnesLikeDefaultDtypeModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ones_like(a)


@register_test_case(module_factory=lambda: OnesLikeDefaultDtypeModule())
def OnesLikeModule_defaultDtype(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))


class OnesLikeIntModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ones_like(a, dtype=torch.int32)


@register_test_case(module_factory=lambda: OnesLikeIntModule())
def OnesLikeModule_int(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10))


class OnesLikeFloatModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ones_like(a, dtype=torch.float32)


@register_test_case(module_factory=lambda: OnesLikeFloatModule())
def OnesLikeModule_float(module, tu: TestUtils):
    module.forward(tu.rand(4, 5))


class OnesLikeFalsePinMemoryModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ones_like(a, dtype=torch.float64, pin_memory=False)


@register_test_case(module_factory=lambda: OnesLikeFalsePinMemoryModule())
def OnesLikeModule_falsePinMemory(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


# ==============================================================================


class NewZerosModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4])


@register_test_case(module_factory=lambda: NewZerosModuleDefaultDtype())
def NewZerosModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewZerosModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4], dtype=torch.int64)


@register_test_case(module_factory=lambda: NewZerosModuleInt2D())
def NewZerosModuleInt2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


class NewZerosModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4, 5], dtype=torch.int64)


@register_test_case(module_factory=lambda: NewZerosModuleInt3D())
def NewZerosModuleInt3D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewZerosModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4], dtype=torch.float32)


@register_test_case(module_factory=lambda: NewZerosModuleFloat2D())
def NewZerosModuleFloat2D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, high=10))


class NewZerosModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4, 5], dtype=torch.float32)


@register_test_case(module_factory=lambda: NewZerosModuleFloat3D())
def NewZerosModuleFloat3D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


class NewZerosModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4],
                                        dtype=torch.float32,
                                        pin_memory=False)


@register_test_case(module_factory=lambda: NewZerosModuleFalsePinMemory())
def NewZerosModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


# ==============================================================================


class NewOnesModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4])


@register_test_case(module_factory=lambda: NewOnesModuleDefaultDtype())
def NewOnesModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewOnesModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4], dtype=torch.int64)


@register_test_case(module_factory=lambda: NewOnesModuleInt2D())
def NewOnesModuleInt2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


class NewOnesModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4, 5], dtype=torch.int64)


@register_test_case(module_factory=lambda: NewOnesModuleInt3D())
def NewOnesModuleInt3D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewOnesModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4], dtype=torch.float32)


@register_test_case(module_factory=lambda: NewOnesModuleFloat2D())
def NewOnesModuleFloat2D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, high=10))


class NewOnesModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4, 5], dtype=torch.float32)


@register_test_case(module_factory=lambda: NewOnesModuleFloat3D())
def NewOnesModuleFloat3D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


class NewOnesModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4],
                                       dtype=torch.float32,
                                       pin_memory=False)


@register_test_case(module_factory=lambda: NewOnesModuleFalsePinMemory())
def NewOnesModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


# ==============================================================================


class FullModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([2, 3], 5.0)


@register_test_case(module_factory=lambda: FullModuleDefaultDtype())
def FullModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward()


class FullModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([10, 5], 10.5, dtype=torch.int64)


@register_test_case(module_factory=lambda: FullModuleInt2D())
def FullModuleInt2D_basic(module, tu: TestUtils):
    module.forward()


class FullModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([2, 3, 4], 5)


@register_test_case(module_factory=lambda: FullModuleInt3D())
def FullModuleInt3D_basic(module, tu: TestUtils):
    module.forward()


class FullModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([10, 5], 10, dtype=torch.float32)


@register_test_case(module_factory=lambda: FullModuleFloat2D())
def FullModuleFloat2D_basic(module, tu: TestUtils):
    module.forward()


class FullModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([2, 3, 4], 5.0)


@register_test_case(module_factory=lambda: FullModuleFloat3D())
def FullModuleFloat3D_basic(module, tu: TestUtils):
    module.forward()


class FullModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        return torch.ops.aten.full([2, 3],
                                   5.0,
                                   dtype=torch.int64,
                                   pin_memory=False)


@register_test_case(module_factory=lambda: FullModuleFalsePinMemory())
def FullModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward()


# ==============================================================================


class FullLikeModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 5)


@register_test_case(module_factory=lambda: FullLikeModuleDefaultDtype())
def FullLikeModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class FullLikeModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 10.5)


@register_test_case(module_factory=lambda: FullLikeModuleInt2D())
def FullLikeModuleInt2D_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, high=10))


class FullLikeModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 5.0, dtype=torch.int64)


@register_test_case(module_factory=lambda: FullLikeModuleInt3D())
def FullLikeModuleInt3D_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 4, 5, high=100).to(torch.int32))


class FullLikeModuleInt2DStatic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([4, 5], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 10)


@register_test_case(module_factory=lambda: FullLikeModuleInt2DStatic())
def FullLikeModuleInt2DStatic_basic(module, tu: TestUtils):
    module.forward(tu.randint(4, 5, high=10))


class FullLikeModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 10)


@register_test_case(module_factory=lambda: FullLikeModuleFloat2D())
def FullLikeModuleFloat2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


class FullLikeModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 15, dtype=torch.float32)


@register_test_case(module_factory=lambda: FullLikeModuleFloat3D())
def FullLikeModuleFloat3D_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


class FullLikeModuleFloat3DStatic(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([3, 4, 5], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a, 15.3, dtype=torch.float32)


@register_test_case(module_factory=lambda: FullLikeModuleFloat3DStatic())
def FullLikeModuleFloat3DStatic_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5).to(torch.float64))


class FullLikeModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.full_like(a,
                                        5,
                                        dtype=torch.int64,
                                        pin_memory=False)


@register_test_case(module_factory=lambda: FullLikeModuleFalsePinMemory())
def FullLikeModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 4, high=100))


# ==============================================================================


class ZeroFloat32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.zero_(tensor)


@register_test_case(module_factory=lambda: ZeroFloat32Module())
def ZeroFloat32Module_basic(module, tu: TestUtils):
    module.forward(torch.rand(3, 2))


class ZeroInt32Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.zero_(tensor)


@register_test_case(module_factory=lambda: ZeroInt32Module())
def ZeroInt32Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 4, high=100).to(dtype=torch.int32))


class ZeroInt64Module(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.zero_(tensor)


@register_test_case(module_factory=lambda: ZeroInt64Module())
def ZeroInt64Module_basic(module, tu: TestUtils):
    module.forward(tu.randint(10, 4, high=100))


# ==============================================================================


class NewEmptyModuleDefaultDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4]).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleDefaultDtype())
def NewEmptyModuleDefaultDtype_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewEmptyModuleInt2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4], dtype=torch.int64).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleInt2D())
def NewEmptyModuleInt2D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4))


class NewEmptyModuleInt3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4, 5],
                                        dtype=torch.int64).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleInt3D())
def NewEmptyModuleInt3D_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3))


class NewEmptyModuleFloat2D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4],
                                        dtype=torch.float32).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleFloat2D())
def NewEmptyModuleFloat2D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, 4, high=10))


class NewEmptyModuleFloat3D(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4, 5],
                                        dtype=torch.float32).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleFloat3D())
def NewEmptyModuleFloat3D_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


class NewEmptyModuleFalsePinMemory(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4],
                                        dtype=torch.float32,
                                        pin_memory=False).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleFalsePinMemory())
def NewEmptyModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10))


class NewEmptyModuleNonDefaultFloatDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4]).fill_(0)


@register_test_case(
    module_factory=lambda: NewEmptyModuleNonDefaultFloatDtype())
def NewEmptyModuleNonDefaultFloatDtype_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3).to(torch.float64))


class NewEmptyModuleNonDefaultIntDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4]).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleNonDefaultIntDtype())
def NewEmptyModuleNonDefaultIntDtype_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10).to(torch.int32))


class NewEmptyModuleLayoutIntDtype(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_empty(a, [3, 4], layout=0).fill_(0)


@register_test_case(module_factory=lambda: NewEmptyModuleLayoutIntDtype())
def NewEmptyModuleLayoutIntDtype_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, high=10).to(torch.int32))


# ==============================================================================


class MaskedFillScalarDefaultModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x, mask):
        return torch.ops.aten.masked_fill(x, mask, value=0.5)


@register_test_case(module_factory=lambda: MaskedFillScalarDefaultModule())
def MaskedFillScalarDefaultModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3),
                   tu.randint(2, 3, high=2).to(dtype=torch.bool))


class MaskedFillScalarIntValueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x, mask):
        return torch.ops.aten.masked_fill(x, mask, value=5)


@register_test_case(module_factory=lambda: MaskedFillScalarIntValueModule())
def MaskedFillScalarIntValueModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3),
                   tu.randint(2, 3, high=2).to(dtype=torch.bool))


class MaskedFillScalarFloatValueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.bool, True),
    ])
    def forward(self, x, mask):
        return torch.ops.aten.masked_fill(x, mask, value=-0.01)


@register_test_case(module_factory=lambda: MaskedFillScalarFloatValueModule())
def MaskedFillScalarFloatValueModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, low=-10, high=10),
                   tu.randint(2, 3, high=2).to(dtype=torch.bool))


class MaskedFillTensorFloatValueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.bool, True),
        ([], torch.float32, True),
    ])
    def forward(self, x, mask, value):
        return torch.ops.aten.masked_fill(x, mask, value=value)


@register_test_case(module_factory=lambda: MaskedFillTensorFloatValueModule())
def MaskedFillTensorFloatValueModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(2, 3, low=-10, high=10),
                   tu.randint(2, 3, high=2).to(dtype=torch.bool), tu.rand())
