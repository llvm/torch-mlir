# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

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
    module.forward(torch.randint(10, (3, 5)))


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
    module.forward(torch.randint(10, (3, 5)))


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
    module.forward(torch.randint(10, (3, 5)))


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

class Fill_TensorFloat64WithFloat32(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3.0)

@register_test_case(module_factory=lambda: Fill_TensorFloat64WithFloat32())
def Fill_TensorFloat64WithFloat32_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4))


class Fill_TensorFloat64WithFloat64(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3.0)

@register_test_case(module_factory=lambda: Fill_TensorFloat64WithFloat64())
def Fill_TensorFloat64WithFloat64_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4).to(torch.float64))


class Fill_TensorFloat64WithInt64(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float64, True),
    ])
    def forward(self, tensor):
        return torch.ops.aten.fill_(tensor, 3)

@register_test_case(module_factory=lambda: Fill_TensorFloat64WithInt64())
def Fill_TensorFloat64WithInt64_basic(module, tu: TestUtils):
    module.forward(torch.randn(3, 2, 4).to(torch.float64))


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
    module.forward(torch.randint(10, (2, 3, 4)))


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
    module.forward(torch.randint(10, (2, 3)))


class NewZerosModuleFalsePinMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_zeros(a, [3, 4], dtype=torch.float32, pin_memory=False)

@register_test_case(module_factory=lambda: NewZerosModuleFalsePinMemory())
def NewZerosModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (2, 3)))

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
    module.forward(torch.randint(10, (2, 3, 4)))


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
    module.forward(torch.randint(10, (2, 3)))


class NewOnesModuleFalsePinMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, a):
        return torch.ops.aten.new_ones(a, [3, 4], dtype=torch.float32, pin_memory=False)

@register_test_case(module_factory=lambda: NewOnesModuleFalsePinMemory())
def NewOnesModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (2, 3)))

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
        return torch.ops.aten.full([2, 3], 5.0, dtype=torch.int64, pin_memory=False)

@register_test_case(module_factory=lambda: FullModuleFalsePinMemory())
def FullModuleFalsePinMemory_basic(module, tu: TestUtils):
    module.forward()
