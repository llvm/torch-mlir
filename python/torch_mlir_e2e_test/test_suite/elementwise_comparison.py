# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class ElementwiseGtFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.gt(x, 0.6)


@register_test_case(module_factory=lambda: ElementwiseGtFloatScalarModule())
def ElementwiseGtFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseGtIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.gt(x, 10)


@register_test_case(module_factory=lambda: ElementwiseGtIntScalarModule())
def ElementwiseGtIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15))

# ==============================================================================

class ElementwiseGtMixed2ScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.gt(x, 7)


@register_test_case(module_factory=lambda: ElementwiseGtMixed2ScalarModule())
def ElementwiseGtMixed2ScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15).to(torch.int32))

# ==============================================================================

class ElementwiseGeFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ge(x, 0.6)


@register_test_case(module_factory=lambda: ElementwiseGeFloatScalarModule())
def ElementwiseGeFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseGeIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ge(x, 10)


@register_test_case(module_factory=lambda: ElementwiseGeIntScalarModule())
def ElementwiseGeIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15))

# ==============================================================================

class ElementwiseGeMixedIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.ge(x, 7)


@register_test_case(module_factory=lambda: ElementwiseGeMixedIntScalarModule())
def ElementwiseGeMixedIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15).to(torch.int32))

# ==============================================================================

class ElementwiseGeFloatIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ge(x, 7)


@register_test_case(module_factory=lambda: ElementwiseGeFloatIntScalarModule())
def ElementwiseGeFloatIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseGtFloatTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.gt(x, y)


@register_test_case(module_factory=lambda: ElementwiseGtFloatTensorModule())
def ElementwiseGtFloatTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(5))

# ==============================================================================

class ElementwiseGtIntTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.gt(x, y)


@register_test_case(module_factory=lambda: ElementwiseGtIntTensorModule())
def ElementwiseGtIntTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10), tu.randint(5, high=10))

# ==============================================================================

class ElementwiseLtFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.lt(x, 0.6)


@register_test_case(module_factory=lambda: ElementwiseLtFloatScalarModule())
def ElementwiseLtFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseLtIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.lt(x, 0)


@register_test_case(module_factory=lambda: ElementwiseLtIntScalarModule())
def ElementwiseLtIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15))

# ==============================================================================

class ElementwiseLtDiffWidthScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.lt(x, 2)


@register_test_case(
    module_factory=lambda: ElementwiseLtDiffWidthScalarModule())
def ElementwiseLtDiffWidthScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15).to(torch.int32))

# ==============================================================================

class ElementwiseLeFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.le(x, 0.6)


@register_test_case(module_factory=lambda: ElementwiseLeFloatScalarModule())
def ElementwiseLeFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseLeIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.le(x, 10)


@register_test_case(module_factory=lambda: ElementwiseLeIntScalarModule())
def ElementwiseLeIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15))

# ==============================================================================

class ElementwiseLeMixedIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.le(x, 7)


@register_test_case(module_factory=lambda: ElementwiseLeMixedIntScalarModule())
def ElementwiseLeMixedIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 4, low=-10, high=15).to(torch.int32))

# ==============================================================================

class ElementwiseLeFloatIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.le(x, 7)


@register_test_case(module_factory=lambda: ElementwiseLeFloatIntScalarModule())
def ElementwiseLeFloatIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================

class ElementwiseLtFloatTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.lt(x, y)


@register_test_case(module_factory=lambda: ElementwiseLtFloatTensorModule())
def ElementwiseLtFloatTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5), tu.rand(5))

# ==============================================================================

class ElementwiseLtIntTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.lt(x, y)


@register_test_case(module_factory=lambda: ElementwiseLtIntTensorModule())
def ElementwiseLtIntTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(3, 5, high=10), tu.randint(5, high=10))

# ==============================================================================

class ElementwiseEqFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.eq(x, 6.0)


@register_test_case(module_factory=lambda: ElementwiseEqFloatScalarModule())
def ElementwiseEqFloatScalarModule_basic(module, tu: TestUtils):
    module.forward(
        torch.tensor([[1.0, 2.2, 6.0], [6.0, 2.0, 3.1]]).to(torch.float32))

# ==============================================================================

class ElementwiseEqIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.eq(x, 2)


@register_test_case(module_factory=lambda: ElementwiseEqIntScalarModule())
def ElementwiseEqIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5, 8, low=2, high=4))

# ==============================================================================

class ElementwiseEqDiffWidthScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int32, True),
    ])
    def forward(self, x):
        return torch.eq(x, 2)


@register_test_case(
    module_factory=lambda: ElementwiseEqDiffWidthScalarModule())
def ElementwiseEqDiffWidthScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(5, 8, low=2, high=4).to(torch.int32))

# ==============================================================================

class ElementwiseEqFloatTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, y):
        return torch.eq(x, y)


@register_test_case(module_factory=lambda: ElementwiseEqFloatTensorModule())
def ElementwiseEqFloatTensorModule_basic(module, tu: TestUtils):
    module.forward(
        torch.tensor([[1.0, 2.2, 6.0], [6.0, 2.0, 3.1]]).to(torch.float32),
        torch.tensor([1.0, 2.4, 6.0]).to(torch.float32))

# ==============================================================================

class ElementwiseEqIntTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1], torch.int64, True),
    ])
    def forward(self, x, y):
        return torch.eq(x, y)


@register_test_case(module_factory=lambda: ElementwiseEqIntTensorModule())
def ElementwiseEqIntTensorModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(8, 5, low=2, high=4), tu.randint(5, low=2, high=4))

# ==============================================================================

class ElementwiseNeFloatScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ne(x, 2.0)


@register_test_case(module_factory=lambda: ElementwiseNeFloatScalarModule())
def ElementwiseNeFloatTensorModule_basic(module, tu: TestUtils):
    module.forward(
        torch.tensor([[1.0, 2.2, 2.0], [6.0, 2.0, 3.1]]).to(torch.float32))

# ==============================================================================

class ElementwiseNeIntScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])
    def forward(self, x):
        return torch.ne(x, 3)


@register_test_case(module_factory=lambda: ElementwiseNeIntScalarModule())
def ElementwiseNeIntScalarModule_basic(module, tu: TestUtils):
    module.forward(tu.randint(8, 5, low=2, high=4))

# ==============================================================================

class AnyBoolTrueModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        input = [False, False, True]
        return torch.ops.aten.any(input)


@register_test_case(module_factory=lambda: AnyBoolTrueModule())
def AnyBoolTrueModule_basic(module, tu: TestUtils):
    module.forward()


class AnyBoolFalseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        input = [False, False, False]
        return torch.ops.aten.any(input)


@register_test_case(module_factory=lambda: AnyBoolFalseModule())
def AnyBoolFalseModule_basic(module, tu: TestUtils):
    module.forward()

# =================================================================================

class AllBoolTrueModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        input = [True, True, True, True, True]
        return torch.ops.aten.all(input)


@register_test_case(module_factory=lambda: AllBoolTrueModule())
def AllBoolTrueModule_basic(module, tu: TestUtils):
    module.forward()

# =================================================================================

class AllBoolFalseModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
    ])
    def forward(self):
        input = [True, False, True, True, False]
        return torch.ops.aten.all(input)
        
@register_test_case(module_factory=lambda: AllBoolFalseModule())
def AllBoolFalseModule_basic(module, tu: TestUtils):
    module.forward()
