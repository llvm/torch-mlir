# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================


class Threshold1dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 1, 2)

@register_test_case(module_factory=lambda: Threshold1dIntModule())
def Threshold1dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4,)))


class Threshold2dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 0.5, 2)

@register_test_case(module_factory=lambda: Threshold2dIntModule())
def Threshold2dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4, 5)))


class Threshold3dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 1, 2.2)

@register_test_case(module_factory=lambda: Threshold3dIntModule())
def Threshold3dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4, 5, 6)))


class Threshold1dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 1, 2)

@register_test_case(module_factory=lambda: Threshold1dFloatModule())
def Threshold1dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4))


class Threshold2dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 0.5, 2)

@register_test_case(module_factory=lambda: Threshold2dFloatModule())
def Threshold2dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5))


class Threshold3dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])

    def forward(self, input):
        return torch.ops.aten.threshold(input, 1.4, 2.0)

@register_test_case(module_factory=lambda: Threshold3dFloatModule())
def Threshold3dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6))


class ThresholdBackward1dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.int64, True),
        ([-1], torch.int64, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1)

@register_test_case(module_factory=lambda: ThresholdBackward1dIntModule())
def ThresholdBackward1dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4,)), torch.randint(8, (4,)))


class ThresholdBackward2dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.int64, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 0.5)

@register_test_case(module_factory=lambda: ThresholdBackward2dIntModule())
def ThresholdBackward2dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4, 5)), torch.randint(8, (4, 5)))


class ThresholdBackward3dIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.int64, True),
        ([-1, -1, -1], torch.int64, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1)

@register_test_case(module_factory=lambda: ThresholdBackward3dIntModule())
def ThresholdBackward3dIntModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(10, (4, 5, 6)), torch.randint(8, (4, 5, 6)))


class ThresholdBackward1dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1)

@register_test_case(module_factory=lambda: ThresholdBackward1dFloatModule())
def ThresholdBackward1dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4), torch.randn(4))


class ThresholdBackward2dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 0.5)

@register_test_case(module_factory=lambda: ThresholdBackward2dFloatModule())
def ThresholdBackward2dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5), torch.randn(4, 5))


class ThresholdBackward3dFloatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1.4)

@register_test_case(module_factory=lambda: ThresholdBackward3dFloatModule())
def ThresholdBackward3dFloatModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.randn(4, 5, 6))


class ThresholdBackward1dMixedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([-1], torch.int64, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1)

@register_test_case(module_factory=lambda: ThresholdBackward1dMixedModule())
def ThresholdBackward1dMixedModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4), torch.randint(10, (4,)))


class ThresholdBackward2dMixedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.int64, True),
        ([-1, -1], torch.float32, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 0.5)

@register_test_case(module_factory=lambda: ThresholdBackward2dMixedModule())
def ThresholdBackward2dMixedModule_basic(module, tu: TestUtils):
    module.forward(torch.randint(20, (4, 5)), torch.randn(4, 5))


class ThresholdBackward3dMixedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.int64, True),
    ])

    def forward(self, grad, input):
        return torch.ops.aten.threshold_backward(grad, input, 1.4)

@register_test_case(module_factory=lambda: ThresholdBackward3dMixedModule())
def ThresholdBackward3dMixedModule_basic(module, tu: TestUtils):
    module.forward(torch.randn(4, 5, 6), torch.randint(10, (4, 5, 6)))
