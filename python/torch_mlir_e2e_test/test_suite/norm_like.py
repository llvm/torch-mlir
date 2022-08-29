# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

class BatchNorm1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1d = torch.nn.BatchNorm1d(4)
        self.bn1d.eval()
        self.bn1d.running_mean = torch.tensor([0.5, 0.4, 0.3, 0.6])
        self.bn1d.running_var = torch.tensor([3.0, 2.0, 4.0, 5.0])
        self.bn1d.weight = torch.nn.Parameter(
            torch.tensor([3.0, 2.0, 4.0, 5.0]))
        self.bn1d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4, 0.3, 0.6]))

    @export
    @annotate_args([
        None,
        ([10, 4, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn1d(x)


@register_test_case(module_factory=lambda: BatchNorm1DModule())
def BatchNorm1DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4, 3))

# ==============================================================================

class BatchNorm1DWith2DInputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1d = torch.nn.BatchNorm1d(4)
        self.bn1d.eval()
        self.bn1d.running_mean = torch.tensor([0.5, 0.4, 0.3, 0.6])
        self.bn1d.running_var = torch.tensor([3.0, 2.0, 4.0, 5.0])
        self.bn1d.weight = torch.nn.Parameter(
            torch.tensor([3.0, 2.0, 4.0, 5.0]))
        self.bn1d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4, 0.3, 0.6]))

    @export
    @annotate_args([
        None,
        ([10, 4], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn1d(x)


@register_test_case(module_factory=lambda: BatchNorm1DWith2DInputModule())
def BatchNorm1DWith2DInputModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 4))

# ==============================================================================

class BatchNorm2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn2d = torch.nn.BatchNorm2d(2)
        self.bn2d.eval()
        self.bn2d.running_mean = torch.tensor([0.5, 0.4])
        self.bn2d.running_var = torch.tensor([3.0, 2.0])
        self.bn2d.weight = torch.nn.Parameter(torch.tensor([3.0, 2.0]))
        self.bn2d.bias = torch.nn.Parameter(torch.tensor([0.5, 0.4]))

    @export
    @annotate_args([
        None,
        ([10, 2, 3, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn2d(x)


@register_test_case(module_factory=lambda: BatchNorm2DModule())
def BatchNorm2DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 2, 3, 3))

# ==============================================================================

class BatchNorm3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn3d = torch.nn.BatchNorm3d(5)
        self.bn3d.eval()
        self.bn3d.running_mean = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.4])
        self.bn3d.running_var = torch.tensor([3.0, 2.0, 4.0, 2.0, 3.0])
        self.bn3d.weight = torch.nn.Parameter(
            torch.tensor([3.0, 2.0, 4.0, 2.0, 3.0]))
        self.bn3d.bias = torch.nn.Parameter(
            torch.tensor([0.5, 0.4, 0.3, 0.2, 0.4]))

    @export
    @annotate_args([
        None,
        ([2, 5, 3, 6, 4], torch.float32, True),
    ])
    def forward(self, x):
        return self.bn3d(x)


@register_test_case(module_factory=lambda: BatchNorm3DModule())
def BatchNorm3DModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 3, 6, 4))

# ==============================================================================

class NativeBatchNorm1DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten.native_batch_norm(
            x, weight, bias, running_mean, running_var, training=False, 
            momentum=0.1, eps=0.00001)


@register_test_case(module_factory=lambda: NativeBatchNorm1DModule())
def NativeBatchNorm1DModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 5, 3), tu.rand(5), tu.rand(5), tu.rand(5), tu.rand(5))

# ==============================================================================

class NativeBatchNorm2DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten.native_batch_norm(
            x, weight, bias, running_mean, running_var, training=False, 
            momentum=0.1, eps=0.00001)


@register_test_case(module_factory=lambda: NativeBatchNorm2DModule())
def NativeBatchNorm2DModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 5, 2, 3), tu.rand(5), tu.rand(5), tu.rand(5), tu.rand(5))

# ==============================================================================

class NativeBatchNorm3DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten.native_batch_norm(
            x, weight, bias, running_mean, running_var, training=False, 
            momentum=0.1, eps=0.00001)


@register_test_case(module_factory=lambda: NativeBatchNorm3DModule())
def NativeBatchNorm3DModule_basic(module, tu: TestUtils):
    module.forward(
        tu.rand(2, 5, 2, 2, 3), tu.rand(5), tu.rand(5), tu.rand(5), tu.rand(5))

# ==============================================================================

class NativeBatchNormNoneWeightModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, x, bias, running_mean, running_var):
        return torch.ops.aten.native_batch_norm(
            x, None, bias, running_mean, running_var, training=False, 
            momentum=0.1, eps=0.00001)


@register_test_case(module_factory=lambda: NativeBatchNormNoneWeightModule())
def NativeBatchNormNoneWeightModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 2, 2, 3), tu.rand(5), tu.rand(5), tu.rand(5))

# ==============================================================================

class NativeLayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([2, 5, 2, 2, 3], torch.float32, True),
        ([2, 2, 3], torch.float32, True),
        ([2, 2, 3], torch.float32, True),
    ])
    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)


@register_test_case(module_factory=lambda: NativeLayerNormModule())
def NativeLayerNormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 2, 2, 3), tu.rand(2, 2, 3), tu.rand(2, 2, 3))

class NativeLayerNormDynamicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)


@register_test_case(module_factory=lambda: NativeLayerNormDynamicModule())
def NativeLayerNormDynamicModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 2, 2, 3), tu.rand(2, 2, 3), tu.rand(2, 2, 3))

# ==============================================================================

class NativeLayerNormModule4D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([5, 2, 2, 3], torch.float32, True),
        ([2, 2, 3], torch.float32, True),
        ([2, 2, 3], torch.float32, True),
    ])
    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)[0]


@register_test_case(module_factory=lambda: NativeLayerNormModule4D())
def NativeLayerNormModule4D_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 2, 2, 3), tu.rand(2, 2, 3), tu.rand(2, 2, 3))

# ==============================================================================

class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ly = torch.nn.LayerNorm([2, 2, 3])
        self.ly.eval()
        self.ly.weight = torch.nn.Parameter(
            torch.tensor([[[3.0, 2.0, 4.0], [2.0, 3.0, 3.0]],
                          [[3.0, 2.0, 4.0], [2.0, 3.0, 3.0]]]))
        self.ly.bias = torch.nn.Parameter(
            torch.tensor([[[0.5, 0.4, 0.3], [0.2, 0.4, 0.3]],
                          [[0.5, 0.4, 0.3], [0.2, 0.4, 0.3]]]))

    @export
    @annotate_args([
        None,
        ([2, 5, 2, 2, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.ly(x)


@register_test_case(module_factory=lambda: LayerNormModule())
def LayerNormModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 2, 2, 3))

# ==============================================================================

class LayerNormLastDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ly = torch.nn.LayerNorm([3])
        self.ly.eval()
        self.ly.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0, 2.0]))
        self.ly.bias = torch.nn.Parameter(torch.tensor([0.2, 0.4, 0.3]))

    @export
    @annotate_args([
        None,
        ([2, 5, 2, 2, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.ly(x)


@register_test_case(module_factory=lambda: LayerNormLastDimModule())
def LayerNormLastDimModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 5, 2, 2, 3))

# ==============================================================================

class LayerNormNormalizeOverAllDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ly = torch.nn.LayerNorm([2, 2, 3])
        self.ly.eval()
        self.ly.weight = torch.nn.Parameter(
            torch.tensor([[[3.0, 2.0, 4.0], [2.0, 3.0, 3.0]],
                          [[3.0, 2.0, 4.0], [2.0, 3.0, 3.0]]]))
        self.ly.bias = torch.nn.Parameter(
            torch.tensor([[[0.5, 0.4, 0.3], [0.2, 0.4, 0.3]],
                          [[0.5, 0.4, 0.3], [0.2, 0.4, 0.3]]]))

    @export
    @annotate_args([
        None,
        ([2, 2, 3], torch.float32, True),
    ])
    def forward(self, x):
        return self.ly(x)


@register_test_case(module_factory=lambda: LayerNormNormalizeOverAllDimsModule())
def LayerNormNormalizeOverAllDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 2, 3))
